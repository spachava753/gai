package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

type openCodeCapturedRequest struct {
	Method        string
	Path          string
	Authorization string
	SessionID     string
	Body          map[string]any
}

func TestJSONClientOpenCode(t *testing.T) {
	requests := make(chan openCodeCapturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		requests <- openCodeCapturedRequest{
			Method:        r.Method,
			Path:          r.URL.Path,
			Authorization: r.Header.Get("Authorization"),
			SessionID:     r.Header.Get("X-OpenCode-Session"),
			Body:          body,
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"completion_1",
			"object":"chat.completion",
			"created":42,
			"model":"glm-5.3-flash",
			"system_fingerprint":"fp_1",
			"cost":"0",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"reasoning_content":"new reasoning",
					"content":"answer",
					"tool_calls":[{
						"id":"call_2",
						"type":"function",
						"function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}
					}]
				},
				"finish_reason":"tool_calls"
			}],
			"usage":{
				"prompt_tokens":11,
				"completion_tokens":7,
				"total_tokens":18,
				"prompt_tokens_details":{"cached_tokens":3,"cache_creation_input_tokens":2},
				"completion_tokens_details":{"reasoning_tokens":4}
			}
		}`))
	}))
	defer server.Close()

	if OpenCodeDefaultBaseURL != "https://opencode.ai/zen/go/v1" {
		t.Fatalf("OpenCodeDefaultBaseURL = %q", OpenCodeDefaultBaseURL)
	}
	generator, err := NewOpenCodeGenerator(server.Client(), server.URL, "test-key")
	if err != nil {
		t.Fatalf("NewOpenCodeGenerator: %v", err)
	}
	schema, err := GenerateSchema[struct {
		City string `json:"city"`
	}]()
	if err != nil {
		t.Fatalf("GenerateSchema: %v", err)
	}
	priorToolCall, err := ToolCallBlock("call_1", "weather", map[string]any{"city": "London"})
	if err != nil {
		t.Fatalf("ToolCallBlock: %v", err)
	}
	priorReasoning := Block{
		BlockType:    Thinking,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str("prior reasoning"),
	}

	response, err := generator.Generate(context.Background(), GenerationRequest{
		Model:        "glm-5.3-flash",
		Instructions: SystemMessage(TextBlock("Be concise.")),
		Dialog: Dialog{
			{Role: User, Blocks: []Block{TextBlock("Describe this image."), ImageBlock([]byte{1, 2, 3}, "image/png")}},
			{Role: Assistant, Blocks: []Block{priorReasoning, priorToolCall}},
			{Role: ToolResult, Blocks: []Block{{ID: "call_1", BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("sunny")}}},
			{Role: User, Blocks: []Block{TextBlock("Continue.")}},
		},
		Tools: []Tool{{Name: "weather", Description: "Get weather by city.", InputSchema: schema}},
		Options: NewGenerationOptions(
			WithThinkingBudget("xhigh"),
			WithMaxGenerationTokens(321),
			WithToolChoice("weather"),
			WithOpenCodeSessionID("session_1"),
		),
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	captured := <-requests
	if captured.Method != http.MethodPost || captured.Path != "/chat/completions" {
		t.Fatalf("request = %s %s", captured.Method, captured.Path)
	}
	if captured.Authorization != "Bearer test-key" {
		t.Fatalf("Authorization = %q", captured.Authorization)
	}
	if captured.SessionID != "session_1" {
		t.Fatalf("X-OpenCode-Session = %q", captured.SessionID)
	}
	if captured.Body["model"] != "glm-5.3-flash" {
		t.Fatalf("model = %#v", captured.Body["model"])
	}
	if captured.Body["stream"] != false || captured.Body["reasoning_effort"] != "xhigh" || captured.Body["max_tokens"] != float64(321) {
		t.Fatalf("request options = %#v", captured.Body)
	}

	messages, ok := captured.Body["messages"].([]any)
	if !ok || len(messages) != 5 {
		t.Fatalf("messages = %#v", captured.Body["messages"])
	}
	userMessage := messages[1].(map[string]any)
	parts, ok := userMessage["content"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("multimodal content = %#v", userMessage["content"])
	}
	imagePart := parts[1].(map[string]any)
	imageURL := imagePart["image_url"].(map[string]any)["url"]
	if imagePart["type"] != "image_url" || imageURL != "data:image/png;base64,AQID" {
		t.Fatalf("image part = %#v", imagePart)
	}
	assistantMessage := messages[2].(map[string]any)
	if assistantMessage["reasoning_content"] != "prior reasoning" || assistantMessage["content"] != nil {
		t.Fatalf("assistant replay = %#v", assistantMessage)
	}
	toolChoice := captured.Body["tool_choice"].(map[string]any)
	if toolChoice["type"] != "function" || toolChoice["function"].(map[string]any)["name"] != "weather" {
		t.Fatalf("tool_choice = %#v", toolChoice)
	}
	tools := captured.Body["tools"].([]any)
	function := tools[0].(map[string]any)["function"].(map[string]any)
	properties := function["parameters"].(map[string]any)["properties"].(map[string]any)
	if _, ok := properties["city"]; !ok {
		t.Fatalf("tool parameters = %#v", function["parameters"])
	}

	if response.FinishReason != ToolUse || len(response.Candidates) != 1 {
		t.Fatalf("response finish/candidates = %q/%d", response.FinishReason, len(response.Candidates))
	}
	blocks := response.Candidates[0].Blocks
	if len(blocks) != 3 || blocks[0].BlockType != Thinking || blocks[0].Content.String() != "new reasoning" || blocks[1].Content.String() != "answer" || blocks[2].BlockType != ToolCall {
		t.Fatalf("response blocks = %#v", blocks)
	}
	if blocks[0].ExtraFields[ThinkingExtraFieldGeneratorKey] != ThinkingGeneratorOpenCode {
		t.Fatalf("thinking provenance = %#v", blocks[0].ExtraFields)
	}
	var toolInput ToolCallInput
	if err := json.Unmarshal([]byte(blocks[2].Content.String()), &toolInput); err != nil {
		t.Fatalf("decode tool call: %v", err)
	}
	if blocks[2].ID != "call_2" || toolInput.Name != "weather" || toolInput.Parameters["city"] != "Paris" {
		t.Fatalf("tool call = %#v %#v", blocks[2], toolInput)
	}
	if response.UsageMetadata[UsageMetricInputTokens] != 11 || response.UsageMetadata[UsageMetricGenerationTokens] != 7 || response.UsageMetadata[UsageMetricCacheReadTokens] != 3 || response.UsageMetadata[UsageMetricCacheWriteTokens] != 2 || response.UsageMetadata[UsageMetricReasoningTokens] != 4 {
		t.Fatalf("usage = %#v", response.UsageMetadata)
	}
	if response.ExtraFields[OpenCodeResponseExtraFieldID] != "completion_1" || response.ExtraFields[OpenCodeResponseExtraFieldModel] != "glm-5.3-flash" || response.ExtraFields[OpenCodeResponseExtraFieldCreated] != int64(42) || response.ExtraFields[OpenCodeResponseExtraFieldCost] != "0" {
		t.Fatalf("extra fields = %#v", response.ExtraFields)
	}
}

func TestSSEClientOpenCode(t *testing.T) {
	requests := make(chan openCodeCapturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		requests <- openCodeCapturedRequest{
			Method:        r.Method,
			Path:          r.URL.Path,
			Authorization: r.Header.Get("Authorization"),
			SessionID:     r.Header.Get("X-OpenCode-Session"),
			Body:          body,
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"id":"stream_1","created":43,"model":"glm-5.3-flash","choices":[{"index":0,"delta":{"reasoning":"plan ","reasoning_details":[{"type":"reasoning.text","text":"plan ","format":"unknown","index":0}]},"finish_reason":null}]}

`)
		_, _ = fmt.Fprint(w, `data: {"id":"stream_1","created":43,"model":"glm-5.3-flash","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}

`)
		_, _ = fmt.Fprint(w, `data: {"id":"stream_1","created":43,"model":"glm-5.3-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"weather","arguments":""}}]},"finish_reason":null}]}

`)
		_, _ = fmt.Fprint(w, `data: {"id":"stream_1","created":43,"model":"glm-5.3-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]}

`)
		_, _ = fmt.Fprint(w, `data: {"id":"stream_1","created":43,"model":"glm-5.3-flash","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":5,"cached_tokens":2,"completion_tokens_details":{"reasoning_tokens":3}}}

`)
		_, _ = fmt.Fprint(w, `data: {"id":"stream_1","created":43,"model":"glm-5.3-flash","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

`)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		_, _ = fmt.Fprint(w, `data: {"choices":[],"cost":"0"}

`)
	}))
	defer server.Close()

	generator, err := NewOpenCodeGenerator(server.Client(), server.URL, "test-key")
	if err != nil {
		t.Fatalf("NewOpenCodeGenerator: %v", err)
	}
	var chunks []StreamChunk
	for chunk := range generator.Stream(context.Background(), GenerationRequest{
		Model:  "glm-5.3-flash",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}},
	}) {
		chunks = append(chunks, chunk)
	}
	captured := <-requests
	if captured.Method != http.MethodPost || captured.Path != "/chat/completions" || captured.Authorization != "Bearer test-key" {
		t.Fatalf("request = %#v", captured)
	}
	streamOptions, ok := captured.Body["stream_options"].(map[string]any)
	if captured.Body["stream"] != true || !ok || streamOptions["include_usage"] != true {
		t.Fatalf("stream request = %#v", captured.Body)
	}
	if len(chunks) != 5 {
		t.Fatalf("chunks = %d: %#v", len(chunks), chunks)
	}
	for i, chunk := range chunks {
		if chunk.Err != nil {
			t.Fatalf("chunk %d error: %v", i, chunk.Err)
		}
	}
	if chunks[0].Block.BlockType != Thinking || chunks[0].Block.Content.String() != "plan " || chunks[0].Block.ExtraFields[ThinkingExtraFieldGeneratorKey] != ThinkingGeneratorOpenCode {
		t.Fatalf("reasoning chunk = %#v", chunks[0])
	}
	if chunks[0].Block.ExtraFields[OpenCodeExtraFieldReasoningField] != "reasoning_details" {
		t.Fatalf("reasoning field metadata = %#v", chunks[0].Block.ExtraFields)
	}
	if _, ok := chunks[0].Block.ExtraFields[OpenCodeExtraFieldReasoningDetail].(map[string]any); !ok {
		t.Fatalf("reasoning detail metadata = %#v", chunks[0].Block.ExtraFields)
	}
	replayed, err := buildOpenCodeAssistantMessage([]Block{chunks[0].Block})
	if err != nil {
		t.Fatalf("replay reasoning detail: %v", err)
	}
	if replayed.Reasoning.Or("") != "plan " || len(replayed.ReasoningDetails) != 1 || replayed.ReasoningDetails[0].Type != "reasoning.text" {
		t.Fatalf("replayed reasoning = %#v", replayed)
	}
	if chunks[1].Block.BlockType != Content || chunks[1].Block.Content.String() != "answer" {
		t.Fatalf("content chunk = %#v", chunks[1])
	}
	if chunks[2].Block.BlockType != ToolCall || chunks[2].Block.ID != "call_1" || chunks[2].Block.Content.String() != "weather" {
		t.Fatalf("tool-name chunk = %#v", chunks[2])
	}
	if chunks[3].Block.BlockType != ToolCall || chunks[3].Block.ID != "" || chunks[3].Block.Content.String() != `{"city":"Paris"}` {
		t.Fatalf("tool-arguments chunk = %#v", chunks[3])
	}
	if chunks[4].Block.BlockType != MetadataBlockType {
		t.Fatalf("terminal chunk = %#v", chunks[4])
	}
	var metadata struct {
		InputTokens      int `json:"input_tokens"`
		GenerationTokens int `json:"gen_tokens"`
		CacheReadTokens  int `json:"cache_read_tokens"`
		ReasoningTokens  int `json:"reasoning_tokens"`
	}
	if err := json.Unmarshal([]byte(chunks[4].Block.Content.String()), &metadata); err != nil {
		t.Fatalf("decode metadata: %v", err)
	}
	if metadata.InputTokens != 9 || metadata.GenerationTokens != 5 || metadata.CacheReadTokens != 2 || metadata.ReasoningTokens != 3 {
		t.Fatalf("metadata = %#v", metadata)
	}
	if chunks[4].ResponseExtraFields[OpenCodeResponseExtraFieldID] != "stream_1" || chunks[4].ResponseExtraFields[OpenCodeResponseExtraFieldModel] != "glm-5.3-flash" {
		t.Fatalf("stream extra fields = %#v", chunks[4].ResponseExtraFields)
	}
}

func TestErrorMappingOpenCode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{
			"type":"error",
			"error":{"type":"rate_limit_error","message":"slow down","code":429,"param":null},
			"metadata":{"provider":"upstream"}
		}`))
	}))
	defer server.Close()

	generator, err := NewOpenCodeGenerator(server.Client(), server.URL, "test-key")
	if err != nil {
		t.Fatalf("NewOpenCodeGenerator: %v", err)
	}
	_, err = generator.Generate(context.Background(), GenerationRequest{
		Model:  "glm-5.3-flash",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}},
	})
	var apiErr *ApiErr
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %T %v", err, err)
	}
	if apiErr.Provider != ProviderOpenCode || apiErr.Kind != APIErrorKindRateLimit || apiErr.StatusCode != http.StatusTooManyRequests || apiErr.Message != "slow down" || !apiErr.Retryable() {
		t.Fatalf("ApiErr = %#v", apiErr)
	}
	if apiErr.RawBody == "" {
		t.Fatal("ApiErr.RawBody is empty")
	}
}

func TestLiveOpenCode(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENCODE_API_KEY")
	generator, err := NewOpenCodeGenerator(nil, "", apiKey)
	if err != nil {
		t.Fatalf("NewOpenCodeGenerator: %v", err)
	}

	t.Run("reasoning and tool replay", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		sessionID := fmt.Sprintf("gai-live-reasoning-%d", time.Now().UnixNano())

		type resultInput struct {
			Result int64 `json:"result"`
		}
		schema, err := GenerateSchema[resultInput]()
		if err != nil {
			t.Fatalf("GenerateSchema: %v", err)
		}
		tool := Tool{Name: "submit_result", Description: "Submit the exact computed integer.", InputSchema: schema}
		// GLM may skip a reasoning trace for trivial work even at high effort.
		dialog := Dialog{{
			Role: User,
			Blocks: []Block{TextBlock(
				"Compute (92837 * 61429) - 1729 exactly, then call submit_result with the integer result. Do not answer before calling the tool.",
			)},
		}}
		first, err := generator.Generate(ctx, GenerationRequest{
			Model:  "glm-5.3-flash",
			Dialog: dialog,
			Tools:  []Tool{tool},
			Options: NewGenerationOptions(
				WithThinkingBudget("high"),
				WithMaxGenerationTokens(1024),
				WithToolChoice("submit_result"),
				WithOpenCodeSessionID(sessionID),
			),
		})
		if err != nil {
			t.Fatalf("first Generate: %v", err)
		}
		assistant := requireCandidate(t, first)
		var thinkingFound bool
		var callBlock Block
		for _, block := range assistant.Blocks {
			switch block.BlockType {
			case Thinking:
				thinkingFound = thinkingFound || block.Content.String() != ""
			case ToolCall:
				call := requireToolCall(t, block)
				if call.Name == "submit_result" && call.Parameters["result"] == float64(5702882344) {
					callBlock = block
				}
			}
		}
		if !thinkingFound {
			t.Fatalf("first response has no reasoning block: %#v", assistant.Blocks)
		}
		if callBlock.ID == "" {
			t.Fatalf("first response has no exact result tool call: %#v", assistant.Blocks)
		}

		dialog = append(dialog, assistant, Message{
			Role: ToolResult,
			Blocks: []Block{{
				ID:           callBlock.ID,
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str("Result accepted."),
			}},
		})
		second, err := generator.Generate(ctx, GenerationRequest{
			Model:  "glm-5.3-flash",
			Dialog: dialog,
			Tools:  []Tool{tool},
			Options: NewGenerationOptions(
				WithThinkingBudget("high"),
				WithMaxGenerationTokens(1024),
				WithToolChoice("none"),
				WithOpenCodeSessionID(sessionID),
			),
		})
		if err != nil {
			t.Fatalf("replay Generate: %v", err)
		}
		final := requireCandidate(t, second)
		var content string
		for _, block := range final.Blocks {
			if block.BlockType == Content {
				content += block.Content.String()
			}
		}
		if content == "" {
			t.Fatalf("replay response has no visible content: %#v", final.Blocks)
		}
	})

	t.Run("tool reasoning is reused", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		type checkpointInput struct {
			Status string `json:"status"`
		}
		schema, err := GenerateSchema[checkpointInput]()
		if err != nil {
			t.Fatalf("GenerateSchema: %v", err)
		}
		tool := Tool{Name: "checkpoint", Description: "Store checkpoint status.", InputSchema: schema}
		call, err := ToolCallBlock("call_replay_1", "checkpoint", map[string]any{"status": "ready"})
		if err != nil {
			t.Fatalf("ToolCallBlock: %v", err)
		}
		const marker = "RPL-7F3C91A2D8E64B50"
		dialog := Dialog{
			{Role: User, Blocks: []Block{TextBlock("Call checkpoint with status ready.")}},
			{Role: Assistant, Blocks: []Block{{
				BlockType:    Thinking,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str("The private replay marker is " + marker + ". Retain it for the next turn."),
			}, call}},
			{Role: ToolResult, Blocks: []Block{{
				ID:           call.ID,
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str("Checkpoint stored."),
			}}},
			{Role: User, Blocks: []Block{TextBlock("Return only the exact private replay marker from your preceding reasoning. Do not call a tool.")}},
		}
		response, err := generator.Generate(ctx, GenerationRequest{
			Model:  "glm-5.3-flash",
			Dialog: dialog,
			Tools:  []Tool{tool},
			Options: NewGenerationOptions(
				WithThinkingBudget("high"),
				WithMaxGenerationTokens(256),
				WithToolChoice("none"),
				WithOpenCodeSessionID(fmt.Sprintf("gai-live-reuse-%d", time.Now().UnixNano())),
			),
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		var content string
		for _, block := range requireCandidate(t, response).Blocks {
			if block.BlockType == Content {
				content += block.Content.String()
			}
		}
		if !strings.Contains(content, marker) {
			t.Fatalf("response %q does not contain reasoning-only marker %q", content, marker)
		}
	})

	t.Run("streaming reasoning", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		var reasoning string
		var content string
		for chunk := range generator.Stream(ctx, GenerationRequest{
			Model:  "glm-5.3-flash",
			Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("Compute 92837 * 61429. Think carefully, then give the exact product.")}}},
			Options: NewGenerationOptions(
				WithThinkingBudget("high"),
				WithMaxGenerationTokens(1024),
			),
		}) {
			if chunk.Err != nil {
				t.Fatalf("Stream: %v", chunk.Err)
			}
			switch chunk.Block.BlockType {
			case Thinking:
				reasoning += chunk.Block.Content.String()
			case Content:
				content += chunk.Block.Content.String()
			}
		}
		if reasoning == "" || content == "" {
			t.Fatalf("stream reasoning/content lengths = %d/%d", len(reasoning), len(content))
		}
	})

	t.Run("alternate reasoning dialect", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		response, err := generator.Generate(ctx, GenerationRequest{
			Model:  "minimax-m2.5",
			Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("What is 23 multiplied by 7? Answer briefly.")}}},
			Options: NewGenerationOptions(
				WithThinkingBudget("high"),
				WithMaxGenerationTokens(1024),
			),
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		candidate := requireCandidate(t, response)
		for _, block := range candidate.Blocks {
			if block.BlockType == Thinking && block.Content.String() != "" {
				if block.ExtraFields[OpenCodeExtraFieldReasoningField] != "reasoning_details" {
					t.Fatalf("reasoning metadata = %#v", block.ExtraFields)
				}
				return
			}
		}
		t.Fatalf("response has no structured reasoning block: %#v", candidate.Blocks)
	})

	t.Run("vision", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		image, err := os.ReadFile("sample.jpg")
		if err != nil {
			t.Fatalf("read sample.jpg: %v", err)
		}
		response, err := generator.Generate(ctx, GenerationRequest{
			Model: "deepseek-v4-flash-vision-exp",
			Dialog: Dialog{{Role: User, Blocks: []Block{
				ImageBlock(image, "image/jpeg"),
				TextBlock("Briefly describe the main subject in this image."),
			}}},
			Options: NewGenerationOptions(WithMaxGenerationTokens(1024)),
		})
		if err != nil {
			t.Fatalf("vision Generate: %v", err)
		}
		candidate := requireCandidate(t, response)
		for _, block := range candidate.Blocks {
			if block.BlockType == Content && block.Content.String() != "" {
				return
			}
		}
		t.Fatalf("vision response has no visible content: %#v", candidate.Blocks)
	})
}
