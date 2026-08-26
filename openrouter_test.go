package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	openrouterapi "github.com/spachava753/gai/internal/openrouter"
)

func newOpenRouterTestGenerator(t *testing.T, server *httptest.Server) *OpenRouterGenerator {
	t.Helper()
	client, err := openrouterapi.NewClient(
		server.URL,
		openRouterSecuritySource{apiKey: "test-key"},
		openrouterapi.WithClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("create generated OpenRouter client: %v", err)
	}
	return NewOpenRouterGenerator(client, "")
}

func TestOpenRouterGeneratorUsesGeneratedJSONClient(t *testing.T) {
	requests := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			http.Error(w, "unexpected request", http.StatusNotFound)
			return
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			http.Error(w, "missing authorization", http.StatusUnauthorized)
			return
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		requests <- request

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"completion_1","object":"chat.completion","created":1,"model":"test/model",
			"choices":[{
				"index":0,"finish_reason":"tool_calls",
				"message":{
					"role":"assistant","content":"answer",
					"reasoning_details":[{
						"type":"reasoning.text","text":"thinking","id":"reasoning_1",
						"format":"anthropic-claude-v1","index":0,"signature":"signed"
					}],
					"tool_calls":[{
						"id":"call_1","type":"function",
						"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}
					}]
				}
			}],
			"usage":{
				"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,
				"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":3},
				"completion_tokens_details":{"reasoning_tokens":2}
			}
		}`))
	}))
	defer server.Close()

	schema, err := GenerateSchema[struct {
		City string `json:"city" jsonschema:"required"`
	}]()
	if err != nil {
		t.Fatalf("generate schema: %v", err)
	}
	response, err := newOpenRouterTestGenerator(t, server).Generate(t.Context(), GenerationRequest{
		Model:        "test/model",
		Instructions: SystemMessage(TextBlock("Be concise.")),
		Dialog:       Dialog{{Role: User, Blocks: []Block{TextBlock("Weather?")}}},
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "Get weather.",
			InputSchema: schema,
		}},
		Options: NewGenerationOptions(
			WithTemperature(0.2),
			WithTopP(0.8),
			WithFrequencyPenalty(0.1),
			WithPresencePenalty(0.3),
			WithCandidateCount(2),
			WithMaxGenerationTokens(64),
			WithStopSequences("END", "STOP"),
			WithToolChoice("get_weather"),
			WithThinkingBudget("2048"),
		),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if response.FinishReason != ToolUse || len(response.Candidates) != 1 {
		t.Fatalf("response = %+v", response)
	}
	thinking := requireBlockType(t, response, Thinking)
	if thinking.Content.String() != "thinking" || thinking.ID != "reasoning_1" ||
		thinking.ExtraFields[OpenRouterExtraFieldReasoningSignature] != "signed" {
		t.Fatalf("thinking block = %+v", thinking)
	}
	if got := requireBlockType(t, response, Content).Content.String(); got != "answer" {
		t.Fatalf("content = %q, want answer", got)
	}
	var call ToolCallInput
	if err := json.Unmarshal([]byte(requireBlockType(t, response, ToolCall).Content.String()), &call); err != nil {
		t.Fatalf("decode tool call: %v", err)
	}
	if call.Name != "get_weather" || call.Parameters["city"] != "Paris" {
		t.Fatalf("tool call = %+v", call)
	}
	if response.UsageMetadata[UsageMetricInputTokens] != 10 ||
		response.UsageMetadata[UsageMetricGenerationTokens] != 5 ||
		response.UsageMetadata[UsageMetricCacheReadTokens] != 4 ||
		response.UsageMetadata[UsageMetricCacheWriteTokens] != 3 ||
		response.UsageMetadata[UsageMetricReasoningTokens] != 2 ||
		response.UsageMetadata[OpenRouterUsageMetricReasoningDetailsAvailable] != true {
		t.Fatalf("usage metadata = %v", response.UsageMetadata)
	}

	request := <-requests
	if request["stream"] != false || request["temperature"] != 0.2 || request["top_p"] != 0.8 ||
		request["frequency_penalty"] != 0.1 || request["presence_penalty"] != 0.3 ||
		request["n"] != float64(2) || request["max_completion_tokens"] != float64(64) {
		t.Fatalf("request options = %v", request)
	}
	reasoning, ok := request["reasoning"].(map[string]any)
	if !ok || reasoning["max_tokens"] != float64(2048) {
		t.Fatalf("reasoning = %v", request["reasoning"])
	}
	toolChoice, ok := request["tool_choice"].(map[string]any)
	if !ok || toolChoice["type"] != "function" {
		t.Fatalf("tool choice = %v", request["tool_choice"])
	}
}

func TestOpenRouterGeneratorUsesGeneratedSSEClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil || request["stream"] != true {
			http.Error(w, "expected streaming request", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"step \",\"id\":\"reasoning_1\",\"format\":\"openai-responses-v1\",\"index\":0}]},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"one\",\"id\":\"reasoning_1\",\"format\":\"openai-responses-v1\",\"index\":0}]},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.summary\",\"summary\":\"summary\",\"id\":\"reasoning_2\",\"index\":1}]},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"calculate\",\"arguments\":\"{\\\"x\\\":\"}}]},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":5,\"total_tokens\":12,\"prompt_tokens_details\":{\"cached_tokens\":3,\"cache_write_tokens\":2},\"completion_tokens_details\":{\"reasoning_tokens\":4}}}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	defer server.Close()

	var blocks []Block
	var usage map[string]any
	for chunk := range newOpenRouterTestGenerator(t, server).Stream(t.Context(), GenerationRequest{
		Model:  "test/model",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("calculate")}}},
	}) {
		if chunk.Err != nil {
			t.Fatalf("stream returned error: %v", chunk.Err)
		}
		if chunk.CandidatesIndex != 0 {
			t.Fatalf("candidate index = %d, want 0", chunk.CandidatesIndex)
		}
		if chunk.Block.BlockType == MetadataBlockType {
			if err := json.Unmarshal([]byte(chunk.Block.Content.String()), &usage); err != nil {
				t.Fatalf("decode usage metadata: %v", err)
			}
			continue
		}
		blocks = append(blocks, chunk.Block)
	}

	compressed, err := compressStreamingBlocks(blocks)
	if err != nil {
		t.Fatalf("compress stream: %v", err)
	}
	if len(compressed) != 4 {
		t.Fatalf("compressed blocks = %+v", compressed)
	}
	if compressed[0].BlockType != Thinking || compressed[0].Content.String() != "step one" ||
		compressed[1].BlockType != Thinking || compressed[1].Content.String() != "summary" {
		t.Fatalf("reasoning blocks = %+v", compressed[:2])
	}
	if compressed[0].ExtraFields[OpenRouterExtraFieldReasoningType] != "reasoning.text" ||
		compressed[1].ExtraFields[OpenRouterExtraFieldReasoningType] != "reasoning.summary" {
		t.Fatalf("reasoning metadata = %+v, %+v", compressed[0].ExtraFields, compressed[1].ExtraFields)
	}
	if compressed[2].BlockType != Content || compressed[2].Content.String() != "answer" {
		t.Fatalf("content block = %+v", compressed[2])
	}
	var call ToolCallInput
	if compressed[3].BlockType != ToolCall || json.Unmarshal([]byte(compressed[3].Content.String()), &call) != nil {
		t.Fatalf("tool call block = %+v", compressed[3])
	}
	if call.Name != "calculate" || call.Parameters["x"] != float64(1) {
		t.Fatalf("streamed tool call = %+v", call)
	}
	if usage[UsageMetricInputTokens] != float64(7) || usage[UsageMetricGenerationTokens] != float64(5) ||
		usage[UsageMetricCacheReadTokens] != float64(3) || usage[UsageMetricCacheWriteTokens] != float64(2) ||
		usage[UsageMetricReasoningTokens] != float64(4) || usage[OpenRouterUsageMetricReasoningDetailsAvailable] != true {
		t.Fatalf("usage metadata = %v", usage)
	}
}

func TestOpenRouterStreamMapsMidstreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[],\"error\":{\"code\":503,\"message\":\"provider unavailable\",\"error_type\":\"provider_unavailable\"}}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	defer server.Close()

	var streamErr error
	for chunk := range newOpenRouterTestGenerator(t, server).Stream(t.Context(), GenerationRequest{
		Model:  "test/model",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}},
	}) {
		if streamErr != nil {
			t.Fatal("stream yielded after terminal error")
		}
		streamErr = chunk.Err
	}
	var apiErr *ApiErr
	if !errors.As(streamErr, &apiErr) {
		t.Fatalf("stream error = %T %v, want ApiErr", streamErr, streamErr)
	}
	if apiErr.Provider != ProviderOpenRouter || apiErr.Kind != APIErrorKindServiceUnavailable ||
		apiErr.StatusCode != http.StatusServiceUnavailable || apiErr.Message != "provider unavailable" {
		t.Fatalf("API error = %+v", apiErr)
	}
}

func TestOpenRouterBuildRequestReplaysReasoningAndMultimodalInput(t *testing.T) {
	toolCall, err := ToolCallBlock("call_1", "get_weather", map[string]any{"city": "Paris"})
	if err != nil {
		t.Fatalf("create tool call: %v", err)
	}
	thinking := Block{
		ID:           "reasoning_1",
		BlockType:    Thinking,
		ModalityType: Text,
		Content:      Str("private reasoning"),
		ExtraFields: map[string]interface{}{
			ThinkingExtraFieldGeneratorKey:         ThinkingGeneratorOpenRouter,
			OpenRouterExtraFieldReasoningType:      "reasoning.text",
			OpenRouterExtraFieldReasoningFormat:    "anthropic-claude-v1",
			OpenRouterExtraFieldReasoningIndex:     3,
			OpenRouterExtraFieldReasoningSignature: "signed",
		},
	}
	request, err := (&OpenRouterGenerator{}).buildRequest(GenerationRequest{
		Model:        "test/model",
		Instructions: SystemMessage(TextBlock("Be concise.")),
		Dialog: Dialog{
			{Role: User, Blocks: []Block{
				TextBlock("What is shown?"),
				{BlockType: Content, ModalityType: Image, MimeType: "image/png", Content: Str("aW1hZ2U=")},
			}},
			{Role: Assistant, Blocks: []Block{thinking, TextBlock("Checking."), toolCall}},
			ToolResultMessage("call_1", TextBlock("sunny")),
		},
		Options: NewGenerationOptions(WithThinkingBudget("high")),
	})
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	if len(request.Messages) != 4 {
		t.Fatalf("messages = %d, want 4", len(request.Messages))
	}
	user, ok := request.Messages[1].GetUserMessage()
	if !ok {
		t.Fatalf("user message = %+v", request.Messages[1])
	}
	parts, ok := user.Content.GetUserContentPartArray()
	if !ok || len(parts) != 2 {
		t.Fatalf("user content = %+v", user.Content)
	}
	image, ok := parts[1].GetImageContentPart()
	if !ok || image.ImageURL.URL != "data:image/png;base64,aW1hZ2U=" {
		t.Fatalf("image content = %+v", parts[1])
	}
	assistant, ok := request.Messages[2].GetAssistantMessage()
	if !ok || assistant.Content.Or("") != "Checking." || len(assistant.ToolCalls) != 1 || len(assistant.ReasoningDetails) != 1 {
		t.Fatalf("assistant replay = %+v", assistant)
	}
	detail := assistant.ReasoningDetails[0]
	if detail.Type != "reasoning.text" || detail.Text.Or("") != "private reasoning" ||
		detail.ID.Or("") != "reasoning_1" || detail.Index.Or(0) != 3 || detail.Signature.Or("") != "signed" {
		t.Fatalf("reasoning replay = %+v", detail)
	}
	toolResult, ok := request.Messages[3].GetToolMessage()
	if !ok || toolResult.ToolCallID != "call_1" || toolResult.Content != "sunny" {
		t.Fatalf("tool result replay = %+v", toolResult)
	}
	reasoning, ok := request.Reasoning.Get()
	if !ok || reasoning.Effort.Or("") != openrouterapi.ReasoningConfigEffortHigh {
		t.Fatalf("reasoning config = %+v", request.Reasoning)
	}
}

func TestOpenRouterGenerateReturnsContentPolicyErrorForRefusal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"completion_1","object":"chat.completion","created":1,"model":"test-model",
			"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":null,"refusal":"I cannot help with that."}}]
		}`))
	}))
	defer server.Close()
	generator := newOpenRouterTestGenerator(t, server)

	response, err := generator.Generate(context.Background(), GenerationRequest{
		Model:  "test-model",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
	})
	if response.FinishReason != ContentPolicyViolation {
		t.Fatalf("FinishReason = %v, want ContentPolicyViolation", response.FinishReason)
	}
	var policyErr ContentPolicyErr
	if !errors.As(err, &policyErr) {
		t.Fatalf("Generate error = %T %v, want ContentPolicyErr", err, err)
	}
	if !strings.Contains(policyErr.Error(), "I cannot help with that.") {
		t.Fatalf("Generate error = %q, want refusal message", policyErr)
	}
}

func TestOpenRouterGenerateReturnsContentPolicyErrorForContentFilter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"completion_1","object":"chat.completion","created":1,"model":"test-model",
			"choices":[{"index":0,"finish_reason":"content_filter","message":{"role":"assistant","content":null}}]
		}`))
	}))
	defer server.Close()
	generator := newOpenRouterTestGenerator(t, server)

	response, err := generator.Generate(context.Background(), GenerationRequest{
		Model:  "test-model",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
	})
	if response.FinishReason != ContentPolicyViolation {
		t.Fatalf("FinishReason = %v, want ContentPolicyViolation", response.FinishReason)
	}
	var policyErr ContentPolicyErr
	if !errors.As(err, &policyErr) {
		t.Fatalf("Generate error = %T %v, want ContentPolicyErr", err, err)
	}
	if !strings.Contains(policyErr.Error(), "content policy violation detected") {
		t.Fatalf("Generate error = %q, want content filter fallback", policyErr)
	}
}

func TestOpenRouterGenerator_Generate(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	gen := NewOpenRouterGenerator(nil, apiKey)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hi!"),
				},
			},
		},
	}
	request := GenerationRequest{
		Model:        "z-ai/glm-4.6:exacto",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
		Dialog:       dialog,
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The exact response text may vary, so we'll just print a placeholder
	// Customize generation parameters
	request.Options = NewGenerationOptions(WithMaxGenerationTokens(10000))
	resp, err = gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(resp.Candidates); got == 0 {
		t.Fatal("expected at least one item")
	}
}
func TestOpenRouterGenerator_Generate_image(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		t.Skip("could not open sample.jpg")
		return
	}
	imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))
	// Use a vision-capable model through OpenRouter.
	gen := NewOpenRouterGenerator(nil, apiKey)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Image,
					MimeType:     "image/jpeg",
					Content:      imgBase64,
				},
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is in this image? (Hint, it's a character from The Croods, a DreamWorks animated movie.)"),
				},
			},
		},
	}
	resp, err := gen.Generate(context.Background(), GenerationRequest{
		Model:        "qwen/qwen3-vl-235b-a22b-instruct",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
		Options:      NewGenerationOptions(WithMaxGenerationTokens(512)),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) != 1 {
		t.Fatalf("candidates = %d, want 1", len(resp.Candidates))
	}
	if len(resp.Candidates[0].Blocks) < 1 {
		t.Fatalf("blocks = %d, want at least 1", len(resp.Candidates[0].Blocks))
	}
	if !strings.Contains(resp.Candidates[0].Blocks[0].Content.String(), "Crood") {
		t.Fatalf("content does not contain Crood")
	}
}
func TestOpenRouterGenerator_RequestTools(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	gen := NewOpenRouterGenerator(nil, apiKey)
	// Define a request tool
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
			}]()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			return schema
		}(),
	}
	dialog := Dialog{
		{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}},
	}
	request := GenerationRequest{
		Model:        "moonshotai/kimi-k2-0905:exacto",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant that returns the price of a stock and nothing else.")),
		Dialog:       dialog,
		Tools:        []Tool{tickerTool},
		Options:      NewGenerationOptions(WithToolChoice("get_stock_price")),
	}
	// Force the tool call
	resp, err := gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
		t.Fatal("empty response")
		return
	}
	// Find and print the tool call JSON
	var toolCall Block
	for _, b := range resp.Candidates[0].Blocks {
		if b.BlockType == ToolCall {
			toolCall = b
			break
		}
	}
	if got := toolCall.Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	// Append tool result and continue the conversation
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{ID: toolCall.ID, BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")},
		},
	})
	request.Dialog = dialog
	request.Options = nil
	resp, err = gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
	}
}
func TestOpenRouterGenerator_Generate_reasoningModel(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	// Use a reasoning model through OpenRouter.
	gen := NewOpenRouterGenerator(nil, apiKey)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is the square root of 144?"),
				},
			},
		},
	}
	request := GenerationRequest{
		Model:        "z-ai/glm-4.6:exacto",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
		Options:      NewGenerationOptions(WithThinkingBudget("low")),
	}
	// Generate response - reasoning models may return thinking blocks automatically
	resp, err := gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks (from reasoning_details)
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
				// Thinking blocks have reasoning metadata in ExtraFields
				if reasoningType, ok := block.ExtraFields["reasoning_type"].(string); ok {
					_ = reasoningType // reasoning.text, reasoning.summary, or reasoning.encrypted
				}
			}
		}
		if hasThinking {
		}
		// Find the main content block (not thinking)
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "12") {
				}
				break
			}
		}
	}
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: User,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("What is the square root of 225?"),
			},
		},
	})
	request.Dialog = dialog
	// Generate response - reasoning models may return thinking blocks automatically
	resp, err = gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks (from reasoning_details)
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
				// Thinking blocks have reasoning metadata in ExtraFields
				if reasoningType, ok := block.ExtraFields["reasoning_type"].(string); ok {
					_ = reasoningType // reasoning.text, reasoning.summary, or reasoning.encrypted
				}
			}
		}
		if hasThinking {
		}
		// Find the main content block (not thinking)
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "15") {
				}
				break
			}
		}
	}
}
func TestOpenRouterGenerator_Generate_invalidModel(t *testing.T) {
	// This example demonstrates handling of invalid model IDs with OpenRouter.
	// OpenRouter returns a 400 status code with error details in the response body
	// for invalid requests like nonsense model IDs.
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	// Use a nonsense model ID to trigger an error.
	gen := NewOpenRouterGenerator(nil, apiKey)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hello"),
				},
			},
		},
	}
	_, err := gen.Generate(context.Background(), GenerationRequest{
		Model:        "invalid/model-does-not-exist",
		Instructions: SystemMessage(TextBlock("You are helpful")),
		Dialog:       dialog,
	})
	if err == nil {
		t.Fatal("expected invalid model to return an error")
	}
	var apiErr *ApiErr
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *ApiErr", err)
	}
}
