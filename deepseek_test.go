package gai

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newDeepSeekTestGenerator(t *testing.T, server *httptest.Server) *DeepSeekGenerator {
	t.Helper()
	generator, err := NewDeepSeekGenerator(server.Client(), server.URL, "test-key")
	if err != nil {
		t.Fatalf("create DeepSeek generator: %v", err)
	}
	return generator
}

func newLiveDeepSeekGenerator(t *testing.T, apiKey string) *DeepSeekGenerator {
	t.Helper()
	generator, err := NewDeepSeekGenerator(nil, "", apiKey)
	if err != nil {
		t.Fatalf("create DeepSeek generator: %v", err)
	}
	return generator
}

func TestDeepSeekGeneratorUsesGeneratedJSONClient(t *testing.T) {
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
			"id":"completion_1",
			"object":"chat.completion",
			"created":1,
			"model":"deepseek-v4-pro",
			"system_fingerprint":"fp_1",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"reasoning_content":"thinking",
					"content":"answer",
					"tool_calls":[{
						"id":"call_1",
						"type":"function",
						"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}
					}]
				},
				"finish_reason":"tool_calls",
				"logprobs":null
			}],
			"usage":{
				"prompt_tokens":10,
				"completion_tokens":5,
				"total_tokens":15,
				"prompt_cache_hit_tokens":4,
				"prompt_cache_miss_tokens":6,
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
	response, err := newDeepSeekTestGenerator(t, server).Generate(t.Context(), GenerationRequest{
		Model:        "deepseek-v4-pro",
		Instructions: SystemMessage(TextBlock("Be concise.")),
		Dialog:       Dialog{{Role: User, Blocks: []Block{TextBlock("Weather?")}}},
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "Get weather.",
			InputSchema: schema,
		}},
		Options: NewGenerationOptions(
			WithTemperature(0.2),
			WithToolChoice("get_weather"),
			WithThinkingBudget("max"),
			WithDeepSeekThinking(true),
		),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if response.FinishReason != ToolUse || len(response.Candidates) != 1 {
		t.Fatalf("response = %+v", response)
	}
	thinking := requireBlockType(t, response, Thinking)
	if thinking.Content.String() != "thinking" || thinking.ExtraFields[ThinkingExtraFieldGeneratorKey] != ThinkingGeneratorDeepSeek {
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
		response.UsageMetadata[UsageMetricReasoningTokens] != 2 {
		t.Fatalf("usage metadata = %v", response.UsageMetadata)
	}
	if response.ExtraFields[DeepSeekResponseExtraFieldID] != "completion_1" ||
		response.ExtraFields[DeepSeekResponseExtraFieldModel] != "deepseek-v4-pro" ||
		response.ExtraFields[DeepSeekResponseExtraFieldCreated] != 1 ||
		response.ExtraFields[DeepSeekResponseExtraFieldSystemFingerprint] != "fp_1" {
		t.Fatalf("response extra fields = %v", response.ExtraFields)
	}

	request := <-requests
	if request["stream"] != false || request["temperature"] != 0.2 || request["reasoning_effort"] != "max" {
		t.Fatalf("request options = %v", request)
	}
	thinkingConfig, ok := request["thinking"].(map[string]any)
	if !ok || thinkingConfig["type"] != "enabled" {
		t.Fatalf("thinking request = %v", request["thinking"])
	}
	toolChoice, ok := request["tool_choice"].(map[string]any)
	if !ok || toolChoice["type"] != "function" {
		t.Fatalf("tool choice = %v", request["tool_choice"])
	}
	function, ok := toolChoice["function"].(map[string]any)
	if !ok || function["name"] != "get_weather" {
		t.Fatalf("tool choice function = %v", toolChoice["function"])
	}
}

func TestDeepSeekGeneratorUsesGeneratedSSEClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil || request["stream"] != true {
			http.Error(w, "expected streaming request", http.StatusBadRequest)
			return
		}
		streamOptions, ok := request["stream_options"].(map[string]any)
		if !ok || streamOptions["include_usage"] != true {
			http.Error(w, "expected stream usage", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"system_fingerprint\":\"fp_1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking\"},\"finish_reason\":null}],\"usage\":null}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}],\"usage\":null}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"calculate\",\"arguments\":\"{\\\"x\\\":1}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5,\"prompt_cache_hit_tokens\":1,\"prompt_cache_miss_tokens\":2}}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	defer server.Close()

	var thinking, content, toolCall string
	var toolBlocks []Block
	var usage map[string]int
	responseExtraFields := make(map[string]interface{})
	for chunk := range newDeepSeekTestGenerator(t, server).Stream(t.Context(), GenerationRequest{
		Model:  "deepseek-v4-pro",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("calculate")}}},
	}) {
		if chunk.Err != nil {
			t.Fatalf("stream returned error: %v", chunk.Err)
		}
		for key, value := range chunk.ResponseExtraFields {
			responseExtraFields[key] = value
		}
		switch chunk.Block.BlockType {
		case Thinking:
			thinking += chunk.Block.Content.String()
		case Content:
			content += chunk.Block.Content.String()
		case ToolCall:
			toolCall += chunk.Block.Content.String()
			toolBlocks = append(toolBlocks, chunk.Block)
		case MetadataBlockType:
			if err := json.Unmarshal([]byte(chunk.Block.Content.String()), &usage); err != nil {
				t.Fatalf("decode usage metadata: %v", err)
			}
		}
	}
	if thinking != "thinking" || content != "answer" || toolCall != `calculate{"x":1}` {
		t.Fatalf("thinking = %q, content = %q, tool call = %q", thinking, content, toolCall)
	}
	compressed, err := compressStreamingBlocks(toolBlocks)
	if err != nil {
		t.Fatalf("assemble streamed tool call: %v", err)
	}
	if len(compressed) != 1 {
		t.Fatalf("assembled tool blocks = %+v", compressed)
	}
	var streamedCall ToolCallInput
	if err := json.Unmarshal([]byte(compressed[0].Content.String()), &streamedCall); err != nil {
		t.Fatalf("decode streamed tool call: %v", err)
	}
	if streamedCall.Name != "calculate" || streamedCall.Parameters["x"] != float64(1) {
		t.Fatalf("streamed tool call = %+v", streamedCall)
	}
	if usage[UsageMetricInputTokens] != 3 || usage[UsageMetricGenerationTokens] != 2 || usage[UsageMetricCacheReadTokens] != 1 {
		t.Fatalf("usage metadata = %v", usage)
	}
	if responseExtraFields[DeepSeekResponseExtraFieldID] != "chunk_1" ||
		responseExtraFields[DeepSeekResponseExtraFieldModel] != "deepseek-v4-pro" ||
		responseExtraFields[DeepSeekResponseExtraFieldCreated] != 1 ||
		responseExtraFields[DeepSeekResponseExtraFieldSystemFingerprint] != "fp_1" {
		t.Fatalf("response extra fields = %v", responseExtraFields)
	}
}

func TestDeepSeekGeneratorMapsGeneratedError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limit"}}`))
	}))
	defer server.Close()

	_, err := newDeepSeekTestGenerator(t, server).Generate(t.Context(), GenerationRequest{
		Model:  "deepseek-v4-pro",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}},
	})
	var apiErr *ApiErr
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %T %v, want ApiErr", err, err)
	}
	if apiErr.Provider != ProviderDeepSeek || apiErr.Kind != APIErrorKindRateLimit || apiErr.StatusCode != http.StatusTooManyRequests || apiErr.Message != "slow down" {
		t.Fatalf("API error = %+v", apiErr)
	}
}

func TestDeepSeekBuildRequestReplaysThinking(t *testing.T) {
	toolCall, err := ToolCallBlock("call_1", "get_weather", map[string]any{"city": "Paris"})
	if err != nil {
		t.Fatalf("create tool call: %v", err)
	}
	request, err := (&DeepSeekGenerator{}).buildRequest(GenerationRequest{
		Model: "deepseek-v4-pro",
		Dialog: Dialog{
			{Role: User, Blocks: []Block{TextBlock("Weather?")}},
			{Role: Assistant, Blocks: []Block{deepSeekThinkingBlock("private reasoning"), toolCall}},
			ToolResultMessage("call_1", TextBlock("sunny")),
		},
	}, false)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(request.Messages))
	}
	assistant, ok := request.Messages[1].GetAssistantMessage()
	if !ok || assistant.ReasoningContent.Or("") != "private reasoning" || len(assistant.ToolCalls) != 1 {
		t.Fatalf("assistant replay = %+v", assistant)
	}
	toolResult, ok := request.Messages[2].GetToolMessage()
	if !ok || toolResult.ToolCallID != "call_1" || toolResult.Content != "sunny" {
		t.Fatalf("tool result replay = %+v", toolResult)
	}
}

func TestDeepSeekGeneratorLive(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "DEEPSEEK_API_KEY")
	generator := newLiveDeepSeekGenerator(t, apiKey)
	request := GenerationRequest{
		Model:  "deepseek-v4-flash",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("Reply with the word hello.")}}},
		Options: NewGenerationOptions(
			WithMaxGenerationTokens(64),
			WithDeepSeekThinking(false),
		),
	}

	t.Run("Generate", func(t *testing.T) {
		response, err := generator.Generate(t.Context(), request)
		if err != nil {
			t.Fatalf("generate: %v", err)
		}
		if len(response.Candidates) == 0 || len(response.Candidates[0].Blocks) == 0 {
			t.Fatalf("empty response: %+v", response)
		}
	})

	t.Run("Stream", func(t *testing.T) {
		var content strings.Builder
		for chunk := range generator.Stream(t.Context(), request) {
			if chunk.Err != nil {
				t.Fatalf("stream: %v", chunk.Err)
			}
			if chunk.Block.BlockType == Content {
				content.WriteString(chunk.Block.Content.String())
			}
		}
		if content.Len() == 0 {
			t.Fatal("stream returned no content")
		}
	})
}
