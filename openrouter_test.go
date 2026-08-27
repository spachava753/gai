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
	generator, err := NewOpenRouterGenerator(server.Client(), server.URL, "test-key")
	if err != nil {
		t.Fatalf("create OpenRouter generator: %v", err)
	}
	return generator
}

func newLiveOpenRouterGenerator(t *testing.T, apiKey string) *OpenRouterGenerator {
	t.Helper()
	generator, err := NewOpenRouterGenerator(nil, "", apiKey)
	if err != nil {
		t.Fatalf("create OpenRouter generator: %v", err)
	}
	return generator
}

func TestRouterAdapterScenarios(t *testing.T) {
	t.Run("OpenRouterBuildRequestRejectsInvalidProviderOptions", testOpenRouterBuildRequestRejectsInvalidProviderOptions)
	t.Run("OpenRouterBuildRequestReplaysReasoningAndMultimodalInput", testOpenRouterBuildRequestReplaysReasoningAndMultimodalInput)
	t.Run("OpenRouterGenerateReturnsContentPolicyErrorForContentFilter", testOpenRouterGenerateReturnsContentPolicyErrorForContentFilter)
	t.Run("OpenRouterGenerateReturnsContentPolicyErrorForRefusal", testOpenRouterGenerateReturnsContentPolicyErrorForRefusal)
	t.Run("OpenRouterGeneratedOverloadMapping", testOpenRouterGeneratedOverloadMapping)
	t.Run("OpenRouterGeneratorSurfacesProviderOverload", testOpenRouterGeneratorSurfacesProviderOverload)
	t.Run("OpenRouterGeneratorUsesGeneratedJSONClient", testOpenRouterGeneratorUsesGeneratedJSONClient)
	t.Run("OpenRouterGeneratorUsesGeneratedSSEClient", testOpenRouterGeneratorUsesGeneratedSSEClient)
	t.Run("OpenRouterGenerator/Generate", testOpenRouterGenerator_Generate)
	t.Run("OpenRouterGenerator/Generate/image", testOpenRouterGenerator_Generate_image)
	t.Run("OpenRouterGenerator/Generate/invalidModel", testOpenRouterGenerator_Generate_invalidModel)
	t.Run("OpenRouterGenerator/Generate/reasoningModel", testOpenRouterGenerator_Generate_reasoningModel)
	t.Run("OpenRouterGenerator/RequestTools", testOpenRouterGenerator_RequestTools)
	t.Run("OpenRouterStreamMapsMidstreamError", testOpenRouterStreamMapsMidstreamError)
}

func testOpenRouterGeneratorUsesGeneratedJSONClient(t *testing.T) {
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
			"system_fingerprint":"fp_test","service_tier":"priority",
			"openrouter_metadata":{"provider":"ExampleAI","region":"us-east"},
			"choices":[{
				"index":0,"finish_reason":"tool_calls",
				"logprobs":{"content":[{"token":"answer","logprob":-0.1}]},
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
				"cost":0.0012,"is_byok":true,
				"cost_details":{"upstream_inference_cost":0.001,"upstream_inference_prompt_cost":0.0004,"upstream_inference_completions_cost":0.0006},
				"server_tool_use_details":{"tool_calls_requested":2,"tool_calls_executed":1,"web_search_requests":1},
				"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":3,"audio_tokens":2,"video_tokens":1},
				"completion_tokens_details":{"reasoning_tokens":2,"audio_tokens":1,"accepted_prediction_tokens":2,"rejected_prediction_tokens":1}
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
			WithTopK(40),
			WithOpenRouterLogitBias(map[string]float64{"42": -1.5}),
			WithOpenRouterLogprobs(true),
			WithOpenRouterTopLogprobs(3),
			WithOpenRouterMinP(0.05),
			WithOpenRouterFallbackModels("fallback/one", "fallback/two"),
			WithOpenRouterParallelToolCalls(false),
			WithOpenRouterPrediction("known output"),
			WithOpenRouterPromptCacheKey("conversation-1"),
			WithOpenRouterProviderPreferences(map[string]any{"order": []string{"ExampleAI"}, "allow_fallbacks": false}),
			WithOpenRouterRepetitionPenalty(1.1),
			WithOpenRouterResponseFormat(map[string]any{"type": "json_object"}),
			WithOpenRouterSeed(7),
			WithOpenRouterServiceTier(OpenRouterServiceTierPriority),
			WithOpenRouterSessionID("session-1"),
			WithOpenRouterTopA(0.2),
			WithOpenRouterUser("user-1"),
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
	if response.ExtraFields[OpenRouterResponseExtraFieldID] != "completion_1" ||
		response.ExtraFields[OpenRouterResponseExtraFieldModel] != "test/model" ||
		response.ExtraFields[OpenRouterResponseExtraFieldCreated] != int64(1) ||
		response.ExtraFields[OpenRouterResponseExtraFieldSystemFingerprint] != "fp_test" ||
		response.ExtraFields[OpenRouterResponseExtraFieldServiceTier] != "priority" {
		t.Fatalf("response extra fields = %v", response.ExtraFields)
	}
	nativeMetadata, ok := response.ExtraFields[OpenRouterResponseExtraFieldMetadata].(map[string]any)
	if !ok || nativeMetadata["provider"] != "ExampleAI" || nativeMetadata["region"] != "us-east" {
		t.Fatalf("OpenRouter metadata = %v", response.ExtraFields[OpenRouterResponseExtraFieldMetadata])
	}
	logprobs, ok := response.Candidates[0].ExtraFields[OpenRouterMessageExtraFieldLogprobs].(map[string]any)
	if !ok {
		t.Fatalf("candidate logprobs = %v", response.Candidates[0].ExtraFields)
	}
	logprobItems, ok := logprobs["content"].([]any)
	if !ok || len(logprobItems) != 1 {
		t.Fatalf("candidate logprobs = %v", logprobs)
	}
	if response.UsageMetadata[UsageMetricInputTokens] != 10 ||
		response.UsageMetadata[UsageMetricGenerationTokens] != 5 ||
		response.UsageMetadata[UsageMetricCacheReadTokens] != 4 ||
		response.UsageMetadata[UsageMetricCacheWriteTokens] != 3 ||
		response.UsageMetadata[UsageMetricReasoningTokens] != 2 ||
		response.UsageMetadata[OpenRouterUsageMetricCost] != 0.0012 ||
		response.UsageMetadata[OpenRouterUsageMetricIsBYOK] != true ||
		response.UsageMetadata[OpenRouterUsageMetricReasoningDetailsAvailable] != true {
		t.Fatalf("usage metadata = %v", response.UsageMetadata)
	}
	costDetails, ok := response.UsageMetadata[OpenRouterUsageMetricCostDetails].(map[string]any)
	if !ok || costDetails["upstream_inference_cost"] != 0.001 {
		t.Fatalf("cost details = %v", response.UsageMetadata[OpenRouterUsageMetricCostDetails])
	}
	serverToolDetails, ok := response.UsageMetadata[OpenRouterUsageMetricServerToolUseDetails].(map[string]any)
	if !ok || serverToolDetails["tool_calls_requested"] != float64(2) || serverToolDetails["web_search_requests"] != float64(1) {
		t.Fatalf("server tool details = %v", response.UsageMetadata[OpenRouterUsageMetricServerToolUseDetails])
	}
	promptDetails, ok := response.UsageMetadata[OpenRouterUsageMetricPromptTokenDetails].(map[string]any)
	if !ok || promptDetails["audio_tokens"] != float64(2) || promptDetails["video_tokens"] != float64(1) {
		t.Fatalf("prompt token details = %v", response.UsageMetadata[OpenRouterUsageMetricPromptTokenDetails])
	}
	completionDetails, ok := response.UsageMetadata[OpenRouterUsageMetricCompletionTokenDetails].(map[string]any)
	if !ok || completionDetails["accepted_prediction_tokens"] != float64(2) || completionDetails["rejected_prediction_tokens"] != float64(1) {
		t.Fatalf("completion token details = %v", response.UsageMetadata[OpenRouterUsageMetricCompletionTokenDetails])
	}

	request := <-requests
	if request["stream"] != false || request["temperature"] != 0.2 || request["top_p"] != 0.8 ||
		request["top_k"] != float64(40) || request["logprobs"] != true || request["top_logprobs"] != float64(3) ||
		request["min_p"] != 0.05 || request["parallel_tool_calls"] != false ||
		request["prompt_cache_key"] != "conversation-1" || request["repetition_penalty"] != 1.1 ||
		request["seed"] != float64(7) || request["service_tier"] != "priority" ||
		request["session_id"] != "session-1" || request["top_a"] != 0.2 || request["user"] != "user-1" ||
		request["frequency_penalty"] != 0.1 || request["presence_penalty"] != 0.3 ||
		request["n"] != float64(2) || request["max_completion_tokens"] != float64(64) {
		t.Fatalf("request options = %v", request)
	}
	logitBias, ok := request["logit_bias"].(map[string]any)
	if !ok || logitBias["42"] != -1.5 {
		t.Fatalf("logit bias = %v", request["logit_bias"])
	}
	models, ok := request["models"].([]any)
	if !ok || len(models) != 2 || models[0] != "fallback/one" || models[1] != "fallback/two" {
		t.Fatalf("fallback models = %v", request["models"])
	}
	prediction, ok := request["prediction"].(map[string]any)
	if !ok || prediction["content"] != "known output" {
		t.Fatalf("prediction = %v", request["prediction"])
	}
	provider, ok := request["provider"].(map[string]any)
	if !ok || provider["allow_fallbacks"] != false {
		t.Fatalf("provider preferences = %v", request["provider"])
	}
	responseFormat, ok := request["response_format"].(map[string]any)
	if !ok || responseFormat["type"] != "json_object" {
		t.Fatalf("response format = %v", request["response_format"])
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

func testOpenRouterBuildRequestRejectsInvalidProviderOptions(t *testing.T) {
	tests := []struct {
		name    string
		options GenerationOptions
		want    string
	}{
		{
			name:    "top logprobs without logprobs",
			options: NewGenerationOptions(WithOpenRouterTopLogprobs(3)),
			want:    OpenRouterGenerationOptionTopLogprobs,
		},
		{
			name:    "invalid service tier",
			options: NewGenerationOptions(WithOpenRouterServiceTier("slow")),
			want:    OpenRouterGenerationOptionServiceTier,
		},
		{
			name:    "session ID too long",
			options: NewGenerationOptions(WithOpenRouterSessionID(strings.Repeat("x", 257))),
			want:    OpenRouterGenerationOptionSessionID,
		},
		{
			name:    "response format without type",
			options: NewGenerationOptions(WithOpenRouterResponseFormat(map[string]any{"schema": map[string]any{}})),
			want:    OpenRouterGenerationOptionResponseFormat,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := (&OpenRouterGenerator{}).buildRequest(GenerationRequest{
				Model:   "test/model",
				Dialog:  Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}},
				Options: tt.options,
			})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("buildRequest() error = %v, want parameter %q", err, tt.want)
			}
		})
	}
}

func testOpenRouterGeneratorUsesGeneratedSSEClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil || request["stream"] != true {
			http.Error(w, "expected streaming request", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"system_fingerprint\":\"fp_stream\",\"service_tier\":\"flex\",\"openrouter_metadata\":{\"provider\":\"ExampleAI\"},\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"step \",\"id\":\"reasoning_1\",\"format\":\"openai-responses-v1\",\"index\":0}]},\"finish_reason\":null,\"logprobs\":{\"content\":[{\"token\":\"a\",\"logprob\":-0.1}]}}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"one\",\"id\":\"reasoning_1\",\"format\":\"openai-responses-v1\",\"index\":0}]},\"finish_reason\":null,\"logprobs\":{\"content\":[{\"token\":\"b\",\"logprob\":-0.2}]}}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.summary\",\"summary\":\"summary\",\"id\":\"reasoning_2\",\"index\":1}]},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"calculate\",\"arguments\":\"{\\\"x\\\":\"}}]},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n" +
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test/model\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":5,\"total_tokens\":12,\"cost\":0,\"cost_details\":{\"upstream_inference_cost\":null,\"upstream_inference_prompt_cost\":0.1,\"upstream_inference_completions_cost\":0.2},\"is_byok\":false,\"server_tool_use_details\":{\"tool_calls_requested\":1,\"tool_calls_executed\":1,\"web_search_requests\":0},\"prompt_tokens_details\":{\"cached_tokens\":3,\"cache_write_tokens\":2},\"completion_tokens_details\":{\"reasoning_tokens\":4}}}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	defer server.Close()

	var blocks []Block
	var usage map[string]any
	responseExtraFields := make(map[string]interface{})
	messageExtraFields := make(map[string]interface{})
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
		for key, value := range chunk.ResponseExtraFields {
			responseExtraFields[key] = value
		}
		for key, value := range chunk.MessageExtraFields {
			messageExtraFields[key] = value
		}
		if chunk.Block.BlockType == MetadataBlockType {
			if err := json.Unmarshal([]byte(chunk.Block.Content.String()), &usage); err != nil {
				t.Fatalf("decode usage metadata: %v", err)
			}
			continue
		}
		blocks = append(blocks, chunk.Block)
	}
	if responseExtraFields[OpenRouterResponseExtraFieldID] != "chunk_1" ||
		responseExtraFields[OpenRouterResponseExtraFieldSystemFingerprint] != "fp_stream" ||
		responseExtraFields[OpenRouterResponseExtraFieldServiceTier] != "flex" {
		t.Fatalf("stream response extra fields = %v", responseExtraFields)
	}
	streamMetadata, ok := responseExtraFields[OpenRouterResponseExtraFieldMetadata].(map[string]any)
	if !ok || streamMetadata["provider"] != "ExampleAI" {
		t.Fatalf("stream OpenRouter metadata = %v", responseExtraFields[OpenRouterResponseExtraFieldMetadata])
	}
	streamLogprobs, ok := messageExtraFields[OpenRouterMessageExtraFieldLogprobs].(map[string]any)
	if !ok {
		t.Fatalf("stream message extra fields = %v", messageExtraFields)
	}
	streamLogprobItems, ok := streamLogprobs["content"].([]any)
	if !ok || len(streamLogprobItems) != 2 {
		t.Fatalf("stream logprobs = %v", streamLogprobs)
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
		usage[UsageMetricReasoningTokens] != float64(4) || usage[OpenRouterUsageMetricCost] != float64(0) ||
		usage[OpenRouterUsageMetricIsBYOK] != false || usage[OpenRouterUsageMetricReasoningDetailsAvailable] != true {
		t.Fatalf("usage metadata = %v", usage)
	}
	streamCostDetails, ok := usage[OpenRouterUsageMetricCostDetails].(map[string]any)
	if !ok || streamCostDetails["upstream_inference_prompt_cost"] != 0.1 {
		t.Fatalf("stream cost details = %v", usage[OpenRouterUsageMetricCostDetails])
	}
	streamServerTools, ok := usage[OpenRouterUsageMetricServerToolUseDetails].(map[string]any)
	if !ok || streamServerTools["tool_calls_requested"] != float64(1) {
		t.Fatalf("stream server tool details = %v", usage[OpenRouterUsageMetricServerToolUseDetails])
	}
}

func testOpenRouterStreamMapsMidstreamError(t *testing.T) {
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

func testOpenRouterBuildRequestReplaysReasoningAndMultimodalInput(t *testing.T) {
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

func testOpenRouterGenerateReturnsContentPolicyErrorForRefusal(t *testing.T) {
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

func testOpenRouterGenerateReturnsContentPolicyErrorForContentFilter(t *testing.T) {
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

func testOpenRouterGenerator_Generate(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	gen := newLiveOpenRouterGenerator(t, apiKey)
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
func testOpenRouterGenerator_Generate_image(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		t.Skip("could not open sample.jpg")
		return
	}
	imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))
	// Use a vision-capable model through OpenRouter.
	gen := newLiveOpenRouterGenerator(t, apiKey)
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
func testOpenRouterGenerator_RequestTools(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	gen := newLiveOpenRouterGenerator(t, apiKey)
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
func testOpenRouterGenerator_Generate_reasoningModel(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	// Use a reasoning model through OpenRouter.
	gen := newLiveOpenRouterGenerator(t, apiKey)
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
func testOpenRouterGenerator_Generate_invalidModel(t *testing.T) {
	// This example demonstrates handling of invalid model IDs with OpenRouter.
	// OpenRouter returns a 400 status code with error details in the response body
	// for invalid requests like nonsense model IDs.
	apiKey := requireLiveAPIKey(t, "OPENROUTER_API_KEY")
	// Use a nonsense model ID to trigger an error.
	gen := newLiveOpenRouterGenerator(t, apiKey)
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
