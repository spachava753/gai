package gai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
)

func newCerebrasTestGenerator(t *testing.T, server *httptest.Server) *CerebrasGenerator {
	t.Helper()
	generator, err := NewCerebrasGenerator(server.Client(), server.URL, "test-key")
	if err != nil {
		t.Fatalf("create Cerebras generator: %v", err)
	}
	return generator
}

func newLiveCerebrasGenerator(t *testing.T, apiKey string) *CerebrasGenerator {
	t.Helper()
	generator, err := NewCerebrasGenerator(nil, "", apiKey)
	if err != nil {
		t.Fatalf("create Cerebras generator: %v", err)
	}
	return generator
}

func TestCerebrasAdapterScenarios(t *testing.T) {
	t.Run("CerebrasBuildRequestRejectsInvalidProviderOptions", func(t *testing.T) {
		tests := []struct {
			name    string
			options GenerationOptions
			want    string
		}{
			{
				name:    "top logprobs without logprobs",
				options: NewGenerationOptions(WithCerebrasTopLogprobs(3)),
				want:    CerebrasGenerationOptionTopLogprobs,
			},
			{
				name:    "invalid service tier",
				options: NewGenerationOptions(WithCerebrasServiceTier("fast")),
				want:    CerebrasGenerationOptionServiceTier,
			},
			{
				name:    "invalid response format",
				options: NewGenerationOptions(WithCerebrasResponseFormat(map[string]any{"type": "xml"})),
				want:    CerebrasGenerationOptionResponseFormat,
			},
			{
				name:    "invalid logit bias",
				options: NewGenerationOptions(WithCerebrasLogitBias(map[string]float64{"42": 101})),
				want:    CerebrasGenerationOptionLogitBias,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				_, err := (&CerebrasGenerator{}).buildRequest(GenerationRequest{
					Model:   "gpt-oss-120b",
					Dialog:  Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}},
					Options: tt.options,
				})
				if err == nil || !strings.Contains(err.Error(), tt.want) {
					t.Fatalf("buildRequest() error = %v, want parameter %q", err, tt.want)
				}
			})
		}
	})
	t.Run("CerebrasBuildRequestRejectsUnsupportedImageInput", func(t *testing.T) {
		_, err := (&CerebrasGenerator{}).buildRequest(GenerationRequest{
			Model: "gemma-4-31b",
			Dialog: Dialog{{Role: User, Blocks: []Block{{
				BlockType: Content, ModalityType: Image, MimeType: "image/gif", Content: Str("R0lG"),
			}}}},
		})
		if err == nil || !strings.Contains(err.Error(), "unsupported image MIME type") {
			t.Fatalf("error = %v, want unsupported image MIME type", err)
		}
	})
	t.Run("CerebrasBuildRequestReplaysThinking", func(t *testing.T) {
		toolCall, err := ToolCallBlock("call_1", "get_weather", map[string]any{"city": "Paris"})
		if err != nil {
			t.Fatalf("create tool call: %v", err)
		}
		request, err := (&CerebrasGenerator{}).buildRequest(GenerationRequest{
			Model: "gpt-oss-120b",
			Dialog: Dialog{
				{Role: User, Blocks: []Block{TextBlock("Weather?")}},
				{Role: Assistant, Blocks: []Block{cerebrasThinkingBlock("private reasoning"), TextBlock("Checking."), toolCall}},
				ToolResultMessage("call_1", TextBlock("sunny")),
			},
		})
		if err != nil {
			t.Fatalf("build request: %v", err)
		}
		if len(request.Messages) != 3 {
			t.Fatalf("messages = %d, want 3", len(request.Messages))
		}
		assistant, ok := request.Messages[1].GetAssistantMessage()
		if !ok || assistant.Reasoning.Or("") != "private reasoning" ||
			assistant.Content.Or("") != "Checking." || len(assistant.ToolCalls) != 1 {
			t.Fatalf("assistant replay = %+v", assistant)
		}
		toolResult, ok := request.Messages[2].GetToolMessage()
		if !ok || toolResult.ToolCallID != "call_1" || toolResult.Content != "sunny" {
			t.Fatalf("tool result replay = %+v", toolResult)
		}
	})
	t.Run("CerebrasBuildRequestSupportsImageInput", func(t *testing.T) {
		request, err := (&CerebrasGenerator{}).buildRequest(GenerationRequest{
			Model: "gemma-4-31b",
			Dialog: Dialog{{Role: User, Blocks: []Block{
				TextBlock("Describe this image."),
				ImageBlock([]byte("png"), "image/png"),
			}}},
		})
		if err != nil {
			t.Fatalf("build request: %v", err)
		}
		encoded, err := json.Marshal(request)
		if err != nil {
			t.Fatalf("marshal request: %v", err)
		}
		var payload map[string]any
		if err := json.Unmarshal(encoded, &payload); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		messages, ok := payload["messages"].([]any)
		if !ok || len(messages) != 1 {
			t.Fatalf("messages = %v", payload["messages"])
		}
		message, ok := messages[0].(map[string]any)
		if !ok {
			t.Fatalf("message = %T %v", messages[0], messages[0])
		}
		parts, ok := message["content"].([]any)
		if !ok || len(parts) != 2 {
			t.Fatalf("content parts = %v", message["content"])
		}
		textPart, _ := parts[0].(map[string]any)
		imagePart, _ := parts[1].(map[string]any)
		imageURL, _ := imagePart["image_url"].(map[string]any)
		if textPart["type"] != "text" || textPart["text"] != "Describe this image." ||
			imagePart["type"] != "image_url" || imageURL["url"] != "data:image/png;base64,cG5n" {
			t.Fatalf("content parts = %v", parts)
		}
	})
	t.Run("CerebrasGeneratorMapsGeneratedError", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limit"}}`))
		}))
		defer server.Close()

		_, err := newCerebrasTestGenerator(t, server).Generate(t.Context(), GenerationRequest{
			Model:  "gpt-oss-120b",
			Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}},
		})
		var apiErr *ApiErr
		if !errors.As(err, &apiErr) {
			t.Fatalf("error = %T %v, want ApiErr", err, err)
		}
		if apiErr.Provider != ProviderCerebras || apiErr.Kind != APIErrorKindRateLimit ||
			apiErr.StatusCode != http.StatusTooManyRequests || apiErr.Message != "slow down" {
			t.Fatalf("API error = %+v", apiErr)
		}
	})
	t.Run("CerebrasGeneratorReturnsContentPolicyError", func(t *testing.T) {
		tests := []struct {
			name       string
			choiceJSON string
			want       string
		}{
			{
				name:       "content filter finish reason",
				choiceJSON: `{"index":0,"finish_reason":"content_filter","message":{"role":"assistant","content":""}}`,
				want:       "content policy violation detected",
			},
			{
				name:       "message refusal",
				choiceJSON: `{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"","refusal":"I cannot help with that."}}`,
				want:       "I cannot help with that.",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{"id":"chatcmpl_123","object":"chat.completion","created":0,"model":"test","choices":[` + tt.choiceJSON + `]}`))
				}))
				defer server.Close()

				generator := newCerebrasTestGenerator(t, server)
				response, err := generator.Generate(context.Background(), GenerationRequest{
					Model:  "test",
					Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
				})
				if response.FinishReason != ContentPolicyViolation {
					t.Fatalf("FinishReason = %v, want ContentPolicyViolation", response.FinishReason)
				}

				var policyErr ContentPolicyErr
				if !errors.As(err, &policyErr) {
					t.Fatalf("Generate error = %T %v, want ContentPolicyErr", err, err)
				}
				if !strings.Contains(policyErr.Error(), tt.want) {
					t.Fatalf("Generate error = %q, want message containing %q", policyErr, tt.want)
				}
			})
		}
	})
	t.Run("CerebrasGeneratorUsesGeneratedJSONClient", func(t *testing.T) {
		requests := make(chan map[string]any, 1)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
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
			"model":"gpt-oss-120b",
			"system_fingerprint":"fp_test",
			"service_tier":"auto",
			"service_tier_used":"priority",
			"choices":[{
				"index":0,
				"finish_reason":"tool_calls",
				"logprobs":{"content":[{"token":"answer","logprob":-0.1}]},
				"message":{
					"role":"assistant",
					"reasoning":"thinking",
					"content":"answer",
					"tool_calls":[{
						"id":"call_1",
						"type":"function",
						"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}
					}]
				}
			}],
			"usage":{
				"prompt_tokens":10,
				"completion_tokens":5,
				"total_tokens":15,
				"image_tokens":3,
				"prompt_tokens_details":{"cached_tokens":4},
				"completion_tokens_details":{"reasoning_tokens":2}
			},
			"time_info":{
				"queue_time":0.01,
				"prompt_time":0.02,
				"completion_time":0.03,
				"total_time":0.06,
				"created":123.5
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
		response, err := newCerebrasTestGenerator(t, server).Generate(t.Context(), GenerationRequest{
			Model:        "gpt-oss-120b",
			Instructions: SystemMessage(TextBlock("Be concise.")),
			Dialog:       Dialog{{Role: User, Blocks: []Block{TextBlock("Weather?")}}},
			Tools: []Tool{{
				Name:        "get_weather",
				Description: "Get weather.",
				InputSchema: schema,
			}},
			Options: NewGenerationOptions(
				WithTemperature(0.2),
				WithFrequencyPenalty(0.3),
				WithPresencePenalty(-0.4),
				WithCerebrasLogitBias(map[string]float64{"42": -1.5}),
				WithCerebrasLogprobs(true),
				WithCerebrasTopLogprobs(3),
				WithCerebrasParallelToolCalls(false),
				WithCerebrasPrediction("known output"),
				WithCerebrasPromptCacheKey("conversation-1"),
				WithCerebrasResponseFormat(map[string]any{"type": "json_object"}),
				WithCerebrasSeed(7),
				WithCerebrasServiceTier(CerebrasServiceTierAuto),
				WithCerebrasUser("user-1"),
				WithMaxGenerationTokens(64),
				WithStopSequences("END", "STOP"),
				WithToolChoice("get_weather"),
				WithThinkingBudget("medium"),
			),
		})
		if err != nil {
			t.Fatalf("generate: %v", err)
		}
		if response.FinishReason != ToolUse || len(response.Candidates) != 1 {
			t.Fatalf("response = %+v", response)
		}
		thinking := requireBlockType(t, response, Thinking)
		if thinking.Content.String() != "thinking" || thinking.ExtraFields[ThinkingExtraFieldGeneratorKey] != ThinkingGeneratorCerebras {
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
		if response.ExtraFields[CerebrasResponseExtraFieldID] != "completion_1" ||
			response.ExtraFields[CerebrasResponseExtraFieldModel] != "gpt-oss-120b" ||
			response.ExtraFields[CerebrasResponseExtraFieldCreated] != int64(1) ||
			response.ExtraFields[CerebrasResponseExtraFieldSystemFingerprint] != "fp_test" ||
			response.ExtraFields[CerebrasResponseExtraFieldServiceTier] != "auto" ||
			response.ExtraFields[CerebrasResponseExtraFieldServiceTierUsed] != "priority" {
			t.Fatalf("response extra fields = %v", response.ExtraFields)
		}
		logprobs, ok := response.Candidates[0].ExtraFields[CerebrasMessageExtraFieldLogprobs].(map[string]any)
		if !ok {
			t.Fatalf("candidate logprobs = %v", response.Candidates[0].ExtraFields)
		}
		logprobItems, ok := logprobs["content"].([]any)
		if !ok || len(logprobItems) != 1 {
			t.Fatalf("candidate logprobs = %v", response.Candidates[0].ExtraFields)
		}
		if response.UsageMetadata[UsageMetricInputTokens] != 10 ||
			response.UsageMetadata[UsageMetricGenerationTokens] != 5 ||
			response.UsageMetadata[UsageMetricCacheReadTokens] != 4 ||
			response.UsageMetadata[UsageMetricReasoningTokens] != 2 ||
			response.UsageMetadata[CerebrasUsageMetricImageTokens] != 3 ||
			response.UsageMetadata[CerebrasUsageMetricQueueTimeSeconds] != 0.01 ||
			response.UsageMetadata[CerebrasUsageMetricPromptTimeSeconds] != 0.02 ||
			response.UsageMetadata[CerebrasUsageMetricCompletionTimeSeconds] != 0.03 ||
			response.UsageMetadata[CerebrasUsageMetricTotalTimeSeconds] != 0.06 ||
			response.UsageMetadata[CerebrasUsageMetricTimeInfoCreated] != 123.5 {
			t.Fatalf("usage metadata = %v", response.UsageMetadata)
		}

		request := <-requests
		if request["stream"] != false || request["temperature"] != 0.2 ||
			request["frequency_penalty"] != 0.3 || request["presence_penalty"] != -0.4 ||
			request["logprobs"] != true || request["top_logprobs"] != float64(3) ||
			request["parallel_tool_calls"] != false || request["prompt_cache_key"] != "conversation-1" ||
			request["seed"] != float64(7) || request["service_tier"] != "auto" || request["user"] != "user-1" ||
			request["max_completion_tokens"] != float64(64) || request["reasoning_effort"] != "medium" {
			t.Fatalf("request options = %v", request)
		}
		logitBias, ok := request["logit_bias"].(map[string]any)
		if !ok || logitBias["42"] != -1.5 {
			t.Fatalf("logit bias = %v", request["logit_bias"])
		}
		prediction, ok := request["prediction"].(map[string]any)
		if !ok || prediction["type"] != "content" || prediction["content"] != "known output" {
			t.Fatalf("prediction = %v", request["prediction"])
		}
		responseFormat, ok := request["response_format"].(map[string]any)
		if !ok || responseFormat["type"] != "json_object" {
			t.Fatalf("response format = %v", request["response_format"])
		}
		stop, ok := request["stop"].([]any)
		if !ok || len(stop) != 2 || stop[0] != "END" || stop[1] != "STOP" {
			t.Fatalf("stop sequences = %v", request["stop"])
		}
		toolChoice, ok := request["tool_choice"].(map[string]any)
		if !ok || toolChoice["type"] != "function" {
			t.Fatalf("tool choice = %v", request["tool_choice"])
		}
	})
	t.Run("CerebrasGeneratorUsesGeneratedSSEClient", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var request map[string]any
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil || request["stream"] != true {
				http.Error(w, "expected streaming request", http.StatusBadRequest)
				return
			}

			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(
				"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-oss-120b\",\"system_fingerprint\":\"fp_stream\",\"service_tier\":\"default\",\"service_tier_used\":\"flex\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"thinking\"},\"finish_reason\":null,\"logprobs\":{\"content\":[{\"token\":\"a\",\"logprob\":-0.1}]}}]}\n\n" +
					"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-oss-120b\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null,\"logprobs\":{\"content\":[{\"token\":\"b\",\"logprob\":-0.2}]}}]}\n\n" +
					"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-oss-120b\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n\n" +
					"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-oss-120b\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n" +
					"data: {\"id\":\"chunk_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-oss-120b\",\"choices\":[],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":4,\"total_tokens\":10,\"image_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":2},\"completion_tokens_details\":{\"reasoning_tokens\":1}},\"time_info\":{\"queue_time\":0.1,\"prompt_time\":0.2,\"completion_time\":0.3,\"total_time\":0.6,\"created\":456.5}}\n\n" +
					"data: [DONE]\n\n",
			))
		}))
		defer server.Close()

		var thinking, content string
		var toolBlocks []Block
		var usage map[string]float64
		responseExtraFields := make(map[string]interface{})
		messageExtraFields := make(map[string]interface{})
		for chunk := range newCerebrasTestGenerator(t, server).Stream(t.Context(), GenerationRequest{
			Model:  "gpt-oss-120b",
			Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("weather")}}},
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
			switch chunk.Block.BlockType {
			case Thinking:
				thinking += chunk.Block.Content.String()
				if chunk.Block.ExtraFields[ThinkingExtraFieldGeneratorKey] != ThinkingGeneratorCerebras {
					t.Fatalf("thinking block = %+v", chunk.Block)
				}
			case Content:
				content += chunk.Block.Content.String()
			case ToolCall:
				toolBlocks = append(toolBlocks, chunk.Block)
			case MetadataBlockType:
				if err := json.Unmarshal([]byte(chunk.Block.Content.String()), &usage); err != nil {
					t.Fatalf("decode usage metadata: %v", err)
				}
			}
		}
		if thinking != "thinking" || content != "answer" {
			t.Fatalf("thinking = %q, content = %q", thinking, content)
		}
		if responseExtraFields[CerebrasResponseExtraFieldID] != "chunk_1" ||
			responseExtraFields[CerebrasResponseExtraFieldSystemFingerprint] != "fp_stream" ||
			responseExtraFields[CerebrasResponseExtraFieldServiceTier] != "default" ||
			responseExtraFields[CerebrasResponseExtraFieldServiceTierUsed] != "flex" {
			t.Fatalf("stream response extra fields = %v", responseExtraFields)
		}
		streamLogprobs, ok := messageExtraFields[CerebrasMessageExtraFieldLogprobs].(map[string]any)
		if !ok {
			t.Fatalf("stream message extra fields = %v", messageExtraFields)
		}
		logprobContent, ok := streamLogprobs["content"].([]any)
		if !ok || len(logprobContent) != 2 {
			t.Fatalf("stream logprobs = %v", streamLogprobs)
		}
		compressed, err := compressStreamingBlocks(toolBlocks)
		if err != nil {
			t.Fatalf("assemble streamed tool call: %v", err)
		}
		if len(compressed) != 1 || compressed[0].ID != "call_1" {
			t.Fatalf("assembled tool blocks = %+v", compressed)
		}
		var call ToolCallInput
		if err := json.Unmarshal([]byte(compressed[0].Content.String()), &call); err != nil {
			t.Fatalf("decode streamed tool call: %v", err)
		}
		if call.Name != "get_weather" || call.Parameters["city"] != "Paris" {
			t.Fatalf("streamed tool call = %+v", call)
		}
		if usage[UsageMetricInputTokens] != 6 || usage[UsageMetricGenerationTokens] != 4 ||
			usage[UsageMetricCacheReadTokens] != 2 || usage[UsageMetricReasoningTokens] != 1 ||
			usage[CerebrasUsageMetricImageTokens] != 2 ||
			usage[CerebrasUsageMetricQueueTimeSeconds] != 0.1 ||
			usage[CerebrasUsageMetricPromptTimeSeconds] != 0.2 ||
			usage[CerebrasUsageMetricCompletionTimeSeconds] != 0.3 ||
			usage[CerebrasUsageMetricTotalTimeSeconds] != 0.6 ||
			usage[CerebrasUsageMetricTimeInfoCreated] != 456.5 {
			t.Fatalf("usage metadata = %v", usage)
		}
	})
	t.Run("CerebrasGenerator/Generate", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
		gen := newLiveCerebrasGenerator(t, apiKey)
		dialog := Dialog{
			{
				Role:   User,
				Blocks: []Block{TextBlock("Hello!")},
			},
		}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        "gpt-oss-120b",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) != 1 || len(resp.Candidates[0].Blocks) == 0 {
			t.Fatalf("empty response: %+v", resp)
		}
	})
	t.Run("CerebrasGenerator/Generate/reasoning/gemma", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
		// Gemma supports reasoning through the reasoning_effort parameter.
		gen := newLiveCerebrasGenerator(t, apiKey)
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is 15 * 12?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        "gemma-4-31b",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant that explains your reasoning step by step.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithThinkingBudget("medium")),
		}
		// Generate a response with reasoning enabled.
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			// Check if we have thinking blocks (reasoning)
			hasThinking := false
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Thinking {
					hasThinking = true
				}
			}
			if hasThinking {
			}
			// Find the main content block (not thinking)
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Content {
					content := block.Content.String()
					if strings.Contains(content, "180") {
					}
					break
				}
			}
		}
		// Append the previous response and ask a follow-up question to test reasoning retention
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Now what is 20 * 15?"),
				},
			},
		})
		request.Dialog = dialog
		// Generate response with reasoning (the previous reasoning should be retained)
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			// Check if we have thinking blocks
			hasThinking := false
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Thinking {
					hasThinking = true
				}
			}
			if hasThinking {
			}
			// Find the main content block
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Content {
					content := block.Content.String()
					if strings.Contains(content, "300") {
					}
					break
				}
			}
		}
	})
	t.Run("CerebrasGenerator/Generate/reasoning/gptoss", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
		// Use gpt-oss-120b model which supports reasoning with reasoning_effort parameter
		gen := newLiveCerebrasGenerator(t, apiKey)
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
			Model:        "gpt-oss-120b",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant that explains your reasoning step by step.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithThinkingBudget("medium")),
		}
		// Generate response with reasoning enabled (medium effort)
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			// Check if we have thinking blocks (reasoning)
			hasThinking := false
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Thinking {
					hasThinking = true
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
		// Append the previous response and ask a follow-up question to test reasoning retention
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
		// Generate response with reasoning (the previous reasoning should be retained)
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			// Check if we have thinking blocks
			hasThinking := false
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Thinking {
					hasThinking = true
				}
			}
			if hasThinking {
			}
			// Find the main content block
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Content {
					content := block.Content.String()
					if strings.Contains(content, "15") {
					}
					break
				}
			}
		}
	})
	t.Run("CerebrasGenerator/RequestTools", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
		cgen := newLiveCerebrasGenerator(t, apiKey)
		instructions := `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`
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
			Model:        "gpt-oss-120b",
			Instructions: SystemMessage(TextBlock(instructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options:      NewGenerationOptions(WithToolChoice("get_stock_price")),
		}
		// Force the tool call
		resp, err := cgen.Generate(context.Background(), request)
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
		request.Options = NewGenerationOptions(WithToolChoice("none"))
		// Ask model to answer now without calling tools
		resp, err = cgen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
				t.Fatal("expected non-empty content")
			}
		}
	})
}
