package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
)

func newOpenRouterTestGenerator(t *testing.T, server *httptest.Server) *OpenRouterGenerator {
	t.Helper()
	generator, err := NewOpenRouterGenerator(server.Client(), server.URL, "test-key")
	if err != nil {
		t.Fatalf("create OpenRouter generator: %v", err)
	}
	if _, ok := any(generator).(TokenCounter); ok {
		t.Fatal("OpenRouter must not expose OpenAI's local counter")
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
	t.Run("RequestIdentity", func(t *testing.T) {
		for _, tc := range []struct{ name, user, cache string }{
			{"absent", "", ""}, {"user only", "end-user", ""},
			{"cache only", "", "conversation"}, {"both", "end-user", "conversation"},
		} {
			for _, streaming := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%v", tc.name, streaming), func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						var body map[string]json.RawMessage
						if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
							t.Error(err)
							return
						}
						for key, want := range map[string]string{"user": tc.user, "prompt_cache_key": tc.cache} {
							if want == "" {
								if _, exists := body[key]; exists {
									t.Errorf("unexpected %s", key)
								}
								continue
							}
							var got string
							if err := json.Unmarshal(body[key], &got); err != nil || got != want {
								t.Errorf("%s = %s, want %q", key, body[key], want)
							}
						}
						if _, exists := body["safety_identifier"]; exists {
							t.Error("OpenAI identity field leaked")
						}
						if streaming {
							w.Header().Set("Content-Type", "text/event-stream")
							fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
						} else {
							w.Header().Set("Content-Type", "application/json")
							fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
						}
					}))
					defer server.Close()
					g := newOpenRouterTestGenerator(t, server)
					request := GenerationRequest{Model: "test", SafetyIdentifier: tc.user, PromptCacheKey: tc.cache, Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
					if streaming {
						for chunk := range g.Stream(t.Context(), request) {
							if chunk.Err != nil {
								t.Fatal(chunk.Err)
							}
						}
					} else if _, err := g.Generate(t.Context(), request); err != nil {
						t.Fatal(err)
					}
				})
			}
		}
	})
	t.Run("WireReplay", func(t *testing.T) {
		var requests []map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("X-Invocation") != "router" {
				t.Error("missing invocation header")
			}
			var request map[string]any
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Error(err)
			}
			requests = append(requests, request)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"replay","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"Checking.","reasoning_details":[{"type":"reasoning.text","text":"private reasoning","id":"reasoning_1","format":"anthropic-claude-v1","index":3,"signature":"signed","future":null}],"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}}]}`))
		}))
		defer server.Close()
		generator := newOpenRouterTestGenerator(t, server)
		request := GenerationRequest{
			Model: "test/model", Headers: http.Header{"X-Invocation": {"router"}},
			Instructions:     SystemMessage(TextBlock("Never reveal secrets."), TextBlock("Answer concisely.")),
			SafetyIdentifier: "end-user", PromptCacheKey: "conversation",
			Dialog:  Dialog{{Role: User, Blocks: []Block{TextBlock("What is shown?"), ImageBlock([]byte("image"), "image/png"), AudioBlock([]byte("audio"), "audio/wav"), PDFBlock([]byte("pdf"), "paper.pdf")}}},
			Options: NewGenerationOptions(WithReasoningEffort("future-effort")),
		}
		response, err := generator.Generate(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		request.Dialog = append(request.Dialog, response.Candidates[0], ToolResultMessage("call_1", TextBlock("sunny")))
		if _, err := generator.Generate(t.Context(), request); err != nil {
			t.Fatal(err)
		}
		wire := requests[1]
		if wire["user"] != "end-user" || wire["safety_identifier"] != nil || wire["prompt_cache_key"] != "conversation" || wire["reasoning_effort"] != nil {
			t.Fatalf("identity/cache/effort mapping: %v", wire)
		}
		if wire["reasoning"].(map[string]any)["effort"] != "future-effort" {
			t.Fatalf("reasoning: %v", wire)
		}
		messages := wire["messages"].([]any)
		if len(messages) != 4 {
			t.Fatalf("messages: %v", messages)
		}
		if got := messages[0].(map[string]any)["content"]; got != "Never reveal secrets.\n\nAnswer concisely." {
			t.Fatalf("instruction block boundaries lost: %q", got)
		}
		parts := messages[1].(map[string]any)["content"].([]any)
		if len(parts) != 4 || parts[1].(map[string]any)["image_url"].(map[string]any)["url"] != "data:image/png;base64,aW1hZ2U=" || parts[2].(map[string]any)["type"] != "input_audio" || parts[3].(map[string]any)["type"] != "file" {
			t.Fatalf("multimodal parts: %v", parts)
		}
		assistant := messages[2].(map[string]any)
		details := assistant["reasoning_details"].([]any)
		detail := details[0].(map[string]any)
		if detail["signature"] != "signed" || detail["index"] != float64(3) || detail["id"] != "reasoning_1" || detail["text"] != "private reasoning" {
			t.Fatalf("reasoning replay: %v", details)
		}
		if _, present := detail["future"]; !present {
			t.Fatal("lost opaque null")
		}
		if assistant["content"] != "Checking." || len(assistant["tool_calls"].([]any)) != 1 {
			t.Fatalf("assistant: %v", assistant)
		}
		result := messages[3].(map[string]any)
		if result["tool_call_id"] != "call_1" || result["content"] != "sunny" {
			t.Fatalf("tool result: %v", result)
		}
	})
	t.Run("NativeOptionsServerValidated", func(t *testing.T) {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprint(streaming), func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if body["top_logprobs"] != float64(30) || body["service_tier"] != "future-tier" || body["session_id"] != strings.Repeat("x", 257) || body["max_tokens"] != float64(64) {
						t.Errorf("native fields: %v", body)
					}
					if format, ok := body["response_format"].(map[string]any); !ok || format["schema"] == nil || format["type"] != nil {
						t.Errorf("server-validated response format: %v", body["response_format"])
					}
					if body["min_p"] != 0.9 || body["reasoning"].(map[string]any)["effort"] != "explicit" {
						t.Errorf("explicit override: %v", body)
					}
					if body["user"] != "native-user" || body["prompt_cache_key"] != "native-cache" {
						t.Errorf("native identity: %v", body)
					}
					w.Header().Set("Content-Type", "application/json")
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
						return
					}
					_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
				}))
				defer server.Close()
				options := NewGenerationOptions(WithTopLogprobs(30), WithOpenRouterServiceTier("future-tier"), WithOpenRouterSessionID(strings.Repeat("x", 257)), WithOpenRouterResponseFormat(map[string]any{"schema": map[string]any{}}), WithOpenRouterMinP(-1), WithThinkingBudget(-10), WithMaxGenerationTokens(64), WithOpenAIExtraBody(map[string]json.RawMessage{"min_p": json.RawMessage(`0.9`), "reasoning": json.RawMessage(`{"effort":"explicit"}`), "user": json.RawMessage(`"native-user"`), "prompt_cache_key": json.RawMessage(`"native-cache"`)}))
				request := GenerationRequest{Model: "test/model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: options, SafetyIdentifier: "end-user", PromptCacheKey: "conversation"}
				generator := newOpenRouterTestGenerator(t, server)
				if streaming {
					for chunk := range generator.Stream(t.Context(), request) {
						if chunk.Err != nil {
							t.Fatal(chunk.Err)
						}
					}
				} else if _, err := generator.Generate(t.Context(), request); err != nil {
					t.Fatal(err)
				}
				if _, present := options[OpenAIGenerationOptionTokenLimitField]; present {
					t.Fatal("mutated caller options")
				}
				if options[GenerationOptionThinkingBudget] != -10 {
					t.Fatal("mutated budget")
				}
			})
		}
	})
	t.Run("OptionTypesAndConflicts", func(t *testing.T) {
		for _, options := range []GenerationOptions{
			{OpenRouterGenerationOptionModels: "wrong"},
			{OpenRouterGenerationOptionMinP: "wrong"},
			{OpenRouterGenerationOptionProvider: true},
			{GenerationOptionThinkingBudget: "4096"},
			NewGenerationOptions(WithThinkingBudget(4096), WithReasoningEffort("high")),
		} {
			_, err := (&OpenRouterGenerator{}).prepareRequest(GenerationRequest{Options: options})
			var invalid *InvalidParameterErr
			if !errors.As(err, &invalid) {
				t.Fatalf("options %v: expected invalid parameter, got %v", options, err)
			}
		}
	})
	t.Run("OpenRouterGenerateReturnsContentPolicyErrorForContentFilter", func(t *testing.T) {
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
	})
	t.Run("OpenRouterGenerateReturnsContentPolicyErrorForRefusal", func(t *testing.T) {
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
	})
	t.Run("OpenRouterHTTPOverloadMapping", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Retry-After", "11")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"code":503,"message":"Upstream request failed","metadata":{"error_type":"provider_overloaded"}}}`)
		}))
		defer server.Close()
		g, err := NewOpenRouterGenerator(server.Client(), server.URL, "test-key")
		if err != nil {
			t.Fatal(err)
		}
		_, err = g.Generate(t.Context(), GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}})
		var got *ApiErr
		if !errors.As(err, &got) {
			t.Fatalf("Generate() = %T %v, want *ApiErr", err, err)
		}
		if got.Kind != APIErrorKindOverloaded {
			t.Fatalf("Kind = %q, want %q", got.Kind, APIErrorKindOverloaded)
		}
		if !got.Retryable() {
			t.Fatal("Retryable() = false, want true")
		}
		if delay, ok := got.RetryAfter(); !ok || delay != 11*time.Second {
			t.Fatalf("RetryAfter() = (%s, %t), want (11s, true)", delay, ok)
		}
		if got.Provider != ProviderOpenRouter || got.StatusCode != http.StatusServiceUnavailable || !strings.Contains(got.RawBody, "provider_overloaded") {
			t.Fatalf("error details lost: %+v", got)
		}
	})
	t.Run("OpenRouterGeneratorSurfacesProviderOverload", func(t *testing.T) {
		tests := []struct {
			name        string
			body        string
			wantPartial bool
		}{
			{
				name: "top-level error_type",
				body: `{"error":{"code":503,"message":"Upstream request failed","error_type":"provider_overloaded"}}`,
			},
			{
				name: "metadata takes precedence",
				body: `{"error":{"code":503,"message":"Upstream request failed","error_type":"authentication","metadata":{"error_type":"provider_overloaded"}}}`,
			},
			{
				name: "top-level error",
				body: `{"error":{"code":503,"message":"Invalid API key","metadata":{"error_type":"provider_overloaded"}}}`,
			},
			{
				name:        "choice error with interrupted tool arguments",
				body:        `{"choices":[{"index":0,"message":{"role":"assistant","content":"partial output","tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}]},"finish_reason":"error","error":{"code":503,"message":"Upstream request failed","metadata":{"error_type":"provider_overloaded"}}}]}`,
				wantPartial: true,
			},
			{
				name:        "top-level error with interrupted tool arguments",
				body:        `{"error":{"code":503,"message":"Upstream request failed","metadata":{"error_type":"provider_overloaded"}},"choices":[{"index":0,"message":{"role":"assistant","content":"partial output","tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}]},"finish_reason":"error"}]}`,
				wantPartial: true,
			},
			{
				name:        "choice error after partial output",
				body:        `{"id":"gen-123","object":"chat.completion","created":1,"model":"test/model","choices":[{"index":0,"message":{"role":"assistant","content":"partial output"},"finish_reason":"error","error":{"code":503,"message":"Upstream request failed","metadata":{"error_type":"provider_overloaded"}}}]}`,
				wantPartial: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(tt.body))
				}))
				defer server.Close()

				generator := newOpenRouterTestGenerator(t, server)
				response, err := generator.Generate(context.Background(), GenerationRequest{
					Model: "test/model",
					Dialog: Dialog{{
						Role:   User,
						Blocks: []Block{TextBlock("hello")},
					}},
				})
				if tt.wantPartial {
					if len(response.Candidates) != 1 || len(response.Candidates[0].Blocks) != 1 ||
						response.Candidates[0].Blocks[0].Content.String() != "partial output" {
						t.Fatalf("partial response = %+v", response)
					}
				} else if len(response.Candidates) != 0 {
					t.Fatalf("top-level error response = %+v", response)
				}

				var apiErr *ApiErr
				if !errors.As(err, &apiErr) {
					t.Fatalf("Generate() error = %v, want *ApiErr", err)
				}
				if apiErr.Provider != ProviderOpenRouter {
					t.Fatalf("Provider = %q, want %q", apiErr.Provider, ProviderOpenRouter)
				}
				if apiErr.StatusCode != http.StatusServiceUnavailable {
					t.Fatalf("StatusCode = %d, want %d", apiErr.StatusCode, http.StatusServiceUnavailable)
				}
				if apiErr.Kind != APIErrorKindOverloaded {
					t.Fatalf("Kind = %q, want %q", apiErr.Kind, APIErrorKindOverloaded)
				}
				if !apiErr.Retryable() {
					t.Fatal("Retryable() = false, want true")
				}
			})
		}
	})
	t.Run("OpenRouterGeneratorUsesSharedJSONClient", func(t *testing.T) {
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
			Model:            "test/model",
			Instructions:     SystemMessage(TextBlock("Be concise.")),
			SafetyIdentifier: "user-1",
			PromptCacheKey:   "conversation-1",
			Dialog:           Dialog{{Role: User, Blocks: []Block{TextBlock("Weather?")}}},
			Tools: []Tool{{
				Name:        "get_weather",
				Description: "Get weather.",
				InputSchema: schema,
			}},
			Options: NewGenerationOptions(
				WithTemperature(0.2),
				WithTopP(0.8),
				WithTopK(40),
				WithLogitBias(map[string]float64{"42": -1.5}),
				WithLogprobs(true),
				WithTopLogprobs(3),
				WithOpenRouterMinP(0.05),
				WithOpenRouterFallbackModels("fallback/one", "fallback/two"),
				WithParallelToolCalls(false),
				WithOpenRouterPrediction("known output"),
				WithOpenRouterProviderPreferences(map[string]any{"order": []string{"ExampleAI"}, "allow_fallbacks": false}),
				WithOpenRouterRepetitionPenalty(1.1),
				WithOpenRouterResponseFormat(map[string]any{"type": "json_object"}),
				WithSeed(7),
				WithOpenRouterServiceTier(OpenRouterServiceTierPriority),
				WithOpenRouterSessionID("session-1"),
				WithOpenRouterTopA(0.2),
				WithFrequencyPenalty(0.1),
				WithPresencePenalty(0.3),
				WithCandidateCount(2),
				WithMaxGenerationTokens(64),
				WithStopSequences("END", "STOP"),
				WithToolChoice("get_weather"),
				WithThinkingBudget(2048),
			),
		})
		if err != nil {
			t.Fatalf("generate: %v", err)
		}
		if response.FinishReason != ToolUse || len(response.Candidates) != 1 {
			t.Fatalf("response = %+v", response)
		}
		thinking := requireBlockType(t, response, Thinking)
		if thinking.Content.String() != "thinking" || thinking.ExtraFields[OpenAIExtraFieldReasoningSource] != "reasoning_details" {
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
		if routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldWireFields])["id"] != "completion_1" ||
			routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldWireFields])["model"] != "test/model" ||
			routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldWireFields])["created"] != float64(1) ||
			routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldWireFields])["system_fingerprint"] != "fp_test" ||
			routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldWireFields])["service_tier"] != "priority" {
			t.Fatalf("response extra fields = %v", response.ExtraFields)
		}
		nativeMetadata, ok := routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldWireFields])["openrouter_metadata"].(map[string]any)
		if !ok || nativeMetadata["provider"] != "ExampleAI" || nativeMetadata["region"] != "us-east" {
			t.Fatalf("OpenRouter metadata = %v", routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldWireFields])["openrouter_metadata"])
		}
		logprobs, ok := routerRawObject(t, response.Candidates[0].ExtraFields[OpenAIExtraFieldChoice])["logprobs"].(map[string]any)
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
			routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["cost"] != 0.0012 ||
			routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["is_byok"] != true {
			t.Fatalf("usage metadata = %v", response.UsageMetadata)
		}
		costDetails, ok := routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["cost_details"].(map[string]any)
		if !ok || costDetails["upstream_inference_cost"] != 0.001 {
			t.Fatalf("cost details = %v", routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["cost_details"])
		}
		serverToolDetails, ok := routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["server_tool_use_details"].(map[string]any)
		if !ok || serverToolDetails["tool_calls_requested"] != float64(2) || serverToolDetails["web_search_requests"] != float64(1) {
			t.Fatalf("server tool details = %v", routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["server_tool_use_details"])
		}
		promptDetails, ok := routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["prompt_tokens_details"].(map[string]any)
		if !ok || promptDetails["audio_tokens"] != float64(2) || promptDetails["video_tokens"] != float64(1) || promptDetails["cache_write_tokens"] != float64(3) {
			t.Fatalf("prompt token details = %v", routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["prompt_tokens_details"])
		}
		completionDetails, ok := routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["completion_tokens_details"].(map[string]any)
		if !ok || completionDetails["accepted_prediction_tokens"] != float64(2) || completionDetails["rejected_prediction_tokens"] != float64(1) || completionDetails["reasoning_tokens"] != float64(2) {
			t.Fatalf("completion token details = %v", routerRawObject(t, response.ExtraFields[OpenAIResponseExtraFieldUsage])["completion_tokens_details"])
		}

		request := <-requests
		if request["stream"] == true || request["temperature"] != 0.2 || request["top_p"] != 0.8 ||
			request["top_k"] != float64(40) || request["logprobs"] != true || request["top_logprobs"] != float64(3) ||
			request["min_p"] != 0.05 || request["parallel_tool_calls"] != false ||
			request["prompt_cache_key"] != "conversation-1" || request["repetition_penalty"] != 1.1 ||
			request["seed"] != float64(7) || request["service_tier"] != "priority" ||
			request["session_id"] != "session-1" || request["top_a"] != 0.2 || request["user"] != "user-1" ||
			request["frequency_penalty"] != 0.1 || request["presence_penalty"] != 0.3 ||
			request["n"] != float64(2) || request["max_tokens"] != float64(64) {
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
	})
	t.Run("OpenRouterGeneratorUsesSharedSSEClient", func(t *testing.T) {
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
		if routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldWireFields])["id"] != "chunk_1" ||
			routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldWireFields])["system_fingerprint"] != "fp_stream" ||
			routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldWireFields])["service_tier"] != "flex" {
			t.Fatalf("stream response extra fields = %v", responseExtraFields)
		}
		streamMetadata, ok := routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldWireFields])["openrouter_metadata"].(map[string]any)
		if !ok || streamMetadata["provider"] != "ExampleAI" {
			t.Fatalf("stream OpenRouter metadata = %v", routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldWireFields])["openrouter_metadata"])
		}
		streamLogprobs, ok := routerRawObject(t, messageExtraFields[OpenAIExtraFieldChoice])["logprobs"].(map[string]any)
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
		if len(compressed) != 5 {
			t.Fatalf("compressed blocks = %+v", compressed)
		}
		for i, text := range []string{"step ", "one", "summary"} {
			if compressed[i].BlockType != Thinking || compressed[i].Content.String() != text || compressed[i].ExtraFields[OpenAIExtraFieldReasoningSource] != "reasoning_details" {
				t.Fatalf("thinking display = %+v", compressed[i])
			}
		}
		details := routerRawObject(t, messageExtraFields[OpenAIExtraFieldWireFields])["reasoning_details"].([]any)
		if len(details) != 2 || details[0].(map[string]any)["text"] != "step one" || details[1].(map[string]any)["summary"] != "summary" || details[0].(map[string]any)["id"] != "reasoning_1" {
			t.Fatalf("native reasoning = %v", details)
		}
		if compressed[3].BlockType != Content || compressed[3].Content.String() != "answer" {
			t.Fatalf("content block = %+v", compressed[3])
		}
		var call ToolCallInput
		if compressed[4].BlockType != ToolCall || json.Unmarshal([]byte(compressed[4].Content.String()), &call) != nil {
			t.Fatalf("tool call block = %+v", compressed[4])
		}
		if call.Name != "calculate" || call.Parameters["x"] != float64(1) {
			t.Fatalf("streamed tool call = %+v", call)
		}
		nativeUsage := routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldUsage])
		if nativeUsage["prompt_tokens_details"].(map[string]any)["cache_write_tokens"] != float64(2) || nativeUsage["completion_tokens_details"].(map[string]any)["reasoning_tokens"] != float64(4) {
			t.Fatalf("native usage = %v", nativeUsage)
		}
		if usage[UsageMetricInputTokens] != float64(7) || usage[UsageMetricGenerationTokens] != float64(5) ||
			usage[UsageMetricCacheReadTokens] != float64(3) ||
			usage[UsageMetricCacheWriteTokens] != float64(2) || usage[UsageMetricReasoningTokens] != float64(4) || routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldUsage])["cost"] != float64(0) ||
			routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldUsage])["is_byok"] != false {
			t.Fatalf("usage metadata = %v", usage)
		}
		streamCostDetails, ok := routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldUsage])["cost_details"].(map[string]any)
		if !ok || streamCostDetails["upstream_inference_prompt_cost"] != 0.1 {
			t.Fatalf("stream cost details = %v", routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldUsage])["cost_details"])
		}
		streamServerTools, ok := routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldUsage])["server_tool_use_details"].(map[string]any)
		if !ok || streamServerTools["tool_calls_requested"] != float64(1) {
			t.Fatalf("stream server tool details = %v", routerRawObject(t, responseExtraFields[OpenAIResponseExtraFieldUsage])["server_tool_use_details"])
		}
	})
	t.Run("OpenRouterGenerator/Generate", func(t *testing.T) {
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
	})
	t.Run("OpenRouterGenerator/Generate/image", func(t *testing.T) {
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
	})
	t.Run("OpenRouterGenerator/Generate/invalidModel", func(t *testing.T) {
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
	})
	t.Run("OpenRouterGenerator/Generate/reasoningModel", func(t *testing.T) {
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
			Options:      NewGenerationOptions(WithReasoningEffort("low")),
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
	})
	t.Run("OpenRouterGenerator/RequestTools", func(t *testing.T) {
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
	})
	t.Run("OpenRouterStreamMapsMidstreamError", func(t *testing.T) {
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
	})
}

func routerRawObject(t *testing.T, value any) map[string]any {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatal(err)
	}
	return result
}
