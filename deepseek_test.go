package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type compatibleTestGenerator interface {
	Generator
	StreamingGenerator
	prepareRequest(GenerationRequest) (GenerationRequest, error)
}

func compatibleTestProviders() []struct {
	name     string
	provider Provider
	new      func(*http.Client, string, string) (compatibleTestGenerator, error)
	options  []GenerationOption
} {
	return []struct {
		name     string
		provider Provider
		new      func(*http.Client, string, string) (compatibleTestGenerator, error)
		options  []GenerationOption
	}{
		{"zai", ProviderZAI, func(c *http.Client, u, k string) (compatibleTestGenerator, error) { return NewZaiGenerator(c, u, k) }, []GenerationOption{WithZaiThinking(true), WithZaiClearThinking(false), WithZaiDoSample(false), WithZaiToolStream(true)}},
		{"deepseek", ProviderDeepSeek, func(c *http.Client, u, k string) (compatibleTestGenerator, error) {
			return NewDeepSeekGenerator(c, u, k)
		}, []GenerationOption{WithDeepSeekThinking(true)}},
		{"moonshot", ProviderMoonshot, func(c *http.Client, u, k string) (compatibleTestGenerator, error) {
			return NewMoonshotGenerator(c, u, k)
		}, []GenerationOption{WithMoonshotThinking(true), WithMoonshotKeepThinking(true)}},
	}
}

func ExampleNewMoonshotGenerator() {
	generator, err := NewMoonshotGenerator(nil, "", "application-supplied-key")
	if err != nil {
		fmt.Println(err)
		return
	}
	request := GenerationRequest{
		Model:            "kimi-k3",
		SafetyIdentifier: "opaque-application-user-id",
		PromptCacheKey:   "assistant-prefix-v1",
		Dialog:           Dialog{{Role: User, Blocks: []Block{TextBlock("Hello")}}},
		// K3 uses effort; omit thinking.type and thinking.keep.
		Options: NewGenerationOptions(WithReasoningEffort("high")),
	}
	// Call generator.Generate(ctx, request) or generator.Stream(ctx, request).
	fmt.Println(generator != nil, request.Options[GenerationOptionReasoningEffort])
	// Output: true high
}

func ExampleWithThinkingBudget() {
	// Anthropic and OpenRouter accept a numeric thinking-token budget.
	options := NewGenerationOptions(WithThinkingBudget(4096), WithMaxGenerationTokens(8192))
	fmt.Println(options[GenerationOptionThinkingBudget])
	// Output: 4096
}

func TestCompatibleProviders(t *testing.T) {
	t.Run("requests", func(t *testing.T) { testCompatibleProviderRequests(t) })
	t.Run("terminal reasons", func(t *testing.T) { testCompatibleTerminalReasons(t) })
	t.Run("errors", func(t *testing.T) { testCompatibleProviderErrors(t) })
	t.Run("options", func(t *testing.T) { testCompatibleOptionDefaultsAndOverrides(t) })
	t.Run("live", func(t *testing.T) { testCompatibleProvidersLive(t) })
}

func testCompatibleProviderRequests(t *testing.T) {
	for _, provider := range compatibleTestProviders() {
		t.Run(provider.name, func(t *testing.T) {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprint(stream), func(t *testing.T) {
					var received map[string]any
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer test-key" {
							t.Errorf("unexpected request: %s %v", r.URL, r.Header)
						}
						if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
							t.Error(err)
						}
						if stream {
							w.Header().Set("Content-Type", "text/event-stream")
							fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"reason\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n")
						} else {
							w.Header().Set("Content-Type", "application/json")
							fmt.Fprint(w, `{"id":"completion","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"reason","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}`)
						}
					}))
					defer server.Close()
					generator, err := provider.new(server.Client(), server.URL+"/v1", "test-key")
					if err != nil {
						t.Fatal(err)
					}
					opts := NewGenerationOptions(append(provider.options, WithReasoningEffort("high"), WithMaxGenerationTokens(123), WithOpenAIResponseFormat(json.RawMessage(`{"type":"json_object"}`)), WithOpenAILogprobs(true), WithOpenAITopLogprobs(2))...)
					before, _ := json.Marshal(opts)
					request := GenerationRequest{Model: "no-model-heuristics", SafetyIdentifier: "opaque-user", PromptCacheKey: "prefix", Instructions: SystemMessage(TextBlock("instruction")), Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: opts}
					var executor Generator = generator
					if stream {
						executor = &StreamingAdapter{S: generator}
					}
					response, err := executor.Generate(t.Context(), request)
					if err != nil {
						t.Fatal(err)
					}
					requireContentContaining(t, response, "hello")
					requireBlockType(t, response, Thinking)
					if response.FinishReason != EndTurn {
						t.Fatalf("finish = %v", response.FinishReason)
					}
					after, _ := json.Marshal(opts)
					if string(before) != string(after) {
						t.Fatal("mutated caller options")
					}
					if received["reasoning_effort"] != "high" || received["logprobs"] != true || received["top_logprobs"] != float64(2) {
						t.Fatalf("shared fields: %v", received)
					}
					thinking := received["thinking"].(map[string]any)
					if thinking["type"] != "enabled" {
						t.Fatalf("thinking: %v", thinking)
					}
					if provider.provider == ProviderMoonshot {
						if received["max_completion_tokens"] != float64(123) || received["safety_identifier"] != "opaque-user" || received["prompt_cache_key"] != "prefix" || thinking["keep"] != "all" {
							t.Fatalf("Moonshot fields: %v", received)
						}
					} else {
						if received["max_tokens"] != float64(123) || received["user_id"] != "opaque-user" || received["safety_identifier"] != nil || received["prompt_cache_key"] != nil {
							t.Fatalf("provider fields: %v", received)
						}
					}
					if provider.provider == ProviderZAI {
						if received["stream_options"] != nil || thinking["clear_thinking"] != false || received["do_sample"] != false || received["tool_stream"] != true {
							t.Fatalf("ZAI fields: %v", received)
						}
					} else if stream && received["stream_options"] == nil {
						t.Fatal("missing stream usage option")
					}
				})
			}
		})
	}
}

func testCompatibleTerminalReasons(t *testing.T) {
	for _, provider := range compatibleTestProviders() {
		for _, reason := range []string{"network_error", "insufficient_system_resource", "model_context_window_exceeded", "sensitive"} {
			t.Run(provider.name+"/"+reason, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var request map[string]any
					if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
						t.Error(err)
					}
					if request["stream"] == true {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":%q}]}\n\ndata: [DONE]\n\n", reason)
					} else {
						fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","content":"partial"},"finish_reason":%q}]}`, reason)
					}
				}))
				defer server.Close()
				g, err := provider.new(server.Client(), server.URL, "key")
				if err != nil {
					t.Fatal(err)
				}
				request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
				for index, executor := range []Generator{g, &StreamingAdapter{S: g}} {
					response, err := executor.Generate(t.Context(), request)
					if err == nil {
						t.Fatal("terminal failure accepted as success")
					}
					if index == 0 {
						requireContentContaining(t, response, "partial")
					}
					switch reason {
					case "model_context_window_exceeded":
						if !errors.Is(err, ErrMaxGenerationLimit) {
							t.Fatal(err)
						}
					case "sensitive":
						var policy ContentPolicyErr
						if !errors.As(err, &policy) {
							t.Fatal(err)
						}
					default:
						var api *ApiErr
						if !errors.As(err, &api) || api.Provider != provider.provider || !api.Retryable() {
							t.Fatalf("error = %#v", err)
						}
					}
				}
			})
		}
	}
}

func testCompatibleProviderErrors(t *testing.T) {
	for _, provider := range compatibleTestProviders() {
		t.Run(provider.name, func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(429)
				fmt.Fprint(w, `{"error":{"message":"slow down","code":"rate_limit"}}`)
			}))
			defer server.Close()
			g, err := provider.new(server.Client(), server.URL, "key")
			if err != nil {
				t.Fatal(err)
			}
			req := GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
			check := func(err error) {
				t.Helper()
				var api *ApiErr
				if !errors.As(err, &api) || api.Provider != provider.provider || api.StatusCode != 429 || api.Kind != APIErrorKindRateLimit {
					t.Fatalf("error = %#v", err)
				}
			}
			_, err = g.Generate(t.Context(), req)
			check(err)
			for chunk := range g.Stream(t.Context(), req) {
				check(chunk.Err)
			}
			if calls != 2 {
				t.Fatalf("calls = %d; unexpected retry", calls)
			}
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			_, err = g.Generate(ctx, req)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation: %v", err)
			}
		})
	}
}

func testCompatibleOptionDefaultsAndOverrides(t *testing.T) {
	for _, provider := range compatibleTestProviders() {
		t.Run(provider.name, func(t *testing.T) {
			request := GenerationRequest{Options: NewGenerationOptions(WithOpenAIExtraBody(map[string]json.RawMessage{"thinking": json.RawMessage(`null`)}), WithOpenAIStreamUsage(true), WithOpenAITokenLimitField("max_completion_tokens"))}
			for _, option := range provider.options {
				option(request.Options)
			}
			generator, err := provider.new(nil, "", "test-key")
			if err != nil {
				t.Fatal(err)
			}
			result, err := generator.prepareRequest(request)
			if err != nil {
				t.Fatal(err)
			}
			fields := result.Options[OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)
			if string(fields["thinking"]) != "null" || result.Options[OpenAIGenerationOptionStreamUsage] != true || result.Options[OpenAIGenerationOptionTokenLimitField] != "max_completion_tokens" {
				t.Fatalf("explicit options overridden: %v", result.Options)
			}
			result, err = generator.prepareRequest(GenerationRequest{})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Options[OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)) != 0 {
				t.Fatal("thinking default injected")
			}
			// A provider must not validate or translate another provider's options.
			foreign := GenerationOptions{
				ZaiGenerationOptionThinkingEnabled:      "invalid",
				ZaiGenerationOptionClearThinking:        "invalid",
				ZaiGenerationOptionDoSample:             "invalid",
				ZaiGenerationOptionToolStream:           "invalid",
				DeepSeekGenerationOptionThinkingEnabled: "invalid",
				MoonshotGenerationOptionThinkingEnabled: "invalid",
				MoonshotGenerationOptionKeepThinking:    "invalid",
			}
			for key := range NewGenerationOptions(provider.options...) {
				delete(foreign, key)
			}
			result, err = generator.prepareRequest(GenerationRequest{Options: foreign})
			if err != nil {
				t.Fatalf("foreign options rejected: %v", err)
			}
			if len(result.Options[OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)) != 0 {
				t.Fatal("foreign provider options were translated")
			}
		})
	}
	g, err := NewDeepSeekGenerator(nil, "", "key")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := any(g).(TokenCounter); ok {
		t.Fatal("DeepSeek must not implement TokenCounter")
	}
	opts := NewGenerationOptions(WithMoonshotKeepThinking(false))
	req, err := (&MoonshotGenerator{}).prepareRequest(GenerationRequest{Options: opts})
	if err != nil {
		t.Fatal(err)
	}
	if string(req.Options[OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)["thinking"]) != `{"keep":null}` {
		t.Fatal("missing explicit null")
	}
	_, err = (&ZaiGenerator{}).prepareRequest(GenerationRequest{Options: GenerationOptions{ZaiGenerationOptionDoSample: "false"}})
	if err == nil {
		t.Fatal("invalid option accepted")
	}
}

func TestNativeCompatibleTokenCounts(t *testing.T) {
	for _, provider := range compatibleTestProviders() {
		if provider.provider == ProviderDeepSeek {
			continue
		}
		t.Run(provider.name, func(t *testing.T) {
			for _, total := range []string{"42", "0", "-1", "1.5", "null", "18446744073709551616"} {
				t.Run(total, func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						path := "/v1/tokenizer"
						field := "usage"
						if provider.provider == ProviderMoonshot {
							path = "/v1/tokenizers/estimate-token-count"
							field = "data"
						}
						if r.URL.Path != path || r.Header.Get("Authorization") != "Bearer key" {
							t.Errorf("request: %s", r.URL)
						}
						var body map[string]any
						if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
							t.Error(err)
						}
						if len(body) != 2 || body["model"] != "model" || len(body["messages"].([]any)) != 2 {
							t.Errorf("body: %v", body)
						}
						fmt.Fprintf(w, `{"%s":{"total_tokens":%s}}`, field, total)
					}))
					defer server.Close()
					g, err := provider.new(server.Client(), server.URL+"/v1", "key")
					if err != nil {
						t.Fatal(err)
					}
					n, err := g.(TokenCounter).Count(t.Context(), GenerationRequest{Model: "model", Instructions: SystemMessage(TextBlock("system")), Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}})
					valid := total == "42" || total == "0"
					if valid != (err == nil) {
						t.Fatalf("count = %d, %v", n, err)
					}
					if total == "42" && n != 42 {
						t.Fatal(n)
					}
				})
			}
		})
	}
}

func TestTokenizerContracts(t *testing.T) {
	for _, provider := range compatibleTestProviders() {
		if provider.provider == ProviderDeepSeek {
			continue
		}
		t.Run(provider.name, func(t *testing.T) {
			calls := 0
			status := http.StatusOK
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if status != http.StatusOK {
					w.WriteHeader(status)
					fmt.Fprint(w, `{"error":{"message":"quota"}}`)
					return
				}
				if _, ok := body["tools"]; !ok {
					t.Error("Z.AI count omitted tools")
				}
				messages := body["messages"].([]any)
				content := messages[0].(map[string]any)["content"].([]any)
				image := content[0].(map[string]any)["image_url"].(map[string]any)["url"]
				if image != "data:image/png;base64,aGVsbG8=" {
					t.Errorf("image = %v", image)
				}
				fmt.Fprint(w, `{"usage":{"total_tokens":7}}`)
			}))
			defer server.Close()
			g, err := provider.new(server.Client(), server.URL, "key")
			if err != nil {
				t.Fatal(err)
			}
			counter := g.(TokenCounter)
			request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Image, MimeType: "image/png", Content: Str("aGVsbG8=")}}}}, Tools: []Tool{{Name: "lookup"}}}
			count, err := counter.Count(t.Context(), request)
			if provider.provider == ProviderMoonshot {
				if err == nil || calls != 0 {
					t.Fatalf("undocumented tool count: %d %v (%d calls)", count, err, calls)
				}
			} else if err != nil || count != 7 {
				t.Fatalf("tool/image count: %d %v", count, err)
			}
			request.Tools = nil
			status = http.StatusTooManyRequests
			before := calls
			_, err = counter.Count(t.Context(), request)
			var api *ApiErr
			if !errors.As(err, &api) || api.Provider != provider.provider || api.StatusCode != status || calls != before+1 {
				t.Fatalf("count error: %v (%d calls)", err, calls)
			}
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			_, err = counter.Count(ctx, request)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("count cancellation: %v", err)
			}
		})
	}
}

func TestThinkingControlsAreDistinct(t *testing.T) {
	for _, options := range []GenerationOptions{NewGenerationOptions(WithThinkingBudget(0)), NewGenerationOptions(WithThinkingBudget(-1)), NewGenerationOptions(WithThinkingBudget(1024), WithReasoningEffort("high")), NewGenerationOptions(WithReasoningEffort("1024")), {GenerationOptionThinkingBudget: "1024"}} {
		if _, err := thinkingSetting(options); err == nil {
			t.Fatalf("accepted %v", options)
		}
	}
	for _, options := range []GenerationOptions{NewGenerationOptions(WithThinkingBudget(1024)), NewGenerationOptions(WithReasoningEffort("high"))} {
		if _, err := parseAnthropicGenerationOptions(options); err != nil {
			t.Fatal(err)
		}
		params, err := (&OpenRouterGenerator{}).buildRequest(GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: options})
		if err != nil {
			t.Fatal(err)
		}
		reasoning, ok := params.Reasoning.Get()
		if !ok {
			t.Fatal("missing reasoning config")
		}
		if _, budget := options[GenerationOptionThinkingBudget]; budget {
			if reasoning.MaxTokens.Or(0) != 1024 || reasoning.Effort.IsSet() {
				t.Fatalf("numeric budget = %+v", reasoning)
			}
		} else if string(reasoning.Effort.Or("")) != "high" || reasoning.MaxTokens.IsSet() {
			t.Fatalf("effort = %+v", reasoning)
		}
	}
}

func testCompatibleProvidersLive(t *testing.T) {
	for _, provider := range compatibleTestProviders() {
		t.Run(provider.name, func(t *testing.T) {
			keyEnv, model := "Z_API_KEY", "glm-5.3-flash"
			if provider.provider == ProviderDeepSeek {
				keyEnv, model = "DEEPSEEK_API_KEY", "deepseek-flash"
			}
			if provider.provider == ProviderMoonshot {
				keyEnv, model = "MOONSHOT_API_KEY", "kimi-k3"
			}
			key := requireLiveAPIKey(t, keyEnv)
			g, err := provider.new(nil, "", key)
			if err != nil {
				t.Fatal(err)
			}
			request := GenerationRequest{Model: model, Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("Reply hello.")}}}}
			response, err := g.Generate(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			requireContentContaining(t, response, "hello")
			var text strings.Builder
			for chunk := range g.Stream(t.Context(), request) {
				if chunk.Err != nil {
					t.Fatal(chunk.Err)
				}
				if chunk.Block.BlockType == Content {
					text.WriteString(chunk.Block.Content.String())
				}
			}
			if text.Len() == 0 {
				t.Fatal("empty stream")
			}
		})
	}
}
