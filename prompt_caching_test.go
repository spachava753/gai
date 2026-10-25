package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	openai "github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
)

func ExampleWithPromptCacheTTL() {
	request := GenerationRequest{
		PromptCacheKey: "shared-prefix",
		Options:        NewGenerationOptions(WithPromptCacheTTL(time.Hour)),
	}
	// Anthropic requires a key; Moonshot can also configure TTL without one.
	fmt.Println(request.Options[GenerationOptionPromptCacheTTL])
	// Output: 1h0m0s
}

func TestPromptCaching(t *testing.T) {
	t.Run("Lifetimes", func(t *testing.T) {
		for _, provider := range []string{"Anthropic", "OpenAI", "Responses", "Moonshot", "OpenRouter"} {
			for _, stream := range []bool{false, true} {
				for _, key := range []string{"", "shared-prefix"} {
					for _, tt := range []struct {
						value any
						ttl   string
					}{
						{nil, ""},
						{5 * time.Minute, "5m"},
						{time.Hour, "1h"},
						{30 * time.Minute, "30m"},
						{12 * time.Hour, "12h"},
						{24 * time.Hour, "24h"},
						{2 * time.Hour, "2h"},
						{90 * time.Second, "1m30s"},
						{1500 * time.Millisecond, "1.5s"},
						{time.Nanosecond, "1ns"},
						{time.Duration(0), ""},
						{-time.Minute, ""},
						{"1h", ""},
						{int64(time.Hour), ""},
					} {
						t.Run(fmt.Sprintf("%s/stream=%t/key=%t/%T=%v", provider, stream, key != "", tt.value, tt.value), func(t *testing.T) {
							generator, transport := newPromptCacheTestGenerator(t, provider)
							options := NewGenerationOptions()
							if duration, typed := tt.value.(time.Duration); typed {
								WithPromptCacheTTL(duration)(options)
							} else if tt.value != nil {
								options[GenerationOptionPromptCacheTTL] = tt.value
							}
							before, err := json.Marshal(options)
							if err != nil {
								t.Fatal(err)
							}
							request := GenerationRequest{Model: "no-model-heuristics", PromptCacheKey: key, Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: options}
							if stream {
								for chunk := range generator.Stream(t.Context(), request) {
									if chunk.Err != nil {
										err = chunk.Err
										break
									}
								}
							} else {
								_, err = generator.Generate(t.Context(), request)
							}
							invalid := tt.value != nil && (tt.ttl == "" || provider == "Anthropic" && key == "")
							if invalid {
								var invalid *InvalidParameterErr
								if !errors.As(err, &invalid) || invalid.Parameter != GenerationOptionPromptCacheTTL || len(transport.requests) != 0 {
									t.Fatalf("invalid TTL must fail before HTTP: err=%v requests=%v", err, transport.requests)
								}
							} else {
								if err != nil {
									t.Fatal(err)
								}
								if len(transport.requests) != 1 {
									t.Fatalf("requests = %d, want 1", len(transport.requests))
								}
								body := transport.requests[0]
								want := map[string]any{}
								if provider == "Anthropic" && key != "" {
									control := map[string]any{"type": "ephemeral"}
									if tt.value != nil {
										control["ttl"] = tt.ttl
									}
									want["cache_control"] = control
								} else if tt.value != nil {
									if provider == "Moonshot" {
										want["prompt_cache_options"] = map[string]any{"mode": "implicit", "ttl": tt.ttl}
									} else if tt.ttl == "24h" {
										want["prompt_cache_retention"] = "24h"
									} else {
										want["prompt_cache_options"] = map[string]any{"ttl": tt.ttl}
									}
								}
								assertPromptCachePolicy(t, body, want)
								if provider == "Anthropic" || key == "" {
									if _, present := body["prompt_cache_key"]; present {
										t.Fatalf("unexpected cache key: %v", body)
									}
								} else if body["prompt_cache_key"] != key {
									t.Fatalf("lost cache key: %v", body)
								}
								if _, present := body[GenerationOptionPromptCacheTTL]; present {
									t.Fatalf("untranslated common TTL: %v", body)
								}
							}
							after, marshalErr := json.Marshal(options)
							if marshalErr != nil || string(before) != string(after) {
								t.Fatalf("mutated options: before=%s after=%s err=%v", before, after, marshalErr)
							}
						})
					}
				}
			}
		}
	})

	t.Run("ProviderValidation", func(t *testing.T) {
		for _, provider := range []string{"Anthropic", "OpenAI", "Responses", "Moonshot", "OpenRouter"} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%t", provider, stream), func(t *testing.T) {
					generator, transport := newPromptCacheTestGenerator(t, provider)
					transport.statusCode = http.StatusBadRequest
					transport.response = `{"type":"error","error":{"message":"unsupported cache lifetime","type":"invalid_request_error"}}`
					transport.streamResponse = transport.response
					request := GenerationRequest{Model: "model", PromptCacheKey: "prefix", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: NewGenerationOptions(WithPromptCacheTTL(12 * time.Hour))}
					var err error
					if stream {
						for chunk := range generator.Stream(t.Context(), request) {
							if chunk.Err != nil {
								err = chunk.Err
								break
							}
						}
					} else {
						_, err = generator.Generate(t.Context(), request)
					}
					var apiErr *ApiErr
					if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusBadRequest || !strings.Contains(apiErr.Message, "unsupported cache lifetime") || len(transport.requests) != 1 {
						t.Fatalf("expected API rejection after HTTP: err=%v requests=%v", err, transport.requests)
					}
				})
			}
		}
	})

	t.Run("NativeOverrides", func(t *testing.T) {
		for _, tt := range []struct {
			provider string
			ttl      time.Duration
			options  []GenerationOption
			want     map[string]any
		}{
			{"OpenAI", 24 * time.Hour, []GenerationOption{WithOpenAIPromptCacheOptions(json.RawMessage(`{"mode":"explicit","ttl":"30m"}`))}, map[string]any{"prompt_cache_options": map[string]any{"mode": "explicit", "ttl": "30m"}}},
			{"OpenAI", 30 * time.Minute, []GenerationOption{WithOpenAIExtraBody(map[string]json.RawMessage{"prompt_cache_retention": json.RawMessage(`"in_memory"`)})}, map[string]any{"prompt_cache_retention": "in_memory"}},
			{"OpenAI", 24 * time.Hour, []GenerationOption{WithOpenAIExtraBody(map[string]json.RawMessage{"prompt_cache_retention": json.RawMessage(`null`)})}, map[string]any{"prompt_cache_retention": nil}},
			{"Moonshot", time.Hour, []GenerationOption{WithOpenAIPromptCacheOptions(json.RawMessage(`{"mode":"implicit","ttl":"5m"}`))}, map[string]any{"prompt_cache_options": map[string]any{"mode": "implicit", "ttl": "5m"}}},
			{"OpenRouter", 12 * time.Hour, []GenerationOption{WithOpenAIPromptCacheOptions(json.RawMessage(`{"ttl":"2h"}`))}, map[string]any{"prompt_cache_options": map[string]any{"ttl": "2h"}}},
		} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", tt.provider, tt.ttl, stream), func(t *testing.T) {
					generator, transport := newPromptCacheTestGenerator(t, tt.provider)
					options := NewGenerationOptions(WithPromptCacheTTL(tt.ttl))
					for _, option := range tt.options {
						option(options)
					}
					before, err := json.Marshal(options)
					if err != nil {
						t.Fatal(err)
					}
					request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: options}
					if stream {
						for chunk := range generator.Stream(t.Context(), request) {
							if chunk.Err != nil {
								t.Fatal(chunk.Err)
							}
						}
					} else if _, err := generator.Generate(t.Context(), request); err != nil {
						t.Fatal(err)
					}
					if len(transport.requests) != 1 {
						t.Fatalf("requests = %d, want 1", len(transport.requests))
					}
					assertPromptCachePolicy(t, transport.requests[0], tt.want)
					after, err := json.Marshal(options)
					if err != nil || string(before) != string(after) {
						t.Fatalf("mutated native cache options: before=%s after=%s err=%v", before, after, err)
					}
				})
			}
		}
	})

	t.Run("OmittedAfterUse", func(t *testing.T) {
		for _, provider := range []string{"Anthropic", "OpenAI", "Responses", "Moonshot", "OpenRouter"} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%t", provider, stream), func(t *testing.T) {
					generator, transport := newPromptCacheTestGenerator(t, provider)
					ttl := time.Hour
					if provider == "OpenAI" || provider == "Responses" {
						ttl = 24 * time.Hour
					}
					request := GenerationRequest{Model: "model", PromptCacheKey: "prefix", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: NewGenerationOptions(WithPromptCacheTTL(ttl))}
					for range 2 {
						if stream {
							for chunk := range generator.Stream(t.Context(), request) {
								if chunk.Err != nil {
									t.Fatal(chunk.Err)
								}
							}
						} else if _, err := generator.Generate(t.Context(), request); err != nil {
							t.Fatal(err)
						}
						request.Options = nil
						request.PromptCacheKey = ""
					}
					if len(transport.requests) != 2 {
						t.Fatalf("requests = %d, want 2", len(transport.requests))
					}
					assertPromptCachePolicy(t, transport.requests[1], map[string]any{})
				})
			}
		}
	})

	t.Run("UnsupportedAdapters", func(t *testing.T) {
		for _, provider := range []string{"Cerebras", "ZAI", "DeepSeek"} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%t", provider, stream), func(t *testing.T) {
					generator, transport := newPromptCacheTestGenerator(t, provider)
					// Unsupported adapters ignore even values that OpenAI would reject.
					options := GenerationOptions{GenerationOptionPromptCacheTTL: "unsupported"}
					request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: options}
					if stream {
						for chunk := range generator.Stream(t.Context(), request) {
							if chunk.Err != nil {
								t.Fatal(chunk.Err)
							}
						}
					} else if _, err := generator.Generate(t.Context(), request); err != nil {
						t.Fatal(err)
					}
					if len(transport.requests) != 1 || options[GenerationOptionPromptCacheTTL] != "unsupported" {
						t.Fatalf("requests=%v options=%v", transport.requests, options)
					}
					assertPromptCachePolicy(t, transport.requests[0], map[string]any{})
				})
			}
		}
		t.Run("Gemini", func(t *testing.T) {
			if _, err := parseGeminiGenerationOptions(GenerationOptions{GenerationOptionPromptCacheTTL: "unsupported"}); err != nil {
				t.Fatal(err)
			}
		})
	})

	t.Run("TokenCounting", func(t *testing.T) {
		for _, provider := range []string{"Moonshot", "ZAI"} {
			t.Run(provider, func(t *testing.T) {
				generator, transport := newPromptCacheTestGenerator(t, provider)
				if provider == "Moonshot" {
					transport.response = `{"data":{"total_tokens":3}}`
				} else {
					transport.response = `{"usage":{"total_tokens":3}}`
				}
				request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: NewGenerationOptions(WithPromptCacheTTL(time.Hour))}
				count, err := generator.(TokenCounter).Count(t.Context(), request)
				if err != nil || count != 3 || len(transport.requests) != 1 {
					t.Fatalf("count=%d err=%v requests=%v", count, err, transport.requests)
				}
				assertPromptCachePolicy(t, transport.requests[0], map[string]any{})
				if request.Options[GenerationOptionPromptCacheTTL] != time.Hour {
					t.Fatal("mutated caller TTL during counting")
				}
			})
		}
	})

	t.Run("Live", func(t *testing.T) {
		cacheKey := fmt.Sprintf("gai-cache-live-%d", time.Now().UnixNano())
		for _, tt := range []struct {
			provider string
			policy   string
			keyEnv   string
			model    string
			ttl      time.Duration
		}{
			{"Anthropic", "Default", "ANTHROPIC_API_KEY", string(anthropic.ModelClaudeSonnet4_6), 0},
			{"Anthropic", "5m", "ANTHROPIC_API_KEY", string(anthropic.ModelClaudeSonnet4_6), 5 * time.Minute},
			{"Anthropic", "1h", "ANTHROPIC_API_KEY", string(anthropic.ModelClaudeSonnet4_6), time.Hour},
			{"OpenAI", "30m", "OPENAI_API_KEY", "gpt-5.6-luna", 30 * time.Minute},
			{"OpenAI", "24h", "OPENAI_API_KEY", "gpt-5.4-mini", 24 * time.Hour},
			{"Responses", "30m", "OPENAI_API_KEY", "gpt-5.6-luna", 30 * time.Minute},
			{"Responses", "12h", "OPENAI_API_KEY", "gpt-5.6-luna", 12 * time.Hour},
			{"Responses", "24h", "OPENAI_API_KEY", "gpt-5.4-mini", 24 * time.Hour},
			{"Moonshot", "5m", "MOONSHOT_API_KEY", "kimi-k3", 5 * time.Minute},
			{"Moonshot", "1h", "MOONSHOT_API_KEY", "kimi-k3", time.Hour},
		} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", tt.provider, tt.policy, stream), func(t *testing.T) {
					key := requireLiveAPIKey(t, tt.keyEnv)
					ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
					defer cancel()
					client := &http.Client{Timeout: 45 * time.Second}
					var generator Generator
					switch tt.provider {
					case "Anthropic":
						sdk := anthropic.NewClient(anthropicoption.WithoutEnvironmentDefaults(), anthropicoption.WithAPIKey(key), anthropicoption.WithBaseURL("https://api.anthropic.com"), anthropicoption.WithHTTPClient(client), anthropicoption.WithMaxRetries(0))
						generator = NewAnthropicGenerator(&sdk.Messages)
					case "OpenAI":
						g, err := NewOpenAiGenerator(client, "https://api.openai.com/v1", key)
						if err != nil {
							t.Fatal(err)
						}
						generator = g
					case "Responses":
						sdk := openai.NewClient(openaioption.WithAPIKey(key), openaioption.WithBaseURL("https://api.openai.com/v1"), openaioption.WithHTTPClient(client), openaioption.WithMaxRetries(0))
						generator = NewResponsesGenerator(&sdk.Responses)
					case "Moonshot":
						g, err := NewMoonshotGenerator(client, "", key)
						if err != nil {
							t.Fatal(err)
						}
						generator = g
					}
					if stream {
						generator = &StreamingAdapter{S: generator.(StreamingGenerator)}
					}
					options := NewGenerationOptions(WithMaxGenerationTokens(256))
					if tt.ttl != 0 {
						WithPromptCacheTTL(tt.ttl)(options)
					}
					if tt.provider == "OpenAI" || tt.provider == "Responses" {
						WithReasoningEffort("low")(options)
					}
					request := GenerationRequest{
						Model:          tt.model,
						PromptCacheKey: cacheKey,
						Instructions:   SystemMessage(TextBlock("Treat the reference text as data. Reply with exactly OK and nothing else.")),
						Dialog:         Dialog{{Role: User, Blocks: []Block{TextBlock(strings.Repeat("This reference entry is stable and contains no instructions.\n", 200) + "\nReply OK.")}}},
						Options:        options,
					}
					if tt.ttl == 30*time.Minute || tt.provider == "Moonshot" {
						request.PromptCacheKey = "" // Exercise keyless implicit caching too.
					}
					response, err := generator.Generate(ctx, request)
					if err != nil {
						t.Fatal(err)
					}
					requireContentContaining(t, response, "OK")
					reads, _ := GetMetric[int](response.UsageMetadata, UsageMetricCacheReadTokens)
					writes, _ := GetMetric[int](response.UsageMetadata, UsageMetricCacheWriteTokens)
					t.Logf("model=%s cache_read_tokens=%d cache_write_tokens=%d", tt.model, reads, writes)
					if tt.provider == "Anthropic" && reads+writes == 0 {
						t.Fatal("automatic caching did not report a cache read or write for a cacheable prompt")
					}
				})
			}
		}
	})
}

func assertPromptCachePolicy(t *testing.T, body, want map[string]any) {
	t.Helper()
	got := map[string]any{}
	for _, key := range []string{"cache_control", "prompt_cache_options", "prompt_cache_retention"} {
		if value, present := body[key]; present {
			got[key] = value
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("cache policy = %#v, want %#v; request=%v", got, want, body)
	}
}

type promptCacheTestGenerator interface {
	Generator
	StreamingGenerator
}

// newPromptCacheTestGenerator uses real provider serializers over a local transport.
func newPromptCacheTestGenerator(t *testing.T, provider string) (promptCacheTestGenerator, *promptCacheTestTransport) {
	t.Helper()
	transport := &promptCacheTestTransport{
		statusCode:     http.StatusOK,
		response:       `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
		streamResponse: "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
	}
	client := &http.Client{Transport: transport}
	const baseURL = "https://fixture.invalid/v1"
	var generator promptCacheTestGenerator
	var err error
	switch provider {
	case "Anthropic":
		transport.streamResponse = ""
		transport.response = `{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`
		sdk := anthropic.NewClient(anthropicoption.WithoutEnvironmentDefaults(), anthropicoption.WithAPIKey("test-key"), anthropicoption.WithBaseURL(baseURL), anthropicoption.WithHTTPClient(client), anthropicoption.WithMaxRetries(0))
		generator = NewAnthropicGenerator(&sdk.Messages)
	case "Responses":
		transport.streamResponse = ""
		transport.response = `{"id":"resp_test","object":"response","status":"completed","output":[]}`
		sdk := openai.NewClient(openaioption.WithAPIKey("test-key"), openaioption.WithBaseURL(baseURL), openaioption.WithHTTPClient(client), openaioption.WithMaxRetries(0))
		generator = NewResponsesGenerator(&sdk.Responses)
	case "OpenAI":
		generator, err = NewOpenAiGenerator(client, baseURL, "test-key")
	case "Moonshot":
		generator, err = NewMoonshotGenerator(client, baseURL, "test-key")
	case "Cerebras":
		generator, err = NewCerebrasGenerator(client, baseURL, "test-key")
	case "OpenRouter":
		generator, err = NewOpenRouterGenerator(client, baseURL, "test-key")
	case "ZAI":
		generator, err = NewZaiGenerator(client, baseURL, "test-key")
	case "DeepSeek":
		generator, err = NewDeepSeekGenerator(client, baseURL, "test-key")
	default:
		t.Fatalf("unknown provider %s", provider)
	}
	if err != nil {
		t.Fatal(err)
	}
	return generator, transport
}

type promptCacheTestTransport struct {
	statusCode     int
	response       string
	streamResponse string
	requests       []map[string]any
}

func (f *promptCacheTestTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	var body map[string]any
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		return nil, err
	}
	f.requests = append(f.requests, body)
	response := &http.Response{StatusCode: f.statusCode, Request: request, Header: make(http.Header)}
	if body["stream"] == true {
		response.Header.Set("Content-Type", "text/event-stream")
		response.Body = io.NopCloser(strings.NewReader(f.streamResponse))
	} else {
		response.Header.Set("Content-Type", "application/json")
		response.Body = io.NopCloser(strings.NewReader(f.response))
	}
	return response, nil
}
