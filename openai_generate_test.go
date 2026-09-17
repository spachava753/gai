package gai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	oaissestream "github.com/openai/openai-go/v3/packages/ssestream"

	oai "github.com/openai/openai-go/v3"

	wire "github.com/spachava753/gai/internal/openai"
)

func testCommonChatCompletionOptions(t *testing.T) {
	providers := []struct {
		name string
		new  func(*http.Client, string, string) (Generator, error)
	}{
		{"openai", func(c *http.Client, u, k string) (Generator, error) { return NewOpenAiGenerator(c, u, k) }},
		{"cerebras", func(c *http.Client, u, k string) (Generator, error) { return NewCerebrasGenerator(c, u, k) }},
		{"openrouter", func(c *http.Client, u, k string) (Generator, error) { return NewOpenRouterGenerator(c, u, k) }},
		{"zai", func(c *http.Client, u, k string) (Generator, error) { return NewZaiGenerator(c, u, k) }},
		{"deepseek", func(c *http.Client, u, k string) (Generator, error) { return NewDeepSeekGenerator(c, u, k) }},
		{"moonshot", func(c *http.Client, u, k string) (Generator, error) { return NewMoonshotGenerator(c, u, k) }},
	}
	for _, provider := range providers {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", provider.name, streaming), func(t *testing.T) {
				var received map[string]any
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
						t.Error(err)
					}
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
					} else {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
					}
				}))
				defer server.Close()
				generator, err := provider.new(server.Client(), server.URL, "test-key")
				if err != nil {
					t.Fatal(err)
				}
				invoke := func(options GenerationOptions) {
					t.Helper()
					before, _ := json.Marshal(options)
					received = nil
					request := GenerationRequest{Model: "test-model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: options}
					if streaming {
						for chunk := range generator.(StreamingGenerator).Stream(t.Context(), request) {
							if chunk.Err != nil {
								t.Fatal(chunk.Err)
							}
						}
					} else if _, err := generator.Generate(t.Context(), request); err != nil {
						t.Fatal(err)
					}
					after, _ := json.Marshal(options)
					if !bytes.Equal(before, after) {
						t.Fatal("mutated caller options")
					}
				}
				invoke(nil)
				for _, field := range []string{"logit_bias", "logprobs", "top_logprobs", "parallel_tool_calls", "seed", "min_p", "top_a", "repetition_penalty"} {
					if _, present := received[field]; present {
						t.Errorf("omitted option sent: %s", field)
					}
				}
				for _, zero := range []bool{false, true} {
					bias, count, seed, minP, topA, repetition := 123.5, 30, -7, -0.5, 1.5, -2.0
					if zero {
						bias, count, seed, minP, topA, repetition = 0, 0, 0, 0, 0, 0
					}
					options := NewGenerationOptions(WithLogitBias(map[string]float64{"42": bias}), WithLogprobs(!zero), WithTopLogprobs(count), WithParallelToolCalls(!zero), WithSeed(seed), WithOpenRouterMinP(minP), WithOpenRouterTopA(topA), WithOpenRouterRepetitionPenalty(repetition))
					invoke(options)
					if received["logprobs"] != !zero || received["top_logprobs"] != float64(count) || received["parallel_tool_calls"] != !zero || received["seed"] != float64(seed) || !reflect.DeepEqual(received["logit_bias"], map[string]any{"42": bias}) {
						t.Fatalf("common options = %v", received)
					}
					for field, want := range map[string]float64{"min_p": minP, "top_a": topA, "repetition_penalty": repetition} {
						if provider.name == "openrouter" {
							if received[field] != want {
								t.Errorf("%s = %v, want %v", field, received[field], want)
							}
						} else if _, present := received[field]; present {
							t.Errorf("unsupported field sent: %s", field)
						}
					}
					explicit := map[string]json.RawMessage{"logit_bias": json.RawMessage(`{"99":-1}`), "logprobs": json.RawMessage(`false`), "top_logprobs": json.RawMessage(`2`), "parallel_tool_calls": json.RawMessage(`false`), "seed": json.RawMessage(`13`)}
					if provider.name == "openrouter" {
						explicit["min_p"] = json.RawMessage(`0.4`)
						explicit["top_a"] = json.RawMessage(`0.3`)
						explicit["repetition_penalty"] = json.RawMessage(`1.2`)
					}
					WithOpenAIExtraBody(explicit)(options)
					invoke(options)
					for field, raw := range explicit {
						var want any
						_ = json.Unmarshal(raw, &want)
						if !reflect.DeepEqual(received[field], want) {
							t.Errorf("override %s = %v, want %v", field, received[field], want)
						}
					}
				}
				// A count does not silently enable logprobs or rewrite explicit false.
				invoke(NewGenerationOptions(WithTopLogprobs(2)))
				if _, present := received["logprobs"]; present {
					t.Fatal("top-logprobs implicitly enabled logprobs")
				}
				invoke(NewGenerationOptions(WithTopLogprobs(2), WithLogprobs(false)))
				if received["logprobs"] != false {
					t.Fatal("explicit logprobs false ignored")
				}
				keys := []string{GenerationOptionLogitBias, GenerationOptionLogprobs, GenerationOptionTopLogprobs, GenerationOptionParallelToolCalls, GenerationOptionSeed}
				if provider.name == "openrouter" {
					keys = append(keys, OpenRouterGenerationOptionMinP, OpenRouterGenerationOptionTopA, OpenRouterGenerationOptionRepetitionPenalty)
				}
				for _, key := range keys {
					received = nil
					request := GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: GenerationOptions{key: struct{}{}}}
					var err error
					if streaming {
						for chunk := range generator.(StreamingGenerator).Stream(t.Context(), request) {
							err = chunk.Err
						}
					} else {
						_, err = generator.Generate(t.Context(), request)
					}
					var invalid *InvalidParameterErr
					if !errors.As(err, &invalid) || invalid.Parameter != key || received != nil {
						t.Fatalf("invalid %s: err=%v request=%v", key, err, received)
					}
				}
			})
		}
	}
}

// mockChatCompletionService supplies SDK-shaped fixtures over an HTTP transport.
type mockChatCompletionService struct {
	response     *oai.ChatCompletion
	err          error
	streamEvents []oaissestream.Event
	requests     []oai.ChatCompletionNewParams
}

// newTestOpenAIGenerator keeps the existing SDK-shaped fixtures as wire-format
// test data, but sends requests through the real generated HTTP client.
func newTestOpenAIGenerator(t *testing.T, fixture *mockChatCompletionService) *OpenAiGenerator {
	t.Helper()
	if fixture == nil {
		return &OpenAiGenerator{}
	}
	generator, err := NewOpenAiGenerator(&http.Client{Transport: fixture}, "https://fixture.invalid/v1", "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	return generator
}

func (m *mockChatCompletionService) RoundTrip(request *http.Request) (*http.Response, error) {
	if m.err != nil {
		return nil, m.err
	}
	data, err := io.ReadAll(request.Body)
	if err != nil {
		return nil, err
	}
	var body oai.ChatCompletionNewParams
	if err := json.Unmarshal(data, &body); err != nil {
		return nil, err
	}
	m.requests = append(m.requests, body)
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	response := &http.Response{StatusCode: 200, Header: make(http.Header), Request: request}
	if string(raw["stream"]) == "true" {
		var stream strings.Builder
		for _, event := range m.streamEvents {
			fmt.Fprintf(&stream, "data: %s\n\n", event.Data)
		}
		stream.WriteString("data: [DONE]\n\n")
		response.Header.Set("Content-Type", "text/event-stream")
		response.Body = io.NopCloser(strings.NewReader(stream.String()))
	} else {
		data, err := json.Marshal(m.response)
		if err != nil {
			return nil, err
		}
		response.Header.Set("Content-Type", "application/json")
		response.Body = io.NopCloser(bytes.NewReader(data))
	}
	return response, nil
}

// newLiveOpenAIGenerator only constructs the adapter; tests must call
// requireLiveAPIKey before any network operation.
func newLiveOpenAIGenerator(t *testing.T, connection ...string) *OpenAiGenerator {
	t.Helper()
	baseURL, key := "", os.Getenv("OPENAI_API_KEY")
	if len(connection) == 2 {
		baseURL, key = connection[0], connection[1]
	}
	generator, err := NewOpenAiGenerator(nil, baseURL, key)
	if err != nil {
		t.Fatal(err)
	}
	return generator
}

func testOpenAIGenerateReturnsContentPolicyErrorForRefusal(t *testing.T) {
	client := &mockChatCompletionService{response: &oai.ChatCompletion{
		Choices: []oai.ChatCompletionChoice{{
			FinishReason: "stop",
			Message:      oai.ChatCompletionMessage{Refusal: "I cannot help with that."},
		}},
	}}
	generator := newTestOpenAIGenerator(t, client)

	response, err := generator.Generate(context.Background(), GenerationRequest{
		Model:  "gpt-5",
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

func testOpenAIGenerateReturnsContentPolicyErrorForContentFilter(t *testing.T) {
	client := &mockChatCompletionService{response: &oai.ChatCompletion{
		Choices: []oai.ChatCompletionChoice{{FinishReason: "content_filter"}},
	}}
	generator := newTestOpenAIGenerator(t, client)

	response, err := generator.Generate(context.Background(), GenerationRequest{
		Model:  "gpt-5",
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

func testOpenAIStreamReturnsContentPolicyErrorForRefusal(t *testing.T) {
	client := &mockChatCompletionService{streamEvents: []oaissestream.Event{{
		Data: []byte(`{"id":"chatcmpl_123","object":"chat.completion.chunk","created":0,"model":"gpt-5","choices":[{"index":0,"delta":{"refusal":"I cannot help with that."},"finish_reason":""}]}`),
	}}}
	generator := newTestOpenAIGenerator(t, client)

	var gotErr error
	for chunk := range generator.Stream(context.Background(), GenerationRequest{
		Model:  "gpt-5",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
	}) {
		if chunk.Err != nil {
			gotErr = chunk.Err
			break
		}
	}

	var policyErr ContentPolicyErr
	if !errors.As(gotErr, &policyErr) {
		t.Fatalf("Stream error = %T %v, want ContentPolicyErr", gotErr, gotErr)
	}
	if !strings.Contains(policyErr.Error(), "I cannot help with that.") {
		t.Fatalf("Stream error = %q, want refusal message", policyErr)
	}
}

func testOpenAIHTTPClient(t *testing.T) {
	t.Run("native options", func(t *testing.T) {
		request := GenerationRequest{Model: "m", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
		for _, options := range []GenerationOptions{
			NewGenerationOptions(WithOpenAIExtraBody(map[string]json.RawMessage{"messages": json.RawMessage(`[]`)})),
			NewGenerationOptions(WithOpenAIExtraBody(map[string]json.RawMessage{"thinking": json.RawMessage(`{`)})),
			{OpenAIGenerationOptionExtraBody: "wrong type"},
			{OpenAIGenerationOptionStreamUsage: "wrong type"},
		} {
			request.Options = options
			if _, _, err := openAIRequest(request, true); err == nil {
				t.Fatalf("accepted invalid native options: %v", options)
			}
		}
		fields := map[string]json.RawMessage{"thinking": json.RawMessage(`{"type":"enabled"}`), "response_format": json.RawMessage(`null`)}
		request.Options = NewGenerationOptions(WithOpenAIExtraBody(fields), WithOpenAIStreamUsage(false))
		params, _, err := openAIRequest(request, true)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(params)
		if err != nil {
			t.Fatal(err)
		}
		if params.StreamOptions.IsSpecified() || !strings.Contains(string(encoded), `"response_format":null`) {
			t.Fatalf("lost explicit native configuration: %s", encoded)
		}
		fields["thinking"][0] = '!'
		if _, err := json.Marshal(params); err != nil {
			t.Fatalf("request aliases caller JSON bytes: %v", err)
		}
		request.Options = nil
		next, _, err := openAIRequest(request, true)
		if err != nil {
			t.Fatal(err)
		}
		if len(next.AdditionalProperties) != 0 || !next.StreamOptions.IsSpecified() {
			t.Fatal("native options leaked into next request")
		}
	})
	t.Run("tool choice controls", func(t *testing.T) {
		for _, choice := range []string{"auto", "required", "none"} {
			t.Run(choice, func(t *testing.T) {
				request := GenerationRequest{Model: "m", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("Use a tool if allowed.")}}}, Options: NewGenerationOptions(WithToolChoice(choice))}
				for _, stream := range []bool{false, true} {
					params, _, err := openAIRequest(request, stream)
					if err != nil {
						t.Fatal(err)
					}
					data, err := json.Marshal(params)
					if err != nil {
						t.Fatal(err)
					}
					var body map[string]json.RawMessage
					if err := json.Unmarshal(data, &body); err != nil {
						t.Fatal(err)
					}
					if string(body["tool_choice"]) != `"`+choice+`"` {
						t.Fatalf("tool choice encoded as %s", body["tool_choice"])
					}
				}
				if _, err := convertToolsToOpenAI([]Tool{{Name: choice}}); err == nil {
					t.Fatal("accepted reserved tool name")
				}
			})
		}
	})
	t.Run("tool arguments replay as strings", func(t *testing.T) {
		for _, test := range []struct {
			name, arguments string
			wantError       bool
		}{
			{"observed string response", `"{\"n\":9007199254740993}"`, false},
			{"reject object response", `{"n":9007199254740993}`, true},
		} {
			t.Run(test.name, func(t *testing.T) {
				var message wire.ResponseMessage
				data := `{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":` + test.arguments + `}}]}`
				err := json.Unmarshal([]byte(data), &message)
				if test.wantError {
					if err == nil {
						t.Fatal("accepted object-valued tool arguments")
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				converted, err := openAIResponseMessage(message, "")
				if err != nil {
					t.Fatal(err)
				}
				replayed, err := toOpenAIMessage(converted)
				if err != nil {
					t.Fatal(err)
				}
				encoded, err := json.Marshal(replayed)
				if err != nil {
					t.Fatal(err)
				}
				// A string-typed destination rejects object-valued request arguments.
				var request struct {
					ToolCalls []struct {
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				}
				if err := json.Unmarshal(encoded, &request); err != nil {
					t.Fatalf("replayed arguments must be strings: %s: %v", encoded, err)
				}
				if len(request.ToolCalls) != 1 || request.ToolCalls[0].Function.Arguments != `{"n":9007199254740993}` {
					t.Fatalf("arguments changed during replay: %s", encoded)
				}
			})
		}
	})
	t.Run("constructor", func(t *testing.T) {
		if _, err := NewOpenAiGenerator(nil, "", ""); !errors.Is(err, ErrMissingAPIKey) {
			t.Fatalf("missing key: %v", err)
		}
		for _, base := range []string{"relative/path", "file:///tmp/client", "https://example.com?key=value"} {
			if _, err := NewOpenAiGenerator(nil, base, "key"); err == nil {
				t.Fatalf("accepted %q", base)
			}
		}
	})
	t.Run("headers errors and no retry", func(t *testing.T) {
		for _, status := range []int{400, 429, 503} {
			t.Run(fmt.Sprint(status), func(t *testing.T) {
				calls := 0
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls++
					if r.URL.Path != "/prefix/chat/completions" || r.Header.Get("Authorization") != "Bearer fixture" {
						t.Errorf("request: %s %v", r.URL, r.Header)
					}
					w.Header().Set("Content-Type", "text/html")
					w.Header().Set("Retry-After", "3")
					w.WriteHeader(status)
					_, _ = io.WriteString(w, "gateway failure")
				}))
				defer server.Close()
				g, err := NewOpenAiGenerator(server.Client(), server.URL+"/prefix", "fixture")
				if err != nil {
					t.Fatal(err)
				}
				request := GenerationRequest{Model: "m", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hello")}}}}
				_, err = g.Generate(t.Context(), request)
				var api *ApiErr
				if !errors.As(err, &api) || api.StatusCode != status || api.RawBody != "gateway failure" {
					t.Fatalf("error: %#v", err)
				}
				if delay, ok := api.RetryAfter(); !ok || delay != 3*time.Second {
					t.Fatalf("retry timing: %v %v", delay, ok)
				}
				if calls != 1 {
					t.Fatalf("POST retried %d times", calls)
				}
			})
		}
	})
	t.Run("wire replay and token fields", func(t *testing.T) {
		var requests []map[string]json.RawMessage
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var request map[string]json.RawMessage
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Error(err)
			}
			requests = append(requests, request)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("x-request-id", "request-1")
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":null,"reasoning_content":"thinking","reasoning_details":[{"type":"reasoning.encrypted","data":"cipher","signature":"sig"}],"opaque":{"n":9007199254740993},"tool_calls":[{"id":"c","type":"function","function":{"name":"lookup","arguments":"{\"n\":9007199254740993}","signature":"tool-sig"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":0,"completion_tokens":4},"provider_number":9007199254740993}`)
		}))
		defer server.Close()
		g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
		if err != nil {
			t.Fatal(err)
		}
		request := GenerationRequest{Model: "first", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: NewGenerationOptions(WithMaxGenerationTokens(8), WithOpenAITokenLimitField("max_tokens"))}
		response, err := g.Generate(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		if string(requests[0]["max_tokens"]) != "8" || requests[0]["max_completion_tokens"] != nil {
			t.Fatalf("token selection: %s", requests[0])
		}
		if response.ExtraFields[OpenAIResponseExtraFieldHeaders].(http.Header).Get("x-request-id") != "request-1" {
			t.Fatal("lost headers")
		}
		if !strings.Contains(response.Candidates[0].Blocks[1].Content.String(), "9007199254740993") {
			t.Fatalf("tool arguments rounded: %+v", response.Candidates[0])
		}
		// Persist and reload to exercise ExtraFields without Go-specific map types.
		saved, err := json.Marshal(response.Candidates[0].ExtraFields)
		if err != nil {
			t.Fatal(err)
		}
		replay := response.Candidates[0]
		replay.ExtraFields = nil
		decoder := json.NewDecoder(bytes.NewReader(saved))
		decoder.UseNumber()
		if err := decoder.Decode(&replay.ExtraFields); err != nil {
			t.Fatal(err)
		}
		request.Model = "second"
		request.Options = NewGenerationOptions(WithMaxGenerationTokens(9))
		request.Dialog = append(request.Dialog, replay)
		if _, err := g.Generate(t.Context(), request); err != nil {
			t.Fatal(err)
		}
		if requests[1]["max_tokens"] != nil || string(requests[1]["max_completion_tokens"]) != "9" {
			t.Fatalf("stale token field: %s", requests[1])
		}
		messageJSON := string(requests[1]["messages"])
		for _, value := range []string{`"reasoning_content":"thinking"`, `"signature":"sig"`, `"signature":"tool-sig"`, `"arguments":"{\"n\":9007199254740993}"`, `"content":null`, `"opaque":{"n":9007199254740993}`} {
			if !strings.Contains(messageJSON, value) {
				t.Errorf("missing %s in %s", value, messageJSON)
			}
		}
	})
}

func testOpenAIHTTPStreaming(t *testing.T) {
	stream := ": keepalive\r\n\r\n" +
		`data: {"choices":[{"index":0,"delta":{"content":"Hi","reasoning_details":[{"index":0,"type":"reasoning.encrypted","data":"part"}],"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"second","arguments":"{\"n\":"}},{"index":0,"id":"a","type":"function","function":{"name":"first","arguments":"{"}}]}}]}` + "\r\n\r\n" +
		`data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"index":0,"data":"two","signature":"sig"}],"tool_calls":[{"index":0,"function":{"arguments":"}"}},{"index":1,"function":{"arguments":"9007199254740993}"}}]},"finish_reason":"tool_calls","usage":{"completion_tokens":7}}]}` + "\n\n" +
		"data: [DONE]\n\n" + `data: {"cost":"0","choices":[],"usage":{"completion_tokens":7}}` + "\n\n"
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, stream)
	}))
	defer server.Close()
	g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	response, err := (&StreamingAdapter{S: g}).Generate(t.Context(), GenerationRequest{Model: "m", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hi")}}}})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("POST replayed %d times", calls)
	}
	if len(response.Candidates) != 1 || len(response.Candidates[0].Blocks) != 3 {
		t.Fatalf("assembled response: %+v", response)
	}
	blocks := response.Candidates[0].Blocks
	if blocks[1].ID != "a" || blocks[2].ID != "b" || !strings.Contains(blocks[2].Content.String(), "9007199254740993") {
		t.Fatalf("corrupted tools: %+v", blocks)
	}
	fields := response.ExtraFields[OpenAIResponseExtraFieldWireFields].(map[string]json.RawMessage)
	if string(fields["cost"]) != `"0"` {
		t.Fatalf("lost trailing cost: %s", fields)
	}
	candidateFields := response.Candidates[0].ExtraFields[OpenAIExtraFieldWireFields].(map[string]json.RawMessage)
	if !strings.Contains(string(candidateFields["reasoning_details"]), `"data":"parttwo"`) {
		t.Fatalf("lost reasoning fragments: %s", candidateFields)
	}
	if value, ok := OutputTokens(response.UsageMetadata); !ok || value != 7 {
		t.Fatalf("usage: %+v", response.UsageMetadata)
	}
}

func testOpenAIStreamTermination(t *testing.T) {
	for _, test := range []struct {
		name, body string
		want       string
	}{
		{"empty", "", "unexpected EOF"},
		{"incomplete", `data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}` + "\n\n", "unexpected EOF"},
		{"malformed", "data: {\n\n", "decode openai event"},
		{"provider error", `data: {"error":{"message":"quota"}}` + "\n\n", "quota"},
		{"length", `data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":"length"}]}` + "\n\n", "maximum generation limit"},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, test.body)
			}))
			defer server.Close()
			g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
			if err != nil {
				t.Fatal(err)
			}
			var terminal error
			for chunk := range g.Stream(t.Context(), GenerationRequest{Model: "m", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hi")}}}}) {
				if chunk.Err != nil {
					terminal = chunk.Err
				}
			}
			if terminal == nil || !strings.Contains(terminal.Error(), test.want) {
				t.Fatalf("error: %v, want %s", terminal, test.want)
			}
		})
	}
	t.Run("early stop closes body", func(t *testing.T) {
		closed := make(chan struct{})
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"content":"first"}}]}`+"\n\n")
			w.(http.Flusher).Flush()
			<-r.Context().Done()
			close(closed)
		}))
		defer server.Close()
		g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
		defer cancel()
		for chunk := range g.Stream(ctx, GenerationRequest{Model: "m", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hi")}}}}) {
			if chunk.Err != nil {
				t.Fatal(chunk.Err)
			}
			break
		}
		select {
		case <-closed:
		case <-ctx.Done():
			t.Fatal("consumer stop did not close response")
		}
	})
}

func testOpenAIStreamAudioAndCandidates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"content":"first","audio":{"id":"audio-1","data":"AA","transcript":"hel"}}},{"index":1,"delta":{"content":"second"}}]}`+"\n\n"+
			`data: {"choices":[{"index":0,"delta":{"audio":{"data":"BB","transcript":"lo"}},"finish_reason":"stop"},{"index":1,"delta":{},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	var audio Block
	var second bool
	for chunk := range g.Stream(t.Context(), GenerationRequest{Model: "m", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hi")}}}, Options: NewGenerationOptions(WithOpenAIOutputModalities(Text, Audio), WithOpenAIAudioConfig(OpenAIAudioConfig{VoiceName: "alloy", Format: "wav"}))}) {
		if chunk.Err != nil {
			t.Fatal(chunk.Err)
		}
		if chunk.Block.ModalityType == Audio {
			audio = chunk.Block
		}
		if chunk.CandidatesIndex == 1 && chunk.Block.Content != nil && chunk.Block.Content.String() == "second" {
			second = true
		}
	}
	if !second || audio.ID != "audio-1" || audio.Content == nil || audio.Content.String() != "AABB" || audio.MimeType != "audio/wav" {
		t.Fatalf("candidates/audio lost: %v %+v", second, audio)
	}
}

func BenchmarkOpenAIThinkingReplay(b *testing.B) {
	for _, count := range []int{1000, 4000, 8000} {
		b.Run(fmt.Sprint(count), func(b *testing.B) {
			blocks := make([]Block, count)
			for i := range blocks {
				blocks[i] = Block{BlockType: Thinking, ModalityType: Text, Content: Str("part"), ExtraFields: map[string]interface{}{OpenAIExtraFieldReasoningSource: "reasoning_details"}}
			}
			details, err := json.Marshal([]map[string]any{{"type": "reasoning.text", "text": strings.Repeat("part", count)}, {"type": "reasoning.encrypted", "data": strings.Repeat("x", 64*1024)}})
			if err != nil {
				b.Fatal(err)
			}
			message := Message{Role: Assistant, Blocks: blocks, ExtraFields: map[string]interface{}{OpenAIExtraFieldWireFields: map[string]json.RawMessage{"reasoning_details": details}}}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := toOpenAIMessage(message); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func testOpenAIStreamChoiceMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{
			`{"choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null,"logprobs":{"content":[{"token":"A","logprob":-0.1,"extra":9007199254740993}]}}]}`,
			`{"choices":[{"index":0,"delta":{},"logprobs":null}]}`,
			`{"choices":[{"index":0,"delta":{"content":"B"},"logprobs":{"content":[{"token":"B","logprob":-0.2}]}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":{"content":null}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cache_write_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}}`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", event)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()
	g, err := NewOpenAiGenerator(server.Client(), server.URL, "test-key")
	if err != nil {
		t.Fatal(err)
	}
	response, err := (&StreamingAdapter{S: g}).Generate(t.Context(), GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Candidates) != 1 || response.Candidates[0].Blocks[0].Content.String() != "AB" {
		t.Fatalf("response = %+v", response)
	}
	fields := response.Candidates[0].ExtraFields[OpenAIExtraFieldChoice].(map[string]json.RawMessage)
	var logprobs struct {
		Content []struct {
			Token string          `json:"token"`
			Extra json.RawMessage `json:"extra"`
		} `json:"content"`
	}
	if err := json.Unmarshal(fields["logprobs"], &logprobs); err != nil {
		t.Fatal(err)
	}
	if len(logprobs.Content) != 2 || logprobs.Content[0].Token != "A" || logprobs.Content[1].Token != "B" || string(logprobs.Content[0].Extra) != "9007199254740993" {
		t.Fatalf("logprobs = %s", fields["logprobs"])
	}
	if string(fields["finish_reason"]) != `"stop"` {
		t.Fatalf("finish reason = %s", fields["finish_reason"])
	}
	if response.UsageMetadata[UsageMetricCacheWriteTokens] != 3 || response.UsageMetadata[UsageMetricReasoningTokens] != 2 {
		t.Fatalf("usage = %+v", response.UsageMetadata)
	}
}

// testOpenAIEmbeddedErrors checks documented HTTP-200 error envelopes and
// choice errors without making provider calls. Partial answers are not success.
func testOpenAIEmbeddedErrors(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
		kind       APIErrorKind
		partial    bool
	}{
		{"top level", `{"id":"failed","error":{"code":429,"message":"limited"}}`, 429, APIErrorKindRateLimit, false},
		{"choice with partial output", `{"choices":[{"index":0,"message":{"role":"assistant","content":"partial"},"finish_reason":"error","error":{"code":502,"message":"limited"}}],"usage":{"prompt_tokens":5}}`, 502, APIErrorKindServer, true},
		{"choice without message", `{"choices":[{"index":0,"error":{"code":503,"message":"limited"}}]}`, 503, APIErrorKindServiceUnavailable, false},
		{"choice with interrupted tool", `{"choices":[{"index":0,"message":{"role":"assistant","content":"partial","reasoning":"checking","tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}]},"finish_reason":"error","error":{"code":503,"message":"limited"}}]}`, 503, APIErrorKindServiceUnavailable, true},
		{"top level with interrupted tool", `{"error":{"code":503,"message":"limited"},"choices":[{"index":0,"message":{"role":"assistant","content":"partial","reasoning":"checking","tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}]},"finish_reason":"error"}]}`, 503, APIErrorKindServiceUnavailable, true},
		{"string code", `{"error":{"code":"rate_limit_exceeded","message":"limited"}}`, 200, APIErrorKindUnknown, false},
	} {
		for _, mode := range []string{"generate", "stream JSON error", "stream event error"} {
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Retry-After", "7")
					w.Header().Set("X-Request-Id", "request-1")
					if mode == "stream event error" {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\ndata: %s\n\ndata: [DONE]\n\n", tc.body)
					} else {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprint(w, tc.body)
					}
				}))
				defer server.Close()
				g, err := NewOpenAiGenerator(server.Client(), server.URL, "test-key")
				if err != nil {
					t.Fatal(err)
				}
				request := GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
				if mode == "generate" {
					var response Response
					response, err = g.Generate(t.Context(), request)
					if tc.partial && (len(response.Candidates) != 1 || response.Candidates[0].Blocks[0].Content.String() != "partial") {
						t.Fatalf("partial response lost: %+v", response)
					}
					if strings.Contains(tc.name, "interrupted tool") {
						for _, block := range response.Candidates[0].Blocks {
							if block.BlockType == ToolCall {
								t.Fatal("incomplete tool call exposed")
							}
						}
						fields := response.Candidates[0].ExtraFields[OpenAIExtraFieldWireFields].(map[string]json.RawMessage)
						if string(fields["reasoning"]) != `"checking"` {
							t.Fatal("partial reasoning metadata lost")
						}
					}
					if response.ExtraFields[OpenAIResponseExtraFieldHeaders].(http.Header).Get("X-Request-Id") != "request-1" {
						t.Fatal("response headers lost")
					}
				} else {
					var text strings.Builder
					for chunk := range g.Stream(t.Context(), request) {
						if err != nil {
							t.Fatal("stream continued after error")
						}
						if chunk.Err != nil {
							err = chunk.Err
							continue
						}
						if chunk.Block.BlockType == Content {
							text.WriteString(chunk.Block.Content.String())
						}
					}
					if mode == "stream event error" && text.String() != "partial" {
						t.Fatalf("partial stream text = %q", text.String())
					}
				}
				var api *ApiErr
				if !errors.As(err, &api) {
					t.Fatalf("error = %v, want ApiErr", err)
				}
				if api.StatusCode != tc.status || api.Kind != tc.kind || api.Message != "limited" || api.RawBody != tc.body {
					t.Fatalf("API error = %+v", api)
				}
				if delay, ok := api.RetryAfter(); !ok || delay != 7*time.Second {
					t.Fatalf("retry delay = %v, %v", delay, ok)
				}
			})
		}
	}
}

func TestGenerate(t *testing.T) {
	t.Run("PartialThinkingMetadataRemoval", func(t *testing.T) {
		for _, scalar := range []string{"reasoning", "reasoning_content"} {
			t.Run(scalar, func(t *testing.T) {
				fields := map[string]json.RawMessage{scalar: json.RawMessage(`"First "`), "reasoning_details": json.RawMessage(`[{"type":"reasoning.text","text":"First step"}]`)}
				var display openAIThinkingDisplay
				blocks := append(display.update(fields), TextBlock("answer"))
				delete(fields, "reasoning_details")
				message := Message{Role: Assistant, Blocks: blocks, ExtraFields: map[string]interface{}{OpenAIExtraFieldWireFields: fields}}
				encoded, err := toOpenAIMessage(message)
				if err != nil {
					t.Fatal(err)
				}
				got, _ := encoded.ReasoningContent.Get()
				if got != "First step" {
					t.Fatalf("remaining %s: synthesized thinking = %q", scalar, got)
				}
				if string(fields[scalar]) != `"First "` {
					t.Fatal("mutated original wire metadata")
				}
			})
		}
	})
	t.Run("ThinkingReplayOwnership", func(t *testing.T) {
		for _, raw := range []string{`[]`, `null`, `[{"type":"reasoning.encrypted","data":"cipher"}]`, `[{"type":"reasoning.text","text":"derived"}]`} {
			t.Run(raw, func(t *testing.T) {
				fields := map[string]json.RawMessage{"reasoning_details": json.RawMessage(raw)}
				var display openAIThinkingDisplay
				blocks := append(display.update(fields), Block{BlockType: Thinking, ModalityType: Text, Content: Str("manual")}, TextBlock("answer"))
				message := Message{Role: Assistant, Blocks: blocks, ExtraFields: map[string]interface{}{OpenAIExtraFieldWireFields: fields}}
				encoded, err := toOpenAIMessage(message)
				if err != nil {
					t.Fatal(err)
				}
				got, _ := encoded.ReasoningContent.Get()
				if got != "manual" {
					t.Fatalf("manual reasoning = %q", got)
				}
				// Round-trip the block metadata where the source tag lives, as
				// well as message-level wire fields.
				for i := range message.Blocks {
					data, err := json.Marshal(message.Blocks[i].ExtraFields)
					if err != nil {
						t.Fatal(err)
					}
					if err := json.Unmarshal(data, &message.Blocks[i].ExtraFields); err != nil {
						t.Fatal(err)
					}
				}
				metadata, err := json.Marshal(message.ExtraFields)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(metadata, &message.ExtraFields); err != nil {
					t.Fatal(err)
				}
				encoded, err = toOpenAIMessage(message)
				if err != nil {
					t.Fatal(err)
				}
				got, _ = encoded.ReasoningContent.Get()
				if got != "manual" {
					t.Fatalf("round-trip reasoning = %q", got)
				}
				message.ExtraFields = nil
				encoded, err = toOpenAIMessage(message)
				if err != nil {
					t.Fatal(err)
				}
				want := "manual"
				if strings.Contains(raw, `"reasoning.text"`) {
					want = "derivedmanual"
				}
				got, _ = encoded.ReasoningContent.Get()
				if got != want {
					t.Fatalf("metadata removed: %q, want %q", got, want)
				}
			})
		}
	})
	t.Run("ThinkingAcrossCandidates", func(t *testing.T) {
		// A local protocol test, not a recorded multi-candidate provider response.
		// Both answers start alike so shared display state would lose text.
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, fragment := range []struct {
				candidate int
				text      string
			}{{0, "First "}, {1, "First "}, {1, "answer B"}, {0, "answer A"}} {
				fmt.Fprintf(w, "data: {\"choices\":[{\"index\":%d,\"delta\":{\"reasoning\":%q,\"reasoning_details\":[{\"index\":0,\"type\":\"reasoning.text\",\"text\":%q}]}}]}\n\n", fragment.candidate, fragment.text, fragment.text)
			}
			fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"},{\"index\":1,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
		}))
		defer server.Close()
		g, err := NewOpenAiGenerator(server.Client(), server.URL, "test-key")
		if err != nil {
			t.Fatal(err)
		}
		// StreamingAdapter accepts only candidate 0. Consume Stream directly
		// to check the generator's support for separate candidate indices.
		candidates := []Message{{Role: Assistant}, {Role: Assistant}}
		for chunk := range g.Stream(t.Context(), GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}) {
			if chunk.Err != nil {
				t.Fatal(chunk.Err)
			}
			if chunk.CandidatesIndex < 0 || chunk.CandidatesIndex >= len(candidates) {
				t.Fatalf("unexpected candidate index %d", chunk.CandidatesIndex)
			}
			candidate := &candidates[chunk.CandidatesIndex]
			if chunk.Block.BlockType == Thinking {
				candidate.Blocks = append(candidate.Blocks, chunk.Block)
			}
			if len(chunk.MessageExtraFields) > 0 {
				if candidate.ExtraFields == nil {
					candidate.ExtraFields = make(map[string]interface{})
				}
				for key, value := range chunk.MessageExtraFields {
					candidate.ExtraFields[key] = value
				}
			}
		}
		for i, want := range []string{"First answer A", "First answer B"} {
			var thought strings.Builder
			for _, block := range candidates[i].Blocks {
				if block.BlockType == Thinking {
					thought.WriteString(block.Content.String())
				}
			}
			if thought.String() != want {
				t.Errorf("candidate %d thinking = %q, want %q", i, thought.String(), want)
			}
			replay, err := toOpenAIMessage(candidates[i])
			if err != nil {
				t.Fatal(err)
			}
			var got string
			err = json.Unmarshal(replay.AdditionalProperties["reasoning"], &got)
			if err != nil || got != want {
				t.Errorf("candidate %d replay reasoning = %q, %v; want %q", i, got, err, want)
			}
			if got := strings.Join(openAIThinkingDetails(replay.AdditionalProperties), ""); got != want {
				t.Errorf("candidate %d reasoning details = %q, want %q", i, got, want)
			}
		}
	})
	t.Run("StructuredThinking", func(t *testing.T) {
		cases := []struct {
			name    string
			deltas  []string
			message string
			want    string
		}{
			{"text and summary", []string{
				`{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"First "}]}`,
				`{"reasoning_details":[{"index":0,"text":"step","signature":"sig"},{"index":1,"type":"reasoning.summary","summary":"Summary "}]}`,
				`{"reasoning_details":[{"index":1,"summary":"done"},{"index":2,"type":"reasoning.encrypted","data":"secret"}]}`,
			}, `{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"First step","signature":"sig"},{"index":1,"type":"reasoning.summary","summary":"Summary done"},{"index":2,"type":"reasoning.encrypted","data":"secret"}]}`, "First stepSummary done"},
			{"duplicate representations", []string{`{"reasoning":"thought","reasoning_content":"thought","reasoning_details":[{"index":0,"type":"reasoning.text","text":"thought"}]}`}, `{"reasoning":"thought","reasoning_content":"thought","reasoning_details":[{"index":0,"type":"reasoning.text","text":"thought"}]}`, "thought"},
			{"encrypted only", []string{`{"reasoning_details":[{"index":0,"type":"reasoning.encrypted","data":"secret","signature":"sig"}]}`}, `{"reasoning_details":[{"index":0,"type":"reasoning.encrypted","data":"secret","signature":"sig"}]}`, ""},
			{"opaque summary", []string{`{"reasoning_details":[{"index":0,"type":"reasoning.summary","summary":{"opaque":true}}]}`}, `{"reasoning_details":[{"index":0,"type":"reasoning.summary","summary":{"opaque":true}}]}`, ""},
			{"encrypted with scalar", []string{`{"reasoning":"readable","reasoning_details":[{"index":0,"type":"reasoning.encrypted","data":"secret"}]}`}, `{"reasoning":"readable","reasoning_details":[{"index":0,"type":"reasoning.encrypted","data":"secret"}]}`, "readable"},
			// The live GLM and DeepSeek probes returned matching reasoning and
			// reasoning_details text in every reasoning chunk.
			{"duplicate fields across chunks", []string{`{"reasoning":"First ","reasoning_details":[{"index":0,"type":"reasoning.text","text":"First "}]}`, `{"reasoning":"step","reasoning_details":[{"index":0,"type":"reasoning.text","text":"step"}]}`}, `{"reasoning":"First step","reasoning_details":[{"index":0,"type":"reasoning.text","text":"First step"}]}`, "First step"},
			{"nullable text fragments", []string{`{"reasoning_details":[{"index":0,"type":"reasoning.text","text":null,"signature":null,"data":null}]}`, `{"reasoning_details":[{"index":0,"text":"hello","signature":"sig","data":"cipher"}]}`, `{"reasoning_details":[{"index":0,"text":null,"signature":null,"data":null}]}`, `{"reasoning_details":[{"index":0,"text":" world","signature":"2","data":"2"}]}`}, `{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"hello world","signature":"sig2","data":"cipher2"}]}`, "hello world"},
			{"nullable summary fragments", []string{`{"reasoning_details":[{"index":0,"type":"reasoning.summary","summary":null}]}`, `{"reasoning_details":[{"index":0,"summary":"hello"}]}`, `{"reasoning_details":[{"index":0,"summary":null}]}`, `{"reasoning_details":[{"index":0,"summary":" world"}]}`}, `{"reasoning_details":[{"index":0,"type":"reasoning.summary","summary":"hello world"}]}`, "hello world"},
			{"repeated null summary", []string{`{"reasoning_details":[{"index":0,"type":"reasoning.summary","summary":null}]}`, `{"reasoning_details":[{"index":0,"summary":null}]}`}, `{"reasoning_details":[{"index":0,"type":"reasoning.summary","summary":null}]}`, ""},
			{"partial overlap", []string{`{"reasoning":"First "}`, `{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"First step"}]}`}, `{"reasoning":"First ","reasoning_details":[{"index":0,"type":"reasoning.text","text":"First step"}]}`, "First step"},
			{"divergent representations", []string{`{"reasoning":"abc"}`, `{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"abX"}]}`}, `{"reasoning":"abc","reasoning_details":[{"index":0,"type":"reasoning.text","text":"abX"}]}`, ""},
			{"unknown continuation", []string{`{"reasoning_details":[{"index":0,"type":"vendor.opaque","text":"hidden "}]}`, `{"reasoning_details":[{"index":0,"text":"payload"}]}`}, `{"reasoning_details":[{"index":0,"type":"vendor.opaque","text":"hidden payload"}]}`, ""},
			{"encrypted continuation", []string{`{"reasoning_details":[{"index":0,"type":"reasoning.encrypted","text":"hidden "}]}`, `{"reasoning_details":[{"index":0,"text":"payload"}]}`}, `{"reasoning_details":[{"index":0,"type":"reasoning.encrypted","text":"hidden payload"}]}`, ""},
			{"late duplicate", []string{`{"reasoning":"thought"}`, `{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"thought"}]}`}, `{"reasoning":"thought","reasoning_details":[{"index":0,"type":"reasoning.text","text":"thought"}]}`, "thought"},
			{"signature before text", []string{`{"reasoning_details":[{"index":0,"type":"reasoning.text","signature":"sig"}]}`, `{"reasoning_details":[{"index":0,"text":"readable"}]}`}, `{"reasoning_details":[{"index":0,"type":"reasoning.text","signature":"sig","text":"readable"}]}`, "readable"},
		}
		for _, tc := range cases {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%v", tc.name, stream), func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						if stream {
							w.Header().Set("Content-Type", "text/event-stream")
							for _, delta := range tc.deltas {
								fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":%s}]}\n\n", delta)
							}
							fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
						} else {
							w.Header().Set("Content-Type", "application/json")
							fmt.Fprintf(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"answer",%s},"finish_reason":"stop"}]}`, tc.message[1:len(tc.message)-1])
						}
					}))
					defer server.Close()
					g, err := NewOpenAiGenerator(server.Client(), server.URL, "test-key")
					if err != nil {
						t.Fatal(err)
					}
					request := GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
					var response Response
					if stream {
						response, err = (&StreamingAdapter{S: g}).Generate(t.Context(), request)
					} else {
						response, err = g.Generate(t.Context(), request)
					}
					if err != nil {
						t.Fatal(err)
					}
					var thought strings.Builder
					for _, block := range response.Candidates[0].Blocks {
						if block.BlockType == Thinking {
							thought.WriteString(block.Content.String())
						}
					}
					wantThought := tc.want
					if tc.name == "divergent representations" {
						wantThought = "abXabc"
						if stream {
							wantThought = "abcabX"
						}
					}
					if thought.String() != wantThought {
						t.Fatalf("thinking = %q, want %q", thought.String(), wantThought)
					}
					replay, err := toOpenAIMessage(response.Candidates[0])
					if err != nil {
						t.Fatal(err)
					}
					data, err := json.Marshal(replay)
					if err != nil {
						t.Fatal(err)
					}
					var got, want map[string]json.RawMessage
					if err := json.Unmarshal(data, &got); err != nil {
						t.Fatal(err)
					}
					if err := json.Unmarshal([]byte(tc.message), &want); err != nil {
						t.Fatal(err)
					}
					for _, key := range []string{"reasoning", "reasoning_content", "reasoning_details"} {
						var gotValue, wantValue any
						if got[key] != nil {
							if err := json.Unmarshal(got[key], &gotValue); err != nil {
								t.Fatal(err)
							}
						}
						if want[key] != nil {
							if err := json.Unmarshal(want[key], &wantValue); err != nil {
								t.Fatal(err)
							}
						}
						if !reflect.DeepEqual(gotValue, wantValue) {
							t.Errorf("replayed %s = %s, want %s", key, got[key], want[key])
						}
					}
				})
			}
		}
	})
	// Create a simple test dialog
	testDialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hello, how are you?"),
				},
			},
		},
	}

	// Create a test dialog with multiple messages
	multiMessageDialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What's the weather like?"),
				},
			},
		},
		{
			Role: Assistant,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("I'll check the weather for you. Where are you located?"),
				},
			},
		},
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("San Francisco"),
				},
			},
		},
	}

	// Test with single stop sequence
	singleStopOptions := NewGenerationOptions(
		WithTemperature(0.7),
		WithStopSequences("stop"),
	)

	// Create a dialog with a tool result that the assistant will use in its response
	toolResultDialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What's the weather like in London?"),
				},
			},
		},
		{
			Role: Assistant,
			Blocks: []Block{
				{
					ID:           "call_789",
					BlockType:    ToolCall,
					ModalityType: Text,
					Content:      Str(`{"name":"get_weather","parameters":{"location":"London"}}`),
				},
			},
		},
		{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           "call_789",
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("The weather in London is 15°C and cloudy with a 30% chance of rain."),
				},
			},
		},
	}

	// Standard options for tests
	testOptions := NewGenerationOptions(WithTemperature(0.7))

	// Advanced options for testing more parameters
	advancedOptions := NewGenerationOptions(
		WithTemperature(0.5),
		WithTopP(0.9),
		WithTopK(10),
		WithFrequencyPenalty(0.2),
		WithPresencePenalty(0.1),
		WithMaxGenerationTokens(100),
		WithCandidateCount(2),
		WithStopSequences("stop"),
		WithToolChoice(ToolChoiceToolsRequired),
	)

	// Typical successful response
	normalResponse := &oai.ChatCompletion{
		ID:     "chat-123",
		Object: "chat.completion",
		Model:  "gpt-4",
		Choices: []oai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: oai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "I'm doing well, thank you for asking!",
				},
			},
		},
		Usage: oai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 9,
			TotalTokens:      19,
		},
	}

	// Response with a tool call
	toolCallResponse := &oai.ChatCompletion{
		ID:     "chat-456",
		Object: "chat.completion",
		Model:  "gpt-4",
		Choices: []oai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "tool_calls",
				Message: oai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "",
					ToolCalls: []oai.ChatCompletionMessageToolCallUnion{
						{
							ID:   "call_123",
							Type: "function",
							Function: oai.ChatCompletionMessageFunctionToolCallFunction{
								Name:      "get_weather",
								Arguments: `{"location": "London"}`,
							},
						},
					},
				},
			},
		},
		Usage: oai.CompletionUsage{
			PromptTokens:     12,
			CompletionTokens: 15,
			TotalTokens:      27,
		},
	}

	// Response with parallel tool calls
	parallelToolCallsResponse := &oai.ChatCompletion{
		ID:     "chat-789",
		Object: "chat.completion",
		Model:  "gpt-4",
		Choices: []oai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "tool_calls",
				Message: oai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "",
					ToolCalls: []oai.ChatCompletionMessageToolCallUnion{
						{
							ID:   "call_456",
							Type: "function",
							Function: oai.ChatCompletionMessageFunctionToolCallFunction{
								Name:      "get_weather",
								Arguments: `{"location": "London"}`,
							},
						},
						{
							ID:   "call_457",
							Type: "function",
							Function: oai.ChatCompletionMessageFunctionToolCallFunction{
								Name:      "get_time",
								Arguments: `{"timezone": "UTC"}`,
							},
						},
					},
				},
			},
		},
		Usage: oai.CompletionUsage{
			PromptTokens:     12,
			CompletionTokens: 25,
			TotalTokens:      37,
		},
	}

	// Response that uses information from a tool result
	toolResultResponse := &oai.ChatCompletion{
		ID:     "chat-321",
		Object: "chat.completion",
		Model:  "gpt-4",
		Choices: []oai.ChatCompletionChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: oai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "Based on the current weather data, it's 15°C and cloudy in London with a 30% chance of rain. You might want to take an umbrella if you're heading out!",
				},
			},
		},
		Usage: oai.CompletionUsage{
			PromptTokens:     25,
			CompletionTokens: 20,
			TotalTokens:      45,
		},
	}

	tests := []struct {
		name     string
		client   *mockChatCompletionService
		dialog   Dialog
		options  GenerationOptions
		want     Response
		wantErr  bool
		errorMsg string
	}{
		{
			name: "error: API timeout",
			client: &mockChatCompletionService{
				response: nil,
				err:      errors.New("request timeout: deadline exceeded"),
			},
			dialog:   testDialog,
			options:  testOptions,
			want:     Response{},
			wantErr:  true,
			errorMsg: "request timeout: deadline exceeded",
		},
		{
			name: "error: rate limit",
			client: &mockChatCompletionService{
				response: nil,
				err:      errors.New("rate limit exceeded, please try again later"),
			},
			dialog:   testDialog,
			options:  testOptions,
			want:     Response{},
			wantErr:  true,
			errorMsg: "rate limit exceeded, please try again later",
		},
		{
			name: "normal assistant response",
			client: &mockChatCompletionService{
				response: normalResponse,
				err:      nil,
			},
			dialog:  testDialog,
			options: testOptions,
			want: Response{
				Candidates: []Message{
					{
						Role: Assistant,
						Blocks: []Block{
							{
								BlockType:    Content,
								ModalityType: Text,
								Content:      Str("I'm doing well, thank you for asking!"),
							},
						},
					},
				},
				FinishReason: EndTurn,
				UsageMetadata: Metadata{
					UsageMetricInputTokens:      10,
					UsageMetricGenerationTokens: 9,
				},
			},
			wantErr: false,
		},
		{
			name: "tool call response",
			client: &mockChatCompletionService{
				response: toolCallResponse,
				err:      nil,
			},
			dialog:  testDialog,
			options: testOptions,
			want: Response{
				Candidates: []Message{
					{
						Role: Assistant,
						Blocks: []Block{
							{
								ID:           "call_123",
								BlockType:    ToolCall,
								ModalityType: Text,
								Content:      Str(`{"name":"get_weather","parameters":{"location":"London"}}`),
							},
						},
					},
				},
				FinishReason: ToolUse,
				UsageMetadata: Metadata{
					UsageMetricInputTokens:      12,
					UsageMetricGenerationTokens: 15,
				},
			},
			wantErr: false,
		},
		{
			name: "parallel tool calls response",
			client: &mockChatCompletionService{
				response: parallelToolCallsResponse,
				err:      nil,
			},
			dialog:  testDialog,
			options: testOptions,
			want: Response{
				Candidates: []Message{
					{
						Role: Assistant,
						Blocks: []Block{
							{
								ID:           "call_456",
								BlockType:    ToolCall,
								ModalityType: Text,
								Content:      Str(`{"name":"get_weather","parameters":{"location":"London"}}`),
							},
							{
								ID:           "call_457",
								BlockType:    ToolCall,
								ModalityType: Text,
								Content:      Str(`{"name":"get_time","parameters":{"timezone":"UTC"}}`),
							},
						},
					},
				},
				FinishReason: ToolUse,
				UsageMetadata: Metadata{
					UsageMetricInputTokens:      12,
					UsageMetricGenerationTokens: 25,
				},
			},
			wantErr: false,
		},
		{
			name: "normal assistant response with advanced options",
			client: &mockChatCompletionService{
				response: normalResponse,
				err:      nil,
			},
			dialog:  testDialog,
			options: advancedOptions,
			want: Response{
				Candidates: []Message{
					{
						Role: Assistant,
						Blocks: []Block{
							{
								BlockType:    Content,
								ModalityType: Text,
								Content:      Str("I'm doing well, thank you for asking!"),
							},
						},
					},
				},
				FinishReason: EndTurn,
				UsageMetadata: Metadata{
					UsageMetricInputTokens:      10,
					UsageMetricGenerationTokens: 9,
				},
			},
			wantErr: false,
		},
		{
			name: "multi-message dialog",
			client: &mockChatCompletionService{
				response: normalResponse,
				err:      nil,
			},
			dialog:  multiMessageDialog,
			options: testOptions,
			want: Response{
				Candidates: []Message{
					{
						Role: Assistant,
						Blocks: []Block{
							{
								BlockType:    Content,
								ModalityType: Text,
								Content:      Str("I'm doing well, thank you for asking!"),
							},
						},
					},
				},
				FinishReason: EndTurn,
				UsageMetadata: Metadata{
					UsageMetricInputTokens:      10,
					UsageMetricGenerationTokens: 9,
				},
			},
			wantErr: false,
		},
		{
			name: "response using tool result information",
			client: &mockChatCompletionService{
				response: toolResultResponse,
				err:      nil,
			},
			dialog:  toolResultDialog,
			options: testOptions,
			want: Response{
				Candidates: []Message{
					{
						Role: Assistant,
						Blocks: []Block{
							{
								BlockType:    Content,
								ModalityType: Text,
								Content:      Str("Based on the current weather data, it's 15°C and cloudy in London with a 30% chance of rain. You might want to take an umbrella if you're heading out!"),
							},
						},
					},
				},
				FinishReason: EndTurn,
				UsageMetadata: Metadata{
					UsageMetricInputTokens:      25,
					UsageMetricGenerationTokens: 20,
				},
			},
			wantErr: false,
		},
		{
			name: "single stop sequence",
			client: &mockChatCompletionService{
				response: normalResponse,
				err:      nil,
			},
			dialog:  testDialog,
			options: singleStopOptions,
			want: Response{
				Candidates: []Message{
					{
						Role: Assistant,
						Blocks: []Block{
							{
								BlockType:    Content,
								ModalityType: Text,
								Content:      Str("I'm doing well, thank you for asking!"),
							},
						},
					},
				},
				FinishReason: EndTurn,
				UsageMetadata: Metadata{
					UsageMetricInputTokens:      10,
					UsageMetricGenerationTokens: 9,
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create generator with mock client
			g := newTestOpenAIGenerator(t, tt.client)

			// Call Generate
			got, err := g.Generate(context.Background(), GenerationRequest{
				Model:        "gpt-4",
				Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
				Dialog:       tt.dialog,
				Options:      tt.options,
			})

			// Check error cases
			if (err != nil) != tt.wantErr {
				t.Errorf("Generate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err != nil && !strings.Contains(err.Error(), tt.errorMsg) {
				t.Errorf("Generate() error message = %v, want %v", err.Error(), tt.errorMsg)
				return
			}

			// For successful cases, check response
			if !tt.wantErr {
				// Check finish reason
				if got.FinishReason != tt.want.FinishReason {
					t.Errorf("Generate() finish reason = %v, want %v", got.FinishReason, tt.want.FinishReason)
				}

				// Check candidates count
				if len(got.Candidates) != len(tt.want.Candidates) {
					t.Errorf("Generate() candidates count = %d, want %d", len(got.Candidates), len(tt.want.Candidates))
					return
				}

				// For tool call test cases, compare blocks length
				for i, candidate := range got.Candidates {
					wantCandidate := tt.want.Candidates[i]
					if len(candidate.Blocks) != len(wantCandidate.Blocks) {
						t.Errorf("Generate() candidate[%d] blocks count = %d, want %d",
							i, len(candidate.Blocks), len(wantCandidate.Blocks))
						continue
					}

					// Check block types and content
					for j, block := range candidate.Blocks {
						wantBlock := wantCandidate.Blocks[j]
						if block.BlockType != wantBlock.BlockType {
							t.Errorf("Generate() block[%d] type = %s, want %s",
								j, block.BlockType, wantBlock.BlockType)
						}

						// For tool calls, check ID
						if block.BlockType == ToolCall {
							if block.ID != wantBlock.ID {
								t.Errorf("Generate() tool call ID = %s, want %s",
									block.ID, wantBlock.ID)
							}
						}
					}
				}

				// Check usage metrics
				inputTokens, hasInputTokens := InputTokens(got.UsageMetadata)
				wantInputTokens, wantHasInputTokens := InputTokens(tt.want.UsageMetadata)

				if hasInputTokens != wantHasInputTokens {
					t.Errorf("Generate() has input tokens = %v, want %v",
						hasInputTokens, wantHasInputTokens)
				}

				if hasInputTokens && inputTokens != wantInputTokens {
					t.Errorf("Generate() input tokens = %d, want %d",
						inputTokens, wantInputTokens)
				}

				outputTokens, hasOutputTokens := OutputTokens(got.UsageMetadata)
				wantOutputTokens, wantHasOutputTokens := OutputTokens(tt.want.UsageMetadata)

				if hasOutputTokens != wantHasOutputTokens {
					t.Errorf("Generate() has output tokens = %v, want %v",
						hasOutputTokens, wantHasOutputTokens)
				}

				if hasOutputTokens && outputTokens != wantOutputTokens {
					t.Errorf("Generate() output tokens = %d, want %d",
						outputTokens, wantOutputTokens)
				}
			}
		})
	}
}
