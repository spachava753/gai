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
	"strings"
	"testing"
	"time"

	oaissestream "github.com/openai/openai-go/v3/packages/ssestream"

	oai "github.com/openai/openai-go/v3"

	wire "github.com/spachava753/gai/internal/openai"
)

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
	for chunk := range g.Stream(t.Context(), GenerationRequest{Model: "m", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hi")}}}, Options: NewGenerationOptions(WithOutputModalities(Text, Audio), WithAudioConfig(AudioConfig{VoiceName: "alloy", Format: "wav"}))}) {
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

func TestGenerate(t *testing.T) {
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
