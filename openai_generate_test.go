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

	oai "github.com/openai/openai-go/v3"
	oaissestream "github.com/openai/openai-go/v3/packages/ssestream"
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
