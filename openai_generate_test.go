package gai

import (
	"context"
	"errors"
	"strings"
	"testing"

	oaissestream "github.com/openai/openai-go/v3/packages/ssestream"

	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// mockChatCompletionService is a mock implementation of OpenAICompletionService
type mockChatCompletionService struct {
	response     *oai.ChatCompletion
	err          error
	streamEvents []oaissestream.Event
	requests     []oai.ChatCompletionNewParams
}

func (m *mockChatCompletionService) New(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) (*oai.ChatCompletion, error) {
	m.requests = append(m.requests, body)
	return m.response, m.err
}

func (m *mockChatCompletionService) NewStreaming(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) (stream *oaissestream.Stream[oai.ChatCompletionChunk]) {
	return oaissestream.NewStream[oai.ChatCompletionChunk](&openAIStreamDecoder{events: m.streamEvents}, nil)
}

type openAIStreamDecoder struct {
	events []oaissestream.Event
	index  int
	cur    oaissestream.Event
}

func (d *openAIStreamDecoder) Next() bool {
	if d.index >= len(d.events) {
		return false
	}
	d.cur = d.events[d.index]
	d.index++
	return true
}

func (d *openAIStreamDecoder) Event() oaissestream.Event { return d.cur }
func (d *openAIStreamDecoder) Close() error              { return nil }
func (d *openAIStreamDecoder) Err() error                { return nil }

func testOpenAIGenerateReturnsContentPolicyErrorForRefusal(t *testing.T) {
	client := &mockChatCompletionService{response: &oai.ChatCompletion{
		Choices: []oai.ChatCompletionChoice{{
			FinishReason: "stop",
			Message:      oai.ChatCompletionMessage{Refusal: "I cannot help with that."},
		}},
	}}
	generator := NewOpenAiGenerator(client)

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
	generator := NewOpenAiGenerator(client)

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
	generator := NewOpenAiGenerator(client)

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
					Content:      Str(`{"name":"get_weather","arguments":{"location":"London"}}`),
				},
			},
		},
		{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           "call_789",
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
			errorMsg: "failed to create new message: request timeout: deadline exceeded",
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
			errorMsg: "failed to create new message: rate limit exceeded, please try again later",
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
								Content:      Str(`{"name":"get_weather","arguments":{"location":"London"}}`),
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
								Content:      Str(`{"name":"get_weather","arguments":{"location":"London"}}`),
							},
							{
								ID:           "call_457",
								BlockType:    ToolCall,
								ModalityType: Text,
								Content:      Str(`{"name":"get_time","arguments":{"timezone":"UTC"}}`),
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
			g := NewOpenAiGenerator(tt.client)

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

			if tt.wantErr && err != nil && err.Error() != tt.errorMsg {
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
