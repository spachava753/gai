package openai

import (
	"context"
	"errors"
	"testing"

	oai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/spachava753/gai"
)

// mockChatCompletionService is a mock implementation of ChatCompletionService
type mockChatCompletionService struct {
	response *oai.ChatCompletion
	err      error
}

func (m *mockChatCompletionService) New(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) (*oai.ChatCompletion, error) {
	return m.response, m.err
}

func TestGenerate(t *testing.T) {
	// Create a simple test dialog
	testDialog := gai.Dialog{
		{
			Role: gai.User,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      "Hello, how are you?",
				},
			},
		},
	}

	// Create a test dialog with multiple messages
	multiMessageDialog := gai.Dialog{
		{
			Role: gai.User,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      "What's the weather like?",
				},
			},
		},
		{
			Role: gai.Assistant,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      "I'll check the weather for you. Where are you located?",
				},
			},
		},
		{
			Role: gai.User,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      "San Francisco",
				},
			},
		},
	}

	// Test with single stop sequence
	singleStopOptions := &gai.GenOpts{
		Temperature:   0.7,
		StopSequences: []string{"stop"},
	}

	// Create a dialog with a tool result that the assistant will use in its response
	toolResultDialog := gai.Dialog{
		{
			Role: gai.User,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      "What's the weather like in London?",
				},
			},
		},
		{
			Role: gai.Assistant,
			Blocks: []gai.Block{
				{
					ID:           "call_789",
					BlockType:    gai.ToolCall,
					ModalityType: gai.Text,
					Content:      `{"name":"get_weather","arguments":{"location":"London"}}`,
				},
			},
		},
		{
			Role: gai.Assistant,
			Blocks: []gai.Block{
				{
					ID:           "call_789",
					BlockType:    gai.ToolResult,
					ModalityType: gai.Text,
					Content:      "The weather in London is 15°C and cloudy with a 30% chance of rain.",
				},
			},
		},
	}

	// Standard options for tests
	testOptions := &gai.GenOpts{
		Temperature: 0.7,
	}

	// Advanced options for testing more parameters
	advancedOptions := &gai.GenOpts{
		Temperature:         0.5,
		TopP:                0.9,
		TopK:                10,
		FrequencyPenalty:    0.2,
		PresencePenalty:     0.1,
		MaxGenerationTokens: 100,
		N:                   2,
		StopSequences:       []string{"stop"},
		ToolChoice:          gai.ToolChoiceToolsRequired,
	}

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
					ToolCalls: []oai.ChatCompletionMessageToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: oai.ChatCompletionMessageToolCallFunction{
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
					ToolCalls: []oai.ChatCompletionMessageToolCall{
						{
							ID:   "call_456",
							Type: "function",
							Function: oai.ChatCompletionMessageToolCallFunction{
								Name:      "get_weather",
								Arguments: `{"location": "London"}`,
							},
						},
						{
							ID:   "call_457",
							Type: "function",
							Function: oai.ChatCompletionMessageToolCallFunction{
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
		dialog   gai.Dialog
		options  *gai.GenOpts
		want     gai.Response
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
			want:     gai.Response{},
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
			want:     gai.Response{},
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
			want: gai.Response{
				Candidates: []gai.Message{
					{
						Role: gai.Assistant,
						Blocks: []gai.Block{
							{
								BlockType:    gai.Content,
								ModalityType: gai.Text,
								Content:      "I'm doing well, thank you for asking!",
							},
						},
					},
				},
				FinishReason: gai.EndTurn,
				UsageMetrics: gai.Metrics{
					gai.UsageMetricInputTokens:      10,
					gai.UsageMetricGenerationTokens: 9,
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
			want: gai.Response{
				Candidates: []gai.Message{
					{
						Role: gai.Assistant,
						Blocks: []gai.Block{
							{
								ID:           "call_123",
								BlockType:    gai.ToolCall,
								ModalityType: gai.Text,
								Content:      `{"name":"get_weather","arguments":{"location":"London"}}`,
							},
						},
					},
				},
				FinishReason: gai.ToolUse,
				UsageMetrics: gai.Metrics{
					gai.UsageMetricInputTokens:      12,
					gai.UsageMetricGenerationTokens: 15,
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
			want: gai.Response{
				Candidates: []gai.Message{
					{
						Role: gai.Assistant,
						Blocks: []gai.Block{
							{
								ID:           "call_456",
								BlockType:    gai.ToolCall,
								ModalityType: gai.Text,
								Content:      `{"name":"get_weather","arguments":{"location":"London"}}`,
							},
							{
								ID:           "call_457",
								BlockType:    gai.ToolCall,
								ModalityType: gai.Text,
								Content:      `{"name":"get_time","arguments":{"timezone":"UTC"}}`,
							},
						},
					},
				},
				FinishReason: gai.ToolUse,
				UsageMetrics: gai.Metrics{
					gai.UsageMetricInputTokens:      12,
					gai.UsageMetricGenerationTokens: 25,
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
			want: gai.Response{
				Candidates: []gai.Message{
					{
						Role: gai.Assistant,
						Blocks: []gai.Block{
							{
								BlockType:    gai.Content,
								ModalityType: gai.Text,
								Content:      "I'm doing well, thank you for asking!",
							},
						},
					},
				},
				FinishReason: gai.EndTurn,
				UsageMetrics: gai.Metrics{
					gai.UsageMetricInputTokens:      10,
					gai.UsageMetricGenerationTokens: 9,
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
			want: gai.Response{
				Candidates: []gai.Message{
					{
						Role: gai.Assistant,
						Blocks: []gai.Block{
							{
								BlockType:    gai.Content,
								ModalityType: gai.Text,
								Content:      "I'm doing well, thank you for asking!",
							},
						},
					},
				},
				FinishReason: gai.EndTurn,
				UsageMetrics: gai.Metrics{
					gai.UsageMetricInputTokens:      10,
					gai.UsageMetricGenerationTokens: 9,
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
			want: gai.Response{
				Candidates: []gai.Message{
					{
						Role: gai.Assistant,
						Blocks: []gai.Block{
							{
								BlockType:    gai.Content,
								ModalityType: gai.Text,
								Content:      "Based on the current weather data, it's 15°C and cloudy in London with a 30% chance of rain. You might want to take an umbrella if you're heading out!",
							},
						},
					},
				},
				FinishReason: gai.EndTurn,
				UsageMetrics: gai.Metrics{
					gai.UsageMetricInputTokens:      25,
					gai.UsageMetricGenerationTokens: 20,
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
			want: gai.Response{
				Candidates: []gai.Message{
					{
						Role: gai.Assistant,
						Blocks: []gai.Block{
							{
								BlockType:    gai.Content,
								ModalityType: gai.Text,
								Content:      "I'm doing well, thank you for asking!",
							},
						},
					},
				},
				FinishReason: gai.EndTurn,
				UsageMetrics: gai.Metrics{
					gai.UsageMetricInputTokens:      10,
					gai.UsageMetricGenerationTokens: 9,
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create generator with mock client
			g := New(tt.client, "gpt-4", "You are a helpful assistant")

			// Call Generate
			got, err := g.Generate(context.Background(), tt.dialog, tt.options)

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
						if block.BlockType == gai.ToolCall {
							if block.ID != wantBlock.ID {
								t.Errorf("Generate() tool call ID = %s, want %s",
									block.ID, wantBlock.ID)
							}
						}
					}
				}

				// Check usage metrics
				inputTokens, hasInputTokens := gai.InputTokens(got.UsageMetrics)
				wantInputTokens, wantHasInputTokens := gai.InputTokens(tt.want.UsageMetrics)

				if hasInputTokens != wantHasInputTokens {
					t.Errorf("Generate() has input tokens = %v, want %v",
						hasInputTokens, wantHasInputTokens)
				}

				if hasInputTokens && inputTokens != wantInputTokens {
					t.Errorf("Generate() input tokens = %d, want %d",
						inputTokens, wantInputTokens)
				}

				outputTokens, hasOutputTokens := gai.OutputTokens(got.UsageMetrics)
				wantOutputTokens, wantHasOutputTokens := gai.OutputTokens(tt.want.UsageMetrics)

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
