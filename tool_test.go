package gai

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// mockToolCapableGenerator implements ToolCapableGenerator for testing
type mockToolCapableGenerator struct {
	registeredTools map[string]Tool
	generateFunc    func(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error)
}

func newMockToolCapableGenerator() *mockToolCapableGenerator {
	return &mockToolCapableGenerator{
		registeredTools: make(map[string]Tool),
	}
}

func (m *mockToolCapableGenerator) Register(tool Tool) error {
	if tool.Name == "" {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: errors.New("tool name cannot be empty"),
		}
	}

	if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool name cannot be %s", tool.Name),
		}
	}

	if _, exists := m.registeredTools[tool.Name]; exists {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: errors.New("tool already registered"),
		}
	}

	m.registeredTools[tool.Name] = tool
	return nil
}

func (m *mockToolCapableGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, dialog, options)
	}
	return Response{}, nil
}

// mockToolCallback implements ToolCallback for testing
type mockToolCallback struct {
	callFunc func(ctx context.Context, input map[string]any) (any, error)
}

func (m *mockToolCallback) Call(ctx context.Context, input map[string]any) (any, error) {
	if m.callFunc != nil {
		return m.callFunc(ctx, input)
	}
	return "default result", nil
}

func TestToolGenerator_Register(t *testing.T) {
	tests := []struct {
		name      string
		tool      Tool
		callback  ToolCallback
		setupMock func(*mockToolCapableGenerator)
		wantErr   bool
		errType   reflect.Type
	}{
		{
			name: "successful registration",
			tool: Tool{
				Name:        "test_tool",
				Description: "A test tool",
			},
			callback:  &mockToolCallback{},
			setupMock: func(m *mockToolCapableGenerator) {},
			wantErr:   false,
		},
		{
			name: "nil callback",
			tool: Tool{
				Name:        "test_tool",
				Description: "A test tool",
			},
			callback:  nil,
			setupMock: func(m *mockToolCapableGenerator) {},
			wantErr:   true,
			errType:   reflect.TypeOf(&ToolRegistrationErr{}),
		},
		{
			name: "duplicate tool registration",
			tool: Tool{
				Name:        "duplicate_tool",
				Description: "A duplicate tool",
			},
			callback: &mockToolCallback{},
			setupMock: func(m *mockToolCapableGenerator) {
				m.registeredTools["duplicate_tool"] = Tool{Name: "duplicate_tool"}
			},
			wantErr: true,
			errType: reflect.TypeOf(&ToolRegistrationErr{}),
		},
		{
			name: "empty tool name",
			tool: Tool{
				Name:        "",
				Description: "A tool with empty name",
			},
			callback:  &mockToolCallback{},
			setupMock: func(m *mockToolCapableGenerator) {},
			wantErr:   true,
			errType:   reflect.TypeOf(&ToolRegistrationErr{}),
		},
		{
			name: "tool name equals ToolChoiceAuto",
			tool: Tool{
				Name:        ToolChoiceAuto,
				Description: "A tool with reserved name",
			},
			callback:  &mockToolCallback{},
			setupMock: func(m *mockToolCapableGenerator) {},
			wantErr:   true,
			errType:   reflect.TypeOf(&ToolRegistrationErr{}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockGen := newMockToolCapableGenerator()
			tt.setupMock(mockGen)

			toolGen := &ToolGenerator{
				G:             mockGen,
				toolCallbacks: make(map[string]ToolCallback),
			}

			err := toolGen.Register(tt.tool, tt.callback)

			if (err != nil) != tt.wantErr {
				t.Errorf("Register() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil && tt.errType != nil && reflect.TypeOf(err) != tt.errType {
				t.Errorf("Register() error type = %T, want %v", err, tt.errType)
				return
			}

			if !tt.wantErr {
				// Verify the tool was registered with the mock generator
				if _, exists := mockGen.registeredTools[tt.tool.Name]; !exists {
					t.Errorf("Tool not registered with underlying generator")
				}

				// Verify the callback was stored in the ToolGenerator
				storedCallback := toolGen.toolCallbacks[tt.tool.Name]
				if storedCallback == nil {
					t.Errorf("Expected non-nil callback, got nil")
				} else if storedCallback != tt.callback {
					t.Errorf("Callback mismatch: got %v, want %v", storedCallback, tt.callback)
				}
			}
		})
	}
}

func TestToolGenerator_Generate(t *testing.T) {
	tests := []struct {
		name           string
		dialog         Dialog
		options        *GenOpts
		setupTools     func(*ToolGenerator)
		setupMockGen   func(*mockToolCapableGenerator)
		wantErr        bool
		expectedErrMsg string
		validateDialog func(*testing.T, Dialog)
	}{
		{
			name:   "successful generation without tool calls",
			dialog: Dialog{Message{Role: User, Blocks: []Block{{Content: "Hello"}}}},
			options: &GenOpts{
				ToolChoice: ToolChoiceAuto,
			},
			setupTools: func(tg *ToolGenerator) {},
			setupMockGen: func(m *mockToolCapableGenerator) {
				m.generateFunc = func(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
					return Response{
						Candidates: []Message{
							{
								Role: Assistant,
								Blocks: []Block{
									{
										BlockType:    Content,
										ModalityType: Text,
										Content:      "Hi there!",
									},
								},
							},
						},
						FinishReason: EndTurn,
					}, nil
				}
			},
			wantErr: false,
			validateDialog: func(t *testing.T, dialog Dialog) {
				if len(dialog) != 2 {
					t.Errorf("Expected 2 messages in dialog, got %d", len(dialog))
				}
				if dialog[1].Role != Assistant || dialog[1].Blocks[0].Content != "Hi there!" {
					t.Errorf("Expected final message to be Assistant saying 'Hi there!'")
				}
			},
		},
		{
			name:   "invalid tool choice",
			dialog: Dialog{Message{Role: User, Blocks: []Block{{Content: "Hello"}}}},
			options: &GenOpts{
				ToolChoice: "nonexistent_tool",
			},
			setupTools:     func(tg *ToolGenerator) {},
			setupMockGen:   func(m *mockToolCapableGenerator) {},
			wantErr:        true,
			expectedErrMsg: "tool 'nonexistent_tool' not found",
		},
		{
			name:   "successful tool call execution",
			dialog: Dialog{Message{Role: User, Blocks: []Block{{Content: "What time is it?"}}}},
			options: &GenOpts{
				ToolChoice: ToolChoiceAuto,
			},
			setupTools: func(tg *ToolGenerator) {
				tg.Register(Tool{
					Name:        "get_time",
					Description: "Get the current time",
				}, &mockToolCallback{
					callFunc: func(ctx context.Context, input map[string]any) (any, error) {
						return "12:34 PM", nil
					},
				})
			},
			setupMockGen: func(m *mockToolCapableGenerator) {
				// First call: Generate with tool call response
				firstCallExecuted := false
				m.generateFunc = func(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
					if !firstCallExecuted {
						firstCallExecuted = true
						return Response{
							Candidates: []Message{
								{
									Role: Assistant,
									Blocks: []Block{
										{
											ID:           "time_call_1",
											BlockType:    ToolCall,
											ModalityType: Text,
											Content:      `{"name":"get_time","parameters":{}}`,
										},
									},
								},
							},
							FinishReason: ToolUse,
						}, nil
					} else {
						return Response{
							Candidates: []Message{
								{
									Role: Assistant,
									Blocks: []Block{
										{
											BlockType:    Content,
											ModalityType: Text,
											Content:      "The current time is 12:34 PM",
										},
									},
								},
							},
							FinishReason: EndTurn,
						}, nil
					}
				}
			},
			wantErr: false,
			validateDialog: func(t *testing.T, dialog Dialog) {
				// Expected dialog:
				// [0] User: "What time is it?"
				// [1] Assistant: Tool call to get_time
				// [2] Assistant: Tool result "12:34 PM"
				// [3] Assistant: "The current time is 12:34 PM"

				if len(dialog) != 4 {
					t.Errorf("Expected 4 messages in dialog, got %d", len(dialog))
					return
				}

				// Check tool call message
				toolCall := dialog[1]
				if toolCall.Role != Assistant || len(toolCall.Blocks) != 1 || toolCall.Blocks[0].BlockType != ToolCall {
					t.Errorf("Expected tool call message")
				}

				// Check tool result message
				toolResult := dialog[2]
				if toolResult.Role != ToolResult || len(toolResult.Blocks) != 1 {
					t.Errorf("Expected tool result message")
				}
				if toolResult.Blocks[0].Content != "12:34 PM" {
					t.Errorf("Expected tool result '12:34 PM', got %q", toolResult.Blocks[0].Content)
				}

				// Check final response
				final := dialog[3]
				if final.Role != Assistant || len(final.Blocks) != 1 {
					t.Errorf("Expected final message")
				}
				if final.Blocks[0].Content != "The current time is 12:34 PM" {
					t.Errorf("Unexpected final message content: %s", final.Blocks[0].Content)
				}
			},
		},
		{
			name:   "tool call with failed callback",
			dialog: Dialog{Message{Role: User, Blocks: []Block{{Content: "What's the weather?"}}}},
			options: &GenOpts{
				ToolChoice: ToolChoiceAuto,
			},
			setupTools: func(tg *ToolGenerator) {
				tg.Register(Tool{
					Name:        "get_weather",
					Description: "Get the current weather",
					InputSchema: InputSchema{
						Type: Object,
						Properties: map[string]Property{
							"location": {Type: String},
						},
					},
				}, &mockToolCallback{
					callFunc: func(ctx context.Context, input map[string]any) (any, error) {
						return nil, errors.New("API connection failed")
					},
				})
			},
			setupMockGen: func(m *mockToolCapableGenerator) {
				m.generateFunc = func(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
					return Response{
						Candidates: []Message{
							{
								Role: Assistant,
								Blocks: []Block{
									{
										ID:           "weather_call_1",
										BlockType:    ToolCall,
										ModalityType: Text,
										Content:      `{"name":"get_weather","parameters":{"location":"New York"}}`,
									},
								},
							},
						},
						FinishReason: ToolUse,
					}, nil
				}
			},
			wantErr:        true,
			expectedErrMsg: "API connection failed",
			validateDialog: func(t *testing.T, dialog Dialog) {
				// Should have original message and tool call message
				if len(dialog) != 2 {
					t.Errorf("Expected 2 messages in dialog (original + tool call), got %d", len(dialog))
				}
			},
		},
		{
			name:   "multiple parallel tool calls",
			dialog: Dialog{Message{Role: User, Blocks: []Block{{Content: "Compare weather in NYC and LA"}}}},
			options: &GenOpts{
				ToolChoice: ToolChoiceToolsRequired,
			},
			setupTools: func(tg *ToolGenerator) {
				tg.Register(Tool{
					Name:        "get_weather",
					Description: "Get weather by location",
					InputSchema: InputSchema{
						Type: Object,
						Properties: map[string]Property{
							"location": {Type: String},
						},
					},
				}, &mockToolCallback{
					callFunc: func(ctx context.Context, input map[string]any) (any, error) {
						location := input["location"].(string)
						if location == "New York" {
							return "72°F and sunny", nil
						}
						return "68°F and cloudy", nil
					},
				})
			},
			setupMockGen: func(m *mockToolCapableGenerator) {
				firstCallExecuted := false
				m.generateFunc = func(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
					if !firstCallExecuted {
						firstCallExecuted = true
						return Response{
							Candidates: []Message{
								{
									Role: Assistant,
									Blocks: []Block{
										{
											ID:           "weather_call_1",
											BlockType:    ToolCall,
											ModalityType: Text,
											Content:      `{"name":"get_weather","parameters":{"location":"New York"}}`,
										},
										{
											ID:           "weather_call_2",
											BlockType:    ToolCall,
											ModalityType: Text,
											Content:      `{"name":"get_weather","parameters":{"location":"Los Angeles"}}`,
										},
									},
								},
							},
							FinishReason: ToolUse,
						}, nil
					} else {
						return Response{
							Candidates: []Message{
								{
									Role: Assistant,
									Blocks: []Block{
										{
											BlockType:    Content,
											ModalityType: Text,
											Content:      "New York is 72°F and sunny, while Los Angeles is 68°F and cloudy",
										},
									},
								},
							},
							FinishReason: EndTurn,
						}, nil
					}
				}
			},
			wantErr: false,
			validateDialog: func(t *testing.T, dialog Dialog) {
				// Expected dialog:
				// [0] User: "Compare weather in NYC and LA"
				// [1] Assistant: Tool calls to get_weather (2 calls)
				// [2] Assistant: Tool results (2 results)
				// [3] Assistant: Final comparison

				if len(dialog) != 4 {
					t.Errorf("Expected 4 messages in dialog, got %d", len(dialog))
					return
				}

				// Check tool call message has both calls
				toolCalls := dialog[1]
				if toolCalls.Role != Assistant || len(toolCalls.Blocks) != 2 {
					t.Errorf("Expected tool call message with 2 calls")
				}

				// Check tool result message has both results
				toolResults := dialog[2]
				if toolResults.Role != ToolResult || len(toolResults.Blocks) != 2 {
					t.Errorf("Expected tool result message with 2 results")
				}

				// Verify the results
				var foundNY, foundLA bool
				for _, block := range toolResults.Blocks {
					if block.ID == "weather_call_1" && block.Content == "72°F and sunny" {
						foundNY = true
					}
					if block.ID == "weather_call_2" && block.Content == "68°F and cloudy" {
						foundLA = true
					}
				}
				if !foundNY || !foundLA {
					t.Errorf("Missing weather results. Found NY: %v, Found LA: %v", foundNY, foundLA)
				}

				// Check final response
				final := dialog[3]
				if final.Role != Assistant || len(final.Blocks) != 1 {
					t.Errorf("Expected final message with comparison")
				}
				if final.Blocks[0].Content != "New York is 72°F and sunny, while Los Angeles is 68°F and cloudy" {
					t.Errorf("Unexpected final message content: %s", final.Blocks[0].Content)
				}
			},
		},
		{
			name:   "sequential tool calls",
			dialog: Dialog{Message{Role: User, Blocks: []Block{{Content: "What's the weather where I am?"}}}},
			options: &GenOpts{
				ToolChoice: ToolChoiceAuto,
			},
			setupTools: func(tg *ToolGenerator) {
				// Register a location tool
				tg.Register(Tool{
					Name:        "get_location",
					Description: "Get the user's current location",
				}, &mockToolCallback{
					callFunc: func(ctx context.Context, input map[string]any) (any, error) {
						return "New York", nil
					},
				})

				// Register a weather tool
				tg.Register(Tool{
					Name:        "get_weather",
					Description: "Get weather for a location",
					InputSchema: InputSchema{
						Type: Object,
						Properties: map[string]Property{
							"location": {Type: String},
						},
					},
				}, &mockToolCallback{
					callFunc: func(ctx context.Context, input map[string]any) (any, error) {
						location := input["location"].(string)
						if location == "New York" {
							return "72°F and sunny", nil
						}
						return "Unknown location", nil
					},
				})
			},
			setupMockGen: func(m *mockToolCapableGenerator) {
				callCount := 0
				m.generateFunc = func(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
					callCount++

					switch callCount {
					case 1:
						// First call: Ask for location
						return Response{
							Candidates: []Message{
								{
									Role: Assistant,
									Blocks: []Block{
										{
											ID:           "location_call",
											BlockType:    ToolCall,
											ModalityType: Text,
											Content:      `{"name":"get_location","parameters":{}}`,
										},
									},
								},
							},
							FinishReason: ToolUse,
						}, nil

					case 2:
						// Second call: Get weather for the location
						return Response{
							Candidates: []Message{
								{
									Role: Assistant,
									Blocks: []Block{
										{
											ID:           "weather_call",
											BlockType:    ToolCall,
											ModalityType: Text,
											Content:      `{"name":"get_weather","parameters":{"location":"New York"}}`,
										},
									},
								},
							},
							FinishReason: ToolUse,
						}, nil

					case 3:
						// Final response with the answer
						return Response{
							Candidates: []Message{
								{
									Role: Assistant,
									Blocks: []Block{
										{
											BlockType:    Content,
											ModalityType: Text,
											Content:      "The weather in New York is 72°F and sunny.",
										},
									},
								},
							},
							FinishReason: EndTurn,
						}, nil

					default:
						t.Errorf("Unexpected call count: %d", callCount)
						return Response{}, fmt.Errorf("unexpected call count: %d", callCount)
					}
				}
			},
			wantErr: false,
			validateDialog: func(t *testing.T, dialog Dialog) {
				// Expected dialog:
				// [0] User: "What's the weather where I am?"
				// [1] Assistant: Tool call to get_location
				// [2] Assistant: Tool result "New York"
				// [3] Assistant: Tool call to get_weather
				// [4] Assistant: Tool result "72°F and sunny"
				// [5] Assistant: "The weather in New York is 72°F and sunny."

				if len(dialog) != 6 {
					t.Errorf("Expected 6 messages in dialog, got %d", len(dialog))
					return
				}

				// Check location tool call and result
				locationCall := dialog[1]
				if locationCall.Role != Assistant || len(locationCall.Blocks) != 1 || locationCall.Blocks[0].BlockType != ToolCall {
					t.Errorf("Expected location tool call message")
				}

				locationResult := dialog[2]
				if locationResult.Role != ToolResult || len(locationResult.Blocks) != 1 {
					t.Errorf("Expected location tool result message")
				}
				if locationResult.Blocks[0].Content != "New York" {
					t.Errorf("Expected location result 'New York', got %q", locationResult.Blocks[0].Content)
				}

				// Check weather tool call and result
				weatherCall := dialog[3]
				if weatherCall.Role != Assistant || len(weatherCall.Blocks) != 1 || weatherCall.Blocks[0].BlockType != ToolCall {
					t.Errorf("Expected weather tool call message")
				}

				weatherResult := dialog[4]
				if weatherResult.Role != ToolResult || len(weatherResult.Blocks) != 1 {
					t.Errorf("Expected weather tool result message")
				}
				if weatherResult.Blocks[0].Content != "72°F and sunny" {
					t.Errorf("Expected weather result '72°F and sunny', got %q", weatherResult.Blocks[0].Content)
				}

				// Check final response
				final := dialog[5]
				if final.Role != Assistant || len(final.Blocks) != 1 {
					t.Errorf("Expected final message")
				}
				if final.Blocks[0].Content != "The weather in New York is 72°F and sunny." {
					t.Errorf("Unexpected final message content: %s", final.Blocks[0].Content)
				}
			},
		},
		{
			name:   "non-assistant role in response",
			dialog: Dialog{Message{Role: User, Blocks: []Block{{Content: "Hello"}}}},
			options: &GenOpts{
				ToolChoice: ToolChoiceAuto,
			},
			setupTools: func(tg *ToolGenerator) {},
			setupMockGen: func(m *mockToolCapableGenerator) {
				m.generateFunc = func(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
					return Response{
						Candidates: []Message{
							{
								Role: User, // Incorrect role!
								Blocks: []Block{
									{
										ID:           "tool_call_1",
										BlockType:    ToolCall,
										ModalityType: Text,
										Content:      `{"name":"get_time","parameters":{}}`,
									},
								},
							},
						},
						FinishReason: ToolUse,
					}, nil
				}
			},
			wantErr:        true,
			expectedErrMsg: "expected assistant role in response",
			validateDialog: func(t *testing.T, dialog Dialog) {
				// Should only have the original message
				if len(dialog) != 1 {
					t.Errorf("Expected only original message in dialog, got %d messages", len(dialog))
				}
			},
		},
		{
			name:   "multiple candidates in response",
			dialog: Dialog{Message{Role: User, Blocks: []Block{{Content: "Hello"}}}},
			options: &GenOpts{
				ToolChoice: ToolChoiceAuto,
			},
			setupTools: func(tg *ToolGenerator) {},
			setupMockGen: func(m *mockToolCapableGenerator) {
				m.generateFunc = func(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
					return Response{
						Candidates: []Message{
							{
								Role: Assistant,
								Blocks: []Block{
									{
										ID:           "tool_call_1",
										BlockType:    ToolCall,
										ModalityType: Text,
										Content:      `{"name":"get_time","parameters":{}}`,
									},
								},
							},
							{
								Role: Assistant,
								Blocks: []Block{
									{
										ID:           "tool_call_2",
										BlockType:    ToolCall,
										ModalityType: Text,
										Content:      `{"name":"get_weather","parameters":{}}`,
									},
								},
							},
						},
						FinishReason: ToolUse,
					}, nil
				}
			},
			wantErr:        true,
			expectedErrMsg: "expected exactly one candidate in response",
			validateDialog: func(t *testing.T, dialog Dialog) {
				// Should only have the original message
				if len(dialog) != 1 {
					t.Errorf("Expected only original message in dialog, got %d messages", len(dialog))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockGen := newMockToolCapableGenerator()
			tt.setupMockGen(mockGen)

			toolGen := &ToolGenerator{
				G:             mockGen,
				toolCallbacks: make(map[string]ToolCallback),
			}
			tt.setupTools(toolGen)

			dialog, err := toolGen.Generate(context.Background(), tt.dialog, tt.options)

			if (err != nil) != tt.wantErr {
				t.Errorf("Generate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil && tt.expectedErrMsg != "" {
				if !strings.Contains(err.Error(), tt.expectedErrMsg) {
					t.Errorf("Generate() expected error to contain %q, got %q", tt.expectedErrMsg, err.Error())
				}
			}

			if err == nil && tt.validateDialog != nil {
				tt.validateDialog(t, dialog)
			}
		})
	}
}
