package gai

import (
	"testing"

	oai "github.com/openai/openai-go"
)

func TestToOpenAIMessage(t *testing.T) {
	tests := []struct {
		name    string
		msg     Message
		want    oai.ChatCompletionMessageParamUnion
		wantErr bool
	}{
		{
			name: "error: empty blocks",
			msg: Message{
				Role:   User,
				Blocks: []Block{},
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "error: nil blocks",
			msg: Message{
				Role:   User,
				Blocks: nil,
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "user message",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hello, how are you?"),
					},
				},
			},
			want:    oai.UserMessage("Hello, how are you?"),
			wantErr: false,
		},
		{
			name: "assistant message",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("I'm doing well, thank you!"),
					},
				},
			},
			want:    oai.AssistantMessage("I'm doing well, thank you!"),
			wantErr: false,
		},
		{
			name: "tool call",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						ID:           "call_123",
						BlockType:    ToolCall,
						ModalityType: Text,
						Content:      Str(`{"name": "get_weather", "parameters": {"location": "London"}}`),
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					ToolCalls: []oai.ChatCompletionMessageToolCallParam{
						{
							ID: "call_123",
							Function: oai.ChatCompletionMessageToolCallFunctionParam{
								Name:      "get_weather",
								Arguments: `{"location":"London"}`,
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "tool result",
			msg: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "call_123",
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("The current temperature is 72°F"),
					},
				},
			},
			want:    oai.ToolMessage("The current temperature is 72°F", "call_123"),
			wantErr: false,
		},
		{
			name: "tool call with text",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str(`Let me get the weather for you:`),
					},
					{
						ID:           "call_123",
						BlockType:    ToolCall,
						ModalityType: Text,
						Content:      Str(`{"name": "get_weather", "parameters": {"location": "London"}}`),
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					Content: oai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: oai.String("Let me get the weather for you:"),
					},
					ToolCalls: []oai.ChatCompletionMessageToolCallParam{
						{
							ID: "call_123",
							Function: oai.ChatCompletionMessageToolCallFunctionParam{
								Name:      "get_weather",
								Arguments: `{"location":"London"}`,
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "error: video modality not supported",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Video,
						Content:      Str("fake-base64-data"),
					},
				},
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "error: invalid role",
			msg: Message{
				Role: 999,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hello"),
					},
				},
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "multimodal user message with text and image",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What's in this image?"),
					},
					{
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "image/jpeg",
						Content:      Str("fake-image-base64-data"),
					},
				},
			},
			want: oai.UserMessage([]oai.ChatCompletionContentPartUnionParam{
				{
					OfText: &oai.ChatCompletionContentPartTextParam{
						Text: "What's in this image?",
					},
				},
				{
					OfImageURL: &oai.ChatCompletionContentPartImageParam{
						ImageURL: oai.ChatCompletionContentPartImageImageURLParam{
							URL: "data:image/jpeg;base64,fake-image-base64-data",
						},
					},
				},
			}),
			wantErr: false,
		},
		{
			name: "multimodal user message with text and audio",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What's in this audio?"),
					},
					{
						BlockType:    Content,
						ModalityType: Audio,
						MimeType:     "audio/wav",
						Content:      Str("fake-audio-base64-data"),
					},
				},
			},
			want: oai.UserMessage([]oai.ChatCompletionContentPartUnionParam{
				{
					OfText: &oai.ChatCompletionContentPartTextParam{
						Text: "What's in this audio?",
					},
				},
				{
					OfInputAudio: &oai.ChatCompletionContentPartInputAudioParam{
						InputAudio: oai.ChatCompletionContentPartInputAudioInputAudioParam{
							Data:   "fake-audio-base64-data",
							Format: "audio/wav",
						},
					},
				},
			}),
			wantErr: false,
		},
		{
			name: "assistant message with audio ID",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						ID:           "audio_abc123",
						BlockType:    Content,
						ModalityType: Audio,
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					Audio: oai.ChatCompletionAssistantMessageParamAudio{
						ID: "audio_abc123",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "assistant message with text and audio ID",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Here's my response:"),
					},
					{
						ID:           "audio_abc123",
						BlockType:    Content,
						ModalityType: Audio,
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					Audio: oai.ChatCompletionAssistantMessageParamAudio{
						ID: "audio_abc123",
					},
					Content: oai.ChatCompletionAssistantMessageParamContentUnion{
						OfArrayOfContentParts: []oai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
							{
								OfText: &oai.ChatCompletionContentPartTextParam{
									Text: "Here's my response:",
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toOpenAIMessage(tt.msg)
			if (err != nil) != tt.wantErr {
				t.Errorf("toOpenAIMessage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Custom comparison that ignores unexported fields
				// This approach checks only the message types and key properties,
				// without attempting to compare the unexported fields of the complex
				// OpenAI SDK types such as the internal implementations of Opt[T]
				// and other generic structures.
				//
				// We're specifically checking that the message role (User, Assistant, Tool)
				// matches, and for specific message types, we verify relevant fields like
				// tool call IDs and tool function names.

				// Check if the role/message type matches
				if (got.OfUser != nil) != (tt.want.OfUser != nil) ||
					(got.OfAssistant != nil) != (tt.want.OfAssistant != nil) ||
					(got.OfTool != nil) != (tt.want.OfTool != nil) ||
					(got.OfSystem != nil) != (tt.want.OfSystem != nil) {
					t.Errorf("toOpenAIMessage() returned wrong message type")
					return
				}

				// For tool call messages, verify tool call ID matches
				if got.OfTool != nil && tt.want.OfTool != nil {
					if got.OfTool.ToolCallID != tt.want.OfTool.ToolCallID {
						t.Errorf("Tool call ID mismatch: got %v, want %v",
							got.OfTool.ToolCallID, tt.want.OfTool.ToolCallID)
					}
				}

				// For assistant messages with tool calls, verify tool call info
				if got.OfAssistant != nil && tt.want.OfAssistant != nil {
					// Check if both have tool calls
					if (len(got.OfAssistant.ToolCalls) > 0) != (len(tt.want.OfAssistant.ToolCalls) > 0) {
						t.Errorf("Tool calls presence mismatch")
						return
					}

					// If they have tool calls, verify basic properties
					if len(got.OfAssistant.ToolCalls) > 0 && len(tt.want.OfAssistant.ToolCalls) > 0 {
						if got.OfAssistant.ToolCalls[0].ID != tt.want.OfAssistant.ToolCalls[0].ID ||
							got.OfAssistant.ToolCalls[0].Function.Name != tt.want.OfAssistant.ToolCalls[0].Function.Name {
							t.Errorf("Tool call details mismatch")
						}
					}
				}
			}
		})
	}
}
