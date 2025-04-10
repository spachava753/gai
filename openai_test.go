package gai

import (
	"testing"

	"github.com/google/go-cmp/cmp"
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
			want:    nil,
			wantErr: true,
		},
		{
			name: "error: nil blocks",
			msg: Message{
				Role:   User,
				Blocks: nil,
			},
			want:    nil,
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
			want: oai.ChatCompletionMessageParam{
				Role:    oai.F(oai.ChatCompletionMessageParamRoleUser),
				Content: oai.F[interface{}]("Hello, how are you?"),
			},
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
			want: oai.ChatCompletionAssistantMessageParam{
				Role: oai.F(oai.ChatCompletionAssistantMessageParamRoleAssistant),
				ToolCalls: oai.F([]oai.ChatCompletionMessageToolCallParam{
					{
						ID: oai.F("call_123"),
						Function: oai.F(oai.ChatCompletionMessageToolCallFunctionParam{
							Name:      oai.F("get_weather"),
							Arguments: oai.F(`{"location":"London"}`),
						}),
						Type: oai.F(oai.ChatCompletionMessageToolCallTypeFunction),
					},
				}),
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
			want: oai.ChatCompletionMessageParam{
				Role:       oai.F(oai.ChatCompletionMessageParamRoleTool),
				ToolCallID: oai.F("call_123"),
				Content:    oai.F[interface{}]("The current temperature is 72°F"),
			},
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
			want: oai.ChatCompletionAssistantMessageParam{
				Role: oai.F(oai.ChatCompletionAssistantMessageParamRoleAssistant),
				Content: oai.F([]oai.ChatCompletionAssistantMessageParamContentUnion{
					oai.TextPart(`Let me get the weather for you:`),
				}),
				ToolCalls: oai.F([]oai.ChatCompletionMessageToolCallParam{
					{
						ID: oai.F("call_123"),
						Function: oai.F(oai.ChatCompletionMessageToolCallFunctionParam{
							Name:      oai.F("get_weather"),
							Arguments: oai.F(`{"location":"London"}`),
						}),
						Type: oai.F(oai.ChatCompletionMessageToolCallTypeFunction),
					},
				}),
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
			want:    nil,
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
			want:    nil,
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
			want: oai.UserMessageParts(
				oai.ChatCompletionContentPartTextParam{
					Type: oai.F(oai.ChatCompletionContentPartTextTypeText),
					Text: oai.F("What's in this image?"),
				},
				oai.ChatCompletionContentPartImageParam{
					Type: oai.F(oai.ChatCompletionContentPartImageTypeImageURL),
					ImageURL: oai.F(oai.ChatCompletionContentPartImageImageURLParam{
						URL: oai.F("data:image/jpeg;base64,fake-image-base64-data"),
					}),
				},
			),
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
			want: oai.UserMessageParts(
				oai.ChatCompletionContentPartTextParam{
					Type: oai.F(oai.ChatCompletionContentPartTextTypeText),
					Text: oai.F("What's in this audio?"),
				},
				oai.ChatCompletionContentPartInputAudioParam{
					Type: oai.F(oai.ChatCompletionContentPartInputAudioTypeInputAudio),
					InputAudio: oai.F(oai.ChatCompletionContentPartInputAudioInputAudioParam{
						Data:   oai.F("fake-audio-base64-data"),
						Format: oai.F(oai.ChatCompletionContentPartInputAudioInputAudioFormatWAV),
					}),
				},
			),
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
			want: oai.ChatCompletionAssistantMessageParam{
				Role: oai.F(oai.ChatCompletionAssistantMessageParamRoleAssistant),
				Audio: oai.F(oai.ChatCompletionAssistantMessageParamAudio{
					ID: oai.F("audio_abc123"),
				}),
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
			want: oai.ChatCompletionAssistantMessageParam{
				Role: oai.F(oai.ChatCompletionAssistantMessageParamRoleAssistant),
				Content: oai.F([]oai.ChatCompletionAssistantMessageParamContentUnion{
					oai.TextPart("Here's my response:"),
				}),
				Audio: oai.F(oai.ChatCompletionAssistantMessageParamAudio{
					ID: oai.F("audio_abc123"),
				}),
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
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("toOpenAIMessage() mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}
