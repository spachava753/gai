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
						Content:      "Hello, how are you?",
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
						Content:      "I'm doing well, thank you!",
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
						Content:      `{"name": "get_weather", "parameters": {"location": "London"}}`,
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
							Arguments: oai.F(`{"location": "London"}`),
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
				Role: Assistant,
				Blocks: []Block{
					{
						ID:           "call_123",
						BlockType:    ToolResult,
						ModalityType: Text,
						Content:      "The current temperature is 72°F",
					},
				},
			},
			want:    oai.ToolMessage("call_123", "The current temperature is 72°F"),
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
						Content:      `Let me get the weather for you:`,
					},
					{
						ID:           "call_123",
						BlockType:    ToolCall,
						ModalityType: Text,
						Content:      `{"name": "get_weather", "arguments": {"location": "London"}}`,
					},
				},
			},
			want: oai.ChatCompletionAssistantMessageParam{
				Content: oai.F([]oai.ChatCompletionAssistantMessageParamContentUnion{
					oai.TextPart(`Let me get the weather for you:`),
				}),
				ToolCalls: oai.F([]oai.ChatCompletionMessageToolCallParam{
					{
						ID: oai.F("call_123"),
						Function: oai.F(oai.ChatCompletionMessageToolCallFunctionParam{
							Name:      oai.F("get_weather"),
							Arguments: oai.F(`{"location": "London"}`),
						}),
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
						Media:        nil,
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
						Content:      "Hello",
					},
				},
			},
			want:    nil,
			wantErr: true,
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
