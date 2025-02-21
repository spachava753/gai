package openai

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	oai "github.com/openai/openai-go"
	"github.com/spachava753/gai"
)

func TestToOpenAIMessage(t *testing.T) {
	tests := []struct {
		name    string
		msg     gai.Message
		want    oai.ChatCompletionMessageParamUnion
		wantErr bool
	}{
		{
			name: "error: empty blocks",
			msg: gai.Message{
				Role:   gai.User,
				Blocks: []gai.Block{},
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "error: nil blocks",
			msg: gai.Message{
				Role:   gai.User,
				Blocks: nil,
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "user message",
			msg: gai.Message{
				Role: gai.User,
				Blocks: []gai.Block{
					{
						BlockType:    gai.Unstructured,
						ModalityType: gai.Text,
						Content:      "Hello, how are you?",
					},
				},
			},
			want:    oai.UserMessage("Hello, how are you?"),
			wantErr: false,
		},
		{
			name: "assistant message",
			msg: gai.Message{
				Role: gai.Assistant,
				Blocks: []gai.Block{
					{
						BlockType:    gai.Unstructured,
						ModalityType: gai.Text,
						Content:      "I'm doing well, thank you!",
					},
				},
			},
			want:    oai.AssistantMessage("I'm doing well, thank you!"),
			wantErr: false,
		},
		{
			name: "tool call",
			msg: gai.Message{
				Role: gai.Assistant,
				Blocks: []gai.Block{
					{
						ID:           "call_123",
						BlockType:    gai.ToolCall,
						ModalityType: gai.Text,
						Content:      `{"name": "get_weather", "arguments": {"location": "London"}}`,
					},
				},
			},
			want: oai.ChatCompletionAssistantMessageParam{
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
			name: "tool result",
			msg: gai.Message{
				Role: gai.Assistant,
				Blocks: []gai.Block{
					{
						ID:           "call_123",
						BlockType:    gai.ToolResult,
						ModalityType: gai.Text,
						Content:      "The current temperature is 72°F",
					},
				},
			},
			want:    oai.ToolMessage("call_123", "The current temperature is 72°F"),
			wantErr: false,
		},
		{
			name: "tool call with text",
			msg: gai.Message{
				Role: gai.Assistant,
				Blocks: []gai.Block{
					{
						BlockType:    gai.Unstructured,
						ModalityType: gai.Text,
						Content:      `Let me get the weather for you:`,
					},
					{
						ID:           "call_123",
						BlockType:    gai.ToolCall,
						ModalityType: gai.Text,
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
			msg: gai.Message{
				Role: gai.User,
				Blocks: []gai.Block{
					{
						BlockType:    gai.Unstructured,
						ModalityType: gai.Video,
						Media:        nil,
					},
				},
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "error: invalid role",
			msg: gai.Message{
				Role: 999,
				Blocks: []gai.Block{
					{
						BlockType:    gai.Unstructured,
						ModalityType: gai.Text,
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
