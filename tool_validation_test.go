package gai

import (
	"strings"
	"testing"
)

func TestValidateToolResultMessage(t *testing.T) {
	tests := []struct {
		name       string
		message    Message
		toolCallID string
		wantErr    bool
		errSubstr  string
	}{
		{
			name: "valid text message",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    Content,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str("Test content"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    false,
		},
		{
			name: "valid image message",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "img-id",
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "image/jpeg",
						Content:      Str("base64-data"),
					},
				},
			},
			toolCallID: "img-id",
			wantErr:    false,
		},
		{
			name: "valid audio message",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "audio-id",
						BlockType:    Content,
						ModalityType: Audio,
						MimeType:     "audio/mp3",
						Content:      Str("base64-audio-data"),
					},
				},
			},
			toolCallID: "audio-id",
			wantErr:    false,
		},
		{
			name: "valid video message",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "video-id",
						BlockType:    Content,
						ModalityType: Video,
						MimeType:     "video/mp4",
						Content:      Str("base64-video-data"),
					},
				},
			},
			toolCallID: "video-id",
			wantErr:    false,
		},
		{
			name: "valid PDF message",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "pdf-id",
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "application/pdf",
						Content:      Str("base64-pdf-data"),
					},
				},
			},
			toolCallID: "pdf-id",
			wantErr:    false,
		},
		{
			name: "valid multi-block message",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "multi-id",
						BlockType:    Content,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str("Description text"),
					},
					{
						ID:           "multi-id",
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "image/png",
						Content:      Str("base64-image-data"),
					},
				},
			},
			toolCallID: "multi-id",
			wantErr:    false,
		},
		{
			name: "wrong role",
			message: Message{
				Role: User, // Wrong role
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    Content,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str("Test content"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "must have ToolResult role",
		},
		{
			name: "empty blocks",
			message: Message{
				Role:   ToolResult,
				Blocks: []Block{}, // Empty blocks
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "must have at least one block",
		},
		{
			name: "wrong block ID",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "wrong-id", // Wrong ID
						BlockType:    Content,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str("Test content"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "incorrect ID",
		},
		{
			name: "nil content",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    Content,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      nil, // Nil content
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "nil content",
		},
		{
			name: "empty block type",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    "", // Empty block type
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str("Test content"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "missing block type",
		},
		{
			name: "empty MIME type",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    Content,
						ModalityType: Text,
						MimeType:     "", // Empty MIME type
						Content:      Str("Test content"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "missing MIME type",
		},
		{
			name: "invalid modality",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    Content,
						ModalityType: Modality(99), // Invalid modality
						MimeType:     "text/plain",
						Content:      Str("Test content"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "invalid modality type",
		},
		{
			name: "mismatched MIME type for text",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    Content,
						ModalityType: Text,
						MimeType:     "image/jpeg", // Wrong MIME type for text
						Content:      Str("Test content"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "text modality but non-text MIME type",
		},
		{
			name: "mismatched MIME type for image",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "text/plain", // Wrong MIME type for image
						Content:      Str("base64-data"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "image modality but non-image MIME type",
		},
		{
			name: "mismatched MIME type for audio",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    Content,
						ModalityType: Audio,
						MimeType:     "video/mp4", // Wrong MIME type for audio
						Content:      Str("base64-audio-data"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "audio modality but non-audio MIME type",
		},
		{
			name: "mismatched MIME type for video",
			message: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "test-id",
						BlockType:    Content,
						ModalityType: Video,
						MimeType:     "audio/mp3", // Wrong MIME type for video
						Content:      Str("base64-video-data"),
					},
				},
			},
			toolCallID: "test-id",
			wantErr:    true,
			errSubstr:  "video modality but non-video MIME type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateToolResultMessage(&tt.message, tt.toolCallID)

			if (err != nil) != tt.wantErr {
				t.Errorf("validateToolResultMessage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil && tt.errSubstr != "" && !strings.Contains(err.Error(), tt.errSubstr) {
				t.Errorf("validateToolResultMessage() error = %v, expected to contain %q", err, tt.errSubstr)
			}
		})
	}
}
