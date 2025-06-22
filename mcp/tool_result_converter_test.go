package mcp

import (
	"testing"

	"github.com/spachava753/gai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvertCallToolResultToGAIMessage(t *testing.T) {
	tests := []struct {
		name           string
		input          map[string]interface{}
		expectedRole   gai.Role
		expectedError  bool
		expectedBlocks int
		validateFunc   func(t *testing.T, msg gai.Message)
	}{
		{
			name: "text content",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "Current weather in New York:\nTemperature: 72°F\nConditions: Partly cloudy",
					},
				},
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 1,
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.False(t, msg.ToolResultError)
				assert.Equal(t, gai.Text, msg.Blocks[0].ModalityType)
				assert.Equal(t, "text/plain", msg.Blocks[0].MimeType)
				assert.Equal(t, "Current weather in New York:\nTemperature: 72°F\nConditions: Partly cloudy", msg.Blocks[0].Content.String())
			},
		},
		{
			name: "multiple text content blocks",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "First block",
					},
					map[string]interface{}{
						"type": "text",
						"text": "Second block",
					},
				},
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 2,
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.False(t, msg.ToolResultError)
				assert.Equal(t, "First block", msg.Blocks[0].Content.String())
				assert.Equal(t, "Second block", msg.Blocks[1].Content.String())
			},
		},
		{
			name: "image content",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type":     "image",
						"data":     "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
						"mimeType": "image/png",
					},
				},
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 1,
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.False(t, msg.ToolResultError)
				assert.Equal(t, gai.Image, msg.Blocks[0].ModalityType)
				assert.Equal(t, "image/png", msg.Blocks[0].MimeType)
				assert.Equal(t, "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", msg.Blocks[0].Content.String())
			},
		},
		{
			name: "image content without mime type - should error",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type": "image",
						"data": "base64imagedata",
					},
				},
				"isError": false,
			},
			expectedRole:  gai.ToolResult,
			expectedError: true,
			validateFunc:  nil,
		},
		{
			name: "audio content",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type":     "audio",
						"data":     "UklGRjQDAABXQVZFZm10IBAAAAABAAEA",
						"mimeType": "audio/wav",
					},
				},
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 1,
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.Equal(t, gai.Audio, msg.Blocks[0].ModalityType)
				assert.Equal(t, "audio/wav", msg.Blocks[0].MimeType)
				assert.Equal(t, "UklGRjQDAABXQVZFZm10IBAAAAABAAEA", msg.Blocks[0].Content.String())
			},
		},
		{
			name: "audio content without mime type - should error",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type": "audio",
						"data": "audiodata",
					},
				},
				"isError": false,
			},
			expectedRole:  gai.ToolResult,
			expectedError: true,
			validateFunc:  nil,
		},
		{
			name: "resource with text content",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type": "resource",
						"resource": map[string]interface{}{
							"uri":      "file:///example.txt",
							"text":     "File content here",
							"mimeType": "text/plain",
						},
					},
				},
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 1,
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.Equal(t, gai.Text, msg.Blocks[0].ModalityType)
				assert.Equal(t, "text/plain", msg.Blocks[0].MimeType)
				assert.Equal(t, "File content here", msg.Blocks[0].Content.String())
				assert.Equal(t, "file:///example.txt", msg.Blocks[0].ExtraFields["resource_uri"])
			},
		},
		{
			name: "resource with blob content",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type": "resource",
						"resource": map[string]interface{}{
							"uri":      "file:///example.pdf",
							"blob":     "JVBERi0xLjQK",
							"mimeType": "application/pdf",
						},
					},
				},
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 1,
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.Equal(t, gai.Text, msg.Blocks[0].ModalityType) // Resources are converted to text
				assert.Equal(t, "application/pdf", msg.Blocks[0].MimeType)
				assert.Equal(t, "JVBERi0xLjQK", msg.Blocks[0].Content.String())
				assert.Equal(t, "file:///example.pdf", msg.Blocks[0].ExtraFields["resource_uri"])
			},
		},
		{
			name: "resource with only URI",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type": "resource",
						"resource": map[string]interface{}{
							"uri": "https://example.com/resource",
						},
					},
				},
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 1,
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.Equal(t, gai.Text, msg.Blocks[0].ModalityType)
				assert.Equal(t, "text/plain", msg.Blocks[0].MimeType)
				assert.Equal(t, "Resource: https://example.com/resource", msg.Blocks[0].Content.String())
				assert.Equal(t, "https://example.com/resource", msg.Blocks[0].ExtraFields["resource_uri"])
			},
		},
		{
			name: "mixed content types",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "Here's the result:",
					},
					map[string]interface{}{
						"type":     "image",
						"data":     "base64imagedata",
						"mimeType": "image/jpeg",
					},
				},
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 2,
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.Equal(t, gai.Text, msg.Blocks[0].ModalityType)
				assert.Equal(t, "Here's the result:", msg.Blocks[0].Content.String())
				assert.Equal(t, gai.Image, msg.Blocks[1].ModalityType)
				assert.Equal(t, "image/jpeg", msg.Blocks[1].MimeType)
			},
		},
		{
			name: "error result",
			input: map[string]interface{}{
				"content": []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "An error occurred while processing the request",
					},
				},
				"isError": true,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 1,
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.True(t, msg.ToolResultError)
				assert.Equal(t, "An error occurred while processing the request", msg.Blocks[0].Content.String())
			},
		},
		{
			name: "empty content array",
			input: map[string]interface{}{
				"content": []interface{}{},
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 1, // Should create a default empty text block
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.Equal(t, gai.Text, msg.Blocks[0].ModalityType)
				assert.Equal(t, "", msg.Blocks[0].Content.String())
			},
		},
		{
			name: "missing content field",
			input: map[string]interface{}{
				"isError": false,
			},
			expectedRole:   gai.ToolResult,
			expectedError:  false,
			expectedBlocks: 1, // Should create a default empty text block
			validateFunc: func(t *testing.T, msg gai.Message) {
				assert.Equal(t, gai.Text, msg.Blocks[0].ModalityType)
				assert.Equal(t, "", msg.Blocks[0].Content.String())
			},
		},
		{
			name: "malformed result - invalid JSON structure",
			input: map[string]interface{}{
				"content": "not an array",
				"isError": false,
			},
			expectedRole:  gai.ToolResult,
			expectedError: true,
			validateFunc:  nil,
		},
		{
			name: "result with invalid content block structure",
			input: map[string]interface{}{
				"content": []interface{}{
					"not an object",
				},
				"isError": false,
			},
			expectedRole:  gai.ToolResult,
			expectedError: true,
			validateFunc:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertCallToolResultToGAIMessage(tt.input)

			if tt.expectedError {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.expectedRole, result.Role)
			assert.Len(t, result.Blocks, tt.expectedBlocks)

			if tt.validateFunc != nil {
				tt.validateFunc(t, result)
			}
		})
	}
}

func TestConvertContentBlockToGAIBlock(t *testing.T) {
	tests := []struct {
		name          string
		contentBlock  ContentBlock
		index         int
		expectedError bool
		validateFunc  func(t *testing.T, block gai.Block)
	}{
		{
			name: "text block",
			contentBlock: ContentBlock{
				Type: "text",
				Text: "Hello, world!",
			},
			expectedError: false,
			validateFunc: func(t *testing.T, block gai.Block) {
				assert.Equal(t, gai.Text, block.ModalityType)
				assert.Equal(t, "text/plain", block.MimeType)
				assert.Equal(t, "Hello, world!", block.Content.String())
			},
		},
		{
			name: "image block with mime type",
			contentBlock: ContentBlock{
				Type:     "image",
				Data:     "base64data",
				MimeType: "image/png",
			},
			expectedError: false,
			validateFunc: func(t *testing.T, block gai.Block) {
				assert.Equal(t, gai.Image, block.ModalityType)
				assert.Equal(t, "image/png", block.MimeType)
				assert.Equal(t, "base64data", block.Content.String())
			},
		},
		{
			name: "audio block with explicit mime type",
			contentBlock: ContentBlock{
				Type:     "audio",
				Data:     "audiodata",
				MimeType: "audio/wav",
			},
			expectedError: false,
			validateFunc: func(t *testing.T, block gai.Block) {
				assert.Equal(t, gai.Audio, block.ModalityType)
				assert.Equal(t, "audio/wav", block.MimeType)
				assert.Equal(t, "audiodata", block.Content.String())
			},
		},
		{
			name: "image block without mime type",
			contentBlock: ContentBlock{
				Type: "image",
				Data: "imagedata",
			},
			expectedError: true, // Should error because mimeType is missing
			validateFunc:  nil,
		},
		{
			name: "audio block without mime type",
			contentBlock: ContentBlock{
				Type: "audio",
				Data: "audiodata",
			},
			expectedError: true, // Should error because mimeType is missing
			validateFunc:  nil,
		},
		{
			name: "resource with text",
			contentBlock: ContentBlock{
				Type: "resource",
				Resource: struct {
					URI      string `json:"uri,omitempty"`
					Text     string `json:"text,omitempty"`
					MimeType string `json:"mimeType,omitempty"`
					Blob     string `json:"blob,omitempty"`
				}{
					URI:      "file:///test.txt",
					Text:     "file content",
					MimeType: "text/plain",
				},
			},
			expectedError: false,
			validateFunc: func(t *testing.T, block gai.Block) {
				assert.Equal(t, gai.Text, block.ModalityType)
				assert.Equal(t, "text/plain", block.MimeType)
				assert.Equal(t, "file content", block.Content.String())
				assert.Equal(t, "file:///test.txt", block.ExtraFields["resource_uri"])
			},
		},
		{
			name: "unsupported type",
			contentBlock: ContentBlock{
				Type: "unknown",
				Text: "content",
			},
			expectedError: true,
			validateFunc:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertContentBlockToGAIBlock(tt.contentBlock, tt.index)

			if tt.expectedError {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, gai.Content, result.BlockType)

			if tt.validateFunc != nil {
				tt.validateFunc(t, result)
			}
		})
	}
}
