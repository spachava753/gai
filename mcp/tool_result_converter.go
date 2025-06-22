package mcp

import (
	"encoding/json"
	"fmt"

	"github.com/spachava753/gai"
)

// CallToolResult represents the result structure returned by MCP tool calls.
// This matches the TypeScript CallToolResult interface from the MCP specification.
type CallToolResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError,omitempty"`
}

// ContentBlock represents a content block in the tool result.
// This supports the various content types defined in the MCP specification.
type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Data     string `json:"data,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
	Resource struct {
		URI      string `json:"uri,omitempty"`
		Text     string `json:"text,omitempty"`
		MimeType string `json:"mimeType,omitempty"`
		Blob     string `json:"blob,omitempty"`
	} `json:"resource,omitempty"`
}

// convertCallToolResultToGAIMessage converts an MCP tool call result into a gai.Message.
// The function handles the different content types (text, image, audio, resource)
// and creates appropriate blocks for each type.
func convertCallToolResultToGAIMessage(result map[string]interface{}) (gai.Message, error) {
	// Parse the result into our structured format
	resultBytes, err := json.Marshal(result)
	if err != nil {
		return gai.Message{}, fmt.Errorf("failed to marshal tool result: %w", err)
	}

	var callResult CallToolResult
	if err := json.Unmarshal(resultBytes, &callResult); err != nil {
		return gai.Message{}, fmt.Errorf("failed to parse tool result: %w", err)
	}

	// Create the message
	message := gai.Message{
		Role:            gai.ToolResult,
		ToolResultError: callResult.IsError,
		Blocks:          make([]gai.Block, 0, len(callResult.Content)),
	}

	// Convert each content block
	for i, contentBlock := range callResult.Content {
		block, err := convertContentBlockToGAIBlock(contentBlock, i)
		if err != nil {
			return gai.Message{}, fmt.Errorf("failed to convert content block %d: %w", i, err)
		}
		message.Blocks = append(message.Blocks, block)
	}

	// If no content blocks were provided, create a default empty text block
	if len(message.Blocks) == 0 {
		message.Blocks = append(message.Blocks, gai.Block{
			BlockType:    gai.Content,
			ModalityType: gai.Text,
			MimeType:     "text/plain",
			Content:      gai.Str(""),
		})
	}

	return message, nil
}

// convertContentBlockToGAIBlock converts a single MCP content block to a gai.Block.
func convertContentBlockToGAIBlock(contentBlock ContentBlock, index int) (gai.Block, error) {
	block := gai.Block{
		BlockType: gai.Content,
		MimeType:  "text/plain", // default
	}

	switch contentBlock.Type {
	case "text":
		block.ModalityType = gai.Text
		block.MimeType = "text/plain"
		block.Content = gai.Str(contentBlock.Text)

	case "image":
		if contentBlock.MimeType == "" {
			return gai.Block{}, fmt.Errorf("image content block missing required mimeType field")
		}
		block.ModalityType = gai.Image
		block.MimeType = contentBlock.MimeType
		block.Content = gai.Str(contentBlock.Data)

	case "audio":
		if contentBlock.MimeType == "" {
			return gai.Block{}, fmt.Errorf("audio content block missing required mimeType field")
		}
		block.ModalityType = gai.Audio
		block.MimeType = contentBlock.MimeType
		block.Content = gai.Str(contentBlock.Data)

	case "resource":
		// For resources, we convert them to text content with the resource data
		block.ModalityType = gai.Text

		if contentBlock.Resource.Text != "" {
			// Text resource
			if contentBlock.Resource.MimeType != "" {
				block.MimeType = contentBlock.Resource.MimeType
			} else {
				block.MimeType = "text/plain"
			}
			block.Content = gai.Str(contentBlock.Resource.Text)
		} else if contentBlock.Resource.Blob != "" {
			// Binary resource - we'll treat it as text with the MIME type preserved
			if contentBlock.Resource.MimeType != "" {
				block.MimeType = contentBlock.Resource.MimeType
			} else {
				block.MimeType = "application/octet-stream"
			}
			block.Content = gai.Str(contentBlock.Resource.Blob)
		} else {
			// Resource with just URI - create a text description
			block.MimeType = "text/plain"
			resourceDescription := fmt.Sprintf("Resource: %s", contentBlock.Resource.URI)
			block.Content = gai.Str(resourceDescription)
		}

		// Store resource URI in extra fields for reference
		if contentBlock.Resource.URI != "" {
			block.ExtraFields = map[string]interface{}{
				"resource_uri": contentBlock.Resource.URI,
			}
		}

	default:
		return gai.Block{}, fmt.Errorf("unsupported content type: %s", contentBlock.Type)
	}

	return block, nil
}
