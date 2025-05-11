package gai

import (
	"context"
	"encoding/base64"
	"strings"
	"testing"
)

func TestOpenAiGenerator_calculateImageTokens(t *testing.T) {
	// Create an OpenAI generator
	g := OpenAiGenerator{
		model: "gpt-4o",
	}

	// Create a test image block with dimensions in ExtraFields
	block := Block{
		BlockType:    Content,
		ModalityType: Image,
		MimeType:     "image/jpeg",
		Content:      Str("base64-content"),
		ExtraFields: map[string]interface{}{
			"width":  1024,
			"height": 1024,
		},
	}

	// Test high detail image
	tokens, err := g.calculateImageTokens(block)
	if err != nil {
		t.Errorf("Error calculating image tokens: %v", err)
	}
	expectedTokens := 765 // 85 + (170 * 4)
	if tokens != expectedTokens {
		t.Errorf("Expected %d tokens for 1024x1024 image with gpt-4o, got %d", expectedTokens, tokens)
	}

	// Test low detail image
	block.ExtraFields["detail"] = "low"
	tokens, err = g.calculateImageTokens(block)
	if err != nil {
		t.Errorf("Error calculating image tokens: %v", err)
	}
	expectedTokens = 85 // Just base tokens for low detail
	if tokens != expectedTokens {
		t.Errorf("Expected %d tokens for low detail image with gpt-4o, got %d", expectedTokens, tokens)
	}

	// Test minimal model
	g.model = "gpt-4.1-mini"
	block.ExtraFields["detail"] = "high" // Reset to high detail
	tokens, err = g.calculateImageTokens(block)
	if err != nil {
		t.Errorf("Error calculating image tokens: %v", err)
	}
	// 32x32 patches for 1024x1024 image = 32x32 = 1024 patches
	// With 1.62 multiplier for gpt-4.1-mini = 1024 * 1.62 = 1658 (rounded)
	expectedTokens = 1658
	if tokens != expectedTokens {
		t.Errorf("Expected %d tokens for 1024x1024 image with gpt-4.1-mini, got %d", expectedTokens, tokens)
	}

	// Test with actual image data
	// Create a small 2x2 pixel JPEG image in base64
	miniJpegBase64 := "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4QBaRXhpZgAASUkqAAgAAAABAGEAAAEBAAEAAAABAAIAAAECAAEAAAABAAIAAAEDAAEAAAABAAIAAAABBAABAAAAYAAAAP/bAEMAAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAf/bAEMBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAf/AABEIAAIAAQMBEQACEQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEFEiExQQYTUWEHInEUMoGRoQgjQrHBFVLR8CQzYnKCCQoWFxgZGiUmJygpKjQ1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4eLj5OXm5+jp6vHy8/T19vf4+fr/xAAfAQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgv/xAC1EQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGhscEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2gAMAwEAAhEDEQA/AO72j0GM9fvHn8K/6N3Jvkd9rOytdfDv8+lj/gvqxhzy5dLbv4drIlSGWOKKJmfKLhsD7u0YJz/L8a/VOFcvwOX5fhcHlWDw+AwtChRwlHCZbQ+owYehToxVPCZbQkpewowhbD4WjCTs74TCy93+YszxOYYrF4jG43E4jHY3FV6+Kxmb5nWdfMMdi8RWlVr4zMcXWlUqYnFV6k5Vq1Z1atSTtUeq+TQ/yrb9T5D+o3P/AGj/AB3P/9k="
	// Remove the data URL prefix
	miniJpegBase64 = strings.Replace(miniJpegBase64, "data:image/jpeg;base64,", "", 1)

	// Decode the base64 string to ensure it's valid
	_, err = base64.StdEncoding.DecodeString(miniJpegBase64)
	if err != nil {
		t.Fatalf("Failed to decode test image base64: %v", err)
	}

	block = Block{
		BlockType:    Content,
		ModalityType: Image,
		MimeType:     "image/jpeg",
		Content:      Str(miniJpegBase64),
	}

	tokens, err = g.calculateImageTokens(block)
	if err != nil {
		t.Errorf("Error calculating image tokens from actual image data: %v", err)
	}

	// Test case for image without dimensions
	blockWithoutDimensions := Block{
		BlockType:    Content,
		ModalityType: Image,
		MimeType:     "image/jpeg",
		Content:      Str("invalid-base64-content"),
	}

	// This should now return an error
	_, err = g.calculateImageTokens(blockWithoutDimensions)
	if err == nil {
		t.Errorf("Expected error when calculating tokens for image without dimensions, but got none")
	}
}

func TestOpenAiGenerator_Count(t *testing.T) {
	// Mock the tokenizer with a simple generator
	g := OpenAiGenerator{
		model:              "gpt-4o",
		systemInstructions: "You are a helpful assistant.",
	}

	// Create a simple dialog with text and image
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hello!"),
				},
				{
					BlockType:    Content,
					ModalityType: Image,
					MimeType:     "image/jpeg",
					Content:      Str("base64-content"),
					ExtraFields: map[string]interface{}{
						"width":  1024,
						"height": 1024,
					},
				},
			},
		},
	}

	// This is more of an integration test, as it relies on the tiktoken library
	// We're just checking that it runs without errors
	_, err := g.Count(context.Background(), dialog)
	if err != nil {
		t.Errorf("Error counting tokens: %v", err)
	}
}
