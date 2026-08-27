package gai

import (
	"context"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
)

// GenerateSchema derives a JSON Schema for T for use as [Tool.InputSchema]. It
// closes object schemas by setting additionalProperties to false when the
// generated schema does not specify that keyword.
func GenerateSchema[T any]() (*jsonschema.Schema, error) {
	schema, err := jsonschema.For[T](&jsonschema.ForOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to generate schema for type: %w", err)
	}
	// Set additionalProperties to false (disallow additional properties)
	if schema.AdditionalProperties == nil {
		schema.AdditionalProperties = &jsonschema.Schema{Not: &jsonschema.Schema{}}
	}
	return schema, nil
}

// Tool declares a caller-owned function available through
// [GenerationRequest.Tools]. Generators send the declaration to the provider and
// return requested calls as [ToolCall] blocks; they never execute the function.
// Use [GenerateSchema] to derive InputSchema from a Go parameter type.
type Tool struct {
	// Name is the provider-visible function identifier. It must be non-empty,
	// non-reserved, and unique within [GenerationRequest.Tools].
	Name string `json:"name" yaml:"name"`

	// Description tells the model when and how to call the function.
	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	// InputSchema defines parameters with JSON Schema. A nil schema declares no
	// parameters. [GenerateSchema] derives a closed schema from a Go type.
	InputSchema *jsonschema.Schema `json:"input_schema,omitempty" yaml:"input_schema,omitempty"`
}

// ToolCallback represents a function that executes a specific tool call and returns
// the corresponding tool result message.
//
// The callback should return a message with role ToolResult containing
// the result of the tool execution. The message will be validated to ensure
// it has the correct role, at least one block, and that all blocks have:
// - Non-nil content
// - A valid block type
// - A valid modality type
// - A MimeType appropriate for the modality
//
// Example implementation for a stock price tool:
//
//	type StockAPI struct{}
//
//	func (s *StockAPI) Call(ctx context.Context, parameters map[string]any) (Message, error) {
//	    // Context can be used for timeouts and cancellation
//	    if ctx.Err() != nil {
//	        return Message{}, fmt.Errorf("context cancelled: %w", ctx.Err())
//	    }
//
//	    // Read parameters from the decoded tool input
//	    ticker, ok := parameters["ticker"].(string)
//	    if !ok || ticker == "" {
//	        return Message{
//	            Role: ToolResult,
//	            Blocks: []Block{
//	                {
//	                    BlockType:    Content,
//	                    ModalityType: Text,
//	                    MimeType:     "text/plain",
//	                    Content:      Str("Error: ticker is required"),
//	                },
//	            },
//	        }, nil
//	    }
//
//	    price, err := s.fetchPrice(ctx, ticker)
//	    if err != nil {
//	        // Example of expected error - fed back to Generator as a message
//	        return Message{
//	            Role: ToolResult,
//	            Blocks: []Block{
//	                {
//	                    BlockType:    Content,      // Must specify a block type
//	                    ModalityType: Text,
//	                    MimeType:     "text/plain", // Required for all blocks
//	                    Content:      Str(fmt.Sprintf("Error: failed to get price for %s: %v", ticker, err)),
//	                },
//	            },
//	        }, nil
//	    }
//
//	    // Return a successful result as a message
//	    return Message{
//	        Role: ToolResult,
//	        Blocks: []Block{
//	            {
//	                BlockType:    Content,
//	                ModalityType: Text,
//	                MimeType:     "text/plain",
//	                Content:      Str(fmt.Sprintf("$%.2f", price)),
//	            },
//	        },
//	    }, nil
//	}
//
//	// Example of a tool returning an image
//	type ImageGeneratorTool struct{}
//
//	func (t *ImageGeneratorTool) Call(ctx context.Context, parameters map[string]any) (Message, error) {
//	    prompt, ok := parameters["prompt"].(string)
//	    if !ok || prompt == "" {
//	        return Message{}, fmt.Errorf("prompt is required")
//	    }
//
//	    imageData, err := t.generateImage(ctx, prompt)
//	    if err != nil {
//	        return Message{}, err
//	    }
//
//	    // Base64 encode the image data
//	    encodedImage := base64.StdEncoding.EncodeToString(imageData)
//
//	    return Message{
//	        Role: ToolResult,
//	        Blocks: []Block{
//	            {
//	                BlockType:    Content,
//	                ModalityType: Image,           // Image modality
//	                MimeType:     "image/jpeg",    // MimeType is required for all modalities
//	                Content:      Str(encodedImage),
//	            },
//	        },
//	    }, nil
//	}
type ToolCallback interface {
	// Call executes the tool with the given parameters and returns a tool result message.
	// The context should be used for cancellation and timeouts.
	// The parameters map contains the decoded tool parameters as defined by its InputSchema.
	//
	// The returned message must have the ToolResult role and at least one block.
	// Each block must have:
	// - Non-nil Content
	// - A valid BlockType (usually "content")
	// - A valid ModalityType (Text, Image, Audio, or Video)
	// - A MimeType appropriate for the modality (e.g., "text/plain" for text, "image/jpeg" for images)
	//
	// The second return value should only be non-nil if the callback itself fails to execute
	// (e.g., network errors, panics, context cancellation).
	Call(ctx context.Context, parameters map[string]any) (Message, error)
}

// ToolCallInput is the normalized payload encoded in a [ToolCall] block. Use
// [ToolCallBlock] to construct one for replay or tests.
type ToolCallInput struct {
	// Name matches the requested [Tool.Name].
	Name string `json:"name" yaml:"name"`
	// Parameters contains the provider-decoded JSON arguments.
	Parameters map[string]any `json:"parameters" yaml:"parameters"`
}
