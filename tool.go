package gai

import (
	"context"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
)

// GenerateSchema is a helper function to help generate the schema definition for Tool.InputSchema
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

// Tool represents a tool that can be called by a Generator during generation.
// Each tool has a name, description, and a schema defining its input parameters.
//
// Example tools:
//
// A simple tool with a single required string parameter:
//
//	{
//	    Name:        "get_stock_price",
//	    Description: "Get the current stock price for a given ticker symbol.",
//	    InputSchema: &jsonschema.Schema{
//	        Type: "object",
//	        Properties: map[string]*jsonschema.Schema{
//	            "ticker": {
//	                Type:        "string",
//	                Description: "The stock ticker symbol, e.g. AAPL for Apple Inc.",
//	            },
//	        },
//	        Required: []string{"ticker"},
//	    },
//	}
//
// A tool with both required and optional parameters:
//
//	{
//	    Name:        "get_weather",
//	    Description: "Get the current weather in a given location",
//	    InputSchema: &jsonschema.Schema{
//	        Type: "object",
//	        Properties: map[string]*jsonschema.Schema{
//	            "location": {
//	                Type:        "string",
//	                Description: "The city and state, e.g. San Francisco, CA",
//	            },
//	            "unit": {
//	                Type:        "string",
//	                Enum:        []interface{}{"celsius", "fahrenheit"},
//	                Description: "The unit of temperature, either 'celsius' or 'fahrenheit'",
//	            },
//	        },
//	        Required: []string{"location"},
//	    },
//	}
//
// A tool with an array parameter:
//
//	{
//	    Name:        "get_batch_stock_prices",
//	    Description: "Get the current stock prices for a list of ticker symbols.",
//	    InputSchema: &jsonschema.Schema{
//	        Type: "object",
//	        Properties: map[string]*jsonschema.Schema{
//	            "tickers": {
//	                Type:        "array",
//	                Description: "List of stock ticker symbols, e.g. ['AAPL', 'GOOGL', 'MSFT']",
//	                Items: &jsonschema.Schema{
//	                    Type:        "string",
//	                    Description: "A stock ticker symbol",
//	                },
//	            },
//	        },
//	        Required: []string{"tickers"},
//	    },
//	}
//
// A tool with no parameters:
//
//	{
//	    Name:        "get_server_time",
//	    Description: "Get the current server time in UTC.",
//	    InputSchema: nil, // or omit the field entirely
//	}
type Tool struct {
	// Name is the identifier used to reference this tool.
	// It should be unique among all tools provided to a Generator.
	Name string `json:"name" yaml:"name"`

	// Description explains what the tool does.
	// This helps the Generator understand when and how to use the tool.
	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	// InputSchema defines the parameters this tool accepts using JSON Schema.
	// A nil value indicates no parameters are accepted.
	// The schema should typically be of type "object" for parameter definitions.
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

type ToolRegister interface {
	// Register adds a tool to the Generator's available tools.
	//
	// Some Generator implementations may have built-in tools. In such cases, only
	// the Tool.Name needs to match a built-in tool's name to enable its use. The rest
	// of the Tool fields (Description, InputSchema) will be ignored in favor of the
	// built-in tool's definition. The callback behavior remains the same - you can
	// optionally provide a callback for automatic execution.
	//
	// JSON Schema compatibility note:
	// Different generators have different levels of support for the anyOf JSON Schema feature:
	// - OpenAI and Anthropic: Full support for anyOf properties
	// - Gemini: Limited support for anyOf - only supports [Type, null] pattern for nullable fields.
	//   Will error on multiple non-null types in anyOf or null-only anyOf.
	//
	// When using the anyOf property, the most portable approach is to restrict its usage to
	// nullable fields following the pattern: anyOf: [{type: "string"}, {type: "null"}]
	//
	// Returns an error if:
	//  - Tool name is empty
	//  - Tool name conflicts with an already registered tool
	//  - Tool name conflicts with a built-in tool that's already registered
	//  - Tool name matches special values ToolChoiceAuto or ToolChoiceToolsRequired
	//  - Tool schema is invalid (e.g., Array type without Items field)
	//  - Tool schema uses unsupported JSON Schema features for the specific generator
	Register(tool Tool) error
}

type ToolCallingGenerator interface {
	Generator
	ToolRegister
}

// ToolCallInput represents a standardized format for tool use in all generators.
// It contains the name of the tool to use and the parameters to pass to it.
type ToolCallInput struct {
	Name       string         `json:"name" yaml:"name"`
	Parameters map[string]any `json:"parameters" yaml:"parameters"`
}
