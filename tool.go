package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

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
	Name string

	// Description explains what the tool does.
	// This helps the Generator understand when and how to use the tool.
	Description string

	// InputSchema defines the parameters this tool accepts using JSON Schema.
	// A nil value indicates no parameters are accepted.
	// The schema should typically be of type "object" for parameter definitions.
	InputSchema *jsonschema.Schema
}

// ToolCallback represents a function that can be automatically executed by a ToolGenerator
// when a specific tool is called during generation.
//
// The callback should return a message with role ToolResult containing
// the result of the tool execution. The message will be validated to ensure
// it has the correct role, at least one block, and that all blocks have:
// - The correct ID matching the tool call ID
// - Non-nil content
// - A valid block type
// - A valid modality type
// - A MimeType appropriate for the modality
//
// Example implementation for a stock price tool:
//
//	type StockAPI struct{}
//
//	func (s *StockAPI) Call(ctx context.Context, parametersJSON json.RawMessage, toolCallID string) (Message, error) {
//	    // Context can be used for timeouts and cancellation
//	    if ctx.Err() != nil {
//	        return Message{}, fmt.Errorf("context cancelled: %w", ctx.Err())
//	    }
//
//	    // Parse parameters from JSON
//	    var params struct {
//	        Ticker string `json:"ticker"`
//	    }
//	    if err := json.Unmarshal(parametersJSON, &params); err != nil {
//	        return Message{
//	            Role: ToolResult,
//	            Blocks: []Block{
//	                {
//	                    ID:           toolCallID,
//	                    BlockType:    Content,
//	                    ModalityType: Text,
//	                    MimeType:     "text/plain",
//	                    Content:      Str(fmt.Sprintf("Error parsing parameters: %v", err)),
//	                },
//	            },
//	        }, nil
//	    }
//
//	    price, err := s.fetchPrice(ctx, params.Ticker)
//	    if err != nil {
//	        // Example of expected error - fed back to Generator as a message
//	        return Message{
//	            Role: ToolResult,
//	            Blocks: []Block{
//	                {
//	                    ID:           toolCallID,  // Must match the tool call ID
//	                    BlockType:    Content,     // Must specify a block type
//	                    ModalityType: Text,
//	                    MimeType:     "text/plain", // Required for all blocks
//	                    Content:      Str(fmt.Sprintf("Error: failed to get price for %s: %v", params.Ticker, err)),
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
//	                ID:           toolCallID,
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
//	func (t *ImageGeneratorTool) Call(ctx context.Context, parametersJSON json.RawMessage, toolCallID string) (Message, error) {
//	    // Parse parameters
//	    var params struct {
//	        Prompt string `json:"prompt"`
//	    }
//	    if err := json.Unmarshal(parametersJSON, &params); err != nil {
//	        return Message{}, fmt.Errorf("failed to parse parameters: %w", err)
//	    }
//
//	    imageData, err := t.generateImage(ctx, params.Prompt)
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
//	                ID:           toolCallID,
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
	// The parametersJSON contains the tool's parameters as raw JSON as defined by its InputSchema.
	// The toolCallID is the ID of the tool call block that initiated this tool execution.
	//
	// The returned message must have the ToolResult role and at least one block.
	// Each block must have:
	// - ID matching the provided toolCallID
	// - Non-nil Content
	// - A valid BlockType (usually "content")
	// - A valid ModalityType (Text, Image, Audio, or Video)
	// - A MimeType appropriate for the modality (e.g., "text/plain" for text, "image/jpeg" for images)
	//
	// The second return value should only be non-nil if the callback itself fails to execute
	// (e.g., network errors, panics, context cancellation).
	Call(ctx context.Context, parametersJSON json.RawMessage, toolCallID string) (Message, error)
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

type ToolCapableGenerator interface {
	Generator
	ToolRegister
}

// ToolGenerator represents a Generator that can use tools during generation.
// It extends the basic Generator interface with the ability to register tools
// with callbacks for automatic tool execution.
//
// When a tool is called during generation, ToolGenerator will automatically execute
// the registered callback and include both the tool call and its result
// in the returned Message. If the callback returns a value implementing the error
// interface, it will be treated as a tool execution error and fed as a tool result
// into the underlying Generator.
//
// Tools can be registered with nil callbacks, in which case execution will be
// terminated immediately when the tool is called. This is useful for tools like
// "finish_execution" that are meant to interrupt generation and return the dialog.
//
// The behavior of tool usage is controlled via GenOpts.ToolChoice:
//   - ToolChoiceAuto: Generator decides when to use tools
//   - ToolChoiceToolsRequired: Generator must use at least one tool
//   - "<tool-name>": Generator must use the specified tool
//
// Example usage:
//
//	// Create a ToolGenerator with an underlying generator
//	toolGen := &ToolGenerator{
//	    G: myGenerator,
//	    toolCallbacks: make(map[string]ToolCallback),
//	}
//
//	// Register a tool with automatic execution via callback
//	toolGen.Register(stockPriceTool, &StockAPI{})
//
//	// Register a tool that terminates execution when called
//	toolGen.Register(Tool{Name: "finish_execution"}, nil)
type ToolGenerator struct {
	G ToolCapableGenerator

	toolCallbacks map[string]ToolCallback
}

// Register adds a tool to the ToolGenerator's available tools with an optional callback.
// If a callback is provided, it will be automatically executed when the tool is called during generation.
// If the callback is nil, no automatic execution will occur. This is useful for tools that are meant to
// interrupt or terminate execution, such as a "finish_execution" tool that should end the generation process.
//
// Returns an error if:
//   - Tool name is empty
//   - Tool name conflicts with an already registered tool
//   - Tool name matches special values ToolChoiceAuto or ToolChoiceToolsRequired
//   - The underlying ToolCapableGenerator's Register method returns an error
func (t *ToolGenerator) Register(tool Tool, callback ToolCallback) error {
	// Initialize the callbacks map if it doesn't exist
	if t.toolCallbacks == nil {
		t.toolCallbacks = make(map[string]ToolCallback)
	}

	// Check if the tool is already registered with a callback
	if _, exists := t.toolCallbacks[tool.Name]; exists {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool already registered"),
		}
	}

	// Register the tool with the underlying generator
	if err := t.G.Register(tool); err != nil {
		return err
	}

	// Store the callback (which may be nil)
	t.toolCallbacks[tool.Name] = callback
	return nil
}

// GenOptsGenerator is a function that takes a dialog and returns generation options.
// This allows customizing the options based on the current state of the dialog.
type GenOptsGenerator func(dialog Dialog) *GenOpts

// Generate executes the given dialog with the underlying ToolCapableGenerator,
// handling any tool calls by executing their registered callbacks and feeding
// the results back into the generator. It returns the complete dialog including
// all intermediate tool calls, tool results, and the final response.
//
// The optsGen parameter is a function that generates generation options based on
// the current state of the dialog. This allows customizing options like temperature,
// tool choice, or modalities based on the conversation context. If optsGen is nil,
// a default function that returns nil options will be used.
//
// Error Handling:
// If an error occurs during the looped generation process (e.g., tool callback
// execution fails, invalid tool calls, context cancellation), the dialog accumulated
// up to that point is returned along with the error. This partial dialog includes
// all successfully processed messages, tool calls, and tool results that occurred
// before the error, allowing callers to inspect the conversation state when the
// error occurred.
//
// Example usage with dynamic options:
//
//	dialog, err := toolGen.Generate(ctx, dialog, func(d Dialog) *GenOpts {
//	    // Increase temperature after each tool use
//	    toolUses := 0
//	    for _, msg := range d {
//	        if msg.Role == ToolResult {
//	            toolUses++
//	        }
//	    }
//	    return &GenOpts{
//	        Temperature: 0.2 * float64(toolUses),
//	        ToolChoice: ToolChoiceAuto,
//	    }
//	})
//
// Example usage with static options:
//
//	// Always use the same options
//	dialog, err := toolGen.Generate(ctx, dialog, func(d Dialog) *GenOpts {
//	    return &GenOpts{
//	        ToolChoice: ToolChoiceToolsRequired,
//	    }
//	})
//
// Example usage with no options:
//
//	// Use default options (nil)
//	dialog, err := toolGen.Generate(ctx, dialog, nil)
//
// The returned dialog will contain:
// 1. The original input dialog
// 2. Any tool call messages from the generator
// 3. Tool result messages from callback execution
// 4. The final response from the generator
//
// For example, if the generator first calls a location tool and then a weather tool,
// the returned dialog might look like:
//
//	[0] User: "What's the weather where I am?"
//	[1] Assistant: Tool call to get_location
//	[2] Assistant: Tool result "New York"
//	[3] Assistant: Tool call to get_weather with location="New York"
//	[4] Assistant: Tool result "72°F and sunny"
//	[5] Assistant: "The weather in New York is 72°F and sunny"
func (t *ToolGenerator) Generate(ctx context.Context, dialog Dialog, optsGen GenOptsGenerator) (Dialog, error) {
	// Start with a copy of the input dialog
	currentDialog := make(Dialog, len(dialog))
	copy(currentDialog, dialog)

	// If no options generator is provided, use a default one that returns nil
	if optsGen == nil {
		optsGen = func(dialog Dialog) *GenOpts {
			return nil
		}
	}

	// Get the options for the current dialog state
	options := optsGen(currentDialog)

	// Validate the tool choice if provided
	if options != nil && options.ToolChoice != "" && options.ToolChoice != ToolChoiceAuto && options.ToolChoice != ToolChoiceToolsRequired {
		// ToolChoice specifies a specific tool; verify it exists
		if _, exists := t.toolCallbacks[options.ToolChoice]; !exists {
			return nil, InvalidToolChoiceErr(fmt.Sprintf("tool '%s' not found", options.ToolChoice))
		}
	}

	// Loop to handle sequential tool calls
	for {
		// Check for context cancellation before generating a response
		select {
		case <-ctx.Done():
			return currentDialog, ctx.Err()
		default:
		}

		// Call the underlying generator with the current dialog
		resp, err := t.G.Generate(ctx, currentDialog, options)
		if err != nil {
			return currentDialog, err
		}

		// Verify the response has exactly one candidate
		if len(resp.Candidates) != 1 {
			return currentDialog, fmt.Errorf("expected exactly one candidate in response, got: %d", len(resp.Candidates))
		}

		// Verify the response has an Assistant role
		if resp.Candidates[0].Role != Assistant {
			return currentDialog, fmt.Errorf("expected assistant role in response, got: %v", resp.Candidates[0].Role)
		}

		// Get the candidate message
		candidate := resp.Candidates[0]

		// If this isn't a tool call, append the message and return the dialog
		if resp.FinishReason != ToolUse {
			currentDialog = append(currentDialog, candidate)
			return currentDialog, nil
		}

		// Look for tool calls
		var toolCallBlocks []Block

		// First pass: collect all tool call blocks
		for _, block := range candidate.Blocks {
			if block.BlockType == ToolCall {
				toolCallBlocks = append(toolCallBlocks, block)
			}
		}

		// Append the tool call message to the dialog
		currentDialog = append(currentDialog, candidate)

		// If no tool calls, return
		if len(toolCallBlocks) == 0 {
			return currentDialog, nil
		}

		// Process tool calls
		for _, block := range toolCallBlocks {
			// Parse tool call JSON with a single unmarshaling operation
			var toolCallData struct {
				Name       string          `json:"name"`
				Parameters json.RawMessage `json:"parameters"`
			}

			if err := json.Unmarshal([]byte(block.Content.String()), &toolCallData); err != nil {
				return currentDialog, fmt.Errorf("invalid tool call JSON: %w", err)
			}

			if toolCallData.Name == "" {
				return currentDialog, fmt.Errorf("missing tool name")
			}

			// Check if the tool exists
			callback, exists := t.toolCallbacks[toolCallData.Name]
			if !exists {
				return currentDialog, fmt.Errorf("tool '%s' not found", toolCallData.Name)
			}

			// If the callback is nil, this indicates the tool is meant to terminate execution
			// So we return the current dialog with what we have so far
			if callback == nil {
				return currentDialog, nil
			}

			// Set default empty JSON object if parameters is null or not provided
			parametersJSON := toolCallData.Parameters
			if len(parametersJSON) == 0 {
				parametersJSON = json.RawMessage("{}")
			}

			// Execute the callback with the raw parameters JSON
			resultMessage, callErr := callback.Call(ctx, parametersJSON, block.ID)

			// Handle callback execution errors
			if callErr != nil {
				// Callback failed to execute - propagate the error up
				return currentDialog, callErr
			}

			// Validate and sanitize the tool result message
			if err := validateToolResultMessage(&resultMessage, block.ID); err != nil {
				return currentDialog, fmt.Errorf("invalid tool result message: %w", err)
			}

			// Append the validated result message to the dialog
			currentDialog = append(currentDialog, resultMessage)
		}
	}
}

// ToolCallInput represents a standardized format for tool use in all generators.
// It contains the name of the tool to use and the parameters to pass to it.
type ToolCallInput struct {
	Name       string         `json:"name"`
	Parameters map[string]any `json:"parameters"`
}

// validateToolResultMessage validates a tool result message without modifying it.
// It returns an error if any validation check fails.
func validateToolResultMessage(message *Message, toolCallID string) error {
	// Check that the message has the correct role
	if message.Role != ToolResult {
		return fmt.Errorf("message must have ToolResult role, got: %v", message.Role)
	}

	// Check that the message has at least one block
	if len(message.Blocks) == 0 {
		return fmt.Errorf("message must have at least one block")
	}

	// Validate all blocks
	for i, block := range message.Blocks {
		// Check block ID matches the tool call ID
		if block.ID != toolCallID {
			return fmt.Errorf("block %d has incorrect ID: expected %q, got %q", i, toolCallID, block.ID)
		}

		// Ensure each block has content
		if block.Content == nil {
			return fmt.Errorf("block %d has nil content", i)
		}

		// Check block type is set
		if block.BlockType == "" {
			return fmt.Errorf("block %d is missing block type", i)
		}

		// Check MIME type is always required
		if block.MimeType == "" {
			return fmt.Errorf("block %d is missing MIME type", i)
		}

		// Validate modality type
		switch block.ModalityType {
		case Text, Image, Audio, Video:
			// Valid modality, now check if MIME type matches modality
			switch block.ModalityType {
			case Text:
				if !strings.HasPrefix(block.MimeType, "text/") {
					return fmt.Errorf("block %d has text modality but non-text MIME type: %q", i, block.MimeType)
				}
			case Image:
				if !strings.HasPrefix(block.MimeType, "image/") {
					return fmt.Errorf("block %d has image modality but non-image MIME type: %q", i, block.MimeType)
				}
			case Audio:
				if !strings.HasPrefix(block.MimeType, "audio/") {
					return fmt.Errorf("block %d has audio modality but non-audio MIME type: %q", i, block.MimeType)
				}
			case Video:
				if !strings.HasPrefix(block.MimeType, "video/") {
					return fmt.Errorf("block %d has video modality but non-video MIME type: %q", i, block.MimeType)
				}
			}
		default:
			return fmt.Errorf("block %d has invalid modality type: %v", i, block.ModalityType)
		}
	}

	return nil
}
