package gai

import (
	"context"
	"encoding/json"
	"fmt"
)

// PropertyType represents the type of a property in a JSON Schema.
// The zero value is Null.
type PropertyType uint8

const (
	// Null represents JSON Schema's 'null' type
	Null PropertyType = iota

	// Boolean represents JSON Schema's 'boolean' type
	Boolean

	// Object represents JSON Schema's 'object' type
	Object

	// Array represents JSON Schema's 'array' type
	Array

	// Number represents JSON Schema's 'number' type
	Number

	// String represents JSON Schema's 'string' type
	String

	// Integer represents JSON Schema's 'integer' type
	Integer
)

// String implements fmt.Stringer and returns the JSON Schema type name
func (p PropertyType) String() string {
	switch p {
	case Null:
		return "null"
	case Boolean:
		return "boolean"
	case Object:
		return "object"
	case Array:
		return "array"
	case Number:
		return "number"
	case String:
		return "string"
	case Integer:
		return "integer"
	default:
		return fmt.Sprintf("PropertyType(%d)", p)
	}
}

// Property represents a JSON Schema property definition.
// It can describe simple types like strings and numbers,
// as well as complex types like objects and arrays.
type Property struct {
	// Type specifies the JSON Schema type of the property
	Type PropertyType

	// Enum specifies a list of valid values for the property.
	// If non-nil and non-empty, the property value must be one of these values.
	// Most commonly used with String type properties.
	Enum []string

	// Description provides a human-readable explanation of the property
	Description string

	// Properties defines the properties of an Object type.
	// Only valid when Type is Object.
	Properties map[string]Property

	// Required lists which properties are required in an Object type.
	// Only valid when Type is Object.
	Required []string

	// Items defines the type of elements in an Array type.
	// Only valid when Type is Array.
	// If nil when Type is Array, it means the array can contain elements of any type.
	Items *Property
}

// InputSchema represents the JSON Schema that defines the input parameters
// for a tool. It is always an object type schema at the root level,
// containing zero or more properties.
//
// When no parameters are needed, the zero value (Type: Null, nil maps and slices)
// can be used to indicate a parameterless schema.
type InputSchema struct {
	// Type specifies the schema type, typically Object for parameter schemas.
	// Using Null (zero value) indicates no parameters are accepted.
	Type PropertyType

	// Properties defines the available input parameters when Type is Object.
	// A nil value indicates no parameters are accepted.
	Properties map[string]Property

	// Required lists which properties are required.
	// Only valid when Type is Object.
	Required []string
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
//	    InputSchema: InputSchema{
//	        Type: Object,
//	        Properties: map[string]Property{
//	            "ticker": {
//	                Type:        String,
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
//	    InputSchema: InputSchema{
//	        Type: Object,
//	        Properties: map[string]Property{
//	            "location": {
//	                Type:        String,
//	                Description: "The city and state, e.g. San Francisco, CA",
//	            },
//	            "unit": {
//	                Type:        String,
//	                Enum:        []string{"celsius", "fahrenheit"},
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
//	    InputSchema: InputSchema{
//	        Type: Object,
//	        Properties: map[string]Property{
//	            "tickers": {
//	                Type:        Array,
//	                Description: "List of stock ticker symbols, e.g. ['AAPL', 'GOOGL', 'MSFT']",
//	                Items: &Property{
//	                    Type:        String,
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
//	}
type Tool struct {
	// Name is the identifier used to reference this tool.
	// It should be unique among all tools provided to a Generator.
	Name string

	// Description explains what the tool does.
	// This helps the Generator understand when and how to use the tool.
	Description string

	// InputSchema defines the parameters this tool accepts.
	// A zero value (Type: Null, nil maps and slices) indicates no parameters are accepted.
	InputSchema InputSchema
}

// ToolCallback represents a function that can be automatically executed by a ToolGenerator
// when a specific tool is called during generation.
//
// There are two distinct ways a tool execution can signal failure:
//  1. Returning a result that implements the error interface: This indicates the tool executed
//     successfully but failed in an expected way (e.g., invalid stock symbol, malformed input).
//     The error will be fed back to the Generator.
//  2. Returning a non-nil error as the second return value: This indicates the callback itself
//     failed to execute (e.g., network error, panic, context cancellation). This immediately
//     stops generation and the error is propagated up.
//
// Example implementation for a stock price tool:
//
//	type StockAPI struct{}
//
//	func (s *StockAPI) Call(ctx context.Context, input map[string]any) (any, error) {
//	    // Context can be used for timeouts and cancellation
//	    ticker := input["ticker"].(string)
//
//	    // Example of callback error - immediate failure
//	    if ctx.Err() != nil {
//	        return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
//	    }
//
//	    price, err := s.fetchPrice(ctx, ticker)
//	    if err != nil {
//	        // Example of expected error - fed back to Generator
//	        return fmt.Errorf("failed to get price for %s: %v", ticker, err), nil
//	    }
//
//	    return price, nil
//	}
type ToolCallback interface {
	// Call executes the tool with the given parameters.
	// The context should be used for cancellation and timeouts.
	// The input map contains the tool's parameters as defined by its InputSchema.
	// For example, with the stock price tool, input would contain {"ticker": "AAPL"}.
	//
	// The first return value can be of any type:
	// - On success: any value (primitive types, maps, slices, structs)
	// - On tool execution failure: a value that implements the error interface
	//
	// The second return value should only be non-nil if the callback itself
	// fails to execute (e.g., network errors, panics, context cancellation).
	Call(ctx context.Context, input map[string]any) (any, error)
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
	// Returns an error if:
	//  - Tool name is empty
	//  - Tool name conflicts with an already registered tool
	//  - Tool name conflicts with a built-in tool that's already registered
	//  - Tool name matches special values ToolChoiceAuto or ToolChoiceToolsRequired
	//  - Tool schema is invalid (e.g., Array type without Items field)
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
type ToolGenerator struct {
	G ToolCapableGenerator

	toolCallbacks map[string]ToolCallback
}

// Register adds a tool to the ToolGenerator's available tools with a required callback.
// The callback will be automatically executed when the tool is called during generation.
//
// Returns an error if:
//   - Tool name is empty
//   - Tool name conflicts with an already registered tool
//   - Tool name matches special values ToolChoiceAuto or ToolChoiceToolsRequired
//   - The underlying ToolCapableGenerator's Register method returns an error
//   - The callback is nil
func (t *ToolGenerator) Register(tool Tool, callback ToolCallback) error {
	// Initialize the callbacks map if it doesn't exist
	if t.toolCallbacks == nil {
		t.toolCallbacks = make(map[string]ToolCallback)
	}

	// Check if callback is provided
	if callback == nil {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("callback cannot be nil"),
		}
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

	// Store the callback
	t.toolCallbacks[tool.Name] = callback
	return nil
}

// Generate executes the given dialog with the underlying ToolCapableGenerator,
// handling any tool calls by executing their registered callbacks and feeding
// the results back into the generator. It returns the complete dialog including
// all intermediate tool calls, tool results, and the final response.
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
func (t *ToolGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Dialog, error) {
	// Validate the tool choice if provided
	if options != nil && options.ToolChoice != "" && options.ToolChoice != ToolChoiceAuto && options.ToolChoice != ToolChoiceToolsRequired {
		// ToolChoice specifies a specific tool; verify it exists
		if _, exists := t.toolCallbacks[options.ToolChoice]; !exists {
			return nil, InvalidToolChoiceErr(fmt.Sprintf("tool '%s' not found", options.ToolChoice))
		}
	}

	// Start with a copy of the input dialog
	currentDialog := make(Dialog, len(dialog))
	copy(currentDialog, dialog)

	// Loop to handle sequential tool calls
	for {
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

		// Process tool calls and generate result blocks
		var toolResultBlocks []Block

		for _, block := range toolCallBlocks {
			// Parse tool call from the block
			toolName, toolInput, err := parseToolCall(block.Content)
			if err != nil {
				return currentDialog, fmt.Errorf("invalid tool call format: %w", err)
			}

			// Check if the tool exists
			callback, exists := t.toolCallbacks[toolName]
			if !exists {
				return currentDialog, fmt.Errorf("tool '%s' not found", toolName)
			}

			// Execute the callback
			result, callErr := callback.Call(ctx, toolInput)

			// Create a tool result block
			resultBlock := Block{
				ID:           block.ID, // Use the same ID to link call and result
				BlockType:    ToolResult,
				ModalityType: Text,
			}

			// Handle callback execution errors versus tool execution errors
			if callErr != nil {
				// Callback failed to execute - propagate the error up
				return currentDialog, callErr
			} else if resultErr, isErr := result.(error); isErr {
				// Tool executed but returned an error - feed it back to the generator
				resultBlock.Content = fmt.Sprintf("Error: %s", resultErr.Error())
			} else {
				// Tool executed successfully
				resultBlock.Content = fmt.Sprintf("%v", result)
			}

			// Add to tool result blocks
			toolResultBlocks = append(toolResultBlocks, resultBlock)
		}

		// Create a message with tool results and append to dialog
		if len(toolResultBlocks) > 0 {
			resultMessage := Message{
				Role:   Assistant,
				Blocks: toolResultBlocks,
			}

			// Append the result message to the dialog
			currentDialog = append(currentDialog, resultMessage)
		}
	}
}

// parseToolCall extracts the tool name and parameters from a tool call content string.
// Expected format is a JSON object with "name" and "parameters" fields.
// Returns the tool name, input parameters as a map, and an error if parsing fails.
func parseToolCall(content string) (string, map[string]any, error) {
	var callData struct {
		Name       string         `json:"name"`
		Parameters map[string]any `json:"parameters"`
	}

	if err := json.Unmarshal([]byte(content), &callData); err != nil {
		return "", nil, fmt.Errorf("invalid tool call JSON: %w", err)
	}

	if callData.Name == "" {
		return "", nil, fmt.Errorf("missing tool name")
	}

	return callData.Name, callData.Parameters, nil
}
