package gai

import "context"

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

// ToolGenerator represents a Generator that can use tools during generation.
// It extends the basic Generator interface with the ability to register tools
// and optionally provide callbacks for automatic tool execution.
//
// When a tool is registered without a callback, the Generator will pause generation
// when the tool is called and return a Message containing a ToolCall block. This allows
// for several use cases:
//  1. Manual tool execution - caller executes the tool and feeds results back
//  2. Structured output - tool calls provide structure to generated content
//  3. Control flow - tools like "finish_execution" to signal completion
//
// When a tool is registered with a callback, the Generator will automatically execute
// the callback when the tool is called and include both the tool call and its result
// in the returned Message. If the callback returns a value implementing the error
// interface, it will be treated as a tool execution error and fed back to the Generator.
//
// The behavior of tool usage is controlled via GenOpts.ToolChoice:
//  - ToolChoiceAuto: Generator decides when to use tools
//  - ToolChoiceToolsRequired: Generator must use at least one tool
//  - "<tool-name>": Generator must use the specified tool
//
// Example usage:
//
//	// Manual execution - no callback
//	g.RegisterTool(Tool{
//	    Name:        "finish_execution",
//	    Description: "Signal that the execution is complete",
//	}, nil)
//
//	// Automatic execution with callback
//	g.RegisterTool(stockPriceTool, &StockAPI{})
type ToolGenerator interface {
	Generator

	// RegisterTool adds a tool to the Generator's available tools.
	// The callback parameter is optional - if nil, tool calls will require manual execution.
	//
	// Some ToolGenerator implementations may have built-in tools. In such cases, only
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
	//
	// Note: ToolGenerator implementations may or may not be safe for concurrent use.
	// If an implementation is safe for concurrent use, it will be explicitly documented.
	// When using a goroutine-safe ToolGenerator, ensure any provided ToolCallbacks
	// are also safe for concurrent use.
	RegisterTool(tool Tool, callback ToolCallback) error
}
