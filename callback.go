package gai

import (
	"context"
	"encoding/json"
	"fmt"
)

// Validator is an interface that can be implemented by tool parameter types
// to validate their contents after being unmarshaled from JSON.
//
// This interface allows parameter types to perform custom validation
// that goes beyond what JSON Schema validation can provide, such as:
// - Cross-field validations (e.g., field A must be present if field B has a certain value)
// - Range or format validations (e.g., dates must be in a specific format or range)
// - Business rule validations (e.g., certain combinations of values are invalid)
//
// Example implementation:
//
//	type WeatherParams struct {
//	    Location string  `json:"location"`
//	    Unit     string  `json:"unit,omitempty"`
//	}
//
//	func (p *WeatherParams) Validate() error {
//	    if p.Location == "" {
//	        return fmt.Errorf("location is required")
//	    }
//	    if p.Unit != "" && p.Unit != "celsius" && p.Unit != "fahrenheit" {
//	        return fmt.Errorf("unit must be either 'celsius' or 'fahrenheit'")
//	    }
//	    return nil
//	}
type Validator interface {
	// Validate checks if the struct's field values are valid.
	// It returns nil if validation passes, or an error describing the validation failure.
	Validate() error
}

// ToolCallBackFunc is a generic function type that wraps a callback function
// with a strongly-typed parameter struct, implementing the ToolCallback interface.
//
// The type parameter T represents the struct type that will be unmarshaled from
// the tool's JSON parameters. This allows for type-safe tool callbacks without
// the need to manually handle JSON unmarshaling or message creation.
//
// Example usage:
//
//	type WeatherParams struct {
//	    Location string  `json:"location"`
//	    Unit     string  `json:"unit,omitempty"`
//	}
//
//	func getWeather(ctx context.Context, params WeatherParams) (string, error) {
//	    unit := "celsius"
//	    if params.Unit == "fahrenheit" {
//	        unit = "fahrenheit"
//	    }
//	    // Fetch weather data
//	    return fmt.Sprintf("Weather in %s: 72°F", params.Location), nil
//	}
//
//	// Register the tool
//	weatherTool := Tool{
//	    Name: "get_weather",
//	    Description: "Get the current weather for a location",
//	    // InputSchema definition...
//	}
//	toolGen.Register(weatherTool, ToolCallBackFunc(getWeather))
type ToolCallBackFunc[T any] func(ctx context.Context, t T) (string, error)

// Call implements the ToolCallback interface, handling JSON unmarshaling
// and message creation automatically.
//
// It unmarshalals the JSON parameters into the type T,
// optionally validates them if T implements the Validator interface,
// calls the wrapped function with the parsed parameters,
// and constructs a properly formatted ToolResult message from the result.
func (f ToolCallBackFunc[T]) Call(ctx context.Context, parametersJSON json.RawMessage, toolCallID string) (Message, error) {
	var t T
	if err := json.Unmarshal(parametersJSON, &t); err != nil {
		return Message{}, fmt.Errorf("failed to unmarshal parameters: %w", err)
	}

	// Check if T implements Validator and validate if it does
	if validator, ok := any(&t).(Validator); ok {
		if err := validator.Validate(); err != nil {
			return Message{}, fmt.Errorf("parameter validation failed: %w", err)
		}
	}

	// Call the wrapped function
	content, err := f(ctx, t)
	if err != nil {
		return Message{}, err
	}

	// Create and return a text tool result message
	return TextToolResultMessage(toolCallID, content), nil
}
