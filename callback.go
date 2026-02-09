package gai

import (
	"context"
	"encoding/json"
	"errors"
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

// CallbackExecErr is an error type that wraps a real callback execution error.
// When returned from a callback (wrapped in this type), it signals to the caller
// that the error is a hard failure and execution should terminate, rather than being
// returned as an erroneous tool result.
type CallbackExecErr struct {
	Err error `json:"err,omitempty" yaml:"err,omitempty"`
}

// Unwrap allows errors.Unwrap and errors.As to extract the underlying error.
func (c CallbackExecErr) Unwrap() error {
	return c.Err
}

func (c CallbackExecErr) Error() string {
	return c.Err.Error()
}

// ToolCallBackFunc is a generic function type that wraps a callback function
// with a strongly-typed parameter struct, implementing the ToolCallback interface.
//
// The type parameter T represents the struct type that will be unmarshaled from
// the tool's JSON parameters. This allows for type-safe tool callbacks without
// the need to manually handle JSON unmarshaling or message creation.
//
// Callback error handling:
//   - If the callback returns an error value of type CallbackExecErr (or wraps one),
//     this signals a true callback execution error (panic, cancellation, etc.), and execution will terminate.
//   - Any other non-nil error is treated as a tool result error: the error message will be sent as a textual tool result message to the generator, not treated as fatal.
//
// Example usage:
//
//	type WeatherParams struct {
//	    Location string  `json:"location"`
//	    Unit     string  `json:"unit,omitempty"`
//	}
//
//	func getWeather(ctx context.Context, params WeatherParams) (string, error) {
//	    if params.Location == "" {
//	        return "", fmt.Errorf("location is required") // Erroneous tool result
//	    }
//	    // Simulate a callback execution error
//	    // return "", CallbackExecErr{Err: fmt.Errorf("panic occurred")}
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
// It unmarshals the JSON parameters into the type T, optionally validates them if T implements the Validator interface,
// calls the wrapped function with the parsed parameters, and constructs a properly formatted ToolResult message from
// the result.
//
// Error handling:
//   - If the callback returns a non-nil error of type CallbackExecErr (or wrapping one), this signals a real callback execution failure (e.g., panic, context cancellation), and the underlying error is returned. This will typically terminate execution in ToolGenerator.Generate.
//   - If the callback returns any other non-nil error, it is treated as an erroneous tool result, and a text ToolResult message containing the error message is returned instead of terminating execution.
func (f ToolCallBackFunc[T]) Call(ctx context.Context, parametersJSON json.RawMessage, toolCallID string) (Message, error) {
	var t T
	if err := json.Unmarshal(parametersJSON, &t); err != nil {
		return Message{}, fmt.Errorf("failed to unmarshal parameters: %w", err)
	}

	// Check if T implements Validator and validate if it does
	if validator, ok := any(&t).(Validator); ok {
		if err := validator.Validate(); err != nil {
			// Otherwise: Treat as tool error, return as tool result message so execution can continue.
			msg := ToolResultMessage(
				toolCallID,
				TextBlock(fmt.Errorf("parameter validation failed: %s", err).Error()),
			)
			msg.ToolResultError = true
			return msg, nil
		}
	}

	// Call the wrapped function
	content, err := f(ctx, t)
	if err != nil {
		var execErr CallbackExecErr
		if ok := errors.As(err, &execErr); ok {
			// Return the *underlying* error (unwrap), will terminate generation
			return Message{}, execErr.Unwrap()
		}
		// Otherwise: Treat as tool error, return as tool result message so execution can continue.
		msg := ToolResultMessage(toolCallID, TextBlock(err.Error()))
		msg.ToolResultError = true
		return msg, nil
	}

	// Create and return a text tool result message
	return ToolResultMessage(toolCallID, TextBlock(content)), nil
}
