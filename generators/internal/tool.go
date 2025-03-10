package internal

import (
	"context"
	"fmt"
	"github.com/spachava753/gai"
)

// ToolMiddleware wraps a ToolGenerator and provides tool validation.
// It ensures that tools are valid before they are registered with the underlying generator.
// It also enforces that GenOpts.N must be 1 when tools are registered with callbacks.
type ToolMiddleware struct {
	generator gai.ToolGenerator
	// hasToolWithCallback indicates if any tool has been registered with a callback
	hasToolWithCallback bool
}

// validateTool checks if a tool definition is valid.
// Returns an error if:
// - Tool name is empty
// - Tool name conflicts with special values
// - Tool schema is invalid
func validateTool(tool gai.Tool) error {
	if tool.Name == "" {
		return fmt.Errorf("tool name cannot be empty")
	}

	if tool.Name == gai.ToolChoiceAuto || tool.Name == gai.ToolChoiceToolsRequired {
		return fmt.Errorf("tool name cannot be special value %q", tool.Name)
	}

	// Validate schema if parameters are defined
	if tool.InputSchema.Type == gai.Object && tool.InputSchema.Properties != nil {
		for name, prop := range tool.InputSchema.Properties {
			if err := validateProperty(name, prop); err != nil {
				return fmt.Errorf("invalid property %q: %w", name, err)
			}
		}
	}

	return nil
}

// validateProperty recursively validates a property definition.
// Returns an error if:
// - Array type without Items field
// - Object type with invalid properties
func validateProperty(name string, prop gai.Property) error {
	switch prop.Type {
	case gai.Array:
		if prop.Items == nil {
			return fmt.Errorf("array property %q must define Items", name)
		}
		// Recursively validate array item type
		if err := validateProperty(name+"[]", *prop.Items); err != nil {
			return err
		}
	case gai.Object:
		if prop.Properties != nil {
			for subName, subProp := range prop.Properties {
				if err := validateProperty(subName, subProp); err != nil {
					return fmt.Errorf("in object property %q: %w", name, err)
				}
			}
		}
	}
	return nil
}

// Generate implements gai.Generator by delegating to the underlying generator.
// If any tool is registered with a callback, it enforces that GenOpts.N must be 1.
func (v *ToolMiddleware) Generate(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
	// Check if we have any tool with a callback and validate N
	if v.hasToolWithCallback {
		// If N is not set (0), the default is 1, which is valid
		if options != nil && options.N > 1 {
			return gai.Response{}, gai.InvalidParameterErr{
				Parameter: "N",
				Reason:    "value greater than 1 is not supported when tools are registered with callbacks",
			}
		}
	}

	return v.generator.Generate(ctx, dialog, options)
}

// RegisterTool implements gai.ToolGenerator by validating the tool before delegating
// to the underlying generator. It also tracks if any tool has a callback.
func (v *ToolMiddleware) RegisterTool(tool gai.Tool, callback gai.ToolCallback) error {
	// Validate the tool
	if err := validateTool(tool); err != nil {
		return gai.ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: err,
		}
	}

	// Track if this tool has a callback
	if callback != nil {
		v.hasToolWithCallback = true
	}

	// Delegate to the underlying generator
	return v.generator.RegisterTool(tool, callback)
}

// NewToolMiddleware creates a new validationMiddleware that wraps the given generator.
// This is an internal function used by generator implementations to add tool validation.
func NewToolMiddleware(generator gai.ToolGenerator) ToolMiddleware {
	return ToolMiddleware{
		generator: generator,
	}
}

var _ gai.Generator = (*ToolMiddleware)(nil)
var _ gai.ToolGenerator = (*ToolMiddleware)(nil)
