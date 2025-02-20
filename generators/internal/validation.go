package internal

import (
	"context"
	"fmt"
	"github.com/spachava753/gai"
)

// ValidationMiddleware wraps a ToolGenerator and provides tool validation.
// It ensures that tools are valid before they are registered with the underlying generator.
type ValidationMiddleware struct {
	generator gai.ToolGenerator
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

// Generate implements gai.Generator by delegating to the underlying generator
func (v *ValidationMiddleware) Generate(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
	return v.generator.Generate(ctx, dialog, options)
}

// RegisterTool implements gai.ToolGenerator by validating the tool before delegating
// to the underlying generator
func (v *ValidationMiddleware) RegisterTool(tool gai.Tool, callback gai.ToolCallback) error {
	// Validate the tool
	if err := validateTool(tool); err != nil {
		return gai.ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: err,
		}
	}

	// Delegate to the underlying generator
	return v.generator.RegisterTool(tool, callback)
}

// NewValidation creates a new validationMiddleware that wraps the given generator.
// This is an internal function used by generator implementations to add tool validation.
func NewValidation(generator gai.ToolGenerator) ValidationMiddleware {
	return ValidationMiddleware{
		generator: generator,
	}
}

var _ gai.Generator = (*ValidationMiddleware)(nil)
var _ gai.ToolGenerator = (*ValidationMiddleware)(nil)
