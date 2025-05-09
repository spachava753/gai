package gai

import (
	"testing"

	a "github.com/anthropics/anthropic-sdk-go"
	oai "github.com/openai/openai-go"
	"google.golang.org/genai"
)

// Test that OpenAI generator correctly handles anyOf
func TestOpenAIAnyOf(t *testing.T) {
	// Create a property with anyOf
	prop := Property{
		Description: "line number in the file to comment on",
		AnyOf: []Property{
			{Type: Number},
			{Type: Null},
		},
	}

	// Convert to OpenAI format
	result := convertPropertyToMap(prop)

	// Verify that the result has anyOf and no type field
	anyOfVal, hasAnyOf := result["anyOf"]
	if !hasAnyOf {
		t.Errorf("Expected anyOf in result, got: %v", result)
	}

	// Verify that the type is not set when anyOf is present
	if _, hasType := result["type"]; hasType {
		t.Errorf("Expected no type field when anyOf is present, got: %v", result)
	}

	// Verify anyOf structure
	anyOfArr, ok := anyOfVal.([]interface{})
	if !ok {
		t.Errorf("Expected anyOf to be an array, got: %T", anyOfVal)
	}
	if len(anyOfArr) != 2 {
		t.Errorf("Expected anyOf to have 2 elements, got: %d", len(anyOfArr))
	}
}

// Test that Anthropic generator correctly handles anyOf
func TestAnthropicAnyOf(t *testing.T) {
	// Create a property with anyOf
	prop := Property{
		Description: "line number in the file to comment on",
		AnyOf: []Property{
			{Type: Number},
			{Type: Null},
		},
	}

	// Convert to Anthropic format
	result := convertPropertyToAnthropicMap(prop)

	// Verify that the result has anyOf and no type field
	anyOfVal, hasAnyOf := result["anyOf"]
	if !hasAnyOf {
		t.Errorf("Expected anyOf in result, got: %v", result)
	}

	// Verify that the type is not set when anyOf is present
	if _, hasType := result["type"]; hasType {
		t.Errorf("Expected no type field when anyOf is present, got: %v", result)
	}

	// Verify anyOf structure
	anyOfArr, ok := anyOfVal.([]interface{})
	if !ok {
		t.Errorf("Expected anyOf to be an array, got: %T", anyOfVal)
	}
	if len(anyOfArr) != 2 {
		t.Errorf("Expected anyOf to have 2 elements, got: %d", len(anyOfArr))
	}
}

// Test that Gemini generator correctly handles anyOf
func TestGeminiAnyOf(t *testing.T) {
	// Create a property with anyOf (null + another type)
	prop := Property{
		Description: "line number in the file to comment on",
		AnyOf: []Property{
			{Type: Number},
			{Type: Null},
		},
	}

	// Convert to Gemini format
	result, err := convertSinglePropertyToGeminiSchema(prop)
	if err != nil {
		t.Fatalf("Failed to convert property: %v", err)
	}

	// Since Gemini doesn't directly support anyOf, verify our workaround:
	// - It should use the non-null type (Number)
	// - It should set Nullable=true instead of modifying description
	if result.Type != genai.TypeNumber {
		t.Errorf("Expected type to be Number, got: %v", result.Type)
	}
	if !*result.Nullable {
		t.Errorf("Expected Nullable to be true for null+number anyOf property")
	}

	// Test complex anyOf (multiple non-null types) - should fail
	complexProp := Property{
		Description: "Complex anyOf test",
		AnyOf: []Property{
			{Type: String},
			{Type: Number},
			{Type: Boolean},
		},
	}

	// Convert to Gemini format - should error
	_, err = convertSinglePropertyToGeminiSchema(complexProp)
	if err == nil {
		t.Errorf("Expected error for multiple non-null types in anyOf, but got nil")
	}

	// Test null-only anyOf - should fail
	nullOnlyProp := Property{
		Description: "Null-only property",
		AnyOf: []Property{
			{Type: Null},
		},
	}

	// Convert to Gemini format - should error
	_, err = convertSinglePropertyToGeminiSchema(nullOnlyProp)
	if err == nil {
		t.Errorf("Expected error for null-only anyOf, but got nil")
	}
}

// Test creating a tool with anyOf property and registering it with all generator types
func TestToolWithAnyOf(t *testing.T) {
	// Define a tool with properties that use anyOf (based on the example schema provided)
	tool := Tool{
		Name:        "create_pull_request_review",
		Description: "Create a review on a pull request",
		InputSchema: InputSchema{
			Type: Object,
			Properties: map[string]Property{
				"line": {
					Description: "line number in the file to comment on",
					AnyOf: []Property{
						{Type: Number},
						{Type: Null},
					},
				},
				"event": {
					Description: "Review action to perform",
					Type:        String,
					Enum:        []string{"APPROVE", "REQUEST_CHANGES", "COMMENT"},
				},
			},
			Required: []string{"event"},
		},
	}

	// Test OpenAI conversion
	openaiGen := OpenAiGenerator{
		tools: make(map[string]oai.ChatCompletionToolParam),
	}
	if err := openaiGen.Register(tool); err != nil {
		t.Fatalf("Failed to register tool with OpenAI generator: %v", err)
	}

	// Test Anthropic conversion
	anthropicGen := AnthropicGenerator{
		tools: make(map[string]a.ToolParam),
	}
	if err := anthropicGen.Register(tool); err != nil {
		t.Fatalf("Failed to register tool with Anthropic generator: %v", err)
	}

	// Test Gemini conversion
	geminiGen := GeminiGenerator{}
	if err := geminiGen.Register(tool); err != nil {
		t.Fatalf("Failed to register tool with Gemini generator: %v", err)
	}

	// Success if we reach this point without error
	t.Log("Successfully registered tool with anyOf property in all generators")
}
