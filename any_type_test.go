package gai

import (
	"testing"
)

func TestPropertyType_Any_String(t *testing.T) {
	// Test that the Any PropertyType has the correct string representation
	anyType := Any
	if anyType.String() != "any" {
		t.Errorf("Expected Any.String() to return 'any', got '%s'", anyType.String())
	}
}

func TestOpenAI_AnyPropertyType(t *testing.T) {
	// Test that OpenAI generator handles Any type by omitting the type field
	prop := Property{
		Type:        Any,
		Description: "Can be any type",
	}

	result := convertPropertyToMap(prop)

	// Should not have a "type" field
	if _, hasType := result["type"]; hasType {
		t.Error("Expected Any property to not have 'type' field in OpenAI conversion")
	}

	// Should have description
	if desc, ok := result["description"].(string); !ok || desc != "Can be any type" {
		t.Errorf("Expected description 'Can be any type', got %v", result["description"])
	}

	// Should not have any type-specific fields
	if _, hasEnum := result["enum"]; hasEnum {
		t.Error("Expected Any property to not have 'enum' field")
	}
	if _, hasItems := result["items"]; hasItems {
		t.Error("Expected Any property to not have 'items' field")
	}
	if _, hasProperties := result["properties"]; hasProperties {
		t.Error("Expected Any property to not have 'properties' field")
	}
}

func TestAnthropic_AnyPropertyType(t *testing.T) {
	// Test that Anthropic generator handles Any type by omitting the type field
	prop := Property{
		Type:        Any,
		Description: "Can be any type",
	}

	result := convertPropertyToAnthropicMap(prop)

	// Should not have a "type" field
	if _, hasType := result["type"]; hasType {
		t.Error("Expected Any property to not have 'type' field in Anthropic conversion")
	}

	// Should have description
	if desc, ok := result["description"].(string); !ok || desc != "Can be any type" {
		t.Errorf("Expected description 'Can be any type', got %v", result["description"])
	}

	// Should not have any type-specific fields
	if _, hasEnum := result["enum"]; hasEnum {
		t.Error("Expected Any property to not have 'enum' field")
	}
	if _, hasItems := result["items"]; hasItems {
		t.Error("Expected Any property to not have 'items' field")
	}
	if _, hasProperties := result["properties"]; hasProperties {
		t.Error("Expected Any property to not have 'properties' field")
	}
}

func TestGemini_AnyPropertyType_Error(t *testing.T) {
	// Test that Gemini generator returns an error for Any type
	prop := Property{
		Type:        Any,
		Description: "Can be any type",
	}

	_, err := convertSinglePropertyToGeminiSchema(prop)
	if err == nil {
		t.Error("Expected Gemini to return an error for Any property type")
	}

	expectedErrMsg := "Gemini does not support 'any' type properties"
	if err.Error() != expectedErrMsg+". Consider using anyOf with specific types or a more specific type" {
		t.Errorf("Expected specific error message, got: %v", err)
	}
}

func TestOpenAI_AnyPropertyType_InComplexSchema(t *testing.T) {
	// Test Any type within a complex schema (object with mixed property types)
	tool := Tool{
		Name:        "flexible_tool",
		Description: "A tool with flexible parameters",
		InputSchema: InputSchema{
			Type: Object,
			Properties: map[string]Property{
				"fixed_string": {
					Type:        String,
					Description: "A fixed string parameter",
				},
				"flexible_value": {
					Type:        Any,
					Description: "Can be any type",
				},
				"number_param": {
					Type:        Number,
					Description: "A number parameter",
				},
			},
			Required: []string{"fixed_string"},
		},
	}

	openaiTool := convertToolToOpenAI(tool)

	// Check that the parameters are correctly converted
	params := map[string]interface{}(openaiTool.Function.Parameters)

	properties, ok := params["properties"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected properties to be map[string]interface{}, got %T", params["properties"])
	}

	// fixed_string should have type
	fixedStringProp := properties["fixed_string"].(map[string]interface{})
	if fixedStringProp["type"] != "string" {
		t.Errorf("Expected fixed_string to have type 'string', got %v", fixedStringProp["type"])
	}

	// flexible_value should NOT have type
	flexibleValueProp := properties["flexible_value"].(map[string]interface{})
	if _, hasType := flexibleValueProp["type"]; hasType {
		t.Error("Expected flexible_value (Any type) to not have 'type' field")
	}
	if flexibleValueProp["description"] != "Can be any type" {
		t.Errorf("Expected flexible_value description, got %v", flexibleValueProp["description"])
	}

	// number_param should have type
	numberProp := properties["number_param"].(map[string]interface{})
	if numberProp["type"] != "number" {
		t.Errorf("Expected number_param to have type 'number', got %v", numberProp["type"])
	}
}
