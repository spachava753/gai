package gai

import (
	"reflect"
	"testing"

	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// sortFieldsFn creates an option for cmp.Diff to ignore the order of fields in maps
var sortFieldsFn = cmpopts.SortMaps(func(a, b string) bool {
	return a < b
})

func TestConvertToolToAnthropic(t *testing.T) {
	tests := []struct {
		name string
		tool Tool
		want a.ToolParam
	}{
		{
			name: "Simple tool with single parameter",
			tool: Tool{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				InputSchema: InputSchema{
					Type: Object,
					Properties: map[string]Property{
						"location": {
							Type:        String,
							Description: "The city and state, e.g. San Francisco, CA",
						},
					},
					Required: []string{"location"},
				},
			},
			want: a.ToolParam{
				Name:        "get_weather",
				Description: a.String("Get the weather for a location"),
				InputSchema: a.ToolInputSchemaParam{
					Properties: map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The city and state, e.g. San Francisco, CA",
						},
					},
					ExtraFields: map[string]interface{}{
						"required": []string{"location"},
					},
				},
			},
		},
		{
			name: "Tool with multiple parameters and enum",
			tool: Tool{
				Name:        "get_stock_price",
				Description: "Get the current stock price",
				InputSchema: InputSchema{
					Type: Object,
					Properties: map[string]Property{
						"ticker": {
							Type:        String,
							Description: "Stock ticker symbol",
							Enum:        []string{"AAPL", "GOOGL", "MSFT"},
						},
						"currency": {
							Type:        String,
							Description: "Currency to show price in",
							Enum:        []string{"USD", "EUR", "GBP"},
						},
					},
					Required: []string{"ticker"},
				},
			},
			want: a.ToolParam{
				Name:        "get_stock_price",
				Description: a.String("Get the current stock price"),
				InputSchema: a.ToolInputSchemaParam{
					Properties: map[string]interface{}{
						"ticker": map[string]interface{}{
							"type":        "string",
							"description": "Stock ticker symbol",
							"enum":        []string{"AAPL", "GOOGL", "MSFT"},
						},
						"currency": map[string]interface{}{
							"type":        "string",
							"description": "Currency to show price in",
							"enum":        []string{"USD", "EUR", "GBP"},
						},
					},
					ExtraFields: map[string]interface{}{
						"required": []string{"ticker"},
					},
				},
			},
		},
		{
			name: "Tool with nested object parameter",
			tool: Tool{
				Name:        "create_user",
				Description: "Create a new user",
				InputSchema: InputSchema{
					Type: Object,
					Properties: map[string]Property{
						"name": {
							Type:        String,
							Description: "User's full name",
						},
						"address": {
							Type:        Object,
							Description: "User's address",
							Properties: map[string]Property{
								"street": {
									Type:        String,
									Description: "Street address",
								},
								"city": {
									Type:        String,
									Description: "City",
								},
								"zip": {
									Type:        String,
									Description: "Zip code",
								},
							},
							Required: []string{"street", "city"},
						},
					},
					Required: []string{"name", "address"},
				},
			},
			want: a.ToolParam{
				Name:        "create_user",
				Description: a.String("Create a new user"),
				InputSchema: a.ToolInputSchemaParam{
					Properties: map[string]interface{}{
						"name": map[string]interface{}{
							"type":        "string",
							"description": "User's full name",
						},
						"address": map[string]interface{}{
							"type":        "object",
							"description": "User's address",
							"properties": map[string]interface{}{
								"street": map[string]interface{}{
									"type":        "string",
									"description": "Street address",
								},
								"city": map[string]interface{}{
									"type":        "string",
									"description": "City",
								},
								"zip": map[string]interface{}{
									"type":        "string",
									"description": "Zip code",
								},
							},
							"required": []string{"street", "city"},
						},
					},
					ExtraFields: map[string]interface{}{
						"required": []string{"name", "address"},
					},
				},
			},
		},
		{
			name: "Tool with array parameter",
			tool: Tool{
				Name:        "summarize_documents",
				Description: "Summarize multiple documents",
				InputSchema: InputSchema{
					Type: Object,
					Properties: map[string]Property{
						"documents": {
							Type:        Array,
							Description: "Array of documents to summarize",
							Items: &Property{
								Type:        String,
								Description: "Document content",
							},
						},
						"max_length": {
							Type:        Integer,
							Description: "Maximum length of summary",
						},
					},
					Required: []string{"documents"},
				},
			},
			want: a.ToolParam{
				Name:        "summarize_documents",
				Description: a.String("Summarize multiple documents"),
				InputSchema: a.ToolInputSchemaParam{
					Properties: map[string]interface{}{
						"documents": map[string]interface{}{
							"type":        "array",
							"description": "Array of documents to summarize",
							"items": map[string]interface{}{
								"type":        "string",
								"description": "Document content",
							},
						},
						"max_length": map[string]interface{}{
							"type":        "integer",
							"description": "Maximum length of summary",
						},
					},
					ExtraFields: map[string]interface{}{
						"required": []string{"documents"},
					},
				},
			},
		},
		{
			name: "Tool with different property types",
			tool: Tool{
				Name:        "calculate_mortgage",
				Description: "Calculate mortgage payment",
				InputSchema: InputSchema{
					Type: Object,
					Properties: map[string]Property{
						"principal": {
							Type:        Number,
							Description: "Loan amount",
						},
						"interest_rate": {
							Type:        Number,
							Description: "Annual interest rate (percentage)",
						},
						"term_years": {
							Type:        Integer,
							Description: "Loan term in years",
						},
						"insurance_required": {
							Type:        Boolean,
							Description: "Whether mortgage insurance is required",
						},
					},
					Required: []string{"principal", "interest_rate", "term_years"},
				},
			},
			want: a.ToolParam{
				Name:        "calculate_mortgage",
				Description: a.String("Calculate mortgage payment"),
				InputSchema: a.ToolInputSchemaParam{
					Properties: map[string]interface{}{
						"principal": map[string]interface{}{
							"type":        "number",
							"description": "Loan amount",
						},
						"interest_rate": map[string]interface{}{
							"type":        "number",
							"description": "Annual interest rate (percentage)",
						},
						"term_years": map[string]interface{}{
							"type":        "integer",
							"description": "Loan term in years",
						},
						"insurance_required": map[string]interface{}{
							"type":        "boolean",
							"description": "Whether mortgage insurance is required",
						},
					},
					ExtraFields: map[string]interface{}{
						"required": []string{"principal", "interest_rate", "term_years"},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertToolToAnthropic(tt.tool)

			// Use deep comparison for complex structures
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertToolToAnthropic() mismatch; want:\n%s\ngot:\n%s\n", tt.want, got)
			}
		})
	}
}

// TestConvertToolToAnthropicPanics verifies that the function panics with invalid input
func TestConvertToolToAnthropicPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("convertToolToAnthropic() did not panic with invalid tool type")
		}
	}()

	// This should panic because InputSchema.Type is not Object
	invalidTool := Tool{
		Name:        "invalid_tool",
		Description: "This will cause a panic",
		InputSchema: InputSchema{
			Type: String, // Not Object, which should cause panic
		},
	}

	convertToolToAnthropic(invalidTool)
}

func TestConvertPropertyToAnthropicMap(t *testing.T) {
	tests := []struct {
		name     string
		property Property
		want     map[string]interface{}
	}{
		{
			name: "String property with enum",
			property: Property{
				Type:        String,
				Description: "A string property",
				Enum:        []string{"option1", "option2", "option3"},
			},
			want: map[string]interface{}{
				"type":        "string",
				"description": "A string property",
				"enum":        []string{"option1", "option2", "option3"},
			},
		},
		{
			name: "Number property",
			property: Property{
				Type:        Number,
				Description: "A number property",
			},
			want: map[string]interface{}{
				"type":        "number",
				"description": "A number property",
			},
		},
		{
			name: "Object property with nested properties",
			property: Property{
				Type:        Object,
				Description: "An object property",
				Properties: map[string]Property{
					"nested_string": {
						Type:        String,
						Description: "A nested string",
					},
					"nested_number": {
						Type:        Number,
						Description: "A nested number",
					},
				},
				Required: []string{"nested_string"},
			},
			want: map[string]interface{}{
				"type":        "object",
				"description": "An object property",
				"properties": map[string]interface{}{
					"nested_string": map[string]interface{}{
						"type":        "string",
						"description": "A nested string",
					},
					"nested_number": map[string]interface{}{
						"type":        "number",
						"description": "A nested number",
					},
				},
				"required": []string{"nested_string"},
			},
		},
		{
			name: "Array property with items",
			property: Property{
				Type:        Array,
				Description: "An array property",
				Items: &Property{
					Type:        String,
					Description: "Array item",
				},
			},
			want: map[string]interface{}{
				"type":        "array",
				"description": "An array property",
				"items": map[string]interface{}{
					"type":        "string",
					"description": "Array item",
				},
			},
		},
		{
			name: "Boolean property",
			property: Property{
				Type:        Boolean,
				Description: "A boolean property",
			},
			want: map[string]interface{}{
				"type":        "boolean",
				"description": "A boolean property",
			},
		},
		{
			name: "Integer property",
			property: Property{
				Type:        Integer,
				Description: "An integer property",
			},
			want: map[string]interface{}{
				"type":        "integer",
				"description": "An integer property",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertPropertyToAnthropicMap(tt.property)

			// Use deep comparison
			if diff := cmp.Diff(tt.want, got, sortFieldsFn); diff != "" {
				t.Errorf("convertPropertyToAnthropicMap() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
