package openai

import (
	"github.com/google/go-cmp/cmp"
	oai "github.com/openai/openai-go"
	"github.com/spachava753/gai"
	"testing"
)

func TestConvertToolToOpenAI(t *testing.T) {
	tests := []struct {
		name string
		tool gai.Tool
		want oai.ChatCompletionToolParam
	}{
		{
			name: "simple tool with no parameters",
			tool: gai.Tool{
				Name:        "get_server_time",
				Description: "Get the current server time in UTC.",
			},
			want: oai.ChatCompletionToolParam{
				Type: oai.F(oai.ChatCompletionToolTypeFunction),
				Function: oai.F(oai.FunctionDefinitionParam{
					Name:        oai.F("get_server_time"),
					Description: oai.F("Get the current server time in UTC."),
				}),
			},
		},
		{
			name: "tool with single required string parameter",
			tool: gai.Tool{
				Name:        "get_stock_price",
				Description: "Get the current stock price for a given ticker symbol.",
				InputSchema: gai.InputSchema{
					Type: gai.Object,
					Properties: map[string]gai.Property{
						"ticker": {
							Type:        gai.String,
							Description: "The stock ticker symbol, e.g. AAPL for Apple Inc.",
						},
					},
					Required: []string{"ticker"},
				},
			},
			want: oai.ChatCompletionToolParam{
				Type: oai.F(oai.ChatCompletionToolTypeFunction),
				Function: oai.F(oai.FunctionDefinitionParam{
					Name:        oai.F("get_stock_price"),
					Description: oai.F("Get the current stock price for a given ticker symbol."),
					Parameters: oai.F(oai.FunctionParameters(map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"ticker": map[string]interface{}{
								"type":        "string",
								"description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
							},
						},
						"required": []string{"ticker"},
					})),
				}),
			},
		},
		{
			name: "tool with string enum parameter",
			tool: gai.Tool{
				Name:        "get_weather",
				Description: "Get the current weather in a given location",
				InputSchema: gai.InputSchema{
					Type: gai.Object,
					Properties: map[string]gai.Property{
						"location": {
							Type:        gai.String,
							Description: "The city and state, e.g. San Francisco, CA",
						},
						"unit": {
							Type:        gai.String,
							Enum:        []string{"celsius", "fahrenheit"},
							Description: "The unit of temperature to use",
						},
					},
					Required: []string{"location"},
				},
			},
			want: oai.ChatCompletionToolParam{
				Type: oai.F(oai.ChatCompletionToolTypeFunction),
				Function: oai.F(oai.FunctionDefinitionParam{
					Name:        oai.F("get_weather"),
					Description: oai.F("Get the current weather in a given location"),
					Parameters: oai.F(oai.FunctionParameters(map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The city and state, e.g. San Francisco, CA",
							},
							"unit": map[string]interface{}{
								"type":        "string",
								"description": "The unit of temperature to use",
								"enum":        []string{"celsius", "fahrenheit"},
							},
						},
						"required": []string{"location"},
					})),
				}),
			},
		},
		{
			name: "tool with array parameter",
			tool: gai.Tool{
				Name:        "get_batch_stock_prices",
				Description: "Get the current stock prices for a list of ticker symbols.",
				InputSchema: gai.InputSchema{
					Type: gai.Object,
					Properties: map[string]gai.Property{
						"tickers": {
							Type:        gai.Array,
							Description: "List of stock ticker symbols, e.g. ['AAPL', 'GOOGL', 'MSFT']",
							Items: &gai.Property{
								Type:        gai.String,
								Description: "A stock ticker symbol",
							},
						},
					},
					Required: []string{"tickers"},
				},
			},
			want: oai.ChatCompletionToolParam{
				Type: oai.F(oai.ChatCompletionToolTypeFunction),
				Function: oai.F(oai.FunctionDefinitionParam{
					Name:        oai.F("get_batch_stock_prices"),
					Description: oai.F("Get the current stock prices for a list of ticker symbols."),
					Parameters: oai.F(oai.FunctionParameters(map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"tickers": map[string]interface{}{
								"type":        "array",
								"description": "List of stock ticker symbols, e.g. ['AAPL', 'GOOGL', 'MSFT']",
								"items": map[string]interface{}{
									"type":        "string",
									"description": "A stock ticker symbol",
								},
							},
						},
						"required": []string{"tickers"},
					})),
				}),
			},
		},
		{
			name: "tool with nested object parameter",
			tool: gai.Tool{
				Name:        "create_user",
				Description: "Create a new user with the given details.",
				InputSchema: gai.InputSchema{
					Type: gai.Object,
					Properties: map[string]gai.Property{
						"user": {
							Type:        gai.Object,
							Description: "User details",
							Properties: map[string]gai.Property{
								"name": {
									Type:        gai.String,
									Description: "User's full name",
								},
								"age": {
									Type:        gai.Integer,
									Description: "User's age in years",
								},
								"settings": {
									Type:        gai.Object,
									Description: "User preferences",
									Properties: map[string]gai.Property{
										"newsletter": {
											Type:        gai.Boolean,
											Description: "Whether to subscribe to newsletter",
										},
										"theme": {
											Type:        gai.String,
											Description: "UI theme preference",
											Enum:        []string{"light", "dark", "system"},
										},
									},
									Required: []string{"newsletter"},
								},
							},
							Required: []string{"name", "age"},
						},
					},
					Required: []string{"user"},
				},
			},
			want: oai.ChatCompletionToolParam{
				Type: oai.F(oai.ChatCompletionToolTypeFunction),
				Function: oai.F(oai.FunctionDefinitionParam{
					Name:        oai.F("create_user"),
					Description: oai.F("Create a new user with the given details."),
					Parameters: oai.F(oai.FunctionParameters(map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"user": map[string]interface{}{
								"type":        "object",
								"description": "User details",
								"properties": map[string]interface{}{
									"name": map[string]interface{}{
										"type":        "string",
										"description": "User's full name",
									},
									"age": map[string]interface{}{
										"type":        "integer",
										"description": "User's age in years",
									},
									"settings": map[string]interface{}{
										"type":        "object",
										"description": "User preferences",
										"properties": map[string]interface{}{
											"newsletter": map[string]interface{}{
												"type":        "boolean",
												"description": "Whether to subscribe to newsletter",
											},
											"theme": map[string]interface{}{
												"type":        "string",
												"description": "UI theme preference",
												"enum":        []string{"light", "dark", "system"},
											},
										},
										"required": []string{"newsletter"},
									},
								},
								"required": []string{"name", "age"},
							},
						},
						"required": []string{"user"},
					})),
				}),
			},
		},
		{
			name: "tool with all primitive types",
			tool: gai.Tool{
				Name:        "test_primitives",
				Description: "Test all primitive types.",
				InputSchema: gai.InputSchema{
					Type: gai.Object,
					Properties: map[string]gai.Property{
						"string_val": {
							Type:        gai.String,
							Description: "A string value",
						},
						"int_val": {
							Type:        gai.Integer,
							Description: "An integer value",
						},
						"number_val": {
							Type:        gai.Number,
							Description: "A floating point value",
						},
						"bool_val": {
							Type:        gai.Boolean,
							Description: "A boolean value",
						},
						"null_val": {
							Type:        gai.Null,
							Description: "A null value",
						},
					},
				},
			},
			want: oai.ChatCompletionToolParam{
				Type: oai.F(oai.ChatCompletionToolTypeFunction),
				Function: oai.F(oai.FunctionDefinitionParam{
					Name:        oai.F("test_primitives"),
					Description: oai.F("Test all primitive types."),
					Parameters: oai.F(oai.FunctionParameters(map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"string_val": map[string]interface{}{
								"type":        "string",
								"description": "A string value",
							},
							"int_val": map[string]interface{}{
								"type":        "integer",
								"description": "An integer value",
							},
							"number_val": map[string]interface{}{
								"type":        "number",
								"description": "A floating point value",
							},
							"bool_val": map[string]interface{}{
								"type":        "boolean",
								"description": "A boolean value",
							},
							"null_val": map[string]interface{}{
								"type":        "null",
								"description": "A null value",
							},
						},
						"required": []string(nil),
					})),
				}),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertToolToOpenAI(tt.tool)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("convertToolToOpenAI() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}