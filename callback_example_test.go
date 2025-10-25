package gai_test

import (
	"context"
	"fmt"
	"slices"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/spachava753/gai"
)

// Define a parameter struct for our weather tool
type WeatherParams struct {
	Location string `json:"location"`
	Unit     string `json:"unit,omitempty"`
}

func (w WeatherParams) Validate() error {
	knownLocs := []string{"San Francisco", "New York", "London"}
	if !slices.Contains(knownLocs, w.Location) {
		return fmt.Errorf("unknown location: %s", w.Location)
	}
	return nil
}

// ExampleToolCallBackFunc demonstrates how to use ToolCallBackFunc to easily create
// tool callbacks with strongly-typed parameters.
func ExampleToolCallBackFunc() {

	// Create a simple weather function that will be wrapped by ToolCallBackFunc
	getWeather := func(ctx context.Context, params WeatherParams) (string, error) {
		unit := "celsius"
		if params.Unit == "fahrenheit" {
			unit = "fahrenheit"
		}

		// In a real implementation, you would call an external weather API here
		temp := 22.5
		if unit == "fahrenheit" {
			temp = temp*9/5 + 32
		}

		return fmt.Sprintf("Weather in %s: %.1f°%s",
				params.Location,
				temp,
				unit[0:1]), // C or F
			nil
	}

	// Create a tool
	weatherTool := gai.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a location",
		InputSchema: func() *jsonschema.Schema {
			schema, err := gai.GenerateSchema[struct {
				Location string `json:"location" jsonschema:"required" jsonschema_description:"The city and state, e.g. San Francisco, CA"`
				Unit     string `json:"unit" jsonschema:"enum=celsius,enum=fahrenheit" jsonschema_description:"The unit of temperature"`
			}]()
			if err != nil {
				panic(err)
			}
			return schema
		}(),
	}

	// Create an instance of the ToolGenerator
	// In a real application, you would use a real generator like OpenAiGenerator
	toolGen := &gai.ToolGenerator{
		G: &ExampleMockGenerator{},
	}

	// Register the tool with the wrapped callback function
	_ = toolGen.Register(weatherTool, gai.ToolCallBackFunc[WeatherParams](getWeather))

	// The tool is now registered and ready to use
	fmt.Println("Weather tool registered successfully")

	// Output: Weather tool registered successfully
}

// ExampleMockGenerator is a simple mock implementation of the ToolCapableGenerator interface
type ExampleMockGenerator struct{}

func (m *ExampleMockGenerator) Generate(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
	return gai.Response{}, nil
}

func (m *ExampleMockGenerator) Register(tool gai.Tool) error {
	return nil
}
