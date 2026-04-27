package gai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

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
func TestToolCallBackFunc(t *testing.T) {
	getWeather := func(ctx context.Context, params WeatherParams) (string, error) {
		unit := "celsius"
		if params.Unit == "fahrenheit" {
			unit = "fahrenheit"
		}

		temp := 22.5
		if unit == "fahrenheit" {
			temp = temp*9/5 + 32
		}

		return fmt.Sprintf("Weather in %s: %.1f°%s", params.Location, temp, unit[0:1]), nil
	}

	weatherTool := gai.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a location",
		InputSchema: func() *jsonschema.Schema {
			schema, err := gai.GenerateSchema[struct {
				Location string `json:"location" jsonschema:"required" jsonschema_description:"The city and state, e.g. San Francisco, CA"`
				Unit     string `json:"unit" jsonschema:"The unit of temperature"`
			}]()
			if err != nil {
				t.Fatalf("generate weather schema: %v", err)
			}
			return schema
		}(),
	}

	callback := gai.ToolCallBackFunc[WeatherParams](getWeather)
	toolGen := &gai.ToolGenerator{G: &ExampleMockGenerator{}}
	if err := toolGen.Register(weatherTool, callback); err != nil {
		t.Fatalf("register weather tool: %v", err)
	}

	t.Run("calls strongly typed callback", func(t *testing.T) {
		msg, err := callback.Call(context.Background(), json.RawMessage(`{"location":"San Francisco","unit":"fahrenheit"}`), "call-1")
		if err != nil {
			t.Fatalf("call weather callback: %v", err)
		}
		if msg.Role != gai.ToolResult {
			t.Fatalf("role = %v, want %v", msg.Role, gai.ToolResult)
		}
		if len(msg.Blocks) != 1 {
			t.Fatalf("blocks = %d, want 1", len(msg.Blocks))
		}
		if got, want := msg.Blocks[0].ID, "call-1"; got != want {
			t.Fatalf("tool call id = %q, want %q", got, want)
		}
		if got, want := msg.Blocks[0].Content.String(), "Weather in San Francisco: 72.5°f"; got != want {
			t.Fatalf("content = %q, want %q", got, want)
		}
	})

	t.Run("validation errors become tool results", func(t *testing.T) {
		msg, err := callback.Call(context.Background(), json.RawMessage(`{"location":"Atlantis"}`), "call-2")
		if err != nil {
			t.Fatalf("validation error should be returned as a tool result, got fatal error: %v", err)
		}
		if len(msg.Blocks) != 1 {
			t.Fatalf("blocks = %d, want 1", len(msg.Blocks))
		}
		if !strings.Contains(msg.Blocks[0].Content.String(), "unknown location: Atlantis") {
			t.Fatalf("tool result content = %q, want validation error", msg.Blocks[0].Content.String())
		}
	})
}

// ExampleMockGenerator is a simple mock implementation of the ToolCapableGenerator interface
type ExampleMockGenerator struct{}

func (m *ExampleMockGenerator) Generate(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
	return gai.Response{}, nil
}

func (m *ExampleMockGenerator) Register(tool gai.Tool) error {
	return nil
}
