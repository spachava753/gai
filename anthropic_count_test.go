package gai

import (
	"context"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"testing"

	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// mockAnthropicSvc is a mock implementation of AnthropicSvc for testing
type mockAnthropicSvc struct {
	countTokensCalled bool
	lastToolsCount    int
	lastSystemPresent bool
}

func (m *mockAnthropicSvc) NewStreaming(ctx context.Context, params a.MessageNewParams, opts ...option.RequestOption) (stream *ssestream.Stream[a.MessageStreamEventUnion]) {
	return nil
}

func (m *mockAnthropicSvc) CountTokens(ctx context.Context, params a.MessageCountTokensParams, opts ...option.RequestOption) (res *a.MessageTokensCount, err error) {
	m.countTokensCalled = true

	// Check if tools are present
	m.lastToolsCount = len(params.Tools)

	// Check if system is present
	m.lastSystemPresent = len(params.System.OfTextBlockArray) > 0

	// Return mock result
	return &a.MessageTokensCount{
		InputTokens: 10, // Mock value
	}, nil
}

func TestAnthropicGenerator_Count_IncludesTools(t *testing.T) {
	// Create a mock Anthropic service
	mockSvc := &mockAnthropicSvc{}

	// Create a generator with the mock service
	gen := AnthropicGenerator{
		client:             mockSvc,
		model:              "claude-3-haiku-20240307",
		systemInstructions: "You are a helpful assistant",
		tools:              make(map[string]a.ToolParam),
	}

	// Register a tool
	tool := Tool{
		Name:        "test_tool",
		Description: "A test tool",
		InputSchema: InputSchema{
			Type: Object,
			Properties: map[string]Property{
				"arg1": {
					Type:        String,
					Description: "Argument 1",
				},
			},
			Required: []string{"arg1"},
		},
	}
	gen.Register(tool)

	// Create a simple dialog
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hello"),
				},
			},
		},
	}

	// Call Count
	_, err := gen.Count(context.Background(), dialog)
	if err != nil {
		t.Errorf("Count returned error: %v", err)
	}

	// Check that CountTokens was called
	if !mockSvc.countTokensCalled {
		t.Errorf("CountTokens was not called")
	}

	// Check that system instructions were included
	if !mockSvc.lastSystemPresent {
		t.Errorf("System instructions were not included in CountTokens params")
	}

	// Check that system instructions were included
	if mockSvc.lastToolsCount != 1 {
		t.Errorf("Tool definitions were not included in CountTokens params")
	}
}
