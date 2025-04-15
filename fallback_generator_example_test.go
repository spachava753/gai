package gai_test

import (
	"context"
	"fmt"

	"github.com/spachava753/gai"
)

func ExampleFallbackGenerator_Generate() {
	// This example shows how to create a fallback generator that first tries a primary generator,
	// and if that fails with rate limiting or 5xx errors, falls back to a secondary generator.

	// Create mock generators for example purposes
	primaryGen := &MockGenerator{name: "Primary Generator"}
	secondaryGen := &MockGenerator{name: "Secondary Generator"}

	// Create the fallback generator
	// By default, it will fallback on rate limits and 5xx errors
	fallbackGen, err := gai.NewFallbackGenerator(
		[]gai.Generator{primaryGen, secondaryGen},
		nil, // Use default config
	)
	if err != nil {
		fmt.Println("Error creating fallback generator:", err)
		return
	}

	// Create a dialog
	dialog := gai.Dialog{
		{
			Role: gai.User,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      gai.Str("What are the best practices for implementing fallback strategies in AI systems?"),
				},
			},
		},
	}

	// Generate a response
	// The fallback generator will try the primary generator first, and if that fails with a rate limit or 5xx error,
	// it will automatically try the secondary generator instead.
	response, err := fallbackGen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Generation failed:", err)
		return
	}

	// Print the response
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println("Response:", response.Candidates[0].Blocks[0].Content)
	}
}

func ExampleFallbackGenerator_Generate_customFallbackConfig() {
	// This example shows how to create a fallback generator with a custom configuration
	// that falls back on specific HTTP status codes including 400 errors.

	// Create mock generators for example purposes
	mockGen1 := &MockGenerator{name: "Primary Generator"}
	mockGen2 := &MockGenerator{name: "Fallback Generator"}

	// Create a fallback config that also fallbacks on 400 errors
	customConfig := gai.NewHTTPStatusFallbackConfig(400, 429, 500, 502, 503)

	// Create the fallback generator with the custom config
	fallbackGen, err := gai.NewFallbackGenerator(
		[]gai.Generator{mockGen1, mockGen2},
		&customConfig,
	)
	if err != nil {
		fmt.Println("Error creating fallback generator:", err)
		return
	}

	// Use the fallback generator
	dialog := gai.Dialog{
		{
			Role: gai.User,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      gai.Str("Hello"),
				},
			},
		},
	}

	response, err := fallbackGen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Generation failed:", err)
		return
	}

	// Print the response
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println("Response:", response.Candidates[0].Blocks[0].Content)
	}
}

// MockGenerator is a simple mock implementation of the Generator interface for example purposes
type MockGenerator struct {
	name string
}

func (m *MockGenerator) Generate(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
	return gai.Response{
		Candidates: []gai.Message{
			{
				Role: gai.Assistant,
				Blocks: []gai.Block{
					{
						BlockType:    gai.Content,
						ModalityType: gai.Text,
						Content:      gai.Str(fmt.Sprintf("Response from %s", m.name)),
					},
				},
			},
		},
		FinishReason: gai.EndTurn,
	}, nil
}
