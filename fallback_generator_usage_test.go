package gai_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/spachava753/gai"
)

func TestFallbackUsageScenarios(t *testing.T) {
	t.Run("FallbackGenerator/Generate/Example", func(t *testing.T) {
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
			t.Fatalf("create fallback generator: %v", err)
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
		response, err := fallbackGen.Generate(context.Background(), gai.GenerationRequest{Dialog: dialog})
		if err != nil {
			t.Fatalf("generate response: %v", err)
		}

		if len(response.Candidates) == 0 || len(response.Candidates[0].Blocks) == 0 {
			t.Fatal("expected response with at least one block")
		}
		if got, want := response.Candidates[0].Blocks[0].Content.String(), "Response from Primary Generator"; got != want {
			t.Fatalf("response content = %q, want %q", got, want)
		}
	})
	t.Run("FallbackGenerator/Generate/customFallbackConfig", func(t *testing.T) {
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
			t.Fatalf("create fallback generator: %v", err)
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

		response, err := fallbackGen.Generate(context.Background(), gai.GenerationRequest{Dialog: dialog})
		if err != nil {
			t.Fatalf("generate response: %v", err)
		}

		if len(response.Candidates) == 0 || len(response.Candidates[0].Blocks) == 0 {
			t.Fatal("expected response with at least one block")
		}
		if got, want := response.Candidates[0].Blocks[0].Content.String(), "Response from Primary Generator"; got != want {
			t.Fatalf("response content = %q, want %q", got, want)
		}
	})
}

// MockGenerator is a simple mock implementation of the Generator interface for example purposes
type MockGenerator struct {
	name string
}

func (m *MockGenerator) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
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
