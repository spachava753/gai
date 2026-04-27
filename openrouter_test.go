package gai

import (
	"context"
	"encoding/base64"
	"errors"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"os"
	"strings"
	"testing"
)

func TestOpenRouterGenerator_Generate(t *testing.T) {
	// Create an OpenAI client configured for OpenRouter
	apiKey := skipOnMissingEnv(t, "OPENROUTER_API_KEY")
	client := openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1"),
		option.WithAPIKey(apiKey),
	)
	// Instantiate an OpenRouter Generator
	gen := NewOpenRouterGenerator(&client.Chat.Completions, "z-ai/glm-4.6:exacto", "You are a helpful assistant")
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hi!"),
				},
			},
		},
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The exact response text may vary, so we'll just print a placeholder
	// Customize generation parameters
	opts := GenOpts{
		MaxGenerationTokens: Ptr(10000),
	}
	resp, err = gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(resp.Candidates); got == 0 {
		t.Fatal("expected at least one item")
	}
}
func TestOpenRouterGenerator_Generate_image(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "OPENROUTER_API_KEY")
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		t.Skip("could not open sample.jpg")
		return
	}
	imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))
	client := openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1"),
		option.WithAPIKey(apiKey),
	)
	// Use a vision-capable model through OpenRouter
	gen := NewOpenRouterGenerator(
		&client.Chat.Completions,
		"qwen/qwen3-vl-235b-a22b-instruct",
		"You are a helpful assistant.",
	)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Image,
					MimeType:     "image/jpeg",
					Content:      imgBase64,
				},
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is in this image? (Hint, it's a character from The Croods, a DreamWorks animated movie.)"),
				},
			},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: Ptr(512)})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) != 1 {
		t.Fatalf("candidates = %d, want 1", len(resp.Candidates))
	}
	if len(resp.Candidates[0].Blocks) < 1 {
		t.Fatalf("blocks = %d, want at least 1", len(resp.Candidates[0].Blocks))
	}
	if !strings.Contains(resp.Candidates[0].Blocks[0].Content.String(), "Crood") {
		t.Fatalf("content does not contain Crood")
	}
}
func TestOpenRouterGenerator_Register(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "OPENROUTER_API_KEY")
	client := openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1"),
		option.WithAPIKey(apiKey),
	)
	gen := NewOpenRouterGenerator(
		&client.Chat.Completions,
		"moonshotai/kimi-k2-0905:exacto",
		"You are a helpful assistant that returns the price of a stock and nothing else.",
	)
	// Register a tool
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
			}]()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			return schema
		}(),
	}
	if err := gen.Register(tickerTool); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	dialog := Dialog{
		{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}},
	}
	// Force the tool call
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ToolChoice: "get_stock_price"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
		t.Fatal("empty response")
		return
	}
	// Find and print the tool call JSON
	var toolCall Block
	for _, b := range resp.Candidates[0].Blocks {
		if b.BlockType == ToolCall {
			toolCall = b
			break
		}
	}
	if got := toolCall.Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	// Append tool result and continue the conversation
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{ID: toolCall.ID, BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")},
		},
	})
	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
	}
}
func TestOpenRouterGenerator_Generate_reasoningModel(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "OPENROUTER_API_KEY")
	client := openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1"),
		option.WithAPIKey(apiKey),
	)
	// Use a reasoning model through OpenRouter
	// NOTE: Models that support reasoning (like those with extended thinking)
	// will automatically return reasoning_details which are extracted as Thinking blocks
	gen := NewOpenRouterGenerator(
		&client.Chat.Completions,
		"z-ai/glm-4.6:exacto",
		"You are a helpful assistant.",
	)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is the square root of 144?"),
				},
			},
		},
	}
	// Generate response - reasoning models may return thinking blocks automatically
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ThinkingBudget: "low"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks (from reasoning_details)
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
				// Thinking blocks have reasoning metadata in ExtraFields
				if reasoningType, ok := block.ExtraFields["reasoning_type"].(string); ok {
					_ = reasoningType // reasoning.text, reasoning.summary, or reasoning.encrypted
				}
			}
		}
		if hasThinking {
		}
		// Find the main content block (not thinking)
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "12") {
				}
				break
			}
		}
	}
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: User,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("What is the square root of 225?"),
			},
		},
	})
	// Generate response - reasoning models may return thinking blocks automatically
	resp, err = gen.Generate(context.Background(), dialog, &GenOpts{ThinkingBudget: "low"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks (from reasoning_details)
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
				// Thinking blocks have reasoning metadata in ExtraFields
				if reasoningType, ok := block.ExtraFields["reasoning_type"].(string); ok {
					_ = reasoningType // reasoning.text, reasoning.summary, or reasoning.encrypted
				}
			}
		}
		if hasThinking {
		}
		// Find the main content block (not thinking)
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "15") {
				}
				break
			}
		}
	}
}
func TestOpenRouterGenerator_Generate_invalidModel(t *testing.T) {
	// This example demonstrates handling of invalid model IDs with OpenRouter.
	// OpenRouter returns a 400 status code with error details in the response body
	// for invalid requests like nonsense model IDs.
	apiKey := skipOnMissingEnv(t, "OPENROUTER_API_KEY")
	client := openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1"),
		option.WithAPIKey(apiKey),
	)
	// Use a nonsense model ID to trigger an error
	gen := NewOpenRouterGenerator(&client.Chat.Completions, "invalid/model-does-not-exist", "You are helpful")
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
	_, err := gen.Generate(context.Background(), dialog, nil)
	if err == nil {
		t.Fatal("expected invalid model to return an error")
	}
	var apiErr *ApiErr
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *ApiErr", err)
	}
}
