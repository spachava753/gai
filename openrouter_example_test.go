package gai

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func ExampleOpenRouterGenerator_Generate() {
	// Create an OpenAI client configured for OpenRouter
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENROUTER_API_KEY env]")
		return
	}

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
		panic(err.Error())
	}
	// The exact response text may vary, so we'll just print a placeholder
	fmt.Println("Response received")

	// Customize generation parameters
	opts := GenOpts{
		MaxGenerationTokens: 10000,
	}
	resp, err = gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(len(resp.Candidates))

	// Output: Response received
	// 1
}

func ExampleOpenRouterGenerator_Generate_image() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENROUTER_API_KEY env]")
		return
	}

	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.jpg]")
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

	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: 512})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(resp.Candidates) != 1 {
		panic("Expected 1 candidate, got " + fmt.Sprint(len(resp.Candidates)))
	}
	if len(resp.Candidates[0].Blocks) < 1 {
		panic("Expected at least 1 block, got " + fmt.Sprint(len(resp.Candidates[0].Blocks)))
	}
	fmt.Println(strings.Contains(resp.Candidates[0].Blocks[0].Content.String(), "Crood"))
	// Output: true
}

func ExampleOpenRouterGenerator_Register() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENROUTER_API_KEY env]")
		return
	}

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
				panic(err)
			}
			return schema
		}(),
	}
	if err := gen.Register(tickerTool); err != nil {
		fmt.Println("Error:", err)
		return
	}

	dialog := Dialog{
		{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}},
	}

	// Force the tool call
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ToolChoice: "get_stock_price"})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
		fmt.Println("Error: empty response")
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
	fmt.Println(toolCall.Content)

	// Append tool result and continue the conversation
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{ID: toolCall.ID, BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")},
		},
	})

	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		fmt.Println(resp.Candidates[0].Blocks[0].Content)
	}

	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// 123.45
}

func ExampleOpenRouterGenerator_Generate_reasoningModel() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENROUTER_API_KEY env]")
		return
	}

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
		fmt.Println("Error:", err)
		return
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
			fmt.Println("Thinking blocks found")
		}

		// Find the main content block (not thinking)
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "12") {
					fmt.Println("Correct answer found")
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
		fmt.Println("Error:", err)
		return
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
			fmt.Println("Thinking blocks found")
		}

		// Find the main content block (not thinking)
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "15") {
					fmt.Println("Correct answer found")
				}
				break
			}
		}
	}

	// Output: Thinking blocks found
	// Correct answer found
	// Thinking blocks found
	// Correct answer found
}

func ExampleOpenRouterGenerator_Generate_invalidModel() {
	// This example demonstrates handling of invalid model IDs with OpenRouter.
	// OpenRouter returns a 400 status code with error details in the response body
	// for invalid requests like nonsense model IDs.
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENROUTER_API_KEY env]")
		return
	}

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
	if err != nil {
		var apiErr ApiErr
		if errors.As(err, &apiErr) {
			fmt.Println("Handled error")
		} else {
			fmt.Println("Unexpected error type")
		}
		return
	}
	panic("unreachable")

	// Output: Handled error
}
