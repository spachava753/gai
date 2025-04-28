package gai

import (
	"context"
	"encoding/base64"
	"fmt"
	a "github.com/anthropics/anthropic-sdk-go"
	"os"
	"strings"
)

func ExampleAnthropicGenerator_Generate() {
	// Create an Anthropic client
	client := a.NewClient()

	// Demonstration of how to enable system prompt caching
	svc := NewAnthropicServiceWrapper(&client.Messages, EnableSystemCaching)

	// Instantiate an Anthropic Generator
	gen := NewAnthropicGenerator(svc, a.ModelClaude3_5HaikuLatest, "You are a helpful assistant")
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
	// Note that anthropic generator requires that max generation tokens generation param be set
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: 1024})
	if err != nil {
		panic(err.Error())
	}
	// The exact response text may vary, so we'll just print a placeholder
	fmt.Println("Response received")

	// Customize generation parameters
	opts := GenOpts{
		Temperature:         0.7,
		MaxGenerationTokens: 1024,
	}
	resp, err = gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(len(resp.Candidates))

	// Output: Response received
	// 1
}

func ExampleAnthropicGenerator_Generate_thinking() {
	// Create an Anthropic client
	client := a.NewClient()

	// Instantiate an Anthropic Generator
	gen := NewAnthropicGenerator(&client.Messages, a.ModelClaude3_7SonnetLatest, "You are a helpful assistant")
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

	// Use thinking
	opts := GenOpts{
		Temperature:         1.0,
		MaxGenerationTokens: 9000,
		ThinkingBudget:      "5000",
	}
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(len(resp.Candidates))

	dialog = append(dialog, resp.Candidates[0], Message{
		Role: User,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("What can you do?"),
			},
		},
	})
	resp, err = gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(len(resp.Candidates))

	// Output: 1
	// 1
}

func ExampleAnthropicGenerator_Generate_image() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set ANTHROPIC_API_KEY env]")
		return
	}

	// This example assumes that Guycrood.jpg is present in the current directory.
	imgBytes, err := os.ReadFile("Guycrood.jpg")
	if err != nil {
		fmt.Println("[Skipped: could not open Guycrood.jpg]")
		return
	}
	imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))

	client := a.NewClient()

	gen := NewAnthropicGenerator(
		&client.Messages,
		a.ModelClaude3_7SonnetLatest,
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
	if len(resp.Candidates[0].Blocks) != 1 {
		panic("Expected 1 block, got " + fmt.Sprint(len(resp.Candidates[0].Blocks)))
	}
	fmt.Println(strings.Contains(resp.Candidates[0].Blocks[0].Content.String(), "Crood"))
	// Output: true
}

func ExampleAnthropicGenerator_Register() {
	// Create an Anthropic client
	client := a.NewClient()

	// Demonstration of how to enable system and multi turn message prompt caching
	svc := NewAnthropicServiceWrapper(&client.Messages, EnableSystemCaching, EnableMultiTurnCaching)

	// Instantiate an Anthropic Generator
	gen := NewAnthropicGenerator(
		svc,
		a.ModelClaude_3_5_Sonnet_20240620,
		`You are a helpful assistant that returns the price of a stock and nothing else.

Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`,
	)

	// Register tools
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: InputSchema{
			Type: Object,
			Properties: map[string]Property{
				"ticker": {
					Type:        String,
					Description: "The stock ticker symbol, e.g. AAPL for Apple Inc.",
				},
			},
			Required: []string{"ticker"},
		},
	}
	if err := gen.Register(tickerTool); err != nil {
		panic(err.Error())
	}

	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is the price of Apple stock?"),
				},
			},
		},
	}

	// Customize generation parameters
	opts := GenOpts{
		ToolChoice:          "get_stock_price", // Can specify a specific tool to force invoke
		MaxGenerationTokens: 8096,
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[0].ID,
				ModalityType: Text,
				Content:      Str("123.45"),
			},
		},
	})

	resp, err = gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: 8096})
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// 123.45
}

func ExampleAnthropicGenerator_Register_parallelToolUse() {
	// Create an Anthropic client
	client := a.NewClient()

	// Register tools
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: InputSchema{
			Type: Object,
			Properties: map[string]Property{
				"ticker": {
					Type:        String,
					Description: "The stock ticker symbol, e.g. AAPL for Apple Inc.",
				},
			},
			Required: []string{"ticker"},
		},
	}

	// Instantiate an Anthropic Generator
	gen := NewAnthropicGenerator(
		&client.Messages,
		a.ModelClaude3_5SonnetLatest,
		`You are a helpful assistant that compares the price of two stocks and returns the ticker of whichever is greater. 
Only mention one of the stock tickers and nothing else.

Only output the price, like
<example>
User: Which one is more expensive? Apple or NVidia?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: Apple: 123.45; Nvidia: 345.65
Assistant: Nvidia
</example>

<example>
User: Which one is more expensive? Microsft or Netflix?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: MSFT: 876.45; NFLX: 345.65
Assistant: MSFT
</example>
`,
	)

	// Register tools
	tickerTool.Description += "\nYou can call this tool in parallel"
	if err := gen.Register(tickerTool); err != nil {
		panic(err.Error())
	}

	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Which stock, Apple vs. Microsoft, is more expensive?"),
				},
			},
		},
	}

	// Generate a response
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: 8096})
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)
	fmt.Println(resp.Candidates[0].Blocks[1].Content)

	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[0].ID,
				ModalityType: Text,
				Content:      Str("123.45"),
			},
		},
	}, Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[1].ID,
				ModalityType: Text,
				Content:      Str("678.45"),
			},
		},
	})

	resp, err = gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: 8096})
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// {"name":"get_stock_price","parameters":{"ticker":"MSFT"}}
	// MSFT
}
