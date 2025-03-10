package gai

import (
	"context"
	"fmt"
	"github.com/openai/openai-go"
)

func ExampleOpenAiGenerator_Generate() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(client.Chat.Completions, openai.ChatModelGPT4oMini, "You are a helpful assistant")
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      "Hi!",
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
		TopK:                10,
		N:                   2, // Set N to a value higher than 1 to generate multiple responses in a single request
		MaxGenerationTokens: 1024,
	}
	resp, err = gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(len(resp.Candidates))

	// Output: Response received
	// 2
}

func ExampleOpenAiGenerator_Register() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(
		client.Chat.Completions,
		openai.ChatModelGPT4oMini,
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
	weatherTool := Tool{
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
	if err := gen.Register(weatherTool); err != nil {
		panic(err.Error())
	}

	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      "What is the price of Apple stock?",
				},
			},
		},
	}

	// Customize generation parameters
	opts := GenOpts{
		ToolChoice: "get_stock_price", // Can specify a specific tool to force invoke
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	dialog = append(dialog, resp.Candidates[0], Message{
		Role: Assistant,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[0].ID,
				BlockType:    ToolResult,
				ModalityType: Text,
				Content:      "123.45",
			},
		},
	})

	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// 123.45
}
