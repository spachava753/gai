package gai

import (
	"context"
	"fmt"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"os"
)

func ExampleOpenAiGenerator_Generate() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions, openai.ChatModelGPT4oMini, "You are a helpful assistant")
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
		TopK:                10,
		N:                   2, // Set N to a value higher than 1 to generate multiple responses in a single request
		MaxGenerationTokens: 1024,
	}
	resp, err = gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(len(resp.Candidates))

	// Create an OpenAI client for open router
	client = openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1/"),
		option.WithAPIKey(os.Getenv("OPEN_ROUTER_API_KEY")),
	)

	// Instantiate a OpenAI Generator
	gen = NewOpenAiGenerator(
		&client.Chat.Completions,
		"google/gemini-2.5-pro-preview-03-25",
		"You are a helpful assistant",
	)
	dialog = Dialog{
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

	// Customize generation parameters
	opts = GenOpts{
		MaxGenerationTokens: 1024,
	}

	// Generate a response
	resp, err = gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	// The exact response text may vary, so we'll just print a placeholder
	fmt.Println("Response received")

	// Output: Response received
	// 2
	// Response received
}

func ExampleOpenAiGenerator_Generate_openRouter() {

	// Create an OpenAI client for open router
	client := openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1/"),
		option.WithAPIKey(os.Getenv("OPEN_ROUTER_API_KEY")),
	)

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		"google/gemini-2.5-pro-preview-03-25",
		"You are a helpful assistant",
	)
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

	// Customize generation parameters
	opts := GenOpts{
		MaxGenerationTokens: 1024,
	}

	// Generate a response
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	// The exact response text may vary, so we'll just print a placeholder
	fmt.Println("Response received")
	fmt.Println(len(resp.Candidates))

	// Output: Response received
	// 1
}

func ExampleOpenAiGenerator_Generate_thinking() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions, openai.ChatModelO3Mini, "You are a helpful assistant")
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

	// Customize generation parameters
	opts := GenOpts{
		MaxGenerationTokens: 4096,
		ThinkingBudget:      "low",
		Temperature:         1.0,
	}

	// Generate a response
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	// The exact response text may vary, so we'll just print a placeholder
	fmt.Println("Response received")

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

	// Output: Response received
	// 1
}

func ExampleOpenAiGenerator_Register() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
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
		ToolChoice: "get_stock_price", // Can specify a specific tool to force invoke
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

	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// 123.45
}

func ExampleOpenAiGenerator_Register_parallelToolUse() {
	// Create an OpenAI client
	client := openai.NewClient()

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

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		`You are a helpful assistant that compares the price of two stocks and returns the ticker of whichever is greater. 
Only mentioned the ticker and nothing else.

Only output the price, like
<example>
User: Which one is more expensive? Apple or NVidia?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: Apple: 123.45; Nvidia: 345.65
Assistant: Nvidia
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
	resp, err := gen.Generate(context.Background(), dialog, nil)
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

	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// {"name":"get_stock_price","parameters":{"ticker":"MSFT"}}
	// MSFT
}

func ExampleOpenAiGenerator_Register_openRouter() {
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

	// Create an OpenAI client for open router
	client := openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1/"),
		option.WithAPIKey(os.Getenv("OPEN_ROUTER_API_KEY")),
	)

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		"google/gemini-2.5-pro-preview-03-25",
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
		ToolChoice: "get_stock_price", // Can specify a specific tool to force invoke
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

	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// 123.45
}

func ExampleOpenAiGenerator_Register_openRouterParallelToolUse() {
	// Create an OpenAI client
	client := openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1/"),
		option.WithAPIKey(os.Getenv("OPEN_ROUTER_API_KEY")),
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

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		"google/gemini-2.5-pro-preview-03-25",
		`You are a helpful assistant that compares the price of two stocks and returns the ticker of whichever is greater. 
Only mentioned the ticker and nothing else.

Only output the price, like
<example>
User: Which one is more expensive? Apple or NVidia?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: Apple: 123.45; Nvidia: 345.65
Assistant: Nvidia
</example>
`,
	)

	// Register tools
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
	resp, err := gen.Generate(context.Background(), dialog, nil)
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

	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// {"name":"get_stock_price","parameters":{"ticker":"MSFT"}}
	// MSFT
}
