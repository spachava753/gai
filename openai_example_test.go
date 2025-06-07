package gai

import (
	"context"
	"encoding/base64"
	"fmt"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"os"
	"strings"
)

func ExampleOpenAiGenerator_Generate_image() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}
	imgBytes, err := os.ReadFile("Guycrood.jpg")
	if err != nil {
		fmt.Println("[Skipped: could not open Guycrood.jpg]")
		return
	}
	imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4o,
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

func ExampleOpenAiGenerator_Generate_audio() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}

	audioBytes, err := os.ReadFile("sample.wav")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.wav]")
		return
	}
	// Encode as base64 for inline audio usage
	audioBase64 := Str(base64.StdEncoding.EncodeToString(audioBytes))

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oAudioPreview,
		"You are a helpful assistant.",
	)

	// Using inline audio data
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Audio,
					MimeType:     "audio/wav",
					Content:      audioBase64,
				},
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is the name of person in the greeting in this audio? Return a one word response of the name"),
				},
			},
		},
	}

	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{
		MaxGenerationTokens: 128,
	})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		fmt.Println(strings.ToLower(resp.Candidates[0].Blocks[0].Content.String()))
	}

	// Output: friday
}

func ExampleOpenAiGenerator_Generate() {
	// Create an OpenAI client
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

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

	// Output: Response received
	// 2
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

func ExampleOpenAiGenerator_Count() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Create a generator
	generator := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4o,
		"You are a helpful assistant.",
	)

	// Create a dialog with a user message
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is the capital of France?"),
				},
			},
		},
	}

	// Count tokens in the dialog
	tokenCount, err := generator.Count(context.Background(), dialog)
	if err != nil {
		fmt.Printf("Error counting tokens: %v\n", err)
		return
	}

	fmt.Printf("Dialog contains %d tokens\n", tokenCount)

	// Add a response to the dialog
	dialog = append(dialog, Message{
		Role: Assistant,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("The capital of France is Paris. It's known as the 'City of Light' and is famous for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."),
			},
		},
	})

	// Count tokens in the updated dialog
	tokenCount, err = generator.Count(context.Background(), dialog)
	if err != nil {
		fmt.Printf("Error counting tokens: %v\n", err)
		return
	}

	fmt.Printf("Dialog with response contains %d tokens\n", tokenCount)

	// Output: Dialog contains 13 tokens
	// Dialog with response contains 48 tokens
}

func ExampleOpenAiGenerator_Generate_pdf() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}

	pdfBytes, err := os.ReadFile("sample.pdf")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.wav]")
		return
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4_1,
		"You are a helpful assistant.",
	)

	// Create a dialog with PDF content
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("What is the title of this PDF? Just output the title and nothing else"),
				PDFBlock(pdfBytes, "sample.pdf"),
			},
		},
	}

	// Generate a response
	ctx := context.Background()
	response, err := gen.Generate(ctx, dialog, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// The response would contain the model's analysis of the PDF
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println(response.Candidates[0].Blocks[0].Content)
	}
	// Output: Attention Is All You Need
}
