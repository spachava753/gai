package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"strings"

	a "github.com/anthropics/anthropic-sdk-go"
)

func ExampleAnthropicGenerator_Generate() {
	// Create an Anthropic client
	client := a.NewClient()

	// Demonstration of how to enable system prompt caching
	svc := NewAnthropicServiceWrapper(&client.Messages, EnableSystemCaching)

	// Instantiate an Anthropic Generator
	gen := NewAnthropicGenerator(svc, string(a.ModelClaude3_5HaikuLatest), "You are a helpful assistant")
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

func ExampleAnthropicGenerator_Stream() {
	// Create an Anthropic client
	client := a.NewClient()

	// Demonstration of how to enable system prompt caching
	svc := NewAnthropicServiceWrapper(&client.Messages, EnableSystemCaching)

	// Instantiate an Anthropic Generator
	gen := NewAnthropicGenerator(svc, string(a.ModelClaude3_5HaikuLatest), "You are a helpful assistant")
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

	// Stream a response
	var blocks []Block
	for chunk, err := range gen.Stream(context.Background(), dialog, &GenOpts{MaxGenerationTokens: 1024}) {
		if err != nil {
			fmt.Println(err.Error())
			return
		}
		blocks = append(blocks, chunk.Block)
	}

	if len(blocks) > 0 {
		fmt.Println("Response received")
	}

	// Output: Response received
}

func ExampleAnthropicGenerator_Generate_thinking() {
	// Create an Anthropic client
	client := a.NewClient()

	// Instantiate an Anthropic Generator
	gen := NewAnthropicGenerator(&client.Messages, string(a.ModelClaudeSonnet4_0), "You are a helpful assistant")
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

	// This example assumes that sample.jpg is present in the current directory.
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.jpg]")
		return
	}
	imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))

	client := a.NewClient()

	gen := NewAnthropicGenerator(
		&client.Messages,
		string(a.ModelClaude3_7SonnetLatest),
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
		string(a.ModelClaude_3_5_Sonnet_20240620),
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
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
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
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
	}

	// Instantiate an Anthropic Generator
	gen := NewAnthropicGenerator(
		&client.Messages,
		string(a.ModelClaudeSonnet4_0),
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
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{
		MaxGenerationTokens: 8096,
		ThinkingBudget:      "4000",
	})
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[1].Content)
	fmt.Println(resp.Candidates[0].Blocks[2].Content)

	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[1].ID,
				ModalityType: Text,
				Content:      Str("123.45"),
			},
		},
	}, Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[2].ID,
				ModalityType: Text,
				Content:      Str("678.45"),
			},
		},
	})

	resp, err = gen.Generate(context.Background(), dialog, &GenOpts{
		MaxGenerationTokens: 8096,
		ThinkingBudget:      "4000",
	})
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)

	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// {"name":"get_stock_price","parameters":{"ticker":"MSFT"}}
	// MSFT
}

func ExampleAnthropicGenerator_Stream_parallelToolUse() {
	// Create an Anthropic client
	client := a.NewClient()

	// Register tools
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
	}

	// Instantiate an Anthropic Generator
	gen := NewAnthropicGenerator(
		&client.Messages,
		string(a.ModelClaudeSonnet4_0),
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

	// Stream a response
	var blocks []Block
	for chunk, err := range gen.Stream(context.Background(), dialog, &GenOpts{
		MaxGenerationTokens: 32000,
		ThinkingBudget:      "10000",
	}) {
		if err != nil {
			fmt.Println(err.Error())
			return
		}
		blocks = append(blocks, chunk.Block)
	}

	if len(blocks) > 1 {
		fmt.Println("Response received")
	}

	// collect the blocks
	var prevToolCallId string
	var toolCalls []Block
	var toolcallArgs string
	var toolCallInput ToolCallInput
	thinking := Block{
		BlockType:    Thinking,
		ModalityType: Text,
		MimeType:     "text/plain",
		ExtraFields:  make(map[string]interface{}),
	}
	thinkingStr := ""
	for _, block := range blocks {
		if block.BlockType == Thinking {
			if block.Content != nil {
				thinkingStr += block.Content.String()
			}
			maps.Copy(thinking.ExtraFields, block.ExtraFields)
			continue
		}
		if block.ID != "" && block.ID != prevToolCallId {
			if toolcallArgs != "" {
				// Parse the arguments string into a map
				if err := json.Unmarshal([]byte(toolcallArgs), &toolCallInput.Parameters); err != nil {
					panic(err.Error())
				}

				// Marshal back to JSON for consistent representation
				toolUseJSON, err := json.Marshal(toolCallInput)
				if err != nil {
					panic(err.Error())
				}
				toolCalls[len(toolCalls)-1].Content = Str(toolUseJSON)
				toolCallInput = ToolCallInput{}
				toolcallArgs = ""
			}
			prevToolCallId = block.ID
			toolCalls = append(toolCalls, Block{
				ID:           block.ID,
				BlockType:    ToolCall,
				ModalityType: Text,
				MimeType:     "text/plain",
			})
			toolCallInput.Name = block.Content.String()
		} else {
			toolcallArgs += block.Content.String()
		}
	}

	thinking.Content = Str(thinkingStr)

	if toolcallArgs != "" {
		// Parse the arguments string into a map
		if err := json.Unmarshal([]byte(toolcallArgs), &toolCallInput.Parameters); err != nil {
			panic(err.Error())
		}

		// Marshal back to JSON for consistent representation
		toolUseJSON, err := json.Marshal(toolCallInput)
		if err != nil {
			panic(err.Error())
		}
		toolCalls[len(toolCalls)-1].Content = Str(toolUseJSON)
		toolCallInput = ToolCallInput{}
	}

	fmt.Println(len(toolCalls))

	assistantMsg := make([]Block, 0, len(toolCalls)+1)
	assistantMsg = append(assistantMsg, thinking)
	assistantMsg = append(assistantMsg, toolCalls...)

	dialog = append(dialog, Message{
		Role:   Assistant,
		Blocks: assistantMsg,
	},
		Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           toolCalls[0].ID,
					ModalityType: Text,
					Content:      Str("123.45"),
				},
			},
		}, Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           toolCalls[1].ID,
					ModalityType: Text,
					Content:      Str("678.45"),
				},
			},
		})

	// Stream a response
	blocks = nil
	for chunk, err := range gen.Stream(context.Background(), dialog, &GenOpts{
		MaxGenerationTokens: 32000,
		ThinkingBudget:      "10000",
	}) {
		if err != nil {
			fmt.Println(err.Error())
			return
		}
		blocks = append(blocks, chunk.Block)
	}

	if len(blocks) > 0 {
		fmt.Println("Response received")
	}

	// Output: Response received
	// 2
	// Response received
}

func ExampleAnthropicGenerator_Count() {
	// Create an Anthropic client
	client := a.NewClient()

	// Create a generator with system instructions
	generator := NewAnthropicGenerator(
		&client.Messages,
		string(a.ModelClaude3_5SonnetLatest),
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

	fmt.Printf("Dialog contains approximately %d tokens\n", tokenCount)

	// Add a response to the dialog
	dialog = append(dialog, Message{
		Role: Assistant,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("The capital of France is Paris. It's a beautiful city known for its culture, art, and cuisine."),
			},
		},
	})

	// Count tokens in the updated dialog
	tokenCount, err = generator.Count(context.Background(), dialog)
	if err != nil {
		fmt.Printf("Error counting tokens: %v\n", err)
		return
	}

	fmt.Printf("Dialog with response contains approximately %d tokens\n", tokenCount)

	// Output: Dialog contains approximately 20 tokens
	// Dialog with response contains approximately 42 tokens
}

func ExampleAnthropicGenerator_Generate_pdf() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set ANTHROPIC_API_KEY env]")
		return
	}

	// This example assumes that sample.pdf is present in the current directory.
	pdfBytes, err := os.ReadFile("sample.pdf")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.pdf]")
		return
	}

	client := a.NewClient()

	gen := NewAnthropicGenerator(
		&client.Messages,
		string(a.ModelClaudeSonnet4_0),
		"You are a helpful assistant.",
	)

	// Create a dialog with PDF content
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("What is the title of this PDF? Just output the title and nothing else"),
				PDFBlock(pdfBytes, "paper.pdf"),
			},
		},
	}

	// Generate a response
	ctx := context.Background()
	response, err := gen.Generate(ctx, dialog, &GenOpts{MaxGenerationTokens: 1024})
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
