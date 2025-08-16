package gai

import (
	"context"
	"fmt"
	"os"
)

func ExampleCerebrasGenerator_Generate() {
	apiKey := os.Getenv("CEREBRAS_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set CEREBRAS_API_KEY env]")
		return
	}
	gen := NewCerebrasGenerator(nil, "", "qwen-3-32b", "You are a helpful assistant.", apiKey)
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("Hello!")},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: 64})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(resp.Candidates) == 1 && len(resp.Candidates[0].Blocks) >= 1 {
		fmt.Println("Response received")
	}
	// Output: Response received
}

func ExampleCerebrasGenerator_Register() {
	apiKey := os.Getenv("CEREBRAS_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set CEREBRAS_API_KEY env]")
		return
	}
	cgen := NewCerebrasGenerator(nil, "", "qwen-3-32b", `You are a helpful assistant that returns the price of a stock and nothing else.

Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`, apiKey)

	// Register a tool
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
	}
	if err := cgen.Register(tickerTool); err != nil {
		fmt.Println("Error:", err)
		return
	}

	dialog := Dialog{
		{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}},
	}

	// Force the tool call
	resp, err := cgen.Generate(context.Background(), dialog, &GenOpts{ToolChoice: "get_stock_price"})
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

	// Ask model to answer now without calling tools
	resp, err = cgen.Generate(context.Background(), dialog, &GenOpts{ToolChoice: "none"})
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
