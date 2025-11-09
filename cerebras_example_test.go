package gai

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
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
	resp, err := gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(resp.Candidates) == 1 && len(resp.Candidates[0].Blocks) >= 1 {
		fmt.Println("Response received")
	}
	// Output: Response received
}

func ExampleCerebrasGenerator_Generate_reasoning_gptoss() {
	apiKey := os.Getenv("CEREBRAS_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set CEREBRAS_API_KEY env]")
		return
	}

	// Use gpt-oss-120b model which supports reasoning with reasoning_effort parameter
	gen := NewCerebrasGenerator(
		nil,
		"",
		"gpt-oss-120b",
		"You are a helpful assistant that explains your reasoning step by step.",
		apiKey,
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

	// Generate response with reasoning enabled (medium effort)
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{
		ThinkingBudget: "medium",
	})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks (reasoning)
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
				fmt.Println("Reasoning found")
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

	// Append the previous response and ask a follow-up question to test reasoning retention
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

	// Generate response with reasoning (the previous reasoning should be retained)
	resp, err = gen.Generate(context.Background(), dialog, &GenOpts{
		ThinkingBudget: "medium",
	})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
			}
		}

		if hasThinking {
			fmt.Println("Thinking blocks found")
		}

		// Find the main content block
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

	// Output: Reasoning found
	// Thinking blocks found
	// Correct answer found
	// Thinking blocks found
	// Correct answer found
}

func ExampleCerebrasGenerator_Generate_reasoning_zai() {
	apiKey := os.Getenv("CEREBRAS_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set CEREBRAS_API_KEY env]")
		return
	}

	// Use zai-glm-4.6 model which supports reasoning with disable_reasoning parameter
	gen := NewCerebrasGenerator(
		nil,
		"",
		"zai-glm-4.6",
		"You are a helpful assistant that explains your reasoning step by step.",
		apiKey,
	)

	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is 15 * 12?"),
				},
			},
		},
	}

	// Generate response with reasoning enabled (disable_reasoning: false)
	resp, err := gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks (reasoning)
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
				fmt.Println("Reasoning found")
			}
		}

		if hasThinking {
			fmt.Println("Thinking blocks found")
		}

		// Find the main content block (not thinking)
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "180") {
					fmt.Println("Correct answer found")
				}
				break
			}
		}
	}

	// Append the previous response and ask a follow-up question to test reasoning retention
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: User,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("Now what is 20 * 15?"),
			},
		},
	})

	// Generate response with reasoning (the previous reasoning should be retained)
	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
			}
		}

		if hasThinking {
			fmt.Println("Thinking blocks found")
		}

		// Find the main content block
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "300") {
					fmt.Println("Correct answer found")
				}
				break
			}
		}
	}

	// Output: Reasoning found
	// Thinking blocks found
	// Correct answer found
	// Thinking blocks found
	// Correct answer found
}

func ExampleCerebrasGenerator_Register() {
	apiKey := os.Getenv("CEREBRAS_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set CEREBRAS_API_KEY env]")
		return
	}
	cgen := NewCerebrasGenerator(nil, "", "qwen-3-235b-a22b-instruct-2507", `You are a helpful assistant that returns the price of a stock and nothing else.

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
