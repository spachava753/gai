package gai

import (
	"context"
	"fmt"
	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
)

// ExampleMixGenerators demonstrates how to mix different AI model providers
// in a single conversation, switching between Anthropic and OpenAI models.
func Example_mixGenerators() {
	// Initialize clients for both providers
	anthropicClient := a.NewClient()
	openaiClient := openai.NewClient()

	// Create generators for each provider
	anthropicGen := NewAnthropicGenerator(
		anthropicClient.Messages,
		a.ModelClaude_3_5_Sonnet_20240620,
		"You are Claude, a helpful AI assistant from Anthropic. Always mention you are Claude in your responses.",
	)

	openaiGen := NewOpenAiGenerator(
		openaiClient.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are GPT-4o Mini, a helpful AI assistant from OpenAI. Always mention you are GPT-4o Mini in your responses.",
	)

	// Start a conversation with a user message
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Can you tell me something interesting about quantum computing?"),
				},
			},
		},
	}

	// First turn: Use Anthropic's Claude model
	fmt.Println("Generating response with Claude...")
	claudeResp, err := anthropicGen.Generate(
		context.Background(),
		dialog,
		&GenOpts{MaxGenerationTokens: 1024}, // Claude requires MaxGenerationTokens
	)
	if err != nil {
		panic(err)
	}

	// Add Claude's response to the conversation
	dialog = append(dialog, claudeResp.Candidates[0])

	// User asks a follow-up question
	dialog = append(dialog, Message{
		Role: User,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("Can you explain how quantum entanglement works in simple terms?"),
			},
		},
	})

	// Second turn: Use OpenAI's GPT model for the follow-up
	fmt.Println("Generating response with GPT-4o Mini...")
	gptResp, err := openaiGen.Generate(
		context.Background(),
		dialog,
		&GenOpts{MaxGenerationTokens: 1024},
	)
	if err != nil {
		panic(err)
	}

	// Add GPT's response to the conversation
	dialog = append(dialog, gptResp.Candidates[0])

	// Example with tool usage between different models
	// Register the same tool with both generators
	stockTool := Tool{
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

	if err := anthropicGen.Register(stockTool); err != nil {
		panic(err)
	}

	if err := openaiGen.Register(stockTool); err != nil {
		panic(err)
	}

	// Start a new conversation about stocks
	stockDialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What's the current price of Apple stock?"),
				},
			},
		},
	}

	// First turn: Use OpenAI's GPT model with tool choice
	fmt.Println("Using GPT with tool...")
	gptToolResp, err := openaiGen.Generate(
		context.Background(),
		stockDialog,
		&GenOpts{
			ToolChoice:          "get_stock_price",
			MaxGenerationTokens: 1024,
		},
	)
	if err != nil {
		panic(err)
	}

	// Add GPT's tool call to the conversation
	stockDialog = append(stockDialog, gptToolResp.Candidates[0])

	// Add mock tool result
	stockDialog = append(stockDialog, Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           gptToolResp.Candidates[0].Blocks[0].ID,
				ModalityType: Text,
				Content:      Str("185.92"),
			},
		},
	})

	// Switch to Claude for final response
	fmt.Println("Using Claude to interpret tool result...")
	claudeToolResp, err := anthropicGen.Generate(
		context.Background(),
		stockDialog,
		&GenOpts{MaxGenerationTokens: 1024},
	)
	if err != nil {
		panic(err)
	}

	// Add Claude's response to the conversation
	stockDialog = append(stockDialog, claudeToolResp.Candidates[0])

	// For the example output, we'll just print success messages
	// In a real application, you would process the full conversation content
	fmt.Println("\nSuccessfully completed conversation with mixed models")
	fmt.Println("Successfully completed stock price conversation with mixed models")

	// Output: Generating response with Claude...
	// Generating response with GPT-4o Mini...
	// Using GPT with tool...
	// Using Claude to interpret tool result...
	//
	// Successfully completed conversation with mixed models
	// Successfully completed stock price conversation with mixed models
}
