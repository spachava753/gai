package gai

import (
	"context"
	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go/v3"
	"testing"
)

// ExampleMixGenerators demonstrates how to mix different AI model providers
// in a single conversation, switching between Anthropic and OpenAI models.
func Test_mixGenerators(t *testing.T) {
	requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
	requireLiveAPIKey(t, "OPENAI_API_KEY")

	// Initialize clients for both providers
	anthropicClient := a.NewClient()
	openaiClient := openai.NewClient()
	// Create generators for each provider
	anthropicGen := NewAnthropicGenerator(&anthropicClient.Messages)
	openaiGen := NewOpenAiGenerator(&openaiClient.Chat.Completions)
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
	claudeResp, err := anthropicGen.Generate(
		context.Background(),
		GenerationRequest{
			Model:        string(a.ModelClaudeHaiku4_5),
			Instructions: SystemMessage(TextBlock("You are Claude, a helpful AI assistant from Anthropic. Always mention you are Claude in your responses.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(1024)),
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	claudeMsg := requireCandidate(t, claudeResp)
	requireContentBlock(t, requireBlock(t, claudeMsg, 0))
	// Add Claude's response to the conversation
	dialog = append(dialog, claudeMsg)
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
	gptResp, err := openaiGen.Generate(
		context.Background(),
		GenerationRequest{
			Model:        openai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock("You are GPT-4o Mini, a helpful AI assistant from OpenAI. Always mention you are GPT-4o Mini in your responses.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(1024)),
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	gptMsg := requireCandidate(t, gptResp)
	requireContentBlock(t, requireBlock(t, gptMsg, 0))
	// Add GPT's response to the conversation
	dialog = append(dialog, gptMsg)
	// Example with tool usage between different models
	// Register the same tool with both generators
	stockTool := Tool{
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
	gptToolResp, err := openaiGen.Generate(
		context.Background(),
		GenerationRequest{
			Model:        openai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock("You are GPT-4o Mini, a helpful AI assistant from OpenAI. Always mention you are GPT-4o Mini in your responses.")),
			Dialog:       stockDialog,
			Tools:        []Tool{stockTool},
			Options: NewGenerationOptions(
				WithToolChoice("get_stock_price"),
				WithMaxGenerationTokens(1024),
			),
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	gptToolMsg := requireCandidate(t, gptToolResp)
	if len(collectToolCalls(t, gptToolMsg.Blocks)) == 0 {
		t.Fatalf("expected GPT response to contain a tool call: %#v", gptToolMsg.Blocks)
	}
	// Add GPT's tool call to the conversation
	stockDialog = append(stockDialog, gptToolMsg)
	// Add mock tool result
	stockDialog = append(stockDialog, Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           gptToolMsg.Blocks[0].ID,
				ModalityType: Text,
				Content:      Str("185.92"),
			},
		},
	})
	// Switch to Claude for final response
	claudeToolResp, err := anthropicGen.Generate(
		context.Background(),
		GenerationRequest{
			Model:        string(a.ModelClaudeHaiku4_5),
			Instructions: SystemMessage(TextBlock("You are Claude, a helpful AI assistant from Anthropic. Always mention you are Claude in your responses.")),
			Dialog:       stockDialog,
			Tools:        []Tool{stockTool},
			Options:      NewGenerationOptions(WithMaxGenerationTokens(1024)),
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	claudeToolMsg := requireCandidate(t, claudeToolResp)
	requireContentBlock(t, requireBlock(t, claudeToolMsg, 0))
	// Add Claude's response to the conversation
	stockDialog = append(stockDialog, claudeToolMsg)
	if len(stockDialog) != 4 {
		t.Fatalf("stock dialog length = %d, want 4", len(stockDialog))
	}
}
