package gai

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

// TestResponsesGenerator_Generate_Thinking_Logging tests that thinking content is returned
// from the responses generator and logs out the thinking blocks for debugging.
func TestResponsesGenerator_Generate_Thinking_Logging(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping test")
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5, "You are a helpful assistant")

	dialog := Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("What is 2 + 2? Think step by step.")},
	}}

	opts := GenOpts{
		ThinkingBudget: "medium",
		Temperature: Ptr(1.0),
		ExtraArgs: map[string]any{
			ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
		},
	}

	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if len(resp.Candidates) == 0 {
		t.Fatal("Expected at least one candidate, got none")
	}

	// Log all blocks and specifically look for thinking blocks
	t.Log("=== All blocks in response ===")
	hasThinkingBlocks := false
	thinkingBlockCount := 0

	for i, block := range resp.Candidates[0].Blocks {
		t.Logf("Block %d: Type=%s, Modality=%s, MimeType=%s", i, block.BlockType, block.ModalityType, block.MimeType)

		if block.BlockType == Thinking {
			hasThinkingBlocks = true
			thinkingBlockCount++
			t.Log("=== THINKING BLOCK CONTENT ===")
			t.Logf("Thinking content:\n%s", block.Content.String())
			t.Log("=== END THINKING BLOCK ===")

			// Log extra fields if present
			if block.ExtraFields != nil {
				t.Logf("Extra fields: %+v", block.ExtraFields)
				if gen, ok := block.ExtraFields[ThinkingExtraFieldGeneratorKey]; ok {
					t.Logf("Generator: %v", gen)
				}
			}
		} else if block.BlockType == Content {
			t.Log("=== CONTENT BLOCK ===")
			t.Logf("Content:\n%s", block.Content.String())
			t.Log("=== END CONTENT BLOCK ===")
		}
	}

	t.Logf("Total thinking blocks found: %d", thinkingBlockCount)

	if !hasThinkingBlocks {
		t.Error("Expected at least one Thinking block in response, but found none")
	}
}

// TestResponsesGenerator_Stream_Thinking_Logging tests that thinking content is streamed
// from the responses generator and logs out the thinking chunks for debugging.
func TestResponsesGenerator_Stream_Thinking_Logging(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping test")
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Nano, "You are a helpful assistant")

	dialog := Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("What is the capital of France? Think about it briefly.")},
	}}

	opts := GenOpts{
		ThinkingBudget: "medium",
		ExtraArgs: map[string]any{
			ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
		},
	}

	var allBlocks []Block
	var thinkingContent string
	var regularContent string

	t.Log("=== Streaming response ===")
	for chunk, err := range gen.Stream(context.Background(), dialog, &opts) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		allBlocks = append(allBlocks, chunk.Block)

		if chunk.Block.BlockType == Thinking {
			thinkingContent += chunk.Block.Content.String()
			t.Logf("Thinking chunk: %q", chunk.Block.Content.String())
		} else if chunk.Block.BlockType == Content {
			regularContent += chunk.Block.Content.String()
		}
	}

	t.Log("=== Final aggregated thinking content ===")
	t.Logf("Thinking:\n%s", thinkingContent)
	t.Log("=== End thinking ===")

	t.Log("=== Final aggregated regular content ===")
	t.Logf("Content:\n%s", regularContent)
	t.Log("=== End content ===")

	hasThinkingBlocks := false
	for _, block := range allBlocks {
		if block.BlockType == Thinking {
			hasThinkingBlocks = true
			break
		}
	}

	if !hasThinkingBlocks {
		t.Error("Expected at least one Thinking block in streaming response, but found none")
	}

	if thinkingContent == "" {
		t.Error("Thinking content is empty - reasoning may not be returned correctly")
	} else {
		t.Logf("Thinking content length: %d characters", len(thinkingContent))
	}
}

// TestResponsesGenerator_StatelessToolCallWithReasoning tests the core stateless refactor
// scenario: a reasoning model that makes tool calls across multiple turns. The encrypted
// reasoning items must be correctly stored in Thinking block ExtraFields and reconstructed
// as input reasoning items on subsequent calls.
func TestResponsesGenerator_StatelessToolCallWithReasoning(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping test")
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, `You are a helpful assistant that uses tools when needed.
When asked about stock prices, use the get_stock_price tool.
After getting the price, report it to the user.`)

	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
			}]()
			if err != nil {
				t.Fatalf("Failed to generate schema: %v", err)
			}
			return schema
		}(),
	}
	if err := gen.Register(tickerTool); err != nil {
		t.Fatalf("Failed to register tool: %v", err)
	}

	// Turn 1: Ask the model to use a tool with reasoning enabled
	dialog := Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("What is the current price of Apple stock?")},
	}}

	opts := GenOpts{
		ThinkingBudget: "low",
		ExtraArgs: map[string]any{
			ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
		},
	}

	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("Turn 1 Generate failed: %v", err)
	}

	if resp.FinishReason != ToolUse {
		t.Fatalf("Expected ToolUse finish reason, got %v", resp.FinishReason)
	}

	// Verify that thinking blocks have encrypted content
	assistantMsg := resp.Candidates[0]
	hasEncryptedReasoning := false
	var toolCallBlock Block
	for _, blk := range assistantMsg.Blocks {
		if blk.BlockType == Thinking && blk.ExtraFields != nil {
			if ec, ok := blk.ExtraFields[ResponsesExtraFieldEncryptedContent].(string); ok && ec != "" {
				hasEncryptedReasoning = true
				t.Logf("Found encrypted reasoning content (length: %d)", len(ec))
			}
			if rid, ok := blk.ExtraFields[ResponsesExtraFieldReasoningID].(string); ok && rid != "" {
				t.Logf("Found reasoning ID: %s", rid)
			}
		}
		if blk.BlockType == ToolCall {
			toolCallBlock = blk
		}
	}

	if !hasEncryptedReasoning {
		t.Log("Warning: no encrypted reasoning content found in response (may not be present for all models/configs)")
	}

	if toolCallBlock.ID == "" {
		t.Fatal("Expected a tool call block in the response")
	}
	t.Logf("Tool call ID: %s, Content: %s", toolCallBlock.ID, toolCallBlock.Content)

	// Turn 2: Provide tool result and let the model respond
	// This is the critical test: the dialog now contains the full assistant message
	// (including Thinking blocks with encrypted content), and the generator must
	// reconstruct reasoning items when building the input.
	dialog = append(dialog, assistantMsg, Message{
		Role: ToolResult,
		Blocks: []Block{{
			ID:           toolCallBlock.ID,
			ModalityType: Text,
			MimeType:     "text/plain",
			Content:      Str("Apple (AAPL) is currently trading at $187.50"),
		}},
	})

	resp2, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("Turn 2 Generate failed: %v", err)
	}

	if resp2.FinishReason != EndTurn {
		t.Fatalf("Expected EndTurn finish reason, got %v", resp2.FinishReason)
	}

	// Verify the model produced a content response
	var finalContent string
	for _, blk := range resp2.Candidates[0].Blocks {
		if blk.BlockType == Content {
			finalContent = blk.Content.String()
			break
		}
	}

	if finalContent == "" {
		t.Fatal("Expected content in the final response")
	}
	t.Logf("Final response: %s", finalContent)

	// The response should mention the price
	if !strings.Contains(finalContent, "187") {
		t.Errorf("Expected the response to mention the stock price, got: %s", finalContent)
	}
}
