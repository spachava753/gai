package gai

import (
	"context"
	"os"
	"testing"

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
		Temperature:    1.0,
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
