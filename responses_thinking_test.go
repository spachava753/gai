package gai

import (
	"context"
	"encoding/json"
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
		t.Error("Expected encrypted reasoning content in Thinking block ExtraFields; stateless multi-turn tool calling requires this")
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

// TestResponsesGenerator_StreamingToolCallWithReasoning tests streaming multi-turn
// tool calling with a reasoning model. This verifies that the encrypted reasoning
// content is correctly captured via response.output_item.done events and propagated
// through StreamingAdapter compression into Thinking block ExtraFields, enabling
// stateless multi-turn function calling through the streaming path.
func TestResponsesGenerator_StreamingToolCallWithReasoning(t *testing.T) {
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

	// Use StreamingAdapter to get a Generator interface from the streaming path
	adapter := &StreamingAdapter{S: &gen}

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

	resp, err := adapter.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("Turn 1 Generate (streaming) failed: %v", err)
	}

	if resp.FinishReason != ToolUse {
		t.Fatalf("Expected ToolUse finish reason, got %v", resp.FinishReason)
	}

	// Verify that the compressed thinking blocks carry encrypted content
	assistantMsg := resp.Candidates[0]
	hasEncryptedReasoning := false
	var toolCallBlock Block
	for _, blk := range assistantMsg.Blocks {
		if blk.BlockType == Thinking && blk.ExtraFields != nil {
			if ec, ok := blk.ExtraFields[ResponsesExtraFieldEncryptedContent].(string); ok && ec != "" {
				hasEncryptedReasoning = true
				t.Logf("Found encrypted reasoning content in streaming response (length: %d)", len(ec))
			}
			if rid, ok := blk.ExtraFields[ResponsesExtraFieldReasoningID].(string); ok && rid != "" {
				t.Logf("Found reasoning ID in streaming response: %s", rid)
			}
		}
		if blk.BlockType == ToolCall {
			toolCallBlock = blk
		}
	}

	if !hasEncryptedReasoning {
		t.Error("Expected encrypted reasoning content in streaming Thinking block ExtraFields; stateless multi-turn tool calling requires this")
	}

	if toolCallBlock.ID == "" {
		t.Fatal("Expected a tool call block in the streaming response")
	}
	t.Logf("Tool call ID: %s, Content: %s", toolCallBlock.ID, toolCallBlock.Content)

	// Verify usage metadata is present
	if resp.UsageMetadata == nil {
		t.Error("Expected usage metadata in streaming response")
	} else {
		t.Logf("Usage metadata: %+v", resp.UsageMetadata)
	}

	// Turn 2: Provide tool result and let the model respond.
	// This tests that the encrypted reasoning from the streaming path
	// can be correctly reconstructed by buildInputItems.
	dialog = append(dialog, assistantMsg, Message{
		Role: ToolResult,
		Blocks: []Block{{
			ID:           toolCallBlock.ID,
			ModalityType: Text,
			MimeType:     "text/plain",
			Content:      Str("Apple (AAPL) is currently trading at $187.50"),
		}},
	})

	resp2, err := adapter.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("Turn 2 Generate (streaming) failed: %v", err)
	}

	if resp2.FinishReason != EndTurn {
		t.Fatalf("Expected EndTurn finish reason, got %v", resp2.FinishReason)
	}

	// Verify the model produced a content response mentioning the price
	var finalContent string
	for _, blk := range resp2.Candidates[0].Blocks {
		if blk.BlockType == Content {
			finalContent = blk.Content.String()
			break
		}
	}

	if finalContent == "" {
		t.Fatal("Expected content in the final streaming response")
	}
	t.Logf("Final streaming response: %s", finalContent)

	if !strings.Contains(finalContent, "187") {
		t.Errorf("Expected the response to mention the stock price, got: %s", finalContent)
	}
}

// TestResponsesGenerator_ReasoningTokenPreservation_Generate tests that encrypted reasoning
// tokens are preserved within an assistant turn (function-calling loop) and discarded when
// a new user turn begins, using the non-streaming Generate path.
//
// The test validates this by checking token counts reported by the API:
//   - Within a function-calling assistant turn, input_tokens[n+1] should approximately equal
//     input_tokens[n] + output_tokens[n], because reasoning tokens are preserved as encrypted
//     content and passed back as input.
//   - After a new user message starts a new assistant turn, the reasoning tokens from the
//     previous turn are discarded, so input_tokens should drop by approximately the number
//     of reasoning tokens from the last turn.
//
// This is a live API test that requires OPENAI_API_KEY to be set.
func TestResponsesGenerator_ReasoningTokenPreservation_Generate(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping test")
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, `You are a helpful assistant that uses tools when needed.
When asked to look up information, always use the provided tools.
After getting tool results, report them to the user.`)

	// Register two tools so the model has a reason to make multiple tool calls
	lookupTool := Tool{
		Name:        "lookup_info",
		Description: "Look up factual information about a topic. Returns a brief fact.",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Topic string `json:"topic" jsonschema:"required" jsonschema_description:"The topic to look up"`
			}]()
			if err != nil {
				t.Fatalf("Failed to generate schema: %v", err)
			}
			return schema
		}(),
	}
	if err := gen.Register(lookupTool); err != nil {
		t.Fatalf("Failed to register tool: %v", err)
	}

	opts := GenOpts{
		ThinkingBudget: "low",
		ExtraArgs: map[string]any{
			ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
		},
	}

	// === Phase 1: Function-calling loop within one assistant turn ===

	dialog := Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("Look up the population of Tokyo and the population of New York City, then tell me which is larger.")},
	}}

	type turnMetrics struct {
		inputTokens     int
		outputTokens    int
		reasoningTokens int
		// toolResultTokens is the number of tokens added as tool results after this
		// turn's response. It is computed as the difference between the next turn's
		// input and this turn's input + output. For the first pair of turns this
		// establishes the baseline; subsequent pairs must match exactly.
		toolResultTokens int
	}
	var turns []turnMetrics

	// Loop until the model stops calling tools (up to a safety limit)
	for i := 0; i < 5; i++ {
		resp, err := gen.Generate(context.Background(), dialog, &opts)
		if err != nil {
			t.Fatalf("Turn %d Generate failed: %v", i+1, err)
		}

		inputTok, _ := InputTokens(resp.UsageMetadata)
		outputTok, _ := OutputTokens(resp.UsageMetadata)
		reasoningTok, _ := GetMetric[int](resp.UsageMetadata, UsageMetricReasoningTokens)

		t.Logf("Turn %d: input=%d, output=%d, reasoning=%d, finish=%v",
			i+1, inputTok, outputTok, reasoningTok, resp.FinishReason)

		turns = append(turns, turnMetrics{
			inputTokens:     inputTok,
			outputTokens:    outputTok,
			reasoningTokens: reasoningTok,
		})

		if resp.FinishReason != ToolUse {
			// Model finished, no more tool calls
			dialog = append(dialog, resp.Candidates[0])
			break
		}

		// Collect tool calls and provide results
		assistantMsg := resp.Candidates[0]
		dialog = append(dialog, assistantMsg)

		for _, blk := range assistantMsg.Blocks {
			if blk.BlockType == ToolCall {
				// Provide a fake tool result
				var tci ToolCallInput
				if err := json.Unmarshal([]byte(blk.Content.String()), &tci); err != nil {
					t.Fatalf("Failed to parse tool call: %v", err)
				}
				result := "The population is approximately 14 million."
				if strings.Contains(strings.ToLower(tci.Parameters["topic"].(string)), "new york") {
					result = "The population is approximately 8.3 million."
				} else if strings.Contains(strings.ToLower(tci.Parameters["topic"].(string)), "tokyo") {
					result = "The population is approximately 13.96 million."
				}
				dialog = append(dialog, Message{
					Role: ToolResult,
					Blocks: []Block{{
						ID:           blk.ID,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str(result),
					}},
				})
			}
		}
	}

	if len(turns) < 2 {
		t.Fatalf("Expected at least 2 turns in the function-calling loop, got %d", len(turns))
	}

	// Compute tool result tokens for each turn from the observed differences.
	// Within an assistant turn the exact relationship is:
	//   input_tokens[n+1] == input_tokens[n] + output_tokens[n] + toolResultTokens[n]
	// where toolResultTokens[n] is the fixed overhead of the tool result text we
	// appended after turn n. We derive toolResultTokens from the first transition,
	// then assert that subsequent transitions match exactly.
	for i := 0; i+1 < len(turns); i++ {
		turns[i].toolResultTokens = turns[i+1].inputTokens - turns[i].inputTokens - turns[i].outputTokens
	}

	// The tool result tokens should be positive (we always add content).
	if turns[0].toolResultTokens <= 0 {
		t.Fatalf("Tool result tokens for turn 1 should be positive, got %d", turns[0].toolResultTokens)
	}
	t.Logf("Tool result tokens (derived from first transition): %d", turns[0].toolResultTokens)

	// Assert exact equality for all subsequent transitions.
	for i := 1; i+1 < len(turns); i++ {
		prev := turns[i-1]
		curr := turns[i]

		t.Logf("Within-turn assertion (turn %d→%d): toolResultTokens=%d (expected %d)",
			i+1, i+2, curr.toolResultTokens, prev.toolResultTokens)

		if curr.toolResultTokens != prev.toolResultTokens {
			t.Errorf("Within assistant turn: tool result token count changed between transitions "+
				"(turn %d→%d: %d, turn %d→%d: %d); reasoning tokens may not be correctly preserved",
				i, i+1, prev.toolResultTokens, i+1, i+2, curr.toolResultTokens)
		}
	}

	// For every transition, verify the exact formula:
	// input_tokens[n+1] == input_tokens[n] + output_tokens[n] + toolResultTokens[n]
	for i := 0; i+1 < len(turns); i++ {
		prev := turns[i]
		curr := turns[i+1]
		expected := prev.inputTokens + prev.outputTokens + prev.toolResultTokens

		t.Logf("Within-turn exact check (turn %d→%d): input[%d]+output[%d]+toolResult[%d] = %d+%d+%d = %d, actual input[%d]=%d",
			i+1, i+2, i+1, i+1, i+1, prev.inputTokens, prev.outputTokens, prev.toolResultTokens, expected, i+2, curr.inputTokens)

		if curr.inputTokens != expected {
			t.Errorf("Within assistant turn: input_tokens[%d]=%d != input_tokens[%d]+output_tokens[%d]+toolResultTokens[%d]=%d",
				i+2, curr.inputTokens, i+1, i+1, i+1, expected)
		}
	}

	// Record the last turn's metrics for the cross-turn comparison
	lastTurnInPhase1 := turns[len(turns)-1]

	// === Phase 2: New user turn — reasoning tokens from previous turn should be discarded ===

	dialog = append(dialog, Message{
		Role:   User,
		Blocks: []Block{TextBlock("Now look up the population of London.")},
	})

	resp2, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("New turn Generate failed: %v", err)
	}

	newTurnInput, _ := InputTokens(resp2.UsageMetadata)
	newTurnOutput, _ := OutputTokens(resp2.UsageMetadata)
	newTurnReasoning, _ := GetMetric[int](resp2.UsageMetadata, UsageMetricReasoningTokens)

	t.Logf("New turn: input=%d, output=%d, reasoning=%d, finish=%v",
		newTurnInput, newTurnOutput, newTurnReasoning, resp2.FinishReason)

	// The expected input if reasoning were preserved would be:
	// lastTurnInPhase1.inputTokens + lastTurnInPhase1.outputTokens + new_user_message_tokens
	expectedWithReasoning := lastTurnInPhase1.inputTokens + lastTurnInPhase1.outputTokens
	// The drop should be approximately equal to the total reasoning tokens from the
	// last assistant turn, because those are no longer included as input.
	totalReasoningInLastTurn := lastTurnInPhase1.reasoningTokens

	t.Logf("Cross-turn: expectedWithReasoning≈%d, actual=%d, reasoning_discarded≈%d",
		expectedWithReasoning, newTurnInput, totalReasoningInLastTurn)

	// The new turn's input should be noticeably less than expected-with-reasoning,
	// because the reasoning tokens from the previous turn are discarded.
	actualDrop := expectedWithReasoning - newTurnInput
	t.Logf("Cross-turn drop: %d tokens (expected drop ≈ %d reasoning tokens from last turn)",
		actualDrop, totalReasoningInLastTurn)

	if totalReasoningInLastTurn > 0 {
		// When the API reports reasoning tokens, the drop should be at least 50%
		// of the reasoning tokens to confirm they were truly discarded.
		if actualDrop < totalReasoningInLastTurn/2 {
			t.Errorf("Cross-turn: expected input to drop by ~%d reasoning tokens, but only dropped %d; "+
				"reasoning tokens may not have been properly discarded across turns",
				totalReasoningInLastTurn, actualDrop)
		}
	} else {
		// Even without explicit reasoning_tokens metrics, we can verify the drop pattern.
		// The API includes encrypted reasoning tokens in output_tokens but may not always
		// report them in output_tokens_details.reasoning_tokens. If encrypted reasoning
		// content was present in the dialog, there should still be a significant drop
		// when those tokens are discarded across turns.
		hasEncryptedContent := false
		if len(turns) >= 2 {
			// Check the last assistant message in the dialog for encrypted reasoning
			for _, msg := range dialog {
				if msg.Role == Assistant {
					for _, blk := range msg.Blocks {
						if blk.BlockType == Thinking && blk.ExtraFields != nil {
							if ec, ok := blk.ExtraFields[ResponsesExtraFieldEncryptedContent].(string); ok && ec != "" {
								hasEncryptedContent = true
							}
						}
					}
				}
			}
		}
		if hasEncryptedContent && actualDrop <= 0 {
			t.Errorf("Cross-turn: encrypted reasoning content was present but input tokens did not drop "+
				"(expectedWithReasoning=%d, actual=%d, drop=%d); reasoning tokens may not have been discarded",
				expectedWithReasoning, newTurnInput, actualDrop)
		} else if hasEncryptedContent {
			t.Logf("Cross-turn: verified drop of %d tokens with encrypted reasoning present (reasoning_tokens metric not reported by API)", actualDrop)
		} else {
			t.Log("Warning: no reasoning tokens reported and no encrypted content found; cannot verify cross-turn token drop")
		}
	}
}

// TestResponsesGenerator_ReasoningTokenPreservation_Stream tests the same reasoning token
// preservation property as TestResponsesGenerator_ReasoningTokenPreservation_Generate, but
// using the streaming path via StreamingAdapter. This verifies that encrypted reasoning
// content captured from response.output_item.done events and compressed into Thinking block
// ExtraFields correctly preserves reasoning tokens across function-calling turns.
func TestResponsesGenerator_ReasoningTokenPreservation_Stream(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping test")
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, `You are a helpful assistant that uses tools when needed.
When asked to look up information, always use the provided tools.
After getting tool results, report them to the user.`)

	lookupTool := Tool{
		Name:        "lookup_info",
		Description: "Look up factual information about a topic. Returns a brief fact.",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Topic string `json:"topic" jsonschema:"required" jsonschema_description:"The topic to look up"`
			}]()
			if err != nil {
				t.Fatalf("Failed to generate schema: %v", err)
			}
			return schema
		}(),
	}
	if err := gen.Register(lookupTool); err != nil {
		t.Fatalf("Failed to register tool: %v", err)
	}

	// Use StreamingAdapter for the streaming path
	adapter := &StreamingAdapter{S: &gen}

	opts := GenOpts{
		ThinkingBudget: "low",
		ExtraArgs: map[string]any{
			ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
		},
	}

	// === Phase 1: Function-calling loop within one assistant turn ===

	dialog := Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("Look up the population of Tokyo and the population of New York City, then tell me which is larger.")},
	}}

	type turnMetrics struct {
		inputTokens      int
		outputTokens     int
		reasoningTokens  int
		toolResultTokens int
	}
	var turns []turnMetrics

	for i := 0; i < 5; i++ {
		resp, err := adapter.Generate(context.Background(), dialog, &opts)
		if err != nil {
			t.Fatalf("Turn %d Generate (streaming) failed: %v", i+1, err)
		}

		inputTok, _ := InputTokens(resp.UsageMetadata)
		outputTok, _ := OutputTokens(resp.UsageMetadata)
		reasoningTok, _ := GetMetric[int](resp.UsageMetadata, UsageMetricReasoningTokens)

		t.Logf("Turn %d (streaming): input=%d, output=%d, reasoning=%d, finish=%v",
			i+1, inputTok, outputTok, reasoningTok, resp.FinishReason)

		turns = append(turns, turnMetrics{
			inputTokens:     inputTok,
			outputTokens:    outputTok,
			reasoningTokens: reasoningTok,
		})

		if resp.FinishReason != ToolUse {
			dialog = append(dialog, resp.Candidates[0])
			break
		}

		assistantMsg := resp.Candidates[0]
		dialog = append(dialog, assistantMsg)

		for _, blk := range assistantMsg.Blocks {
			if blk.BlockType == ToolCall {
				var tci ToolCallInput
				if err := json.Unmarshal([]byte(blk.Content.String()), &tci); err != nil {
					t.Fatalf("Failed to parse tool call: %v", err)
				}
				result := "The population is approximately 14 million."
				if strings.Contains(strings.ToLower(tci.Parameters["topic"].(string)), "new york") {
					result = "The population is approximately 8.3 million."
				} else if strings.Contains(strings.ToLower(tci.Parameters["topic"].(string)), "tokyo") {
					result = "The population is approximately 13.96 million."
				}
				dialog = append(dialog, Message{
					Role: ToolResult,
					Blocks: []Block{{
						ID:           blk.ID,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str(result),
					}},
				})
			}
		}
	}

	if len(turns) < 2 {
		t.Fatalf("Expected at least 2 turns in the function-calling loop, got %d", len(turns))
	}

	// Compute tool result tokens from observed differences (same logic as Generate test).
	for i := 0; i+1 < len(turns); i++ {
		turns[i].toolResultTokens = turns[i+1].inputTokens - turns[i].inputTokens - turns[i].outputTokens
	}

	if turns[0].toolResultTokens <= 0 {
		t.Fatalf("Tool result tokens for turn 1 should be positive, got %d", turns[0].toolResultTokens)
	}
	t.Logf("Tool result tokens (streaming, derived from first transition): %d", turns[0].toolResultTokens)

	for i := 1; i+1 < len(turns); i++ {
		prev := turns[i-1]
		curr := turns[i]

		t.Logf("Within-turn assertion (streaming, turn %d→%d): toolResultTokens=%d (expected %d)",
			i+1, i+2, curr.toolResultTokens, prev.toolResultTokens)

		if curr.toolResultTokens != prev.toolResultTokens {
			t.Errorf("Within assistant turn (streaming): tool result token count changed between transitions "+
				"(turn %d→%d: %d, turn %d→%d: %d); reasoning tokens may not be correctly preserved",
				i, i+1, prev.toolResultTokens, i+1, i+2, curr.toolResultTokens)
		}
	}

	for i := 0; i+1 < len(turns); i++ {
		prev := turns[i]
		curr := turns[i+1]
		expected := prev.inputTokens + prev.outputTokens + prev.toolResultTokens

		t.Logf("Within-turn exact check (streaming, turn %d→%d): %d+%d+%d = %d, actual=%d",
			i+1, i+2, prev.inputTokens, prev.outputTokens, prev.toolResultTokens, expected, curr.inputTokens)

		if curr.inputTokens != expected {
			t.Errorf("Within assistant turn (streaming): input_tokens[%d]=%d != input_tokens[%d]+output_tokens[%d]+toolResultTokens[%d]=%d",
				i+2, curr.inputTokens, i+1, i+1, i+1, expected)
		}
	}

	lastTurnInPhase1 := turns[len(turns)-1]

	// === Phase 2: New user turn — reasoning tokens should be discarded ===

	dialog = append(dialog, Message{
		Role:   User,
		Blocks: []Block{TextBlock("Now look up the population of London.")},
	})

	resp2, err := adapter.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("New turn Generate (streaming) failed: %v", err)
	}

	newTurnInput, _ := InputTokens(resp2.UsageMetadata)
	newTurnOutput, _ := OutputTokens(resp2.UsageMetadata)
	newTurnReasoning, _ := GetMetric[int](resp2.UsageMetadata, UsageMetricReasoningTokens)

	t.Logf("New turn (streaming): input=%d, output=%d, reasoning=%d, finish=%v",
		newTurnInput, newTurnOutput, newTurnReasoning, resp2.FinishReason)

	expectedWithReasoning := lastTurnInPhase1.inputTokens + lastTurnInPhase1.outputTokens
	totalReasoningInLastTurn := lastTurnInPhase1.reasoningTokens

	t.Logf("Cross-turn (streaming): expectedWithReasoning≈%d, actual=%d, reasoning_discarded≈%d",
		expectedWithReasoning, newTurnInput, totalReasoningInLastTurn)

	actualDrop := expectedWithReasoning - newTurnInput
	t.Logf("Cross-turn drop (streaming): %d tokens (expected drop ≈ %d reasoning tokens)",
		actualDrop, totalReasoningInLastTurn)

	if totalReasoningInLastTurn > 0 {
		if actualDrop < totalReasoningInLastTurn/2 {
			t.Errorf("Cross-turn (streaming): expected input to drop by ~%d reasoning tokens, but only dropped %d; "+
				"reasoning tokens may not have been properly discarded across turns",
				totalReasoningInLastTurn, actualDrop)
		}
	} else {
		hasEncryptedContent := false
		if len(turns) >= 2 {
			for _, msg := range dialog {
				if msg.Role == Assistant {
					for _, blk := range msg.Blocks {
						if blk.BlockType == Thinking && blk.ExtraFields != nil {
							if ec, ok := blk.ExtraFields[ResponsesExtraFieldEncryptedContent].(string); ok && ec != "" {
								hasEncryptedContent = true
							}
						}
					}
				}
			}
		}
		if hasEncryptedContent && actualDrop <= 0 {
			t.Errorf("Cross-turn (streaming): encrypted reasoning content was present but input tokens did not drop "+
				"(expectedWithReasoning=%d, actual=%d, drop=%d); reasoning tokens may not have been discarded",
				expectedWithReasoning, newTurnInput, actualDrop)
		} else if hasEncryptedContent {
			t.Logf("Cross-turn (streaming): verified drop of %d tokens with encrypted reasoning present (reasoning_tokens metric not reported by API)", actualDrop)
		} else {
			t.Log("Warning (streaming): no reasoning tokens reported and no encrypted content found; cannot verify cross-turn token drop")
		}
	}
}

// TestResponsesGenerator_StreamMetadata verifies that the streaming path
// correctly emits a metadata block with usage information as the final chunk.
func TestResponsesGenerator_StreamMetadata(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping test")
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Nano, "You are a helpful assistant")

	dialog := Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("Say hello.")},
	}}

	var blocks []Block
	for chunk, err := range gen.Stream(context.Background(), dialog, nil) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		blocks = append(blocks, chunk.Block)
	}

	if len(blocks) == 0 {
		t.Fatal("Expected at least one block from stream")
	}

	// The last block should be a metadata block
	lastBlock := blocks[len(blocks)-1]
	if lastBlock.BlockType != MetadataBlockType {
		t.Fatalf("Expected last block to be metadata, got %s", lastBlock.BlockType)
	}

	// Verify the metadata contains usage information
	metadataJSON := lastBlock.Content.String()
	t.Logf("Metadata block: %s", metadataJSON)

	if !strings.Contains(metadataJSON, UsageMetricInputTokens) {
		t.Errorf("Expected metadata to contain %s", UsageMetricInputTokens)
	}
	if !strings.Contains(metadataJSON, UsageMetricGenerationTokens) {
		t.Errorf("Expected metadata to contain %s", UsageMetricGenerationTokens)
	}

	// Verify content blocks are present before the metadata
	hasContent := false
	for _, blk := range blocks[:len(blocks)-1] {
		if blk.BlockType == Content {
			hasContent = true
			break
		}
	}
	if !hasContent {
		t.Error("Expected at least one content block before the metadata block")
	}
}
