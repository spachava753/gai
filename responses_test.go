package gai

import (
	"context"
	"encoding/base64"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"os"
	"strings"
	"testing"
)

func TestResponsesGenerator_Generate_pdf(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	pdfBytes, err := os.ReadFile("sample.pdf")
	if err != nil {
		t.Skip("could not open sample.pdf")
		return
	}
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, "You are a helpful assistant.")
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("What is the title of this PDF? Just output the title and nothing else"),
				PDFBlock(pdfBytes, "sample.pdf"),
			},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ThinkingBudget: "low"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Find the first Content block (skip Thinking blocks from reasoning)
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			if got := blk.Content.String(); got == "" {
				t.Fatal("expected non-empty content")
			}
			break
		}
	}
}
func TestResponsesGenerator_Generate_image(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		t.Skip("could not open sample.jpg")
		return
	}
	imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, "You are a helpful assistant.")
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
					Content:      Str("What is in this image? (Hint, it's a character from The Croods, a DreamWorks animated movie.) Answer with only the name of the character"),
				},
			},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: Ptr(512), ThinkingBudget: "high"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) != 1 {
		t.Fatalf("candidates = %d, want 1", len(resp.Candidates))
	}
	if len(resp.Candidates[0].Blocks) == 0 {
		t.Fatal("expected at least one block")
	}
	// Find the first Content block (skip Thinking blocks from reasoning)
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			if !strings.Contains(blk.Content.String(), "Guy") {
				t.Fatalf("content does not contain Guy")
			}
			break
		}
	}
}
func TestResponsesGenerator_Generate(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, "You are a helpful assistant")
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Hi!")}}}
	resp, err := gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(resp.Candidates); got == 0 {
		t.Fatal("expected at least one item")
	}
}
func TestResponsesGenerator_Generate_thinking(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5, "You are a helpful assistant")
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Are LLMs conscious? Think it through and give a comprehensive answer")}}}
	opts := GenOpts{ThinkingBudget: "medium", Temperature: Ptr(1.0), ExtraArgs: map[string]any{
		ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
		ResponsesPromptCacheKeyParam:       "responses-thinking-example:v1",
	}}
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Reuse the same prompt cache key on the follow-up request because both turns
	// share the same system instructions and overall prompt prefix shape.
	// The generator is stateless: just append the assistant response and continue.
	// Reasoning blocks with encrypted content are automatically reconstructed as
	// input reasoning items on the next call.
	dialog = append(dialog, resp.Candidates[0], Message{Role: User, Blocks: []Block{TextBlock("What can you do?")}})
	resp, err = gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(resp.Candidates); got == 0 {
		t.Fatal("expected at least one item")
	}
}
func TestResponsesGenerator_Register(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`)
	tickerTool := Tool{
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
	if err := gen.Register(tickerTool); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}}}
	opts := GenOpts{ToolChoice: "get_stock_price"}
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Find the first ToolCall block (reasoning models may produce Thinking blocks before tool calls)
	var toolCallBlock Block
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == ToolCall {
			toolCallBlock = blk
			break
		}
	}
	if got := toolCallBlock.Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	// Append the assistant's response and the tool result. The generator is stateless
	// and manages conversation context through the dialog.
	dialog = append(dialog, resp.Candidates[0], Message{Role: ToolResult, Blocks: []Block{{ID: toolCallBlock.ID, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")}}})
	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Find the first Content block in the final response
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			if got := blk.Content.String(); got == "" {
				t.Fatal("expected non-empty content")
			}
			break
		}
	}
}
func TestResponsesGenerator_Register_parallelToolUse(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, `You are a helpful assistant that compares the price of two stocks and returns the ticker of whichever is greater.
Only mentioned the ticker and nothing else.
Only output the price, like
<example>
User: Which one is more expensive? Apple or NVidia?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: Apple: 123.45; Nvidia: 345.65
Assistant: Nvidia
</example>
`)
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.\nYou can call this tool in parallel",
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
	if err := gen.Register(tickerTool); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Which stock, Apple vs. Microsoft, is more expensive?")}}}
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ThinkingBudget: "medium"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Collect ToolCall blocks (reasoning models may produce Thinking blocks before tool calls)
	var toolCallBlocks []Block
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == ToolCall {
			toolCallBlocks = append(toolCallBlocks, blk)
		}
	}
	if got := toolCallBlocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	if got := toolCallBlocks[1].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	// Append the assistant's response and tool results. The generator is stateless
	// and manages conversation context through the dialog.
	dialog = append(dialog, resp.Candidates[0], Message{Role: ToolResult, Blocks: []Block{{ID: toolCallBlocks[0].ID, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")}}}, Message{Role: ToolResult, Blocks: []Block{{ID: toolCallBlocks[1].ID, ModalityType: Text, MimeType: "text/plain", Content: Str("678.45")}}})
	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Find the first Content block in the final response
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			if got := blk.Content.String(); got == "" {
				t.Fatal("expected non-empty content")
			}
			break
		}
	}
}

// ExampleStreamingAdapter_responses demonstrates using StreamingAdapter with
// the ResponsesGenerator for stateless multi-turn conversation. The adapter
// compresses streaming chunks into complete Response objects, making it easy
// to append the assistant's response to the dialog for subsequent turns.
func TestStreamingAdapter_responses(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := openai.NewClient(option.WithAPIKey(apiKey))
	// Create the generator and wrap it with StreamingAdapter.
	// StreamingAdapter.Generate streams internally, then compresses chunks into
	// a standard Response — identical to what ResponsesGenerator.Generate returns.
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Nano, "You are a helpful assistant")
	adapter := &StreamingAdapter{S: &gen}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Hi!")}}}
	resp, err := adapter.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The adapter produces a complete Response with Candidates, just like Generate.
	// Append the assistant's message and continue the conversation statelessly.
	dialog = append(dialog, resp.Candidates[0], Message{Role: User, Blocks: []Block{TextBlock("What can you help me with?")}})
	resp, err = adapter.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(resp.Candidates); got == 0 {
		t.Fatal("expected at least one item")
	}
}

// ExampleStreamingAdapter_responses_toolUse demonstrates using StreamingAdapter
// with tool calling on the Responses API. The adapter compresses streaming tool
// call chunks into complete blocks, preserving IDs and Thinking block ExtraFields
// so the dialog can be passed back for subsequent turns without any manual chunk
// reconstruction.
func TestStreamingAdapter_responses_toolUse(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`)
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.\nYou can call this tool in parallel",
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
	if err := gen.Register(tickerTool); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// StreamingAdapter wraps the generator so we get compressed Response objects
	// instead of raw streaming chunks.
	adapter := &StreamingAdapter{S: &gen}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Which stock, Apple vs. Microsoft, is more expensive?")}}}
	// Turn 1: the model should call get_stock_price for both tickers.
	resp, err := adapter.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Collect the tool call blocks from the compressed response.
	var toolCallBlocks []Block
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == ToolCall {
			toolCallBlocks = append(toolCallBlocks, blk)
		}
	}
	if got := toolCallBlocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	if got := toolCallBlocks[1].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	// Append the full assistant message (including any Thinking blocks with encrypted
	// reasoning content) and tool results. This is the key advantage of StreamingAdapter:
	// the compressed Candidates[0] is directly usable in the dialog.
	dialog = append(dialog, resp.Candidates[0],
		Message{Role: ToolResult, Blocks: []Block{{ID: toolCallBlocks[0].ID, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")}}},
		Message{Role: ToolResult, Blocks: []Block{{ID: toolCallBlocks[1].ID, ModalityType: Text, MimeType: "text/plain", Content: Str("678.45")}}},
	)
	// Turn 2: the model responds with the answer.
	resp, err = adapter.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			if got := blk.Content.String(); got == "" {
				t.Fatal("expected non-empty content")
			}
			break
		}
	}
}

// ExampleResponsesGenerator_Stream_thinking demonstrates consuming the raw
// streaming iterator with a reasoning model. The stream yields thinking chunks
// (reasoning deltas) interleaved with content chunks. At the end, a metadata
// block carries usage information. This example also shows how to build a
// dialog-ready assistant message from the streamed blocks using
// compressStreamingBlocks (via StreamingAdapter) for a follow-up turn.
func TestResponsesGenerator_Stream_thinking(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Nano, "You are a helpful assistant")
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the capital of France? Reply with just the city name.")}}}
	opts := &GenOpts{
		ThinkingBudget: "low",
		ExtraArgs: map[string]any{
			ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
		},
	}
	// Use StreamingAdapter so the streamed output is automatically compressed
	// into a proper Response with Thinking blocks carrying ExtraFields (including
	// encrypted reasoning content for stateless multi-turn conversations).
	adapter := &StreamingAdapter{S: &gen}
	resp, err := adapter.Generate(context.Background(), dialog, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The compressed response preserves Thinking blocks from the reasoning model.
	hasThinking := false
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Thinking {
			hasThinking = true
			break
		}
	}
	if !hasThinking {
		t.Fatal("expected compressed response to preserve at least one thinking block")
	}
	// Append the full assistant message to the dialog. Thinking blocks with
	// encrypted content are included, so the next call can reconstruct reasoning
	// input items automatically.
	dialog = append(dialog, resp.Candidates[0], Message{Role: User, Blocks: []Block{TextBlock("And what country is that in?")}})
	resp, err = adapter.Generate(context.Background(), dialog, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Find the content block in the follow-up response.
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			if !strings.Contains(blk.Content.String(), "France") {
				t.Fatalf("content does not contain France")
			}
			break
		}
	}
}
