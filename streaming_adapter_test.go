package gai

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"

	"github.com/google/jsonschema-go/jsonschema"
)

// ExampleStreamingAdapter demonstrates how to use StreamingAdapter to convert
// a StreamingGenerator to a regular Generator. This is useful when you want to
// use streaming internally but present a non-streaming interface to users.
func TestStreamingAdapter(t *testing.T) {
	requireLiveAPIKey(t, "OPENAI_API_KEY")

	client := openai.NewClient()
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are a helpful assistant.",
	)
	adapter := StreamingAdapter{S: &gen}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the capital of France?")}}}

	response, err := adapter.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	content := requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0))
	requireTextContains(t, content, "Paris")
}

// ExampleStreamingAdapter_withTools demonstrates using StreamingAdapter with tool calls.
// The adapter handles the compression of streaming tool call chunks into complete tool calls.
func TestStreamingAdapter_withTools(t *testing.T) {
	requireLiveAPIKey(t, "OPENAI_API_KEY")

	client := openai.NewClient()
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are a helpful weather assistant.",
	)

	weatherTool := Tool{
		Name:        "get_weather",
		Description: "Get the current weather in a given location",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Location string `json:"location" jsonschema:"The city and state, e.g. San Francisco, CA"`
				Unit     string `json:"unit,omitempty" jsonschema:"The unit of temperature"`
			}]()
			if err != nil {
				t.Fatalf("generate weather schema: %v", err)
			}
			return schema
		}(),
	}
	if err := gen.Register(weatherTool); err != nil {
		t.Fatalf("register weather tool: %v", err)
	}

	adapter := StreamingAdapter{S: &gen}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What's the weather like in New York?")}}}
	response, err := adapter.Generate(context.Background(), dialog, &GenOpts{ToolChoice: ToolChoiceAuto})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	calls := collectToolCalls(t, requireCandidate(t, response).Blocks)
	if len(calls) == 0 {
		t.Fatalf("expected at least one tool call, got response %#v", response)
	}
	if calls[0].Name != "get_weather" {
		t.Fatalf("tool name = %q, want get_weather", calls[0].Name)
	}
	location, ok := calls[0].Parameters["location"].(string)
	if !ok || !strings.Contains(location, "New York") {
		t.Fatalf("location parameter = %#v, want a New York location", calls[0].Parameters["location"])
	}
}

// ExampleStreamingAdapter_errorHandling demonstrates how StreamingAdapter handles
// errors that occur during streaming.
func TestStreamingAdapter_errorHandling(t *testing.T) {
	client := openai.NewClient()
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are a helpful assistant.",
	)
	adapter := StreamingAdapter{S: &gen}

	_, err := adapter.Generate(context.Background(), Dialog{}, nil)
	if !errors.Is(err, ErrEmptyDialog) {
		t.Fatalf("Generate error = %v, want %v", err, ErrEmptyDialog)
	}
}

// ExampleStreamingAdapter_multipleBlocks demonstrates how StreamingAdapter handles
// responses with multiple blocks of different types, showing the compression of
// consecutive blocks of the same type.
func TestStreamingAdapter_multipleBlocks(t *testing.T) {
	mockGen := &mockStreamingGenerator{
		chunks: []StreamChunk{
			{Block: Block{BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("The weather in ")}, CandidatesIndex: 0},
			{Block: Block{BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("Paris is ")}, CandidatesIndex: 0},
			{Block: Block{BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("sunny today.")}, CandidatesIndex: 0},
		},
	}
	adapter := StreamingAdapter{S: mockGen}

	response, err := adapter.Generate(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	candidate := requireCandidate(t, response)
	if len(candidate.Blocks) != 1 {
		t.Fatalf("blocks = %d, want 1 compressed block", len(candidate.Blocks))
	}
	if got, want := requireContentBlock(t, candidate.Blocks[0]), "The weather in Paris is sunny today."; got != want {
		t.Fatalf("compressed content = %q, want %q", got, want)
	}
	if response.FinishReason != EndTurn {
		t.Fatalf("finish reason = %v, want %v", response.FinishReason, EndTurn)
	}
}

func TestStreamingAdapter_separatorBlocksAreInternal(t *testing.T) {
	mockGen := &mockStreamingGenerator{
		chunks: []StreamChunk{
			{Block: Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str("first "), ExtraFields: map[string]interface{}{"item": "first"}}},
			{Block: Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str("thinking")}},
			{Block: SeparatorBlock()},
			{Block: Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str("second thinking"), ExtraFields: map[string]interface{}{"item": "second"}}},
		},
	}
	adapter := StreamingAdapter{S: mockGen}

	response, err := adapter.Generate(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	candidate := requireCandidate(t, response)
	if len(candidate.Blocks) != 2 {
		t.Fatalf("blocks = %d, want 2 thinking blocks: %#v", len(candidate.Blocks), candidate.Blocks)
	}
	for i, block := range candidate.Blocks {
		if block.BlockType == Separator {
			t.Fatalf("separator leaked into final response at block %d", i)
		}
		if block.BlockType != Thinking {
			t.Fatalf("block %d type = %q, want %q", i, block.BlockType, Thinking)
		}
	}
	if got := candidate.Blocks[0].Content.String(); got != "first thinking" {
		t.Fatalf("first thinking content = %q, want %q", got, "first thinking")
	}
	if got := candidate.Blocks[1].Content.String(); got != "second thinking" {
		t.Fatalf("second thinking content = %q, want %q", got, "second thinking")
	}
	if got := candidate.Blocks[0].ExtraFields["item"]; got != "first" {
		t.Fatalf("first thinking extra item = %v, want first", got)
	}
	if got := candidate.Blocks[1].ExtraFields["item"]; got != "second" {
		t.Fatalf("second thinking extra item = %v, want second", got)
	}
}

// ExampleStreamingAdapter_parallelToolCalls demonstrates how StreamingAdapter handles
// parallel tool calls, showing the compression of multiple tool call chunks.
func TestStreamingAdapter_parallelToolCalls(t *testing.T) {
	requireLiveAPIKey(t, "OPENAI_API_KEY")

	client := openai.NewClient()
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are a helpful stock price assistant.",
	)

	stockTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
			}]()
			if err != nil {
				t.Fatalf("generate stock schema: %v", err)
			}
			return schema
		}(),
	}
	if err := gen.Register(stockTool); err != nil {
		t.Fatalf("register stock tool: %v", err)
	}

	adapter := StreamingAdapter{S: &gen}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What are the current prices of Apple and Microsoft stocks?")}}}
	response, err := adapter.Generate(context.Background(), dialog, &GenOpts{ToolChoice: ToolChoiceAuto})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if response.FinishReason != ToolUse {
		t.Fatalf("finish reason = %v, want %v", response.FinishReason, ToolUse)
	}

	calls := collectToolCalls(t, requireCandidate(t, response).Blocks)
	if len(calls) < 2 {
		t.Fatalf("tool calls = %d, want at least 2", len(calls))
	}
	requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "AAPL")
	requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "MSFT")
}

// ExampleStreamingAdapter_customUsage shows how to create a custom generator
// that implements StreamingGenerator and use it with StreamingAdapter.
func TestStreamingAdapter_customUsage(t *testing.T) {
	customGen := &customStreamingGenerator{systemPrompt: "You are a helpful assistant."}
	adapter := StreamingAdapter{S: customGen}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Hello!")}}}

	response, err := adapter.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	content := requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0))
	content = strings.TrimPrefix(content, "Mock response: ")
	if got, want := content, "Hello! How can I help you today?"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
}

// customStreamingGenerator is an example of a custom StreamingGenerator implementation
type customStreamingGenerator struct {
	systemPrompt string
}

func (c *customStreamingGenerator) Stream(ctx context.Context, dialog Dialog, options *GenOpts) iter.Seq2[StreamChunk, error] {
	return func(yield func(StreamChunk, error) bool) {
		// Validate input
		if len(dialog) == 0 {
			yield(StreamChunk{}, ErrEmptyDialog)
			return
		}

		// Simulate streaming response chunks
		responseChunks := []string{"Mock response: ", "Hello! ", "How can I ", "help you ", "today?"}

		for _, chunk := range responseChunks {
			if !yield(StreamChunk{
				Block: Block{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(chunk),
				},
				CandidatesIndex: 0,
			}, nil) {
				return // User stopped iteration
			}
		}
	}
}
