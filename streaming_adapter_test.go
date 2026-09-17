package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type streamingGeneratorFunc func(context.Context, GenerationRequest) iter.Seq[StreamChunk]

func (f streamingGeneratorFunc) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return f(ctx, request)
}

// ExampleStreamingAdapter demonstrates how to use StreamingAdapter to convert
// a StreamingGenerator to a regular Generator. This is useful when you want to
// use streaming internally but present a non-streaming interface to users.
func TestStreamAssemblyScenarios(t *testing.T) {
	t.Run("StreamChunkErrorIsNotSerialized", func(t *testing.T) {
		chunk := StreamChunk{
			Block: TextBlock("partial"),
			Err:   errors.New("sensitive failure"),
		}

		encoded, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		if strings.Contains(string(encoded), "sensitive failure") || strings.Contains(string(encoded), `"Err"`) || strings.Contains(string(encoded), `"err"`) {
			t.Fatalf("serialized chunk contains runtime error: %s", encoded)
		}
	})
	t.Run("StreamingAdapter", func(t *testing.T) {
		requireLiveAPIKey(t, "OPENAI_API_KEY")
		gen := newLiveOpenAIGenerator(t)
		adapter := StreamingAdapter{S: gen}
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the capital of France?")}}}

		response, err := adapter.Generate(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("Generate returned error: %v", err)
		}

		content := requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0))
		requireTextContains(t, content, "Paris")
	})
	t.Run("StreamingAdapterBlockCompression", func(t *testing.T) {
		t.Run("compresses consecutive thinking blocks into one", func(t *testing.T) {
			blocks := []Block{
				{BlockType: Thinking, Content: Str("I think ")},
				{BlockType: Thinking, Content: Str("therefore ")},
				{BlockType: Thinking, Content: Str("I am.")},
			}
			compressed, err := compressStreamingBlocks(blocks)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(compressed) != 1 || compressed[0].BlockType != Thinking || compressed[0].Content.String() != "I think therefore I am." {
				t.Errorf("expected 1 thinking block with merged content, got %+v", compressed)
			}
		})

		t.Run("separator preserves consecutive thinking block boundaries", func(t *testing.T) {
			blocks := []Block{
				{BlockType: Thinking, ModalityType: Text, Content: Str("first "), ExtraFields: map[string]interface{}{"item": "a"}},
				{BlockType: Thinking, ModalityType: Text, Content: Str("block")},
				SeparatorBlock(),
				{BlockType: Thinking, ModalityType: Text, Content: Str("second"), ExtraFields: map[string]interface{}{"item": "b"}},
			}
			compressed, err := compressStreamingBlocks(blocks)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(compressed) != 2 {
				t.Fatalf("expected 2 thinking blocks, got %+v", compressed)
			}
			if got := compressed[0].Content.String(); got != "first block" {
				t.Fatalf("first thinking content = %q, want %q", got, "first block")
			}
			if got := compressed[1].Content.String(); got != "second" {
				t.Fatalf("second thinking content = %q, want %q", got, "second")
			}
			if got := compressed[0].ExtraFields["item"]; got != "a" {
				t.Fatalf("first thinking extra item = %v, want a", got)
			}
			if got := compressed[1].ExtraFields["item"]; got != "b" {
				t.Fatalf("second thinking extra item = %v, want b", got)
			}
		})

		t.Run("separator preserves consecutive content block boundaries", func(t *testing.T) {
			blocks := []Block{
				TextBlock("first"),
				SeparatorBlock(),
				TextBlock("second"),
			}
			compressed, err := compressStreamingBlocks(blocks)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(compressed) != 2 {
				t.Fatalf("expected 2 content blocks, got %+v", compressed)
			}
			if got := compressed[0].Content.String(); got != "first" {
				t.Fatalf("first content = %q, want first", got)
			}
			if got := compressed[1].Content.String(); got != "second" {
				t.Fatalf("second content = %q, want second", got)
			}
		})

		t.Run("compresses consecutive text/content blocks into one", func(t *testing.T) {
			blocks := []Block{
				{BlockType: Content, Content: Str("Hello, ")},
				{BlockType: Content, Content: Str("world!")},
			}
			compressed, err := compressStreamingBlocks(blocks)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(compressed) != 1 || compressed[0].BlockType != Content || compressed[0].Content.String() != "Hello, world!" {
				t.Errorf("expected 1 content block with merged content, got %+v", compressed)
			}
		})

		t.Run("preserves thought images between thinking text runs", func(t *testing.T) {
			image := Block{BlockType: Thinking, ModalityType: Image, MimeType: "image/png", Content: Str("AQ=="), ExtraFields: map[string]interface{}{"signature": "thought-signature"}}
			blocks := []Block{{BlockType: Thinking, Content: Str("before ")}, {BlockType: Thinking, Content: Str("image")}, image, image, {BlockType: Thinking, Content: Str("after ")}, {BlockType: Thinking, Content: Str("image")}}
			compressed, err := compressStreamingBlocks(blocks)
			if err != nil {
				t.Fatal(err)
			}
			if len(compressed) != 4 || compressed[0].Content.String() != "before image" || compressed[3].Content.String() != "after image" {
				t.Fatalf("unexpected compressed blocks: %+v", compressed)
			}
			if !reflect.DeepEqual(compressed[1:3], blocks[2:4]) {
				t.Fatalf("thought images or metadata changed: %+v", compressed[1:3])
			}
		})

		t.Run("preserves separate media payloads between text runs", func(t *testing.T) {
			image := Block{BlockType: Content, ModalityType: Image, MimeType: "image/png", Content: Str("AQ=="), ExtraFields: map[string]interface{}{"signature": "image-signature"}}
			audio := Block{BlockType: Content, ModalityType: Audio, MimeType: "audio/pcm;rate=24000", Content: Str("Ag=="), ExtraFields: map[string]interface{}{"signature": "audio-signature"}}
			blocks := []Block{TextBlock("before "), TextBlock("image"), image, image, audio, TextBlock("after "), TextBlock("audio")}
			compressed, err := compressStreamingBlocks(blocks)
			if err != nil {
				t.Fatal(err)
			}
			if len(compressed) != 5 || compressed[0].Content.String() != "before image" || compressed[4].Content.String() != "after audio" {
				t.Fatalf("unexpected compressed blocks: %+v", compressed)
			}
			if !reflect.DeepEqual(compressed[1:4], blocks[2:5]) {
				t.Fatalf("media payloads or metadata changed: %+v", compressed[1:4])
			}
		})

		t.Run("compresses tool call block deltas (single call simple)", func(t *testing.T) {
			// Tool call: id set and tool name, then partial parameter chunks, e.g. '{"param":"va', 'lue"}'
			partial1 := `{"param":"va` // not full JSON
			partial2 := `lue"}`
			id := "call_123"
			blocks := []Block{
				// 'header' block signals tool call start
				{BlockType: ToolCall, ID: id, Content: Str("weather"), ModalityType: Text},
				// chunked tool call parameters -- these are seen in streaming
				{BlockType: ToolCall, ID: "", Content: Str(partial1), ModalityType: Text},
				{BlockType: ToolCall, ID: "", Content: Str(partial2), ModalityType: Text},
			}
			compressed, err := compressStreamingBlocks(blocks)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(compressed) != 1 || compressed[0].BlockType != ToolCall {
				t.Fatalf("expected 1 compressed tool call block, got %+v", compressed)
			}
			want := ToolCallInput{Name: "weather", Parameters: map[string]any{"param": "value"}}
			var got ToolCallInput
			if err := json.Unmarshal([]byte(compressed[0].Content.String()), &got); err != nil {
				t.Fatalf("tool call content not valid json: %v", err)
			}
			if got.Name != want.Name {
				t.Errorf("expected tool name %q, got %q", want.Name, got.Name)
			}
			if fmt.Sprintf("%v", got.Parameters) != fmt.Sprintf("%v", want.Parameters) {
				t.Errorf("expected tool parameters %v, got %v", want.Parameters, got.Parameters)
			}
		})

		t.Run("compresses two different tool call blocks (with chunked deltas)", func(t *testing.T) {
			id1, id2 := "call_a", "call_b"
			blocks := []Block{
				{BlockType: ToolCall, ID: id1, Content: Str("foo"), ModalityType: Text},
				{BlockType: ToolCall, ID: "", Content: Str(`{"a":"b`), ModalityType: Text},
				{BlockType: ToolCall, ID: "", Content: Str(`ar"}`), ModalityType: Text},
				{BlockType: ToolCall, ID: id2, Content: Str("bar"), ModalityType: Text},
				{BlockType: ToolCall, ID: "", Content: Str(`{"x":"1"}`), ModalityType: Text},
			}
			compressed, err := compressStreamingBlocks(blocks)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(compressed) != 2 {
				t.Fatalf("expected 2 compressed tool call blocks, got %+v", compressed)
			}
			var ci1, ci2 ToolCallInput
			if err := json.Unmarshal([]byte(compressed[0].Content.String()), &ci1); err != nil {
				t.Fatalf("tool call 1 not valid json: %v", err)
			}
			if err := json.Unmarshal([]byte(compressed[1].Content.String()), &ci2); err != nil {
				t.Fatalf("tool call 2 not valid json: %v", err)
			}
			if ci1.Name != "foo" || ci2.Name != "bar" {
				t.Errorf("unexpected tool call names: %+v %+v", ci1, ci2)
			}
			if ci1.Parameters["a"] != "bar" {
				t.Errorf("unexpected param for call 1: %+v", ci1.Parameters)
			}
			if ci2.Parameters["x"] != "1" {
				t.Errorf("unexpected param for call 2: %+v", ci2.Parameters)
			}
		})

		t.Run("returns error on unknown/unsupported block type", func(t *testing.T) {
			blocks := []Block{{BlockType: "xyz_customtype", Content: Str("no")}}
			_, err := compressStreamingBlocks(blocks)
			if err == nil {
				t.Fatalf("expected error on unsupported block type, got nil")
			}
			if !strings.Contains(err.Error(), "unsupported") {
				t.Errorf("unexpected error: %v", err)
			}
		})
	})
	t.Run("StreamingAdapterGeneratePreservesMessageExtraFields", func(t *testing.T) {
		adapter := &StreamingAdapter{S: &mockStreamingGenerator{chunks: []StreamChunk{
			{
				Block: Block{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str("Thinking out loud"),
				},
				MessageExtraFields: map[string]interface{}{
					ResponsesMessageExtraFieldPhase: ResponsesMessagePhaseCommentary,
				},
			},
			{
				Block: MetadataBlock(Metadata{UsageMetricInputTokens: 1}),
				MessageExtraFields: map[string]interface{}{
					ResponsesMessageExtraFieldPhase: ResponsesMessagePhaseCommentary,
				},
			},
		}}}

		resp, err := adapter.Generate(context.Background(), GenerationRequest{Dialog: Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("hello")},
		}}})
		if err != nil {
			t.Fatalf("StreamingAdapter.Generate failed: %v", err)
		}
		got, ok := resp.Candidates[0].ExtraFields[ResponsesMessageExtraFieldPhase].(string)
		if !ok {
			t.Fatalf("expected message extra fields to be preserved, got %+v", resp.Candidates[0].ExtraFields)
		}
		if got != ResponsesMessagePhaseCommentary {
			t.Fatalf("expected phase %q, got %q", ResponsesMessagePhaseCommentary, got)
		}
	})
	t.Run("StreamingAdapterGeneratePreservesToolOnlyMessageExtraFieldsForReplay", func(t *testing.T) {
		adapter := &StreamingAdapter{S: &mockStreamingGenerator{chunks: []StreamChunk{
			{
				Block: Block{
					ID:           "call_stream_1",
					BlockType:    ToolCall,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str("lookup"),
				},
				MessageExtraFields: map[string]interface{}{
					ResponsesMessageExtraFieldPhase: ResponsesMessagePhaseCommentary,
				},
			},
			{
				Block: Block{
					BlockType:    ToolCall,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(`{"topic":"population"}`),
				},
				MessageExtraFields: map[string]interface{}{
					ResponsesMessageExtraFieldPhase: ResponsesMessagePhaseCommentary,
				},
			},
			{
				Block: MetadataBlock(Metadata{UsageMetricInputTokens: 1}),
				MessageExtraFields: map[string]interface{}{
					ResponsesMessageExtraFieldPhase: ResponsesMessagePhaseCommentary,
				},
			},
		}}}

		resp, err := adapter.Generate(context.Background(), GenerationRequest{Dialog: Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("hello")},
		}}})
		if err != nil {
			t.Fatalf("StreamingAdapter.Generate failed: %v", err)
		}
		candidate := resp.Candidates[0]
		got, ok := candidate.ExtraFields[ResponsesMessageExtraFieldPhase].(string)
		if !ok {
			t.Fatalf("expected message extra fields to be preserved, got %+v", candidate.ExtraFields)
		}
		if got != ResponsesMessagePhaseCommentary {
			t.Fatalf("expected phase %q, got %q", ResponsesMessagePhaseCommentary, got)
		}

		var gotToolCall ToolCallInput
		if err := json.Unmarshal([]byte(candidate.Blocks[0].Content.String()), &gotToolCall); err != nil {
			t.Fatalf("expected compressed tool call content to be valid JSON: %v", err)
		}
		if gotToolCall.Name != "lookup" {
			t.Fatalf("expected compressed tool call name %q, got %q", "lookup", gotToolCall.Name)
		}
		if gotToolCall.Parameters["topic"] != "population" {
			t.Fatalf("expected compressed tool call topic %q, got %+v", "population", gotToolCall.Parameters)
		}

		gen := NewResponsesGenerator(nil)
		items, err := gen.buildInputItems(Dialog{candidate})
		if err != nil {
			t.Fatalf("buildInputItems failed: %v", err)
		}
		if len(items) != 2 {
			t.Fatalf("expected output_message + tool call on replay, got %d items", len(items))
		}
		if items[0].OfOutputMessage == nil {
			t.Fatalf("expected first replay item to be output_message, got %+v", items[0])
		}
		if got := string(items[0].OfOutputMessage.Phase); got != ResponsesMessagePhaseCommentary {
			t.Fatalf("expected replayed phase %q, got %q", ResponsesMessagePhaseCommentary, got)
		}
		if items[1].OfFunctionCall == nil {
			t.Fatalf("expected second replay item to be function call, got %+v", items[1])
		}
		if items[1].OfFunctionCall.Name != "lookup" {
			t.Fatalf("expected replayed function name %q, got %q", "lookup", items[1].OfFunctionCall.Name)
		}
		if items[1].OfFunctionCall.Arguments != `{"topic":"population"}` {
			t.Fatalf("expected replayed arguments %s, got %s", `{"topic":"population"}`, items[1].OfFunctionCall.Arguments)
		}
	})
	t.Run("StreamingAdapterPreservesResponseExtraFields", func(t *testing.T) {
		adapter := &StreamingAdapter{S: &mockStreamingGenerator{chunks: []StreamChunk{
			{
				Block:               TextBlock("hello"),
				ResponseExtraFields: map[string]interface{}{"request_id": "req_123"},
			},
			{
				Block:               MetadataBlock(Metadata{UsageMetricInputTokens: 1}),
				ResponseExtraFields: map[string]interface{}{"request_id": "req_123", "model": "test-model"},
			},
		}}}

		response, err := adapter.Generate(t.Context(), GenerationRequest{})
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if response.ExtraFields["request_id"] != "req_123" || response.ExtraFields["model"] != "test-model" {
			t.Fatalf("response extra fields = %v", response.ExtraFields)
		}
	})
	t.Run("StreamingAdapterRejectsConflictingResponseExtraFields", func(t *testing.T) {
		adapter := &StreamingAdapter{S: &mockStreamingGenerator{chunks: []StreamChunk{
			{Block: TextBlock("hello"), ResponseExtraFields: map[string]interface{}{"request_id": "req_1"}},
			{Block: TextBlock(" world"), ResponseExtraFields: map[string]interface{}{"request_id": "req_2"}},
		}}}

		_, err := adapter.Generate(t.Context(), GenerationRequest{})
		if err == nil || !strings.Contains(err.Error(), "conflicting response extra field") {
			t.Fatalf("Generate() error = %v, want conflicting response extra field", err)
		}
	})
	t.Run("StreamingAdapterStopsOnErrorChunk", func(t *testing.T) {
		wantErr := errors.New("stream failed")
		continuedAfterError := false
		generator := streamingGeneratorFunc(func(context.Context, GenerationRequest) iter.Seq[StreamChunk] {
			return func(yield func(StreamChunk) bool) {
				if !yield(StreamChunk{Err: wantErr}) {
					return
				}
				continuedAfterError = true
				yield(StreamChunk{Block: TextBlock("unexpected")})
			}
		})

		_, err := (&StreamingAdapter{S: generator}).Generate(context.Background(), GenerationRequest{})
		if !errors.Is(err, wantErr) {
			t.Fatalf("Generate() error = %v, want %v", err, wantErr)
		}
		if continuedAfterError {
			t.Fatal("stream continued after terminal error chunk")
		}
	})
	t.Run("StreamingAdapter/customUsage", func(t *testing.T) {
		customGen := &customStreamingGenerator{systemPrompt: "You are a helpful assistant."}
		adapter := StreamingAdapter{S: customGen}
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Hello!")}}}

		response, err := adapter.Generate(context.Background(), GenerationRequest{Dialog: dialog})
		if err != nil {
			t.Fatalf("Generate returned error: %v", err)
		}

		content := requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0))
		content = strings.TrimPrefix(content, "Mock response: ")
		if got, want := content, "Hello! How can I help you today?"; got != want {
			t.Fatalf("content = %q, want %q", got, want)
		}
	})
	t.Run("StreamingAdapter/errorHandling", func(t *testing.T) {
		gen := newTestOpenAIGenerator(t, &mockChatCompletionService{})
		adapter := StreamingAdapter{S: gen}

		_, err := adapter.Generate(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		})
		if !errors.Is(err, ErrEmptyDialog) {
			t.Fatalf("Generate error = %v, want %v", err, ErrEmptyDialog)
		}
	})
	t.Run("StreamingAdapter/multipleBlocks", func(t *testing.T) {
		mockGen := &mockStreamingGenerator{
			chunks: []StreamChunk{
				{Block: Block{BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("The weather in ")}, CandidatesIndex: 0},
				{Block: Block{BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("Paris is ")}, CandidatesIndex: 0},
				{Block: Block{BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("sunny today.")}, CandidatesIndex: 0},
			},
		}
		adapter := StreamingAdapter{S: mockGen}

		response, err := adapter.Generate(context.Background(), GenerationRequest{})
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
	})
	t.Run("StreamingAdapter/parallelToolCalls", func(t *testing.T) {
		requireLiveAPIKey(t, "OPENAI_API_KEY")
		gen := newLiveOpenAIGenerator(t)

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

		adapter := StreamingAdapter{S: gen}
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What are the current prices of Apple and Microsoft stocks?")}}}
		response, err := adapter.Generate(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock("You are a helpful stock price assistant.")),
			Dialog:       dialog,
			Tools:        []Tool{stockTool},
			Options:      NewGenerationOptions(WithToolChoice(ToolChoiceAuto)),
		})
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
	})
	t.Run("StreamingAdapter/responses", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := openai.NewClient(option.WithAPIKey(apiKey))
		// Create the generator and wrap it with StreamingAdapter.
		// StreamingAdapter.Generate streams internally, then compresses chunks into
		// a standard Response — identical to what ResponsesGenerator.Generate returns.
		gen := NewResponsesGenerator(&client.Responses)
		adapter := &StreamingAdapter{S: gen}
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Hi!")}}}
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5Nano,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
		}
		resp, err := adapter.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// The adapter produces a complete Response with Candidates, just like Generate.
		// Append the assistant's message and continue the conversation statelessly.
		dialog = append(dialog, resp.Candidates[0], Message{Role: User, Blocks: []Block{TextBlock("What can you help me with?")}})
		request.Dialog = dialog
		resp, err = adapter.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := len(resp.Candidates); got == 0 {
			t.Fatal("expected at least one item")
		}
	})
	t.Run("StreamingAdapter/responses/toolUse", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)
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
		// StreamingAdapter wraps the generator so we get compressed Response objects
		// instead of raw streaming chunks.
		adapter := &StreamingAdapter{S: gen}
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Which stock, Apple vs. Microsoft, is more expensive?")}}}
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock(openAIStockInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
		}
		// Turn 1: the model should call get_stock_price for both tickers.
		resp, err := adapter.Generate(context.Background(), request)
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
		request.Dialog = dialog
		resp, err = adapter.Generate(context.Background(), request)
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
	})
	t.Run("StreamingAdapter/separatorBlocksAreInternal", func(t *testing.T) {
		mockGen := &mockStreamingGenerator{
			chunks: []StreamChunk{
				{Block: Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str("first "), ExtraFields: map[string]interface{}{"item": "first"}}},
				{Block: Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str("thinking")}},
				{Block: SeparatorBlock()},
				{Block: Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str("second thinking"), ExtraFields: map[string]interface{}{"item": "second"}}},
			},
		}
		adapter := StreamingAdapter{S: mockGen}

		response, err := adapter.Generate(context.Background(), GenerationRequest{})
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
	})
	t.Run("StreamingAdapter/withTools", func(t *testing.T) {
		requireLiveAPIKey(t, "OPENAI_API_KEY")
		gen := newLiveOpenAIGenerator(t)

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

		adapter := StreamingAdapter{S: gen}
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What's the weather like in New York?")}}}
		response, err := adapter.Generate(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock("You are a helpful weather assistant.")),
			Dialog:       dialog,
			Tools:        []Tool{weatherTool},
			Options:      NewGenerationOptions(WithToolChoice(ToolChoiceAuto)),
		})
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
	})
}

// customStreamingGenerator is an example of a custom StreamingGenerator implementation
type customStreamingGenerator struct {
	systemPrompt string
}

func (c *customStreamingGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		// Validate input
		if len(request.Dialog) == 0 {
			yield(StreamChunk{Err: ErrEmptyDialog})
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
			}) {
				return // User stopped iteration
			}
		}
	}
}

// mockStreamingGenerator is a test helper that yields pre-defined chunks
type mockStreamingGenerator struct {
	chunks []StreamChunk
	err    error
}

func (m *mockStreamingGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		for _, chunk := range m.chunks {
			if !yield(chunk) {
				return
			}
		}
		if m.err != nil {
			yield(StreamChunk{Err: m.err})
		}
	}
}
