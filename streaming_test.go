package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"testing"
)

func TestStreamingAdapterBlockCompression(t *testing.T) {
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
}

func TestStreamingAdapterPreservesResponseExtraFields(t *testing.T) {
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
}

func TestStreamingAdapterRejectsConflictingResponseExtraFields(t *testing.T) {
	adapter := &StreamingAdapter{S: &mockStreamingGenerator{chunks: []StreamChunk{
		{Block: TextBlock("hello"), ResponseExtraFields: map[string]interface{}{"request_id": "req_1"}},
		{Block: TextBlock(" world"), ResponseExtraFields: map[string]interface{}{"request_id": "req_2"}},
	}}}

	_, err := adapter.Generate(t.Context(), GenerationRequest{})
	if err == nil || !strings.Contains(err.Error(), "conflicting response extra field") {
		t.Fatalf("Generate() error = %v, want conflicting response extra field", err)
	}
}

func TestStreamChunkErrorIsNotSerialized(t *testing.T) {
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
