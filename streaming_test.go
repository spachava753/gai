package gai

import (
	"encoding/json"
	"fmt"
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
