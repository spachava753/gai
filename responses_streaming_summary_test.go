package gai

import (
	"context"
	"fmt"
	"testing"

	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
)

type responsesStreamService struct {
	events []ssestream.Event
}

func (s *responsesStreamService) New(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (*responses.Response, error) {
	return nil, fmt.Errorf("New not implemented")
}

func (s *responsesStreamService) NewStreaming(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) *ssestream.Stream[responses.ResponseStreamEventUnion] {
	return ssestream.NewStream[responses.ResponseStreamEventUnion](&responsesStreamDecoder{events: s.events}, nil)
}

type responsesStreamDecoder struct {
	events []ssestream.Event
	index  int
	cur    ssestream.Event
}

func (d *responsesStreamDecoder) Next() bool {
	if d.index >= len(d.events) {
		return false
	}
	d.cur = d.events[d.index]
	d.index++
	return true
}

func (d *responsesStreamDecoder) Event() ssestream.Event { return d.cur }
func (d *responsesStreamDecoder) Close() error           { return nil }
func (d *responsesStreamDecoder) Err() error             { return nil }

func responseSSEvent(eventType, data string) ssestream.Event {
	return ssestream.Event{Type: eventType, Data: []byte(data)}
}

func TestResponsesStreamingAdapterPreservesReasoningSummaryParts(t *testing.T) {
	const reasoningID = "rs_123"
	const encrypted = "encrypted-reasoning"

	svc := &responsesStreamService{events: []ssestream.Event{
		responseSSEvent("response.reasoning_summary_part.added", `{"type":"response.reasoning_summary_part.added","sequence_number":1,"item_id":"rs_123","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}`),
		responseSSEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","sequence_number":2,"item_id":"rs_123","output_index":0,"summary_index":0,"delta":"first "}`),
		responseSSEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","sequence_number":3,"item_id":"rs_123","output_index":0,"summary_index":0,"delta":"summary"}`),
		responseSSEvent("response.reasoning_summary_text.done", `{"type":"response.reasoning_summary_text.done","sequence_number":4,"item_id":"rs_123","output_index":0,"summary_index":0,"text":"first summary"}`),
		responseSSEvent("response.reasoning_summary_part.done", `{"type":"response.reasoning_summary_part.done","sequence_number":5,"item_id":"rs_123","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":"first summary"}}`),
		responseSSEvent("response.reasoning_summary_part.added", `{"type":"response.reasoning_summary_part.added","sequence_number":6,"item_id":"rs_123","output_index":0,"summary_index":1,"part":{"type":"summary_text","text":""}}`),
		responseSSEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","sequence_number":7,"item_id":"rs_123","output_index":0,"summary_index":1,"delta":"second summary"}`),
		responseSSEvent("response.reasoning_summary_text.done", `{"type":"response.reasoning_summary_text.done","sequence_number":8,"item_id":"rs_123","output_index":0,"summary_index":1,"text":"second summary"}`),
		responseSSEvent("response.reasoning_summary_part.done", `{"type":"response.reasoning_summary_part.done","sequence_number":9,"item_id":"rs_123","output_index":0,"summary_index":1,"part":{"type":"summary_text","text":"second summary"}}`),
		responseSSEvent("response.output_item.done", `{"type":"response.output_item.done","sequence_number":10,"output_index":0,"item":{"id":"rs_123","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"first summary"},{"type":"summary_text","text":"second summary"}],"encrypted_content":"encrypted-reasoning"}}`),
	}}
	gen := NewResponsesGenerator(svc, "gpt-5", "")
	adapter := StreamingAdapter{S: &gen}

	resp, err := adapter.Generate(context.Background(), Dialog{{Role: User, Blocks: []Block{TextBlock("hard question")}}}, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	candidate := requireCandidate(t, resp)
	if len(candidate.Blocks) != 3 {
		t.Fatalf("blocks = %d, want 2 summary thinking blocks plus 1 metadata block: %#v", len(candidate.Blocks), candidate.Blocks)
	}
	for i := range 2 {
		block := candidate.Blocks[i]
		if block.BlockType != Thinking {
			t.Fatalf("block %d type = %q, want %q", i, block.BlockType, Thinking)
		}
		if got := block.ExtraFields[ResponsesExtraFieldReasoningID]; got != reasoningID {
			t.Fatalf("block %d reasoning ID = %v, want %s", i, got, reasoningID)
		}
		if got := block.ExtraFields[ResponsesExtraFieldSummaryIndex]; got != int64(i) {
			t.Fatalf("block %d summary index = %v, want %d", i, got, i)
		}
	}
	metadataOnly := candidate.Blocks[2]
	if metadataOnly.BlockType != Thinking {
		t.Fatalf("metadata-only block type = %q, want %q", metadataOnly.BlockType, Thinking)
	}
	if got := metadataOnly.Content.String(); got != "" {
		t.Fatalf("metadata-only block content = %q, want empty", got)
	}
	if got := metadataOnly.ExtraFields[ResponsesExtraFieldReasoningID]; got != reasoningID {
		t.Fatalf("metadata-only reasoning ID = %v, want %s", got, reasoningID)
	}
	if got := metadataOnly.ExtraFields[ResponsesExtraFieldEncryptedContent]; got != encrypted {
		t.Fatalf("metadata-only encrypted content = %v, want %s", got, encrypted)
	}
	if got := candidate.Blocks[0].Content.String(); got != "first summary" {
		t.Fatalf("first summary = %q, want %q", got, "first summary")
	}
	if got := candidate.Blocks[1].Content.String(); got != "second summary" {
		t.Fatalf("second summary = %q, want %q", got, "second summary")
	}
}
