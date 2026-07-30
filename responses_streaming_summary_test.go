package gai

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/cenkalti/backoff/v5"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
)

type responsesStreamService struct {
	events         []ssestream.Event
	eventsByStream [][]ssestream.Event
	streamCalls    int
}

func (s *responsesStreamService) New(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (*responses.Response, error) {
	return nil, fmt.Errorf("New not implemented")
}

func (s *responsesStreamService) NewStreaming(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) *ssestream.Stream[responses.ResponseStreamEventUnion] {
	s.streamCalls++
	events := s.events
	if len(s.eventsByStream) >= s.streamCalls {
		events = s.eventsByStream[s.streamCalls-1]
	}
	return ssestream.NewStream[responses.ResponseStreamEventUnion](&responsesStreamDecoder{events: events}, nil)
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
	if len(candidate.Blocks) != 2 {
		t.Fatalf("blocks = %d, want 2 summary thinking blocks: %#v", len(candidate.Blocks), candidate.Blocks)
	}
	for i, block := range candidate.Blocks {
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
	if _, ok := candidate.Blocks[0].ExtraFields[ResponsesExtraFieldEncryptedContent]; ok {
		t.Fatalf("first summary should not have encrypted content merged across separator")
	}
	if got := candidate.Blocks[1].ExtraFields[ResponsesExtraFieldEncryptedContent]; got != encrypted {
		t.Fatalf("last summary encrypted content = %v, want %s", got, encrypted)
	}
	if got := candidate.Blocks[0].Content.String(); got != "first summary" {
		t.Fatalf("first summary = %q, want %q", got, "first summary")
	}
	if got := candidate.Blocks[1].Content.String(); got != "second summary" {
		t.Fatalf("second summary = %q, want %q", got, "second summary")
	}
}

func TestResponsesGeneratorStreamRetriesServerOverloadSSEError(t *testing.T) {
	const overloadPayload = `{"type":"service_unavailable_error","code":"server_is_overloaded","message":"Our servers are currently overloaded. Please try again later.","param":null}`

	svc := &responsesStreamService{eventsByStream: [][]ssestream.Event{
		{responseSSEvent("error", `{"error":`+overloadPayload+`}`)},
		{},
	}}
	generator := NewResponsesGenerator(svc, "gpt-5", "")

	var notified error
	retryingGenerator := NewRetryGenerator(
		&generator,
		backoff.NewConstantBackOff(time.Millisecond),
		backoff.WithMaxTries(2),
		backoff.WithNotify(func(err error, _ time.Duration) {
			notified = err
		}),
	)

	for _, err := range retryingGenerator.Stream(t.Context(), Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("Hello")},
	}}, nil) {
		if err != nil {
			t.Fatalf("Stream() error = %v, want retry to succeed", err)
		}
	}

	if svc.streamCalls != 2 {
		t.Fatalf("stream calls = %d, want 2", svc.streamCalls)
	}
	if notified == nil {
		t.Fatal("retry notification error = nil, want classified overload error")
	}

	var apiErr *ApiErr
	if !errors.As(notified, &apiErr) {
		t.Fatalf("retry notification error = %T %v, want *ApiErr", notified, notified)
	}
	if apiErr.Provider != ProviderResponses {
		t.Fatalf("Provider = %q, want %q", apiErr.Provider, ProviderResponses)
	}
	if apiErr.Kind != APIErrorKindOverloaded {
		t.Fatalf("Kind = %q, want %q", apiErr.Kind, APIErrorKindOverloaded)
	}
	if apiErr.Message != "Our servers are currently overloaded. Please try again later." {
		t.Fatalf("Message = %q, want overload message", apiErr.Message)
	}
	if apiErr.RawBody != overloadPayload {
		t.Fatalf("RawBody = %q, want %q", apiErr.RawBody, overloadPayload)
	}
	if !apiErr.Retryable() {
		t.Fatal("Retryable() = false, want true")
	}
}
