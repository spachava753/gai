package gai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

// mockAnthropicSvc is a mock implementation of AnthropicSvc for testing.
type mockAnthropicSvc struct {
	countTokensCalled bool
	lastToolsCount    int
	lastSystemPresent bool
	response          *a.Message
	streamEvents      []ssestream.Event
	streamFactory     func() *ssestream.Stream[a.MessageStreamEventUnion]
	streamCalls       int
}

func (m *mockAnthropicSvc) New(ctx context.Context, body a.MessageNewParams, opts ...option.RequestOption) (res *a.Message, err error) {
	return m.response, nil
}

func (m *mockAnthropicSvc) NewStreaming(ctx context.Context, params a.MessageNewParams, opts ...option.RequestOption) (stream *ssestream.Stream[a.MessageStreamEventUnion]) {
	m.streamCalls++
	if m.streamFactory != nil {
		return m.streamFactory()
	}
	return ssestream.NewStream[a.MessageStreamEventUnion](&anthropicStreamDecoder{events: m.streamEvents}, nil)
}

func (m *mockAnthropicSvc) CountTokens(ctx context.Context, params a.MessageCountTokensParams, opts ...option.RequestOption) (res *a.MessageTokensCount, err error) {
	m.countTokensCalled = true

	// Check if tools are present
	m.lastToolsCount = len(params.Tools)

	// Check if system is present
	m.lastSystemPresent = len(params.System.OfTextBlockArray) > 0

	// Return mock result
	return &a.MessageTokensCount{
		InputTokens: 10, // Mock value
	}, nil
}

func testAnthropicGenerateReturnsContentPolicyErrorForRefusal(t *testing.T) {
	service := &mockAnthropicSvc{response: &a.Message{
		StopReason: a.StopReasonRefusal,
		StopDetails: a.RefusalStopDetails{
			Explanation: "Request violates policy.",
		},
	}}
	generator := NewAnthropicGenerator(service)

	response, err := generator.Generate(context.Background(), GenerationRequest{
		Model:  "claude",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
	})
	if response.FinishReason != ContentPolicyViolation {
		t.Fatalf("FinishReason = %v, want ContentPolicyViolation", response.FinishReason)
	}
	assertContentPolicyErrorContains(t, err, "Request violates policy.")
}

func testAnthropicStreamReturnsContentPolicyErrorForRefusal(t *testing.T) {
	service := &mockAnthropicSvc{streamEvents: []ssestream.Event{{
		Type: "message_delta",
		Data: []byte(`{"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"general_harms","explanation":"Request violates policy."}},"usage":{"output_tokens":1}}`),
	}}}
	generator := NewAnthropicGenerator(service)

	var gotErr error
	for chunk := range generator.Stream(context.Background(), GenerationRequest{
		Model:  "claude",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
	}) {
		if chunk.Err != nil {
			gotErr = chunk.Err
			break
		}
	}
	assertContentPolicyErrorContains(t, gotErr, "Request violates policy.")
}

func testAnthropicGeneratorStreamRetriesOverloadedSSEError(t *testing.T) {
	const streamPayload = `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`

	newStream := func(data string) *ssestream.Stream[a.MessageStreamEventUnion] {
		response := &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(data)),
			Request:    &http.Request{},
		}
		return ssestream.NewStream[a.MessageStreamEventUnion](ssestream.NewDecoder(response), nil)
	}

	service := &mockAnthropicSvc{}
	service.streamFactory = func() *ssestream.Stream[a.MessageStreamEventUnion] {
		if service.streamCalls == 1 {
			return newStream("event: error\ndata: " + streamPayload + "\n\n")
		}
		return newStream("")
	}
	generator := NewAnthropicGenerator(service)

	var notified error
	retryingGenerator := NewRetryGenerator(generator, RetryConfig{
		Backoff:     func(uint) (time.Duration, bool) { return time.Millisecond, true },
		MaxAttempts: 2,
		Notify: func(err error, _ time.Duration) {
			notified = err
		},
	})

	for chunk := range retryingGenerator.Stream(t.Context(), GenerationRequest{
		Model: "claude",
		Dialog: Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("Hello")},
		}},
	}) {
		if chunk.Err != nil {
			t.Fatalf("Stream() error = %v, want retry to succeed", chunk.Err)
		}
	}

	if service.streamCalls != 2 {
		t.Fatalf("stream calls = %d, want 2", service.streamCalls)
	}
	if notified == nil {
		t.Fatal("retry notification error = nil, want classified overload error")
	}

	var apiErr *ApiErr
	if !errors.As(notified, &apiErr) {
		t.Fatalf("retry notification error = %T %v, want *ApiErr", notified, notified)
	}
	if apiErr.Provider != ProviderAnthropic {
		t.Fatalf("Provider = %q, want %q", apiErr.Provider, ProviderAnthropic)
	}
	if apiErr.Kind != APIErrorKindOverloaded {
		t.Fatalf("Kind = %q, want %q", apiErr.Kind, APIErrorKindOverloaded)
	}
	if apiErr.StatusCode != http.StatusOK {
		t.Fatalf("StatusCode = %d, want %d", apiErr.StatusCode, http.StatusOK)
	}
	if apiErr.Message != "Overloaded" {
		t.Fatalf("Message = %q, want %q", apiErr.Message, "Overloaded")
	}
	if got := strings.TrimSpace(apiErr.RawBody); got != streamPayload {
		t.Fatalf("RawBody = %q, want %q", got, streamPayload)
	}
	if !apiErr.Retryable() {
		t.Fatal("Retryable() = false, want true")
	}
}

func assertContentPolicyErrorContains(t *testing.T, err error, want string) {
	t.Helper()
	var policyErr ContentPolicyErr
	if !errors.As(err, &policyErr) {
		t.Fatalf("error = %T %v, want ContentPolicyErr", err, err)
	}
	if !strings.Contains(policyErr.Error(), want) {
		t.Fatalf("error = %q, want message containing %q", policyErr, want)
	}
}

type anthropicStreamDecoder struct {
	events []ssestream.Event
	index  int
	cur    ssestream.Event
}

func (d *anthropicStreamDecoder) Next() bool {
	if d.index >= len(d.events) {
		return false
	}
	d.cur = d.events[d.index]
	d.index++
	return true
}

func (d *anthropicStreamDecoder) Event() ssestream.Event { return d.cur }
func (d *anthropicStreamDecoder) Close() error           { return nil }
func (d *anthropicStreamDecoder) Err() error             { return nil }

func testAnthropicGenerator_Count_IncludesTools(t *testing.T) {
	// Create a mock Anthropic service
	mockSvc := &mockAnthropicSvc{}

	// Create a generator with the mock service.
	gen := AnthropicGenerator{client: mockSvc}

	tool := Tool{
		Name:        "test_tool",
		Description: "A test tool",
	}
	// Create a simple dialog
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hello"),
				},
			},
		},
	}

	// Call Count with the same request fields used for generation.
	_, err := gen.Count(context.Background(), GenerationRequest{
		Model:        "claude-3-haiku-20240307",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
		Dialog:       dialog,
		Tools:        []Tool{tool},
	})
	if err != nil {
		t.Errorf("Count returned error: %v", err)
	}

	// Check that CountTokens was called
	if !mockSvc.countTokensCalled {
		t.Errorf("CountTokens was not called")
	}

	// Check that system instructions were included
	if !mockSvc.lastSystemPresent {
		t.Errorf("System instructions were not included in CountTokens params")
	}

	// Check that system instructions were included
	if mockSvc.lastToolsCount != 1 {
		t.Errorf("Tool definitions were not included in CountTokens params")
	}
}
