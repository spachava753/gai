package gai

import (
	"context"
	"errors"
	"strings"
	"testing"

	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

// mockAnthropicSvc is a mock implementation of AnthropicSvc for testing
type mockAnthropicSvc struct {
	countTokensCalled bool
	lastToolsCount    int
	lastSystemPresent bool
	response          *a.Message
	streamEvents      []ssestream.Event
}

func (m *mockAnthropicSvc) New(ctx context.Context, body a.MessageNewParams, opts ...option.RequestOption) (res *a.Message, err error) {
	return m.response, nil
}

func (m *mockAnthropicSvc) NewStreaming(ctx context.Context, params a.MessageNewParams, opts ...option.RequestOption) (stream *ssestream.Stream[a.MessageStreamEventUnion]) {
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

func TestAnthropicGenerateReturnsContentPolicyErrorForRefusal(t *testing.T) {
	service := &mockAnthropicSvc{response: &a.Message{
		StopReason: a.StopReasonRefusal,
		StopDetails: a.RefusalStopDetails{
			Explanation: "Request violates policy.",
		},
	}}
	generator := NewAnthropicGenerator(service, "claude", "")

	_, err := generator.Generate(context.Background(), Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}}, nil)
	assertContentPolicyErrorContains(t, err, "Request violates policy.")
}

func TestAnthropicStreamReturnsContentPolicyErrorForRefusal(t *testing.T) {
	service := &mockAnthropicSvc{streamEvents: []ssestream.Event{{
		Type: "message_delta",
		Data: []byte(`{"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"general_harms","explanation":"Request violates policy."}},"usage":{"output_tokens":1}}`),
	}}}
	generator := NewAnthropicGenerator(service, "claude", "")

	var gotErr error
	for _, err := range generator.Stream(context.Background(), Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}}, nil) {
		if err != nil {
			gotErr = err
			break
		}
	}
	assertContentPolicyErrorContains(t, gotErr, "Request violates policy.")
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

func TestAnthropicGenerator_Count_IncludesTools(t *testing.T) {
	// Create a mock Anthropic service
	mockSvc := &mockAnthropicSvc{}

	// Create a generator with the mock service
	gen := AnthropicGenerator{
		client:             mockSvc,
		model:              "claude-3-haiku-20240307",
		systemInstructions: "You are a helpful assistant",
		tools:              make(map[string]a.ToolParam),
	}

	// Register a tool
	tool := Tool{
		Name:        "test_tool",
		Description: "A test tool",
	}
	gen.Register(tool)

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

	// Call Count
	_, err := gen.Count(context.Background(), dialog)
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
