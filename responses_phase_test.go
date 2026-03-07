package gai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
)

type mockResponsesService struct {
	response *responses.Response
	err      error
	lastBody responses.ResponseNewParams
}

func (m *mockResponsesService) New(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (*responses.Response, error) {
	m.lastBody = body
	return m.response, m.err
}

func (m *mockResponsesService) NewStreaming(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) *ssestream.Stream[responses.ResponseStreamEventUnion] {
	panic("unimplemented")
}

func TestResponsesGeneratorBuildInputItemsPreservesAssistantMessagePhase(t *testing.T) {
	gen := NewResponsesGenerator(nil, "gpt-5", "")
	dialog := Dialog{{
		Role: Assistant,
		Blocks: []Block{
			TextBlock("Interim update"),
		},
		ExtraFields: map[string]interface{}{
			ResponsesMessageExtraFieldPhase: ResponsesMessagePhaseCommentary,
		},
	}}

	items, err := gen.buildInputItems(dialog)
	if err != nil {
		t.Fatalf("buildInputItems failed: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 input item, got %d", len(items))
	}
	if items[0].OfOutputMessage == nil {
		t.Fatalf("expected assistant message to be encoded as output_message, got %+v", items[0])
	}
	if got := string(items[0].OfOutputMessage.Phase); got != ResponsesMessagePhaseCommentary {
		t.Fatalf("expected phase %q, got %q", ResponsesMessagePhaseCommentary, got)
	}
}

func TestResponsesGeneratorBuildInputItemsRejectsInvalidAssistantMessagePhase(t *testing.T) {
	gen := NewResponsesGenerator(nil, "gpt-5", "")
	dialog := Dialog{{
		Role: Assistant,
		Blocks: []Block{
			TextBlock("Interim update"),
		},
		ExtraFields: map[string]interface{}{
			ResponsesMessageExtraFieldPhase: "bad_phase",
		},
	}}

	_, err := gen.buildInputItems(dialog)
	if err == nil {
		t.Fatal("expected invalid phase to return an error")
	}
}

func TestResponsesGeneratorBuildInputItemsPreservesAssistantMessagePhaseWithoutTextContent(t *testing.T) {
	gen := NewResponsesGenerator(nil, "gpt-5", "")

	t.Run("tool-call-only assistant", func(t *testing.T) {
		toolCallJSON, err := json.Marshal(ToolCallInput{
			Name:       "lookup",
			Parameters: map[string]any{"query": "weather"},
		})
		if err != nil {
			t.Fatalf("marshal tool call: %v", err)
		}

		dialog := Dialog{{
			Role: Assistant,
			Blocks: []Block{{
				ID:           "call_123",
				BlockType:    ToolCall,
				ModalityType: Text,
				MimeType:     "application/json",
				Content:      Str(toolCallJSON),
			}},
			ExtraFields: map[string]interface{}{
				ResponsesMessageExtraFieldPhase: ResponsesMessagePhaseCommentary,
			},
		}}

		items, err := gen.buildInputItems(dialog)
		if err != nil {
			t.Fatalf("buildInputItems failed: %v", err)
		}
		if len(items) != 2 {
			t.Fatalf("expected phase message + tool call, got %d items", len(items))
		}
		if items[0].OfOutputMessage == nil {
			t.Fatalf("expected first item to be output_message, got %+v", items[0])
		}
		if got := string(items[0].OfOutputMessage.Phase); got != ResponsesMessagePhaseCommentary {
			t.Fatalf("expected phase %q, got %q", ResponsesMessagePhaseCommentary, got)
		}
		payload, err := json.Marshal(items[0].OfOutputMessage)
		if err != nil {
			t.Fatalf("marshal output message: %v", err)
		}
		if !strings.Contains(string(payload), `"content":[]`) {
			t.Fatalf("expected phase-only replay to encode empty content array, got %s", string(payload))
		}
		if items[1].OfFunctionCall == nil {
			t.Fatalf("expected second item to be function call, got %+v", items[1])
		}
	})

	t.Run("thinking-only assistant", func(t *testing.T) {
		dialog := Dialog{{
			Role: Assistant,
			Blocks: []Block{{
				BlockType:    Thinking,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str(""),
				ExtraFields: map[string]interface{}{
					ThinkingExtraFieldGeneratorKey:      ThinkingGeneratorResponses,
					ResponsesExtraFieldReasoningID:      "rs_123",
					ResponsesExtraFieldEncryptedContent: "enc_123",
				},
			}},
			ExtraFields: map[string]interface{}{
				ResponsesMessageExtraFieldPhase: ResponsesMessagePhaseCommentary,
			},
		}}

		items, err := gen.buildInputItems(dialog)
		if err != nil {
			t.Fatalf("buildInputItems failed: %v", err)
		}
		if len(items) != 2 {
			t.Fatalf("expected reasoning item + phase message, got %d items", len(items))
		}
		if items[0].OfReasoning == nil {
			t.Fatalf("expected first item to be reasoning, got %+v", items[0])
		}
		if items[1].OfOutputMessage == nil {
			t.Fatalf("expected second item to be output_message, got %+v", items[1])
		}
		if got := string(items[1].OfOutputMessage.Phase); got != ResponsesMessagePhaseCommentary {
			t.Fatalf("expected phase %q, got %q", ResponsesMessagePhaseCommentary, got)
		}
	})
}

func TestResponsesGeneratorGeneratePreservesAssistantMessagePhase(t *testing.T) {
	var apiResp responses.Response
	if err := json.Unmarshal([]byte(`{
		"id":"resp_123",
		"created_at":0,
		"output":[{
			"id":"msg_123",
			"type":"message",
			"role":"assistant",
			"status":"completed",
			"phase":"final_answer",
			"content":[{
				"type":"output_text",
				"text":"Done.",
				"annotations":[],
				"logprobs":[]
			}]
		}],
		"usage":{
			"input_tokens":12,
			"input_tokens_details":{"cached_tokens":0},
			"output_tokens":3,
			"output_tokens_details":{"reasoning_tokens":0},
			"total_tokens":15
		}
	}`), &apiResp); err != nil {
		t.Fatalf("failed to decode mock response: %v", err)
	}

	svc := &mockResponsesService{response: &apiResp}
	gen := NewResponsesGenerator(svc, "gpt-5", "")

	resp, err := gen.Generate(context.Background(), Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("Say done")},
	}}, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if len(resp.Candidates) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(resp.Candidates))
	}
	got, ok := resp.Candidates[0].ExtraFields[ResponsesMessageExtraFieldPhase].(string)
	if !ok {
		t.Fatalf("expected assistant message to include %q in ExtraFields, got %+v", ResponsesMessageExtraFieldPhase, resp.Candidates[0].ExtraFields)
	}
	if got != ResponsesMessagePhaseFinalAnswer {
		t.Fatalf("expected phase %q, got %q", ResponsesMessagePhaseFinalAnswer, got)
	}
}

func TestResponsesGeneratorGeneratePreservesAssistantToolOnlyPhaseRoundTrip(t *testing.T) {
	var apiResp responses.Response
	if err := json.Unmarshal([]byte(`{
		"id":"resp_456",
		"created_at":0,
		"output":[
			{
				"id":"msg_456",
				"type":"message",
				"role":"assistant",
				"status":"completed",
				"phase":"commentary",
				"content":[]
			},
			{
				"id":"fc_456",
				"type":"function_call",
				"call_id":"call_456",
				"name":"lookup",
				"arguments":"{\"city\":\"Paris\"}"
			}
		],
		"usage":{
			"input_tokens":12,
			"input_tokens_details":{"cached_tokens":0},
			"output_tokens":3,
			"output_tokens_details":{"reasoning_tokens":0},
			"total_tokens":15
		}
	}`), &apiResp); err != nil {
		t.Fatalf("failed to decode mock response: %v", err)
	}

	svc := &mockResponsesService{response: &apiResp}
	gen := NewResponsesGenerator(svc, "gpt-5", "")

	resp, err := gen.Generate(context.Background(), Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("Look up Paris")},
	}}, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if len(resp.Candidates) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(resp.Candidates))
	}
	candidate := resp.Candidates[0]
	gotPhase, ok := candidate.ExtraFields[ResponsesMessageExtraFieldPhase].(string)
	if !ok {
		t.Fatalf("expected assistant phase in ExtraFields, got %+v", candidate.ExtraFields)
	}
	if gotPhase != ResponsesMessagePhaseCommentary {
		t.Fatalf("expected phase %q, got %q", ResponsesMessagePhaseCommentary, gotPhase)
	}

	items, err := gen.buildInputItems(Dialog{candidate})
	if err != nil {
		t.Fatalf("buildInputItems failed: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected output_message + function call on replay, got %d items", len(items))
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
}

func TestStreamingAdapterGeneratePreservesMessageExtraFields(t *testing.T) {
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

	resp, err := adapter.Generate(context.Background(), Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("hello")},
	}}, nil)
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
}

func TestStreamingAdapterGeneratePreservesToolOnlyMessageExtraFieldsForReplay(t *testing.T) {
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

	resp, err := adapter.Generate(context.Background(), Dialog{{
		Role:   User,
		Blocks: []Block{TextBlock("hello")},
	}}, nil)
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

	gen := NewResponsesGenerator(nil, "gpt-5", "")
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
}
