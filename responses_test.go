package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
)

func TestEventAPIAdapterScenarios(t *testing.T) {
	t.Run("CommonLogprobOptions", func(t *testing.T) {
		generator := &ResponsesGenerator{}
		for _, enabled := range []bool{false, true} {
			values := NewGenerationOptions(WithLogprobs(enabled), WithTopLogprobs(0), WithParallelToolCalls(enabled))
			params, err := generator.buildParams(nil, GenerationRequest{Model: "model", Options: values})
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(params)
			if err != nil {
				t.Fatal(err)
			}
			var body map[string]any
			if err := json.Unmarshal(data, &body); err != nil {
				t.Fatal(err)
			}
			if body["parallel_tool_calls"] != enabled || body["top_logprobs"] != float64(0) {
				t.Fatalf("body: %s", data)
			}
			if strings.Contains(string(data), "message.output_text.logprobs") != enabled {
				t.Fatalf("include: %s", data)
			}
			if !strings.Contains(string(data), "reasoning.encrypted_content") {
				t.Fatalf("lost reasoning include: %s", data)
			}
			if values[GenerationOptionLogprobs] != enabled || len(values) != 3 {
				t.Fatal("modified options")
			}
		}
		for _, key := range []string{GenerationOptionLogprobs, GenerationOptionTopLogprobs, GenerationOptionParallelToolCalls} {
			if _, err := generator.buildParams(nil, GenerationRequest{Options: GenerationOptions{key: "invalid"}}); err == nil {
				t.Fatalf("accepted invalid %s", key)
			}
		}
	})
	t.Run("LogprobMetadata", func(t *testing.T) {
		var output []responses.ResponseOutputItemUnion
		if err := json.Unmarshal([]byte(`[{"type":"message","content":[{"type":"output_text","text":"ok","logprobs":[{"token":"ok","logprob":-0.1,"top_logprobs":[]}]}]}]`), &output); err != nil {
			t.Fatal(err)
		}
		request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hi")}}}}
		generated, err := NewResponsesGenerator(&mockResponsesService{response: &responses.Response{Output: output}}).Generate(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		if tokens, ok := generated.Candidates[0].Blocks[0].ExtraFields[ResponsesExtraFieldLogprobs].([]json.RawMessage); !ok || len(tokens) != 1 {
			t.Fatalf("metadata: %#v", generated)
		}
		svc := &responsesStreamService{events: []ssestream.Event{
			responseSSEvent("response.content_part.added", `{"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}`),
			responseSSEvent("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"ok","logprobs":[{"token":"ok","logprob":-0.1,"top_logprobs":[]}]}`),
			responseSSEvent("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"","logprobs":[{"token":"!","logprob":-0.2,"top_logprobs":[]}]}`),
			responseSSEvent("response.output_text.done", `{"type":"response.output_text.done","output_index":0,"content_index":0,"text":"ok","logprobs":[{"token":"ok","logprob":-0.1,"top_logprobs":[]},{"token":"!","logprob":-0.2,"top_logprobs":[]}]}`),
		}}
		gen := NewResponsesGenerator(svc)
		var chunks []StreamChunk
		for chunk := range gen.Stream(t.Context(), request) {
			if chunk.Err != nil {
				t.Fatal(chunk.Err)
			}
			if chunk.Block.BlockType != Separator {
				chunks = append(chunks, chunk)
			}
		}
		if len(chunks) != 2 {
			t.Fatalf("chunks: %#v", chunks)
		}
		if chunks[0].Block.ExtraFields != nil || len(chunks[1].Block.ExtraFields[ResponsesExtraFieldLogprobs].([]json.RawMessage)) != 2 {
			t.Fatal("expected incremental text followed by one final token list")
		}
		adapter := StreamingAdapter{S: gen}
		result, err := adapter.Generate(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Candidates[0].Blocks[0].ExtraFields[ResponsesExtraFieldLogprobs].([]json.RawMessage)) != 2 {
			t.Fatalf("assembled: %#v", result)
		}
		t.Run("linear metadata storage", func(t *testing.T) {
			for _, count := range []int{1000, 2000} {
				const token = `{"token":"x","logprob":-0.1,"top_logprobs":[]}`
				var events []ssestream.Event
				for part := 0; part < 2; part++ {
					events = append(events, responseSSEvent("response.content_part.added", fmt.Sprintf(`{"type":"response.content_part.added","output_index":0,"content_index":%d,"part":{"type":"output_text","text":""}}`, part)))
					for i := 0; i < count/2; i++ {
						events = append(events, responseSSEvent("response.output_text.delta", fmt.Sprintf(`{"type":"response.output_text.delta","output_index":0,"content_index":%d,"delta":"x","logprobs":[%s]}`, part, token)))
					}
					tokens := strings.TrimSuffix(strings.Repeat(token+",", count/2), ",")
					events = append(events, responseSSEvent("response.output_text.done", fmt.Sprintf(`{"type":"response.output_text.done","output_index":0,"content_index":%d,"text":%q,"logprobs":[%s]}`, part, strings.Repeat("x", count/2), tokens)))
				}
				generator := NewResponsesGenerator(&responsesStreamService{events: events})
				// Keep all yielded blocks, as StreamingAdapter does. There must be
				// one retained token list per part, not a growing copy per delta.
				var retained []StreamChunk
				for chunk := range generator.Stream(t.Context(), request) {
					if chunk.Err != nil {
						t.Fatal(chunk.Err)
					}
					retained = append(retained, chunk)
				}
				entries, lists := 0, 0
				for _, chunk := range retained {
					if tokens, ok := chunk.Block.ExtraFields[ResponsesExtraFieldLogprobs].([]json.RawMessage); ok {
						entries += len(tokens)
						lists++
					}
				}
				if entries != count || lists != 2 {
					t.Fatalf("%d tokens retained %d entries in %d lists", count, entries, lists)
				}
				response, err := (&StreamingAdapter{S: generator}).Generate(t.Context(), request)
				if err != nil {
					t.Fatal(err)
				}
				blocks := response.Candidates[0].Blocks
				if len(blocks) != 2 {
					t.Fatalf("got %d text parts", len(blocks))
				}
				for _, block := range blocks {
					if len(block.Content.String()) != count/2 || len(block.ExtraFields[ResponsesExtraFieldLogprobs].([]json.RawMessage)) != count/2 {
						t.Fatal("text-part metadata mixed or lost")
					}
				}
			}
		})
	})
	t.Run("ResponsesErrorEventClassification", func(t *testing.T) {
		tests := []struct {
			code string
			want APIErrorKind
		}{
			{code: "invalid_api_key", want: APIErrorKindAuthentication},
			{code: string(responses.ResponseErrorCodeRateLimitExceeded), want: APIErrorKindRateLimit},
			{code: "content_policy", want: APIErrorKindContentPolicy},
			{code: "unknown", want: APIErrorKindUnknown},
		}

		for _, tt := range tests {
			t.Run(tt.code, func(t *testing.T) {
				if got := mapResponsesErrorEvent(tt.code, "request failed", ""); got.Kind != tt.want {
					t.Fatalf("Kind = %q, want %q", got.Kind, tt.want)
				}
			})
		}
	})
	t.Run("ResponsesFailureClassification", func(t *testing.T) {
		tests := []struct {
			code responses.ResponseErrorCode
			want APIErrorKind
		}{
			{code: responses.ResponseErrorCodeServerError, want: APIErrorKindServer},
			{code: responses.ResponseErrorCodeRateLimitExceeded, want: APIErrorKindRateLimit},
			{code: responses.ResponseErrorCodeVectorStoreTimeout, want: APIErrorKindTimeout},
			{code: responses.ResponseErrorCodeBioPolicy, want: APIErrorKindContentPolicy},
			{code: responses.ResponseErrorCodeImageContentPolicyViolation, want: APIErrorKindContentPolicy},
			{code: responses.ResponseErrorCodeImageFileNotFound, want: APIErrorKindNotFound},
			{code: responses.ResponseErrorCodeImageFileTooLarge, want: APIErrorKindRequestTooLarge},
			{code: responses.ResponseErrorCodeInvalidImageMode, want: APIErrorKindInvalidRequest},
			{code: responses.ResponseErrorCode("unknown"), want: APIErrorKindUnknown},
		}

		for _, tt := range tests {
			t.Run(string(tt.code), func(t *testing.T) {
				got := mapResponsesFailure(responses.ResponseError{Code: tt.code, Message: "request failed"}, "")
				if got.Kind != tt.want {
					t.Fatalf("Kind = %q, want %q", got.Kind, tt.want)
				}
			})
		}
	})
	t.Run("ResponsesGeneratorBuildInputItemsPreservesAssistantMessagePhase", func(t *testing.T) {
		gen := NewResponsesGenerator(nil)
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
	})
	t.Run("ResponsesGeneratorBuildInputItemsPreservesAssistantMessagePhaseWithoutTextContent", func(t *testing.T) {
		gen := NewResponsesGenerator(nil)

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
	})
	t.Run("ResponsesGeneratorBuildInputItemsRejectsInvalidAssistantMessagePhase", func(t *testing.T) {
		gen := NewResponsesGenerator(nil)
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
	})
	t.Run("ResponsesGeneratorBuildParamsServiceTier", func(t *testing.T) {
		withServiceTier := func(value any) GenerationOptions {
			return GenerationOptions{ResponsesServiceTierParam: value}
		}
		withServiceTierHelper := func(value string) GenerationOptions {
			return NewGenerationOptions(WithResponsesServiceTier(value))
		}

		tests := []struct {
			name    string
			options GenerationOptions
			want    responses.ResponseNewParamsServiceTier
			wantErr string
		}{
			{name: "unset"},
			{name: "auto", options: withServiceTierHelper("auto"), want: responses.ResponseNewParamsServiceTierAuto},
			{name: "default", options: withServiceTierHelper("default"), want: responses.ResponseNewParamsServiceTierDefault},
			{name: "flex", options: withServiceTierHelper("flex"), want: responses.ResponseNewParamsServiceTierFlex},
			{name: "scale", options: withServiceTierHelper("scale"), want: responses.ResponseNewParamsServiceTierScale},
			{name: "priority", options: withServiceTierHelper("priority"), want: responses.ResponseNewParamsServiceTierPriority},
			{name: "fast", options: withServiceTierHelper("fast"), want: responses.ResponseNewParamsServiceTierFast},
			{name: "ultrafast", options: withServiceTierHelper("ultrafast"), want: responses.ResponseNewParamsServiceTierUltrafast},
			{
				name:    "SDK value",
				options: withServiceTier(responses.ResponseNewParamsServiceTierPriority),
				want:    responses.ResponseNewParamsServiceTierPriority,
			},
			{name: "invalid value", options: withServiceTier("turbo"), wantErr: "must be one of"},
			{name: "invalid type", options: withServiceTier(1), wantErr: "must be a string"},
		}

		generator := NewResponsesGenerator(nil)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				params, err := generator.buildParams(nil, GenerationRequest{Model: "gpt-5", Options: tt.options})
				if tt.wantErr != "" {
					if err == nil {
						t.Fatalf("buildParams() error = nil, want error containing %q", tt.wantErr)
					}
					if !strings.Contains(err.Error(), tt.wantErr) {
						t.Fatalf("buildParams() error = %q, want error containing %q", err, tt.wantErr)
					}
					return
				}
				if err != nil {
					t.Fatalf("buildParams() error = %v", err)
				}
				if params.ServiceTier != tt.want {
					t.Fatalf("ServiceTier = %q, want %q", params.ServiceTier, tt.want)
				}
			})
		}
	})
	t.Run("ResponsesGeneratorGeneratePreservesAssistantMessagePhase", func(t *testing.T) {
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
		gen := NewResponsesGenerator(svc)

		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model: "gpt-5",
			Dialog: Dialog{{
				Role:   User,
				Blocks: []Block{TextBlock("Say done")},
			}},
		})
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
	})
	t.Run("ResponsesGeneratorGeneratePreservesAssistantToolOnlyPhaseRoundTrip", func(t *testing.T) {
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
		gen := NewResponsesGenerator(svc)

		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model: "gpt-5",
			Dialog: Dialog{{
				Role:   User,
				Blocks: []Block{TextBlock("Look up Paris")},
			}},
		})
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
	})
	t.Run("ResponsesGeneratorGenerateReturnsContentPolicyFinishReason", func(t *testing.T) {
		const rawResponse = `{
		"id":"resp_filtered",
		"created_at":0,
		"status":"incomplete",
		"incomplete_details":{"reason":"content_filter"},
		"output":[]
	}`
		var apiResp responses.Response
		if err := json.Unmarshal([]byte(rawResponse), &apiResp); err != nil {
			t.Fatalf("failed to decode mock response: %v", err)
		}

		gen := NewResponsesGenerator(&mockResponsesService{response: &apiResp})
		response, err := gen.Generate(context.Background(), GenerationRequest{
			Model:  "gpt-5",
			Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
		})
		if response.FinishReason != ContentPolicyViolation {
			t.Fatalf("FinishReason = %v, want ContentPolicyViolation", response.FinishReason)
		}
		var policyErr ContentPolicyErr
		if !errors.As(err, &policyErr) {
			t.Fatalf("Generate error = %T %v, want ContentPolicyErr", err, err)
		}
	})
	t.Run("ResponsesGeneratorGenerateReturnsFailedResponseError", func(t *testing.T) {
		const rawResponse = `{
		"id":"resp_failed",
		"created_at":0,
		"status":"failed",
		"error":{"code":"server_error","message":"The server had an error processing your request."},
		"output":[]
	}`
		var apiResp responses.Response
		if err := json.Unmarshal([]byte(rawResponse), &apiResp); err != nil {
			t.Fatalf("failed to decode mock response: %v", err)
		}

		gen := NewResponsesGenerator(&mockResponsesService{response: &apiResp})
		_, err := gen.Generate(context.Background(), GenerationRequest{
			Model: "gpt-5",
			Dialog: Dialog{{
				Role:   User,
				Blocks: []Block{TextBlock("Hello")},
			}},
		})

		var apiErr *ApiErr
		if !errors.As(err, &apiErr) {
			t.Fatalf("Generate error = %v, want *ApiErr", err)
		}
		if apiErr.Provider != ProviderResponses {
			t.Fatalf("Provider = %q, want %q", apiErr.Provider, ProviderResponses)
		}
		if apiErr.Kind != APIErrorKindServer {
			t.Fatalf("Kind = %q, want %q", apiErr.Kind, APIErrorKindServer)
		}
		if apiErr.Message != "The server had an error processing your request." {
			t.Fatalf("Message = %q, want server error message", apiErr.Message)
		}
		if apiErr.RawBody != rawResponse {
			t.Fatalf("RawBody = %q, want original response JSON", apiErr.RawBody)
		}
		if !apiErr.Retryable() {
			t.Fatal("Retryable() = false, want true")
		}
	})
	t.Run("ResponsesGeneratorStreamRetriesServerOverloadSSEError", func(t *testing.T) {
		const overloadPayload = `{"type":"service_unavailable_error","code":"server_is_overloaded","message":"Our servers are currently overloaded. Please try again later.","param":null}`

		svc := &responsesStreamService{eventsByStream: [][]ssestream.Event{
			{responseSSEvent("error", `{"error":`+overloadPayload+`}`)},
			{},
		}}
		generator := NewResponsesGenerator(svc)

		var notified error
		retryingGenerator := NewRetryGenerator(generator, RetryConfig{
			Backoff:     func(uint) (time.Duration, bool) { return time.Millisecond, true },
			MaxAttempts: 2,
			Notify: func(err error, _ time.Duration) {
				notified = err
			},
		})

		for chunk := range retryingGenerator.Stream(t.Context(), GenerationRequest{
			Model: "gpt-5",
			Dialog: Dialog{{
				Role:   User,
				Blocks: []Block{TextBlock("Hello")},
			}},
		}) {
			if chunk.Err != nil {
				t.Fatalf("Stream() error = %v, want retry to succeed", chunk.Err)
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
	})
	t.Run("ResponsesGenerator/Generate", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Hi!")}}}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := len(resp.Candidates); got == 0 {
			t.Fatal("expected at least one item")
		}
	})
	t.Run("ResponsesGenerator/Generate/Thinking/Logging", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")

		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)

		dialog := Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("What is 2 + 2? Think step by step.")},
		}}

		options := NewGenerationOptions(
			WithReasoningEffort("medium"),
			WithTemperature(1.0),
		)
		options[ResponsesThoughtSummaryDetailParam] = responses.ReasoningSummaryDetailed

		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT5,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      options,
		})
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}

		if len(resp.Candidates) == 0 {
			t.Fatal("Expected at least one candidate, got none")
		}

		// Log all blocks and specifically look for thinking blocks
		t.Log("=== All blocks in response ===")
		hasThinkingBlocks := false
		thinkingBlockCount := 0

		for i, block := range resp.Candidates[0].Blocks {
			t.Logf("Block %d: Type=%s, Modality=%s, MimeType=%s", i, block.BlockType, block.ModalityType, block.MimeType)

			if block.BlockType == Thinking {
				hasThinkingBlocks = true
				thinkingBlockCount++
				t.Log("=== THINKING BLOCK CONTENT ===")
				t.Logf("Thinking content:\n%s", block.Content.String())
				t.Log("=== END THINKING BLOCK ===")

				// Log extra fields if present
				if block.ExtraFields != nil {
					t.Logf("Extra fields: %+v", block.ExtraFields)
				}
			} else if block.BlockType == Content {
				t.Log("=== CONTENT BLOCK ===")
				t.Logf("Content:\n%s", block.Content.String())
				t.Log("=== END CONTENT BLOCK ===")
			}
		}

		t.Logf("Total thinking blocks found: %d", thinkingBlockCount)

		if !hasThinkingBlocks {
			t.Error("Expected at least one Thinking block in response, but found none")
		}
	})
	t.Run("ResponsesGenerator/Generate/image", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		imgBytes, err := os.ReadFile("sample.jpg")
		if err != nil {
			t.Skip("could not open sample.jpg")
			return
		}
		imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))
		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)
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
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Options: NewGenerationOptions(
				WithMaxGenerationTokens(512),
				WithReasoningEffort("high"),
			),
		})
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
	})
	t.Run("ResponsesGenerator/Generate/pdf", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		pdfBytes, err := os.ReadFile("sample.pdf")
		if err != nil {
			t.Skip("could not open sample.pdf")
			return
		}
		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					TextBlock("What is the title of this PDF? Just output the title and nothing else"),
					PDFBlock(pdfBytes, "sample.pdf"),
				},
			},
		}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithReasoningEffort("low")),
		})
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
	})
	t.Run("ResponsesGenerator/Generate/thinking", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Are LLMs conscious? Think it through and give a comprehensive answer")}}}
		options := NewGenerationOptions(
			WithReasoningEffort("medium"),
			WithTemperature(1.0),
			WithResponsesThoughtSummaryDetail("detailed"),
		)
		request := GenerationRequest{
			Model:          openai.ChatModelGPT5,
			PromptCacheKey: "responses-thinking-example:v1",
			Instructions:   SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:         dialog,
			Options:        options,
		}
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Reuse the same prompt cache key on the follow-up request because both turns
		// share the same system instructions and overall prompt prefix shape.
		// The generator is stateless: just append the assistant response and continue.
		// Reasoning blocks with encrypted content are automatically reconstructed as
		// input reasoning items on the next call.
		dialog = append(dialog, resp.Candidates[0], Message{Role: User, Blocks: []Block{TextBlock("What can you do?")}})
		request.Dialog = dialog
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := len(resp.Candidates); got == 0 {
			t.Fatal("expected at least one item")
		}
	})
	t.Run("ResponsesGenerator/ReasoningTokenPreservation/Generate", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")

		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)

		// Register two tools so the model has a reason to make multiple tool calls
		lookupTool := Tool{
			Name:        "lookup_info",
			Description: "Look up factual information about a topic. Returns a brief fact.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Topic string `json:"topic" jsonschema:"required" jsonschema_description:"The topic to look up"`
				}]()
				if err != nil {
					t.Fatalf("Failed to generate schema: %v", err)
				}
				return schema
			}(),
		}

		// === Phase 1: Function-calling loop within one assistant turn ===

		dialog := Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("Look up the population of Tokyo and the population of New York City, then tell me which is larger.")},
		}}

		options := NewGenerationOptions(WithReasoningEffort("low"))
		options[ResponsesThoughtSummaryDetailParam] = responses.ReasoningSummaryDetailed
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock(responsesLookupToolInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{lookupTool},
			Options:      options,
		}

		type turnMetrics struct {
			inputTokens     int
			outputTokens    int
			reasoningTokens int
			// toolResultTokens is the number of tokens added as tool results after this
			// turn's response. It is computed as the difference between the next turn's
			// input and this turn's input + output. For the first pair of turns this
			// establishes the baseline; subsequent pairs must match exactly.
			toolResultTokens int
		}
		var turns []turnMetrics

		// Loop until the model stops calling tools (up to a safety limit)
		for i := 0; i < 5; i++ {
			request.Dialog = dialog
			resp, err := gen.Generate(context.Background(), request)
			if err != nil {
				t.Fatalf("Turn %d Generate failed: %v", i+1, err)
			}

			inputTok, _ := InputTokens(resp.UsageMetadata)
			outputTok, _ := OutputTokens(resp.UsageMetadata)
			reasoningTok, _ := GetMetric[int](resp.UsageMetadata, UsageMetricReasoningTokens)

			t.Logf("Turn %d: input=%d, output=%d, reasoning=%d, finish=%v",
				i+1, inputTok, outputTok, reasoningTok, resp.FinishReason)

			turns = append(turns, turnMetrics{
				inputTokens:     inputTok,
				outputTokens:    outputTok,
				reasoningTokens: reasoningTok,
			})

			if resp.FinishReason != ToolUse {
				// Model finished, no more tool calls
				dialog = append(dialog, resp.Candidates[0])
				break
			}

			// Collect tool calls and provide results
			assistantMsg := resp.Candidates[0]
			dialog = append(dialog, assistantMsg)

			for _, blk := range assistantMsg.Blocks {
				if blk.BlockType == ToolCall {
					// Provide a fake tool result
					var tci ToolCallInput
					if err := json.Unmarshal([]byte(blk.Content.String()), &tci); err != nil {
						t.Fatalf("Failed to parse tool call: %v", err)
					}
					result := "The population is approximately 14 million."
					if strings.Contains(strings.ToLower(tci.Parameters["topic"].(string)), "new york") {
						result = "The population is approximately 8.3 million."
					} else if strings.Contains(strings.ToLower(tci.Parameters["topic"].(string)), "tokyo") {
						result = "The population is approximately 13.96 million."
					}
					dialog = append(dialog, Message{
						Role: ToolResult,
						Blocks: []Block{{
							ID:           blk.ID,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(result),
						}},
					})
				}
			}
		}

		if len(turns) < 2 {
			t.Fatalf("Expected at least 2 turns in the function-calling loop, got %d", len(turns))
		}

		// Compute tool result tokens for each turn from the observed differences.
		// Within an assistant turn the exact relationship is:
		//   input_tokens[n+1] == input_tokens[n] + output_tokens[n] + toolResultTokens[n]
		// where toolResultTokens[n] is the fixed overhead of the tool result text we
		// appended after turn n. We derive toolResultTokens from the first transition,
		// then assert that subsequent transitions match exactly.
		for i := 0; i+1 < len(turns); i++ {
			turns[i].toolResultTokens = turns[i+1].inputTokens - turns[i].inputTokens - turns[i].outputTokens
		}

		// The tool result tokens should be positive (we always add content).
		if turns[0].toolResultTokens <= 0 {
			t.Fatalf("Tool result tokens for turn 1 should be positive, got %d", turns[0].toolResultTokens)
		}
		t.Logf("Tool result tokens (derived from first transition): %d", turns[0].toolResultTokens)

		// Assert exact equality for all subsequent transitions.
		for i := 1; i+1 < len(turns); i++ {
			prev := turns[i-1]
			curr := turns[i]

			t.Logf("Within-turn assertion (turn %d→%d): toolResultTokens=%d (expected %d)",
				i+1, i+2, curr.toolResultTokens, prev.toolResultTokens)

			if curr.toolResultTokens != prev.toolResultTokens {
				t.Errorf("Within assistant turn: tool result token count changed between transitions "+
					"(turn %d→%d: %d, turn %d→%d: %d); reasoning tokens may not be correctly preserved",
					i, i+1, prev.toolResultTokens, i+1, i+2, curr.toolResultTokens)
			}
		}

		// For every transition, verify the exact formula:
		// input_tokens[n+1] == input_tokens[n] + output_tokens[n] + toolResultTokens[n]
		for i := 0; i+1 < len(turns); i++ {
			prev := turns[i]
			curr := turns[i+1]
			expected := prev.inputTokens + prev.outputTokens + prev.toolResultTokens

			t.Logf("Within-turn exact check (turn %d→%d): input[%d]+output[%d]+toolResult[%d] = %d+%d+%d = %d, actual input[%d]=%d",
				i+1, i+2, i+1, i+1, i+1, prev.inputTokens, prev.outputTokens, prev.toolResultTokens, expected, i+2, curr.inputTokens)

			if curr.inputTokens != expected {
				t.Errorf("Within assistant turn: input_tokens[%d]=%d != input_tokens[%d]+output_tokens[%d]+toolResultTokens[%d]=%d",
					i+2, curr.inputTokens, i+1, i+1, i+1, expected)
			}
		}

		// Record the last turn's metrics for the cross-turn comparison
		lastTurnInPhase1 := turns[len(turns)-1]

		// === Phase 2: New user turn — reasoning tokens from previous turn should be discarded ===

		dialog = append(dialog, Message{
			Role:   User,
			Blocks: []Block{TextBlock("Now look up the population of London.")},
		})

		request.Dialog = dialog
		resp2, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("New turn Generate failed: %v", err)
		}

		newTurnInput, _ := InputTokens(resp2.UsageMetadata)
		newTurnOutput, _ := OutputTokens(resp2.UsageMetadata)
		newTurnReasoning, _ := GetMetric[int](resp2.UsageMetadata, UsageMetricReasoningTokens)

		t.Logf("New turn: input=%d, output=%d, reasoning=%d, finish=%v",
			newTurnInput, newTurnOutput, newTurnReasoning, resp2.FinishReason)

		// The expected input if reasoning were preserved would be:
		// lastTurnInPhase1.inputTokens + lastTurnInPhase1.outputTokens + new_user_message_tokens
		expectedWithReasoning := lastTurnInPhase1.inputTokens + lastTurnInPhase1.outputTokens
		// The drop should be approximately equal to the total reasoning tokens from the
		// last assistant turn, because those are no longer included as input.
		totalReasoningInLastTurn := lastTurnInPhase1.reasoningTokens

		t.Logf("Cross-turn: expectedWithReasoning≈%d, actual=%d, reasoning_discarded≈%d",
			expectedWithReasoning, newTurnInput, totalReasoningInLastTurn)

		// The new turn's input should be noticeably less than expected-with-reasoning,
		// because the reasoning tokens from the previous turn are discarded.
		actualDrop := expectedWithReasoning - newTurnInput
		t.Logf("Cross-turn drop: %d tokens (expected drop ≈ %d reasoning tokens from last turn)",
			actualDrop, totalReasoningInLastTurn)

		if totalReasoningInLastTurn > 0 {
			// When the API reports reasoning tokens, the drop should be at least 50%
			// of the reasoning tokens to confirm they were truly discarded.
			if actualDrop < totalReasoningInLastTurn/2 {
				t.Errorf("Cross-turn: expected input to drop by ~%d reasoning tokens, but only dropped %d; "+
					"reasoning tokens may not have been properly discarded across turns",
					totalReasoningInLastTurn, actualDrop)
			}
		} else {
			// Even without explicit reasoning_tokens metrics, we can verify the drop pattern.
			// The API includes encrypted reasoning tokens in output_tokens but may not always
			// report them in output_tokens_details.reasoning_tokens. If encrypted reasoning
			// content was present in the dialog, there should still be a significant drop
			// when those tokens are discarded across turns.
			hasEncryptedContent := false
			if len(turns) >= 2 {
				// Check the last assistant message in the dialog for encrypted reasoning
				for _, msg := range dialog {
					if msg.Role == Assistant {
						for _, blk := range msg.Blocks {
							if blk.BlockType == Thinking && blk.ExtraFields != nil {
								if ec, ok := blk.ExtraFields[ResponsesExtraFieldEncryptedContent].(string); ok && ec != "" {
									hasEncryptedContent = true
								}
							}
						}
					}
				}
			}
			if hasEncryptedContent && actualDrop <= 0 {
				t.Errorf("Cross-turn: encrypted reasoning content was present but input tokens did not drop "+
					"(expectedWithReasoning=%d, actual=%d, drop=%d); reasoning tokens may not have been discarded",
					expectedWithReasoning, newTurnInput, actualDrop)
			} else if hasEncryptedContent {
				t.Logf("Cross-turn: verified drop of %d tokens with encrypted reasoning present (reasoning_tokens metric not reported by API)", actualDrop)
			} else {
				t.Log("Warning: no reasoning tokens reported and no encrypted content found; cannot verify cross-turn token drop")
			}
		}
	})
	t.Run("ResponsesGenerator/ReasoningTokenPreservation/Stream", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")

		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)

		lookupTool := Tool{
			Name:        "lookup_info",
			Description: "Look up factual information about a topic. Returns a brief fact.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Topic string `json:"topic" jsonschema:"required" jsonschema_description:"The topic to look up"`
				}]()
				if err != nil {
					t.Fatalf("Failed to generate schema: %v", err)
				}
				return schema
			}(),
		}

		// Use StreamingAdapter for the streaming path
		adapter := &StreamingAdapter{S: gen}

		// === Phase 1: Function-calling loop within one assistant turn ===

		dialog := Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("Look up the population of Tokyo and the population of New York City, then tell me which is larger.")},
		}}

		options := NewGenerationOptions(WithReasoningEffort("low"))
		options[ResponsesThoughtSummaryDetailParam] = responses.ReasoningSummaryDetailed
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock(responsesLookupToolInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{lookupTool},
			Options:      options,
		}

		type turnMetrics struct {
			inputTokens      int
			outputTokens     int
			reasoningTokens  int
			toolResultTokens int
		}
		var turns []turnMetrics

		for i := 0; i < 5; i++ {
			request.Dialog = dialog
			resp, err := adapter.Generate(context.Background(), request)
			if err != nil {
				t.Fatalf("Turn %d Generate (streaming) failed: %v", i+1, err)
			}

			inputTok, _ := InputTokens(resp.UsageMetadata)
			outputTok, _ := OutputTokens(resp.UsageMetadata)
			reasoningTok, _ := GetMetric[int](resp.UsageMetadata, UsageMetricReasoningTokens)

			t.Logf("Turn %d (streaming): input=%d, output=%d, reasoning=%d, finish=%v",
				i+1, inputTok, outputTok, reasoningTok, resp.FinishReason)

			turns = append(turns, turnMetrics{
				inputTokens:     inputTok,
				outputTokens:    outputTok,
				reasoningTokens: reasoningTok,
			})

			if resp.FinishReason != ToolUse {
				dialog = append(dialog, resp.Candidates[0])
				break
			}

			assistantMsg := resp.Candidates[0]
			dialog = append(dialog, assistantMsg)

			for _, blk := range assistantMsg.Blocks {
				if blk.BlockType == ToolCall {
					var tci ToolCallInput
					if err := json.Unmarshal([]byte(blk.Content.String()), &tci); err != nil {
						t.Fatalf("Failed to parse tool call: %v", err)
					}
					result := "The population is approximately 14 million."
					if strings.Contains(strings.ToLower(tci.Parameters["topic"].(string)), "new york") {
						result = "The population is approximately 8.3 million."
					} else if strings.Contains(strings.ToLower(tci.Parameters["topic"].(string)), "tokyo") {
						result = "The population is approximately 13.96 million."
					}
					dialog = append(dialog, Message{
						Role: ToolResult,
						Blocks: []Block{{
							ID:           blk.ID,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(result),
						}},
					})
				}
			}
		}

		if len(turns) < 2 {
			t.Fatalf("Expected at least 2 turns in the function-calling loop, got %d", len(turns))
		}

		// Compute tool result tokens from observed differences (same logic as Generate test).
		for i := 0; i+1 < len(turns); i++ {
			turns[i].toolResultTokens = turns[i+1].inputTokens - turns[i].inputTokens - turns[i].outputTokens
		}

		if turns[0].toolResultTokens <= 0 {
			t.Fatalf("Tool result tokens for turn 1 should be positive, got %d", turns[0].toolResultTokens)
		}
		t.Logf("Tool result tokens (streaming, derived from first transition): %d", turns[0].toolResultTokens)

		for i := 1; i+1 < len(turns); i++ {
			prev := turns[i-1]
			curr := turns[i]

			t.Logf("Within-turn assertion (streaming, turn %d→%d): toolResultTokens=%d (expected %d)",
				i+1, i+2, curr.toolResultTokens, prev.toolResultTokens)

			if curr.toolResultTokens != prev.toolResultTokens {
				t.Errorf("Within assistant turn (streaming): tool result token count changed between transitions "+
					"(turn %d→%d: %d, turn %d→%d: %d); reasoning tokens may not be correctly preserved",
					i, i+1, prev.toolResultTokens, i+1, i+2, curr.toolResultTokens)
			}
		}

		for i := 0; i+1 < len(turns); i++ {
			prev := turns[i]
			curr := turns[i+1]
			expected := prev.inputTokens + prev.outputTokens + prev.toolResultTokens

			t.Logf("Within-turn exact check (streaming, turn %d→%d): %d+%d+%d = %d, actual=%d",
				i+1, i+2, prev.inputTokens, prev.outputTokens, prev.toolResultTokens, expected, curr.inputTokens)

			if curr.inputTokens != expected {
				t.Errorf("Within assistant turn (streaming): input_tokens[%d]=%d != input_tokens[%d]+output_tokens[%d]+toolResultTokens[%d]=%d",
					i+2, curr.inputTokens, i+1, i+1, i+1, expected)
			}
		}

		lastTurnInPhase1 := turns[len(turns)-1]

		// === Phase 2: New user turn — reasoning tokens should be discarded ===

		dialog = append(dialog, Message{
			Role:   User,
			Blocks: []Block{TextBlock("Now look up the population of London.")},
		})

		request.Dialog = dialog
		resp2, err := adapter.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("New turn Generate (streaming) failed: %v", err)
		}

		newTurnInput, _ := InputTokens(resp2.UsageMetadata)
		newTurnOutput, _ := OutputTokens(resp2.UsageMetadata)
		newTurnReasoning, _ := GetMetric[int](resp2.UsageMetadata, UsageMetricReasoningTokens)

		t.Logf("New turn (streaming): input=%d, output=%d, reasoning=%d, finish=%v",
			newTurnInput, newTurnOutput, newTurnReasoning, resp2.FinishReason)

		expectedWithReasoning := lastTurnInPhase1.inputTokens + lastTurnInPhase1.outputTokens
		totalReasoningInLastTurn := lastTurnInPhase1.reasoningTokens

		t.Logf("Cross-turn (streaming): expectedWithReasoning≈%d, actual=%d, reasoning_discarded≈%d",
			expectedWithReasoning, newTurnInput, totalReasoningInLastTurn)

		actualDrop := expectedWithReasoning - newTurnInput
		t.Logf("Cross-turn drop (streaming): %d tokens (expected drop ≈ %d reasoning tokens)",
			actualDrop, totalReasoningInLastTurn)

		if totalReasoningInLastTurn > 0 {
			if actualDrop < totalReasoningInLastTurn/2 {
				t.Errorf("Cross-turn (streaming): expected input to drop by ~%d reasoning tokens, but only dropped %d; "+
					"reasoning tokens may not have been properly discarded across turns",
					totalReasoningInLastTurn, actualDrop)
			}
		} else {
			hasEncryptedContent := false
			if len(turns) >= 2 {
				for _, msg := range dialog {
					if msg.Role == Assistant {
						for _, blk := range msg.Blocks {
							if blk.BlockType == Thinking && blk.ExtraFields != nil {
								if ec, ok := blk.ExtraFields[ResponsesExtraFieldEncryptedContent].(string); ok && ec != "" {
									hasEncryptedContent = true
								}
							}
						}
					}
				}
			}
			if hasEncryptedContent && actualDrop <= 0 {
				t.Errorf("Cross-turn (streaming): encrypted reasoning content was present but input tokens did not drop "+
					"(expectedWithReasoning=%d, actual=%d, drop=%d); reasoning tokens may not have been discarded",
					expectedWithReasoning, newTurnInput, actualDrop)
			} else if hasEncryptedContent {
				t.Logf("Cross-turn (streaming): verified drop of %d tokens with encrypted reasoning present (reasoning_tokens metric not reported by API)", actualDrop)
			} else {
				t.Log("Warning (streaming): no reasoning tokens reported and no encrypted content found; cannot verify cross-turn token drop")
			}
		}
	})
	t.Run("ResponsesGenerator/RequestTools", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)
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
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}}}
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock(openAIStockInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options:      NewGenerationOptions(WithToolChoice("get_stock_price")),
		}
		resp, err := gen.Generate(context.Background(), request)
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
		request.Dialog = dialog
		request.Options = nil
		resp, err = gen.Generate(context.Background(), request)
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
	})
	t.Run("ResponsesGenerator/RequestTools/parallelToolUse", func(t *testing.T) {
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
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Which stock, Apple vs. Microsoft, is more expensive?")}}}
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock(openAIStockComparisonInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options:      NewGenerationOptions(WithReasoningEffort("medium")),
		}
		resp, err := gen.Generate(context.Background(), request)
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
		request.Dialog = dialog
		request.Options = nil
		resp, err = gen.Generate(context.Background(), request)
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
	})
	t.Run("ResponsesGenerator/StatelessToolCallWithReasoning", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")

		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)

		tickerTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("Failed to generate schema: %v", err)
				}
				return schema
			}(),
		}

		// Turn 1: Ask the model to use a tool with reasoning enabled
		dialog := Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("What is the current price of Apple stock?")},
		}}

		options := NewGenerationOptions(WithReasoningEffort("low"))
		options[ResponsesThoughtSummaryDetailParam] = responses.ReasoningSummaryDetailed
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock(responsesStockToolInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options:      options,
		}

		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("Turn 1 Generate failed: %v", err)
		}

		if resp.FinishReason != ToolUse {
			t.Fatalf("Expected ToolUse finish reason, got %v", resp.FinishReason)
		}

		// Verify that thinking blocks have encrypted content
		assistantMsg := resp.Candidates[0]
		hasEncryptedReasoning := false
		var toolCallBlock Block
		for _, blk := range assistantMsg.Blocks {
			if blk.BlockType == Thinking && blk.ExtraFields != nil {
				if ec, ok := blk.ExtraFields[ResponsesExtraFieldEncryptedContent].(string); ok && ec != "" {
					hasEncryptedReasoning = true
					t.Logf("Found encrypted reasoning content (length: %d)", len(ec))
				}
				if rid, ok := blk.ExtraFields[ResponsesExtraFieldReasoningID].(string); ok && rid != "" {
					t.Logf("Found reasoning ID: %s", rid)
				}
			}
			if blk.BlockType == ToolCall {
				toolCallBlock = blk
			}
		}

		if !hasEncryptedReasoning {
			t.Error("Expected encrypted reasoning content in Thinking block ExtraFields; stateless multi-turn tool calling requires this")
		}

		if toolCallBlock.ID == "" {
			t.Fatal("Expected a tool call block in the response")
		}
		t.Logf("Tool call ID: %s, Content: %s", toolCallBlock.ID, toolCallBlock.Content)

		// Turn 2: Provide tool result and let the model respond
		// This is the critical test: the dialog now contains the full assistant message
		// (including Thinking blocks with encrypted content), and the generator must
		// reconstruct reasoning items when building the input.
		dialog = append(dialog, assistantMsg, Message{
			Role: ToolResult,
			Blocks: []Block{{
				ID:           toolCallBlock.ID,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str("Apple (AAPL) is currently trading at $187.50"),
			}},
		})

		request.Dialog = dialog
		resp2, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("Turn 2 Generate failed: %v", err)
		}

		if resp2.FinishReason != EndTurn {
			t.Fatalf("Expected EndTurn finish reason, got %v", resp2.FinishReason)
		}

		// Verify the model produced a content response
		var finalContent string
		for _, blk := range resp2.Candidates[0].Blocks {
			if blk.BlockType == Content {
				finalContent = blk.Content.String()
				break
			}
		}

		if finalContent == "" {
			t.Fatal("Expected content in the final response")
		}
		t.Logf("Final response: %s", finalContent)

		// The response should mention the price
		if !strings.Contains(finalContent, "187") {
			t.Errorf("Expected the response to mention the stock price, got: %s", finalContent)
		}
	})
	t.Run("ResponsesGenerator/StreamMetadata", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")

		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)

		dialog := Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("Say hello.")},
		}}

		var blocks []Block
		for chunk := range gen.Stream(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT5Nano,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
		}) {
			if chunk.Err != nil {
				t.Fatalf("Stream error: %v", chunk.Err)
			}
			blocks = append(blocks, chunk.Block)
		}

		if len(blocks) == 0 {
			t.Fatal("Expected at least one block from stream")
		}

		// The last block should be a metadata block
		lastBlock := blocks[len(blocks)-1]
		if lastBlock.BlockType != MetadataBlockType {
			t.Fatalf("Expected last block to be metadata, got %s", lastBlock.BlockType)
		}

		// Verify the metadata contains usage information
		metadataJSON := lastBlock.Content.String()
		t.Logf("Metadata block: %s", metadataJSON)

		if !strings.Contains(metadataJSON, UsageMetricInputTokens) {
			t.Errorf("Expected metadata to contain %s", UsageMetricInputTokens)
		}
		if !strings.Contains(metadataJSON, UsageMetricGenerationTokens) {
			t.Errorf("Expected metadata to contain %s", UsageMetricGenerationTokens)
		}

		// Verify content blocks are present before the metadata
		hasContent := false
		for _, blk := range blocks[:len(blocks)-1] {
			if blk.BlockType == Content {
				hasContent = true
				break
			}
		}
		if !hasContent {
			t.Error("Expected at least one content block before the metadata block")
		}
	})
	t.Run("ResponsesGenerator/Stream/Thinking/Logging", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")

		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)

		dialog := Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("What is the capital of France? Think about it briefly.")},
		}}

		options := NewGenerationOptions(WithReasoningEffort("medium"))
		options[ResponsesThoughtSummaryDetailParam] = responses.ReasoningSummaryDetailed

		var allBlocks []Block
		var thinkingContent string
		var regularContent string

		t.Log("=== Streaming response ===")
		for chunk := range gen.Stream(context.Background(), GenerationRequest{
			Model:        openai.ChatModelGPT5Nano,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      options,
		}) {
			if chunk.Err != nil {
				t.Fatalf("Stream error: %v", chunk.Err)
			}
			allBlocks = append(allBlocks, chunk.Block)

			if chunk.Block.BlockType == Thinking {
				thinkingContent += chunk.Block.Content.String()
				t.Logf("Thinking chunk: %q", chunk.Block.Content.String())
			} else if chunk.Block.BlockType == Content {
				regularContent += chunk.Block.Content.String()
			}
		}

		t.Log("=== Final aggregated thinking content ===")
		t.Logf("Thinking:\n%s", thinkingContent)
		t.Log("=== End thinking ===")

		t.Log("=== Final aggregated regular content ===")
		t.Logf("Content:\n%s", regularContent)
		t.Log("=== End content ===")

		hasThinkingBlocks := false
		for _, block := range allBlocks {
			if block.BlockType == Thinking {
				hasThinkingBlocks = true
				break
			}
		}

		if !hasThinkingBlocks {
			t.Error("Expected at least one Thinking block in streaming response, but found none")
		}

		if thinkingContent == "" {
			t.Error("Thinking content is empty - reasoning may not be returned correctly")
		} else {
			t.Logf("Thinking content length: %d characters", len(thinkingContent))
		}
	})
	t.Run("ResponsesGenerator/Stream/thinking", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the capital of France? Reply with just the city name.")}}}
		options := NewGenerationOptions(
			WithReasoningEffort("low"),
			WithResponsesThoughtSummaryDetail("detailed"),
		)
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5Nano,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      options,
		}
		// Use StreamingAdapter so the streamed output is automatically compressed
		// into a proper Response with Thinking blocks carrying ExtraFields (including
		// encrypted reasoning content for stateless multi-turn conversations).
		adapter := &StreamingAdapter{S: gen}
		resp, err := adapter.Generate(context.Background(), request)
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
		request.Dialog = dialog
		resp, err = adapter.Generate(context.Background(), request)
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
	})
	t.Run("ResponsesGenerator/StreamingAdapter/LiveThinkingSummaryFormatting", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")

		reasoningModel := os.Getenv("GAI_RESPONSES_THINKING_LIVE_MODEL")
		if reasoningModel == "" {
			reasoningModel = "gpt-5.5"
		}
		verifierModel := os.Getenv("GAI_RESPONSES_THINKING_VERIFIER_MODEL")
		if verifierModel == "" {
			verifierModel = "gpt-5.4-nano"
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)
		adapter := StreamingAdapter{S: gen}

		dialog := Dialog{{
			Role: User,
			Blocks: []Block{TextBlock(`Solve this deliberately difficult reasoning problem. You must reason through the constraints carefully before answering.

Five houses stand in a row, numbered 1 through 5 from left to right. Each house has exactly one color, nationality, drink, pet, and hobby. All values in each category are used exactly once.

Colors: red, green, ivory, yellow, blue.
Nationalities: Englishman, Spaniard, Ukrainian, Norwegian, Japanese.
Drinks: coffee, tea, milk, orange juice, water.
Pets: dog, snails, fox, horse, zebra.
Hobbies: painter, sculptor, diplomat, violinist, doctor.

Constraints:
1. The Englishman lives in the red house.
2. The Spaniard owns the dog.
3. Coffee is drunk in the green house.
4. The Ukrainian drinks tea.
5. The green house is immediately to the right of the ivory house.
6. The sculptor owns snails.
7. The diplomat lives in the yellow house.
8. Milk is drunk in the middle house.
9. The Norwegian lives in the first house.
10. The doctor lives next to the fox owner.
11. The diplomat lives next to the horse owner.
12. The violinist drinks orange juice.
13. The Japanese is the painter.
14. The Norwegian lives next to the blue house.

Find who drinks water and who owns the zebra. Keep the final visible answer concise: one sentence for water and one sentence for zebra.`)},
		}}

		options := NewGenerationOptions(
			WithReasoningEffort("medium"),
			WithMaxGenerationTokens(12000),
			WithTemperature(1.0),
		)
		options[ResponsesThoughtSummaryDetailParam] = responses.ReasoningSummaryDetailed

		resp, err := adapter.Generate(ctx, GenerationRequest{
			Model:        reasoningModel,
			Instructions: SystemMessage(TextBlock("You solve hard reasoning tasks carefully and expose detailed reasoning summaries when requested.")),
			Dialog:       dialog,
			Options:      options,
		})
		if err != nil {
			t.Fatalf("StreamingAdapter Generate failed: %v", err)
		}
		if len(resp.Candidates) != 1 {
			t.Fatalf("expected 1 candidate, got %d", len(resp.Candidates))
		}

		var thinking strings.Builder
		var answer strings.Builder
		thinkingBlockCount := 0
		for i, block := range resp.Candidates[0].Blocks {
			t.Logf("StreamingAdapter block %d: type=%s modality=%s mime=%s len=%d", i, block.BlockType, block.ModalityType, block.MimeType, len(block.Content.String()))
			switch block.BlockType {
			case Thinking:
				thinkingBlockCount++
				thinking.WriteString(block.Content.String())
				t.Logf("=== STREAMING ADAPTER THINKING BLOCK %d ===\n%s\n=== END THINKING BLOCK %d ===", thinkingBlockCount, block.Content.String(), thinkingBlockCount)
			case Content:
				answer.WriteString(block.Content.String())
			}
		}
		if thinkingBlockCount == 0 || strings.TrimSpace(thinking.String()) == "" {
			t.Fatalf("expected non-empty thinking summaries from live streaming response; blocks=%+v", resp.Candidates[0].Blocks)
		}
		t.Logf("=== STREAMING ADAPTER COMBINED THINKING OUTPUT ===\n%s\n=== END COMBINED THINKING OUTPUT ===", thinking.String())
		t.Logf("=== STREAMING ADAPTER ANSWER OUTPUT ===\n%s\n=== END ANSWER OUTPUT ===", answer.String())

		verdict, err := verifyThinkingSummaryFormatting(ctx, &client, verifierModel, thinking.String())
		if err != nil {
			t.Fatalf("verifier failed: %v", err)
		}
		t.Logf("Verifier verdict: %+v", verdict)
		if !verdict.ProperlyFormatted || verdict.HasWeirdInternalSplintering {
			t.Fatalf("streaming adapter thinking output formatting rejected: %+v\nThinking output:\n%s", verdict, thinking.String())
		}
	})
	t.Run("ResponsesGenerator/StreamingToolCallWithReasoning", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")

		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)

		tickerTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("Failed to generate schema: %v", err)
				}
				return schema
			}(),
		}

		// Use StreamingAdapter to get a Generator interface from the streaming path
		adapter := &StreamingAdapter{S: gen}

		// Turn 1: Ask the model to use a tool with reasoning enabled
		dialog := Dialog{{
			Role:   User,
			Blocks: []Block{TextBlock("What is the current price of Apple stock?")},
		}}

		options := NewGenerationOptions(WithReasoningEffort("low"))
		options[ResponsesThoughtSummaryDetailParam] = responses.ReasoningSummaryDetailed
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5Mini,
			Instructions: SystemMessage(TextBlock(responsesStockToolInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options:      options,
		}

		resp, err := adapter.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("Turn 1 Generate (streaming) failed: %v", err)
		}

		if resp.FinishReason != ToolUse {
			t.Fatalf("Expected ToolUse finish reason, got %v", resp.FinishReason)
		}

		// Verify that the compressed thinking blocks carry encrypted content
		assistantMsg := resp.Candidates[0]
		hasEncryptedReasoning := false
		var toolCallBlock Block
		for _, blk := range assistantMsg.Blocks {
			if blk.BlockType == Thinking && blk.ExtraFields != nil {
				if ec, ok := blk.ExtraFields[ResponsesExtraFieldEncryptedContent].(string); ok && ec != "" {
					hasEncryptedReasoning = true
					t.Logf("Found encrypted reasoning content in streaming response (length: %d)", len(ec))
				}
				if rid, ok := blk.ExtraFields[ResponsesExtraFieldReasoningID].(string); ok && rid != "" {
					t.Logf("Found reasoning ID in streaming response: %s", rid)
				}
			}
			if blk.BlockType == ToolCall {
				toolCallBlock = blk
			}
		}

		if !hasEncryptedReasoning {
			t.Error("Expected encrypted reasoning content in streaming Thinking block ExtraFields; stateless multi-turn tool calling requires this")
		}

		if toolCallBlock.ID == "" {
			t.Fatal("Expected a tool call block in the streaming response")
		}
		t.Logf("Tool call ID: %s, Content: %s", toolCallBlock.ID, toolCallBlock.Content)

		// Verify usage metadata is present
		if resp.UsageMetadata == nil {
			t.Error("Expected usage metadata in streaming response")
		} else {
			t.Logf("Usage metadata: %+v", resp.UsageMetadata)
		}

		// Turn 2: Provide tool result and let the model respond.
		// This tests that the encrypted reasoning from the streaming path
		// can be correctly reconstructed by buildInputItems.
		dialog = append(dialog, assistantMsg, Message{
			Role: ToolResult,
			Blocks: []Block{{
				ID:           toolCallBlock.ID,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str("Apple (AAPL) is currently trading at $187.50"),
			}},
		})

		request.Dialog = dialog
		resp2, err := adapter.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("Turn 2 Generate (streaming) failed: %v", err)
		}

		if resp2.FinishReason != EndTurn {
			t.Fatalf("Expected EndTurn finish reason, got %v", resp2.FinishReason)
		}

		// Verify the model produced a content response mentioning the price
		var finalContent string
		for _, blk := range resp2.Candidates[0].Blocks {
			if blk.BlockType == Content {
				finalContent = blk.Content.String()
				break
			}
		}

		if finalContent == "" {
			t.Fatal("Expected content in the final streaming response")
		}
		t.Logf("Final streaming response: %s", finalContent)

		if !strings.Contains(finalContent, "187") {
			t.Errorf("Expected the response to mention the stock price, got: %s", finalContent)
		}
	})
	t.Run("ResponsesProviderOptionHelpers", func(t *testing.T) {
		generator := NewResponsesGenerator(nil)
		params, err := generator.buildParams(nil, GenerationRequest{
			Model:            "gpt-5",
			PromptCacheKey:   "cache-key",
			SafetyIdentifier: "opaque-user",
			Options: NewGenerationOptions(
				WithReasoningEffort("low"),
				WithResponsesThoughtSummaryDetail("detailed"),
				WithResponsesServiceTier("fast"),
			),
		})
		if err != nil {
			t.Fatalf("buildParams() error = %v", err)
		}
		encoded, err := json.Marshal(params)
		if err != nil {
			t.Fatalf("marshal params: %v", err)
		}
		var wire map[string]any
		if err := json.Unmarshal(encoded, &wire); err != nil {
			t.Fatalf("decode params: %v", err)
		}
		reasoning, ok := wire["reasoning"].(map[string]any)
		if !ok || reasoning["effort"] != "low" || reasoning["summary"] != "detailed" {
			t.Fatalf("reasoning = %#v", wire["reasoning"])
		}
		if wire["prompt_cache_key"] != "cache-key" || wire["safety_identifier"] != "opaque-user" || wire["service_tier"] != "fast" {
			t.Fatalf("provider options = %#v", wire)
		}

		_, err = generator.buildParams(nil, GenerationRequest{
			Model: "gpt-5",
			Options: GenerationOptions{
				ResponsesThoughtSummaryDetailParam: "verbose",
			},
		})
		var invalid *InvalidParameterErr
		if !errors.As(err, &invalid) || invalid.Parameter != ResponsesThoughtSummaryDetailParam {
			t.Fatalf("invalid summary error = %T %v", err, err)
		}
	})
	t.Run("ResponsesStreamingAdapterPreservesReasoningSummaryParts", func(t *testing.T) {
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
		gen := NewResponsesGenerator(svc)
		adapter := StreamingAdapter{S: gen}

		resp, err := adapter.Generate(context.Background(), GenerationRequest{
			Model:  "gpt-5",
			Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hard question")}}},
		})
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
	})
}

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

const responsesStockToolInstructions = `You are a helpful assistant that uses tools when needed.
When asked about stock prices, use the get_stock_price tool.
After getting the price, report it to the user.`

const responsesLookupToolInstructions = `You are a helpful assistant that uses tools when needed.
When asked to look up information, always use the provided tools.
After getting tool results, report them to the user.`

type thinkingSummaryFormattingVerdict struct {
	ProperlyFormatted           bool   `json:"properly_formatted"`
	HasWeirdInternalSplintering bool   `json:"has_weird_internal_splintering"`
	Reason                      string `json:"reason"`
}

func verifyThinkingSummaryFormatting(ctx context.Context, client *openai.Client, model, thinking string) (thinkingSummaryFormattingVerdict, error) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"properly_formatted":             map[string]any{"type": "boolean"},
			"has_weird_internal_splintering": map[string]any{"type": "boolean"},
			"reason":                         map[string]any{"type": "string"},
		},
		"required":             []string{"properly_formatted", "has_weird_internal_splintering", "reason"},
		"additionalProperties": false,
	}

	prompt := `You are checking the formatting of a compressed reasoning-summary transcript produced by a streaming adapter.

Return properly_formatted=true only if the text reads like coherent completed summary paragraphs or sections. It is OK for separate completed summaries to be separated by newlines. Return has_weird_internal_splintering=true if line breaks appear to split tiny streaming fragments or phrase pieces, such as one or two words per line, mid-sentence line breaks that look like token chunks, or arbitrary all-caps/chopped fragments. Do not judge whether the reasoning is correct; judge formatting only.

Transcript to inspect:
---
` + thinking + `
---`

	resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: openai.ResponsesModel(model),
		Input: responses.ResponseNewParamsInputUnion{OfString: openai.String(prompt)},
		Text: responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   "thinking_summary_formatting_verdict",
					Schema: schema,
					Strict: openai.Bool(true),
				},
			},
		},
		MaxOutputTokens: openai.Int(512),
	}, option.WithRequestTimeout(2*time.Minute))
	if err != nil {
		return thinkingSummaryFormattingVerdict{}, err
	}

	var verdict thinkingSummaryFormattingVerdict
	if err := json.Unmarshal([]byte(resp.OutputText()), &verdict); err != nil {
		return thinkingSummaryFormattingVerdict{}, err
	}
	return verdict, nil
}
