package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
)

func newCerebrasTestGenerator(t *testing.T, server *httptest.Server) *CerebrasGenerator {
	t.Helper()
	generator, err := NewCerebrasGenerator(server.Client(), server.URL, "test-key")
	if err != nil {
		t.Fatalf("create Cerebras generator: %v", err)
	}
	return generator
}

func newLiveCerebrasGenerator(t *testing.T, apiKey string) *CerebrasGenerator {
	t.Helper()
	generator, err := NewCerebrasGenerator(nil, "", apiKey)
	if err != nil {
		t.Fatalf("create Cerebras generator: %v", err)
	}
	return generator
}

func TestCerebrasAdapterScenarios(t *testing.T) {
	t.Run("SharedClient", func(t *testing.T) {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path != "/chat/completions" || r.Method != http.MethodPost || r.Header.Get("Authorization") != "Bearer test-key" {
						t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
					}
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
						return
					}
					for key, want := range map[string]any{"model": "test-model", "max_completion_tokens": float64(128), "user": "native-user", "prompt_cache_key": "native-cache", "reasoning_effort": "high", "parallel_tool_calls": false, "logprobs": true, "top_logprobs": float64(3), "seed": float64(7), "service_tier": "flex"} {
						if !reflect.DeepEqual(body[key], want) {
							t.Errorf("%s = %#v, want %#v", key, body[key], want)
						}
					}
					if _, ok := body["safety_identifier"]; ok {
						t.Error("sent OpenAI safety identifier")
					}
					if !reflect.DeepEqual(body["prediction"], map[string]any{"type": "content", "content": "known"}) {
						t.Errorf("prediction = %v", body["prediction"])
					}
					if !reflect.DeepEqual(body["response_format"], map[string]any{"type": "json_object"}) {
						t.Errorf("response_format = %v", body["response_format"])
					}
					if !reflect.DeepEqual(body["logit_bias"], map[string]any{"42": -1.5}) {
						t.Errorf("logit_bias = %v", body["logit_bias"])
					}
					if got, _ := body["stream"].(bool); got != stream {
						t.Errorf("stream = %v, want %v", body["stream"], stream)
					}
					if stream {
						if !reflect.DeepEqual(body["stream_options"], map[string]any{"include_usage": true}) {
							t.Errorf("stream_options = %v", body["stream_options"])
						}
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprint(w, "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning\":\"think\",\"content\":\"answer\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"image_tokens\":1},\"time_info\":{\"queue_time\":0.1}}\n\ndata: [DONE]\n\n")
					} else {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprint(w, `{"id":"c1","choices":[{"index":0,"message":{"role":"assistant","reasoning":"think","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"image_tokens":1},"time_info":{"queue_time":0.1}}`)
					}
				}))
				defer server.Close()
				g := newCerebrasTestGenerator(t, server)
				if _, ok := any(g).(TokenCounter); ok {
					t.Fatal("exposes OpenAI counter")
				}
				request := GenerationRequest{Model: "test-model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, SafetyIdentifier: "native-user", PromptCacheKey: "native-cache", Options: NewGenerationOptions(
					WithMaxGenerationTokens(128), WithReasoningEffort("high"), WithCerebrasPrediction("known"), WithCerebrasLogitBias(map[string]float64{"42": -1.5}), WithCerebrasLogprobs(true), WithCerebrasTopLogprobs(3), WithCerebrasParallelToolCalls(false), WithCerebrasSeed(7), WithCerebrasServiceTier(CerebrasServiceTierFlex), WithCerebrasResponseFormat(map[string]any{"type": "json_object"}),
				)}
				var response Response
				var err error
				if stream {
					response, err = (&StreamingAdapter{S: g}).Generate(t.Context(), request)
				} else {
					response, err = g.Generate(t.Context(), request)
				}
				if err != nil {
					t.Fatal(err)
				}
				requireContentContaining(t, response, "answer")
				requireBlockType(t, response, Thinking)
				if response.UsageMetadata[UsageMetricInputTokens] != 3 {
					t.Errorf("usage = %v", response.UsageMetadata)
				}
				raw, _ := json.Marshal(response.ExtraFields)
				if !strings.Contains(string(raw), `"image_tokens":1`) || !strings.Contains(string(raw), `"queue_time":0.1`) {
					t.Errorf("lost native metadata: %s", raw)
				}
			})
		}
	})
	t.Run("RequestTranslation", func(t *testing.T) {
		g := &CerebrasGenerator{}
		options := NewGenerationOptions(WithOpenAIExtraBody(map[string]json.RawMessage{"user": json.RawMessage(`"explicit"`)}))
		before, _ := json.Marshal(options)
		request, err := g.prepareRequest(GenerationRequest{Options: options, SafetyIdentifier: "identity", PromptCacheKey: "cache-group"})
		if err != nil {
			t.Fatal(err)
		}
		fields := request.Options[OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)
		if string(fields["user"]) != `"explicit"` || request.SafetyIdentifier != "" || request.PromptCacheKey != "cache-group" {
			t.Fatalf("request = %+v", request)
		}
		after, _ := json.Marshal(options)
		if string(before) != string(after) {
			t.Fatal("mutated caller options")
		}
		request, err = g.prepareRequest(GenerationRequest{SafetyIdentifier: "identity"})
		if err != nil || string(request.Options[OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)["user"]) != `"identity"` {
			t.Fatalf("identity translation: %+v, %v", request, err)
		}
		for _, key := range []string{CerebrasGenerationOptionLogitBias, CerebrasGenerationOptionLogprobs, CerebrasGenerationOptionParallelToolCalls, CerebrasGenerationOptionPrediction, CerebrasGenerationOptionResponseFormat, CerebrasGenerationOptionSeed, CerebrasGenerationOptionServiceTier, CerebrasGenerationOptionTopLogprobs, OpenAIGenerationOptionExtraBody} {
			_, err := g.prepareRequest(GenerationRequest{Options: GenerationOptions{key: struct{}{}}})
			if err == nil {
				t.Errorf("accepted invalid %s", key)
			}
		}
	})
	t.Run("SharedClientErrors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"message":"quota exceeded"}}`)
		}))
		defer server.Close()
		g := newCerebrasTestGenerator(t, server)
		request := GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
		check := func(err error) {
			t.Helper()
			var apiErr *ApiErr
			if !errors.As(err, &apiErr) || apiErr.Provider != ProviderCerebras || apiErr.StatusCode != 429 {
				t.Fatalf("error = %v", err)
			}
		}
		_, err := g.Generate(t.Context(), request)
		check(err)
		for chunk := range g.Stream(t.Context(), request) {
			check(chunk.Err)
		}
		if _, err := (&CerebrasGenerator{}).Generate(t.Context(), request); err == nil {
			t.Fatal("accepted uninitialized generator")
		}
		for chunk := range (&CerebrasGenerator{}).Stream(t.Context(), request) {
			if chunk.Err == nil {
				t.Fatal("accepted uninitialized generator")
			}
		}
	})
	t.Run("CerebrasGenerator/Generate", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
		gen := newLiveCerebrasGenerator(t, apiKey)
		dialog := Dialog{
			{
				Role:   User,
				Blocks: []Block{TextBlock("Hello!")},
			},
		}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        "gpt-oss-120b",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) != 1 || len(resp.Candidates[0].Blocks) == 0 {
			t.Fatalf("empty response: %+v", resp)
		}
	})
	t.Run("CerebrasGenerator/Generate/reasoning/gemma", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
		// Gemma supports reasoning through the reasoning_effort parameter.
		gen := newLiveCerebrasGenerator(t, apiKey)
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is 15 * 12?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        "gemma-4-31b",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant that explains your reasoning step by step.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithReasoningEffort("medium")),
		}
		// Generate a response with reasoning enabled.
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			// Check if we have thinking blocks (reasoning)
			hasThinking := false
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Thinking {
					hasThinking = true
				}
			}
			if hasThinking {
			}
			// Find the main content block (not thinking)
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Content {
					content := block.Content.String()
					if strings.Contains(content, "180") {
					}
					break
				}
			}
		}
		// Append the previous response and ask a follow-up question to test reasoning retention
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Now what is 20 * 15?"),
				},
			},
		})
		request.Dialog = dialog
		// Generate response with reasoning (the previous reasoning should be retained)
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			// Check if we have thinking blocks
			hasThinking := false
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Thinking {
					hasThinking = true
				}
			}
			if hasThinking {
			}
			// Find the main content block
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Content {
					content := block.Content.String()
					if strings.Contains(content, "300") {
					}
					break
				}
			}
		}
	})
	t.Run("CerebrasGenerator/Generate/reasoning/gptoss", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
		// Use gpt-oss-120b model which supports reasoning with reasoning_effort parameter
		gen := newLiveCerebrasGenerator(t, apiKey)
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is the square root of 144?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        "gpt-oss-120b",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant that explains your reasoning step by step.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithReasoningEffort("medium")),
		}
		// Generate response with reasoning enabled (medium effort)
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			// Check if we have thinking blocks (reasoning)
			hasThinking := false
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Thinking {
					hasThinking = true
				}
			}
			if hasThinking {
			}
			// Find the main content block (not thinking)
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Content {
					content := block.Content.String()
					if strings.Contains(content, "12") {
					}
					break
				}
			}
		}
		// Append the previous response and ask a follow-up question to test reasoning retention
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is the square root of 225?"),
				},
			},
		})
		request.Dialog = dialog
		// Generate response with reasoning (the previous reasoning should be retained)
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			// Check if we have thinking blocks
			hasThinking := false
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Thinking {
					hasThinking = true
				}
			}
			if hasThinking {
			}
			// Find the main content block
			for _, block := range resp.Candidates[0].Blocks {
				if block.BlockType == Content {
					content := block.Content.String()
					if strings.Contains(content, "15") {
					}
					break
				}
			}
		}
	})
	t.Run("CerebrasGenerator/RequestTools", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
		cgen := newLiveCerebrasGenerator(t, apiKey)
		instructions := `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`
		// Define a request tool
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
		dialog := Dialog{
			{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}},
		}
		request := GenerationRequest{
			Model:        "gpt-oss-120b",
			Instructions: SystemMessage(TextBlock(instructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options:      NewGenerationOptions(WithToolChoice("get_stock_price")),
		}
		// Force the tool call
		resp, err := cgen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
			t.Fatal("empty response")
			return
		}
		// Find and print the tool call JSON
		var toolCall Block
		for _, b := range resp.Candidates[0].Blocks {
			if b.BlockType == ToolCall {
				toolCall = b
				break
			}
		}
		if got := toolCall.Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		// Append tool result and continue the conversation
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: ToolResult,
			Blocks: []Block{
				{ID: toolCall.ID, BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")},
			},
		})
		request.Dialog = dialog
		request.Options = NewGenerationOptions(WithToolChoice("none"))
		// Ask model to answer now without calling tools
		resp, err = cgen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
				t.Fatal("expected non-empty content")
			}
		}
	})
}
