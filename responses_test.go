package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

func TestEventAPIAdapterScenarios(t *testing.T) {
	t.Run("ResponsesErrorEventClassification", func(t *testing.T) { testResponsesErrorEventClassification(t) })
	t.Run("ResponsesFailureClassification", func(t *testing.T) { testResponsesFailureClassification(t) })
	t.Run("ResponsesGeneratorBuildInputItemsPreservesAssistantMessagePhase", func(t *testing.T) { testResponsesGeneratorBuildInputItemsPreservesAssistantMessagePhase(t) })
	t.Run("ResponsesGeneratorBuildInputItemsPreservesAssistantMessagePhaseWithoutTextContent", func(t *testing.T) {
		testResponsesGeneratorBuildInputItemsPreservesAssistantMessagePhaseWithoutTextContent(t)
	})
	t.Run("ResponsesGeneratorBuildInputItemsRejectsInvalidAssistantMessagePhase", func(t *testing.T) { testResponsesGeneratorBuildInputItemsRejectsInvalidAssistantMessagePhase(t) })
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
	t.Run("ResponsesGeneratorGeneratePreservesAssistantMessagePhase", func(t *testing.T) { testResponsesGeneratorGeneratePreservesAssistantMessagePhase(t) })
	t.Run("ResponsesGeneratorGeneratePreservesAssistantToolOnlyPhaseRoundTrip", func(t *testing.T) { testResponsesGeneratorGeneratePreservesAssistantToolOnlyPhaseRoundTrip(t) })
	t.Run("ResponsesGeneratorGenerateReturnsContentPolicyFinishReason", func(t *testing.T) { testResponsesGeneratorGenerateReturnsContentPolicyFinishReason(t) })
	t.Run("ResponsesGeneratorGenerateReturnsFailedResponseError", func(t *testing.T) { testResponsesGeneratorGenerateReturnsFailedResponseError(t) })
	t.Run("ResponsesGeneratorStreamRetriesServerOverloadSSEError", func(t *testing.T) { testResponsesGeneratorStreamRetriesServerOverloadSSEError(t) })
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
	t.Run("ResponsesGenerator/Generate/Thinking/Logging", func(t *testing.T) { testResponsesGenerator_Generate_Thinking_Logging(t) })
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
				WithThinkingBudget("high"),
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
			Options:      NewGenerationOptions(WithThinkingBudget("low")),
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
			WithThinkingBudget("medium"),
			WithTemperature(1.0),
			WithResponsesThoughtSummaryDetail("detailed"),
			WithResponsesPromptCacheKey("responses-thinking-example:v1"),
		)
		request := GenerationRequest{
			Model:        openai.ChatModelGPT5,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      options,
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
	t.Run("ResponsesGenerator/ReasoningTokenPreservation/Generate", func(t *testing.T) { testResponsesGenerator_ReasoningTokenPreservation_Generate(t) })
	t.Run("ResponsesGenerator/ReasoningTokenPreservation/Stream", func(t *testing.T) { testResponsesGenerator_ReasoningTokenPreservation_Stream(t) })
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
			Options:      NewGenerationOptions(WithThinkingBudget("medium")),
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
	t.Run("ResponsesGenerator/StatelessToolCallWithReasoning", func(t *testing.T) { testResponsesGenerator_StatelessToolCallWithReasoning(t) })
	t.Run("ResponsesGenerator/StreamMetadata", func(t *testing.T) { testResponsesGenerator_StreamMetadata(t) })
	t.Run("ResponsesGenerator/Stream/Thinking/Logging", func(t *testing.T) { testResponsesGenerator_Stream_Thinking_Logging(t) })
	t.Run("ResponsesGenerator/Stream/thinking", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := openai.NewClient(option.WithAPIKey(apiKey))
		gen := NewResponsesGenerator(&client.Responses)
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the capital of France? Reply with just the city name.")}}}
		options := NewGenerationOptions(
			WithThinkingBudget("low"),
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
	t.Run("ResponsesGenerator/StreamingAdapter/LiveThinkingSummaryFormatting", func(t *testing.T) { testResponsesGenerator_StreamingAdapter_LiveThinkingSummaryFormatting(t) })
	t.Run("ResponsesGenerator/StreamingToolCallWithReasoning", func(t *testing.T) { testResponsesGenerator_StreamingToolCallWithReasoning(t) })
	t.Run("ResponsesProviderOptionHelpers", func(t *testing.T) {
		generator := NewResponsesGenerator(nil)
		params, err := generator.buildParams(nil, GenerationRequest{
			Model: "gpt-5",
			Options: NewGenerationOptions(
				WithThinkingBudget("low"),
				WithResponsesThoughtSummaryDetail("detailed"),
				WithResponsesPromptCacheKey("cache-key"),
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
		if wire["prompt_cache_key"] != "cache-key" || wire["service_tier"] != "fast" {
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
	t.Run("ResponsesStreamingAdapterPreservesReasoningSummaryParts", func(t *testing.T) { testResponsesStreamingAdapterPreservesReasoningSummaryParts(t) })
}

// ExampleStreamingAdapter_responses demonstrates using StreamingAdapter with
// the ResponsesGenerator for stateless multi-turn conversation. The adapter
// compresses streaming chunks into complete Response objects, making it easy
// to append the assistant's response to the dialog for subsequent turns.
func testStreamingAdapter_responses(t *testing.T) {
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
}

// ExampleStreamingAdapter_responses_toolUse demonstrates using StreamingAdapter
// with tool calling on the Responses API. The adapter compresses streaming tool
// call chunks into complete blocks, preserving IDs and Thinking block ExtraFields
// so the dialog can be passed back for subsequent turns without any manual chunk
// reconstruction.
func testStreamingAdapter_responses_toolUse(t *testing.T) {
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
}
