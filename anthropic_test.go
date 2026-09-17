package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	anthropicshared "github.com/anthropics/anthropic-sdk-go/shared"
	"github.com/google/jsonschema-go/jsonschema"
)

const anthropicStockComparisonInstructions = `You are a helpful assistant that compares the price of two stocks and returns the ticker of whichever is greater.
Only mention one of the stock tickers and nothing else.
Only output the price, like
<example>
User: Which one is more expensive? Apple or NVidia?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: Apple: 123.45; Nvidia: 345.65
Assistant: Nvidia
</example>
<example>
User: Which one is more expensive? Microsft or Netflix?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: MSFT: 876.45; NFLX: 345.65
Assistant: MSFT
</example>
`

func TestClaudeAdapterScenarios(t *testing.T) {
	t.Run("AnthropicErrorClassification", func(t *testing.T) {
		tests := []struct {
			name       string
			statusCode int
			typeCode   anthropicshared.ErrorType
			want       APIErrorKind
		}{
			{
				name:     "SDK overload type without status",
				typeCode: anthropicshared.ErrorTypeOverloadedError,
				want:     APIErrorKindOverloaded,
			},
			{
				name:       "Anthropic overload status without type",
				statusCode: 529,
				want:       APIErrorKindOverloaded,
			},
			{
				name:       "HTTP fallback",
				statusCode: http.StatusUnauthorized,
				want:       APIErrorKindAuthentication,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := classifyAnthropicError(tt.statusCode, tt.typeCode); got != tt.want {
					t.Fatalf("classifyAnthropicError() = %q, want %q", got, tt.want)
				}
			})
		}
	})
	t.Run("AnthropicGenerateReturnsContentPolicyErrorForRefusal", func(t *testing.T) {
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
	})
	t.Run("AnthropicGeneratorStreamRetriesOverloadedSSEError", func(t *testing.T) {
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
	})
	t.Run("AnthropicGenerator/Count", func(t *testing.T) {
		requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
		// Create an Anthropic client
		client := a.NewClient()
		generator := NewAnthropicGenerator(&client.Messages)
		// Create a dialog with a user message
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is the capital of France?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        string(a.ModelClaudeHaiku4_5),
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		}
		// Count tokens in the dialog
		tokenCount, err := generator.Count(context.Background(), request)
		if err != nil {
			t.Fatalf("count tokens: %v", err)
		}
		if tokenCount == 0 {
			t.Fatal("expected non-zero token count")
		}
		// Add a response to the dialog
		dialog = append(dialog, Message{
			Role: Assistant,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("The capital of France is Paris. It's a beautiful city known for its culture, art, and cuisine."),
				},
			},
		})
		request.Dialog = dialog
		// Count tokens in the updated dialog
		tokenCount, err = generator.Count(context.Background(), request)
		if err != nil {
			t.Fatalf("count tokens: %v", err)
		}
		if tokenCount == 0 {
			t.Fatal("expected non-zero token count")
		}
	})
	t.Run("AnthropicGenerator/Count/IncludesTools", func(t *testing.T) {
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
	})
	t.Run("AnthropicGenerator/Generate", func(t *testing.T) {
		requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
		// Create an Anthropic client
		client := a.NewClient()
		// Demonstration of how to enable system prompt caching
		svc := NewAnthropicServiceWrapper(&client.Messages, EnableSystemCaching)
		// Instantiate an Anthropic Generator
		gen := NewAnthropicGenerator(svc)
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hi!"),
					},
				},
			},
		}
		// Generate a response
		// Note that anthropic generator requires that max generation tokens generation param be set
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        string(a.ModelClaudeHaiku4_5),
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(1024)),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// The exact response text may vary, so we'll just print a placeholder
		// Customize generation parameters
		options := NewGenerationOptions(
			WithTemperature(0.7),
			WithMaxGenerationTokens(1024),
		)
		resp, err = gen.Generate(context.Background(), GenerationRequest{
			Model:        string(a.ModelClaudeHaiku4_5),
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      options,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := len(resp.Candidates); got == 0 {
			t.Fatal("expected at least one item")
		}
	})
	t.Run("AnthropicGenerator/Generate/image", func(t *testing.T) {
		requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
		// This example assumes that sample.jpg is present in the current directory.
		imgBytes, err := os.ReadFile("sample.jpg")
		if err != nil {
			t.Skip("could not open sample.jpg")
			return
		}
		imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))
		client := a.NewClient()
		gen := NewAnthropicGenerator(&client.Messages)
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
						Content:      Str("What is in this image? (Hint, it's a character from The Croods, a DreamWorks animated movie.)"),
					},
				},
			},
		}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        string(a.ModelClaudeHaiku4_5),
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(512)),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) != 1 {
			t.Fatalf("candidates = %d, want 1", len(resp.Candidates))
		}
		if len(resp.Candidates[0].Blocks) != 1 {
			t.Fatalf("blocks = %d, want 1", len(resp.Candidates[0].Blocks))
		}
		if !strings.Contains(resp.Candidates[0].Blocks[0].Content.String(), "Crood") {
			t.Fatalf("content does not contain Crood")
		}
	})
	t.Run("AnthropicGenerator/Generate/pdf", func(t *testing.T) {
		requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
		// This example assumes that sample.pdf is present in the current directory.
		pdfBytes, err := os.ReadFile("sample.pdf")
		if err != nil {
			t.Skip("could not open sample.pdf")
			return
		}
		client := a.NewClient()
		gen := NewAnthropicGenerator(&client.Messages)
		// Create a dialog with PDF content
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					TextBlock("What is the title of this PDF? Just output the title and nothing else"),
					PDFBlock(pdfBytes, "paper.pdf"),
				},
			},
		}
		// Generate a response
		ctx := context.Background()
		response, err := gen.Generate(ctx, GenerationRequest{
			Model:        string(a.ModelClaudeSonnet4_6),
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(1024)),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// The response would contain the model's analysis of the PDF
		if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
			if got := response.Candidates[0].Blocks[0].Content.String(); got == "" {
				t.Fatal("expected non-empty content")
			}
		}
	})
	t.Run("AnthropicGenerator/Generate/thinking", func(t *testing.T) {
		requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
		// Create an Anthropic client
		client := a.NewClient()
		// Instantiate an Anthropic Generator
		gen := NewAnthropicGenerator(&client.Messages)
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hi!"),
					},
				},
			},
		}
		// Use thinking
		options := NewGenerationOptions(
			WithTemperature(1.0),
			WithMaxGenerationTokens(9000),
			WithThinkingBudget(5000),
		)
		request := GenerationRequest{
			Model:        string(a.ModelClaudeSonnet4_6),
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      options,
		}
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := len(resp.Candidates); got == 0 {
			t.Fatal("expected at least one item")
		}
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What can you do?"),
				},
			},
		})
		request.Dialog = dialog
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := len(resp.Candidates); got == 0 {
			t.Fatal("expected at least one item")
		}
	})
	t.Run("AnthropicGenerator/RequestTools", func(t *testing.T) {
		requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
		// Create an Anthropic client
		client := a.NewClient()
		// Demonstration of how to enable system and multi turn message prompt caching
		svc := NewAnthropicServiceWrapper(&client.Messages, EnableSystemCaching, EnableMultiTurnCaching)
		// Instantiate an Anthropic Generator
		gen := NewAnthropicGenerator(svc)
		instructions := `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`
		// Define request tools
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
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is the price of Apple stock?"),
					},
				},
			},
		}
		// Customize generation parameters
		request := GenerationRequest{
			Model:        string(a.ModelClaudeSonnet4_5),
			Instructions: SystemMessage(TextBlock(instructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options: NewGenerationOptions(
				WithToolChoice("get_stock_price"),
				WithMaxGenerationTokens(8096),
			),
		}
		// Generate a response
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           resp.Candidates[0].Blocks[0].ID,
					ModalityType: Text,
					Content:      Str("123.45"),
				},
			},
		})
		request.Dialog = dialog
		request.Options = NewGenerationOptions(WithMaxGenerationTokens(8096))
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
	})
	t.Run("AnthropicGenerator/RequestTools/parallelToolUse", func(t *testing.T) {
		requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
		// Create an Anthropic client
		client := a.NewClient()
		// Define request tools
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
		gen := NewAnthropicGenerator(&client.Messages)
		tickerTool.Description += "\nYou can call this tool in parallel"
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Which stock, Apple vs. Microsoft, is more expensive?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        string(a.ModelClaudeSonnet4_6),
			Instructions: SystemMessage(TextBlock(anthropicStockComparisonInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options: NewGenerationOptions(
				WithMaxGenerationTokens(8096),
				WithThinkingBudget(4000),
			),
		}
		// Generate a response
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[1].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		if got := resp.Candidates[0].Blocks[2].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           resp.Candidates[0].Blocks[1].ID,
					ModalityType: Text,
					Content:      Str("123.45"),
				},
			},
		}, Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           resp.Candidates[0].Blocks[2].ID,
					ModalityType: Text,
					Content:      Str("678.45"),
				},
			},
		})
		request.Dialog = dialog
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
	})
	t.Run("AnthropicGenerator/Stream", func(t *testing.T) {
		requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
		// Create an Anthropic client
		client := a.NewClient()
		// Demonstration of how to enable system prompt caching
		svc := NewAnthropicServiceWrapper(&client.Messages, EnableSystemCaching)
		// Instantiate an Anthropic Generator
		gen := NewAnthropicGenerator(svc)
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hi!"),
					},
				},
			},
		}
		// Stream a response
		var blocks []Block
		for chunk := range gen.Stream(context.Background(), GenerationRequest{
			Model:        string(a.ModelClaudeHaiku4_5),
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(1024)),
		}) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			blocks = append(blocks, chunk.Block)
		}
		if len(blocks) > 0 {
		}
	})
	t.Run("AnthropicGenerator/Stream/parallelToolUse", func(t *testing.T) {
		requireLiveAPIKey(t, "ANTHROPIC_API_KEY")
		// Create an Anthropic client
		client := a.NewClient()
		// Define request tools
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
		gen := NewAnthropicGenerator(&client.Messages)
		tickerTool.Description += "\nYou can call this tool in parallel"
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Which stock, Apple vs. Microsoft, is more expensive?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        string(a.ModelClaudeSonnet4_6),
			Instructions: SystemMessage(TextBlock(anthropicStockComparisonInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options: NewGenerationOptions(
				WithMaxGenerationTokens(32000),
				WithThinkingBudget(10000),
			),
		}
		// Stream a response
		var blocks []Block
		for chunk := range gen.Stream(context.Background(), request) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			blocks = append(blocks, chunk.Block)
		}
		if len(blocks) > 1 {
		}
		// collect the blocks
		var prevToolCallId string
		var toolCalls []Block
		var toolcallArgs string
		var toolCallInput ToolCallInput
		thinking := Block{
			BlockType:    Thinking,
			ModalityType: Text,
			MimeType:     "text/plain",
			ExtraFields:  make(map[string]interface{}),
		}
		thinkingStr := ""
		for _, block := range blocks {
			if block.BlockType == Thinking {
				if block.Content != nil {
					thinkingStr += block.Content.String()
				}
				maps.Copy(thinking.ExtraFields, block.ExtraFields)
				continue
			}
			// Skip metadata blocks
			if block.BlockType == MetadataBlockType {
				continue
			}
			if block.ID != "" && block.ID != prevToolCallId {
				if toolcallArgs != "" {
					// Parse the arguments string into a map
					if err := json.Unmarshal([]byte(toolcallArgs), &toolCallInput.Parameters); err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					// Marshal back to JSON for consistent representation
					toolUseJSON, err := json.Marshal(toolCallInput)
					if err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					toolCalls[len(toolCalls)-1].Content = Str(toolUseJSON)
					toolCallInput = ToolCallInput{}
					toolcallArgs = ""
				}
				prevToolCallId = block.ID
				toolCalls = append(toolCalls, Block{
					ID:           block.ID,
					BlockType:    ToolCall,
					ModalityType: Text,
					MimeType:     "text/plain",
				})
				toolCallInput.Name = block.Content.String()
			} else {
				toolcallArgs += block.Content.String()
			}
		}
		thinking.Content = Str(thinkingStr)
		if toolcallArgs != "" {
			// Parse the arguments string into a map
			if err := json.Unmarshal([]byte(toolcallArgs), &toolCallInput.Parameters); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// Marshal back to JSON for consistent representation
			toolUseJSON, err := json.Marshal(toolCallInput)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			toolCalls[len(toolCalls)-1].Content = Str(toolUseJSON)
			toolCallInput = ToolCallInput{}
		}
		if got := len(toolCalls); got == 0 {
			t.Fatal("expected at least one item")
		}
		assistantMsg := make([]Block, 0, len(toolCalls)+1)
		assistantMsg = append(assistantMsg, thinking)
		assistantMsg = append(assistantMsg, toolCalls...)
		dialog = append(dialog, Message{
			Role:   Assistant,
			Blocks: assistantMsg,
		},
			Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           toolCalls[0].ID,
						ModalityType: Text,
						Content:      Str("123.45"),
					},
				},
			}, Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           toolCalls[1].ID,
						ModalityType: Text,
						Content:      Str("678.45"),
					},
				},
			})
		// Stream a response
		blocks = nil
		request.Dialog = dialog
		for chunk := range gen.Stream(context.Background(), request) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			blocks = append(blocks, chunk.Block)
		}
		if len(blocks) > 0 {
		}
	})
	t.Run("AnthropicStreamReturnsContentPolicyErrorForRefusal", func(t *testing.T) {
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
	})
}

// mockAnthropicSvc is a mock implementation of AnthropicSvc for testing.
type mockAnthropicSvc struct {
	lastParams        a.MessageNewParams
	countTokensCalled bool
	lastToolsCount    int
	lastSystemPresent bool
	response          *a.Message
	streamEvents      []ssestream.Event
	streamFactory     func() *ssestream.Stream[a.MessageStreamEventUnion]
	streamCalls       int
}

func (m *mockAnthropicSvc) New(ctx context.Context, body a.MessageNewParams, opts ...option.RequestOption) (res *a.Message, err error) {
	m.lastParams = body
	return m.response, nil
}

func (m *mockAnthropicSvc) NewStreaming(ctx context.Context, params a.MessageNewParams, opts ...option.RequestOption) (stream *ssestream.Stream[a.MessageStreamEventUnion]) {
	m.lastParams = params
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
