package gai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"

	zaiapi "github.com/spachava753/gai/internal/zai"
)

func requireContentContaining(t *testing.T, resp Response, want string) {
	t.Helper()
	if len(resp.Candidates) == 0 {
		t.Fatal("no candidates returned")
	}
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == Content && strings.Contains(strings.ToLower(block.Content.String()), strings.ToLower(want)) {
			return
		}
	}
	t.Fatalf("no content block contained %q; response: %+v", want, resp)
}

func requireBlockType(t *testing.T, resp Response, blockType string) Block {
	t.Helper()
	if len(resp.Candidates) == 0 {
		t.Fatal("no candidates returned")
	}
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == blockType {
			return block
		}
	}
	t.Fatalf("no %s block found; response: %+v", blockType, resp)
	return Block{}
}

func TestZaiGeneratorUsesGeneratedSSEClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/paas/v4/chat/completions" {
			http.Error(w, "unexpected request", http.StatusNotFound)
			return
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			http.Error(w, "missing authorization", http.StatusUnauthorized)
			return
		}
		var request struct {
			Stream bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil || !request.Stream {
			http.Error(w, "expected streaming request", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\n" +
				"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"}}]}\n\n" +
				"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"calculate\",\"arguments\":\"{\\\"x\\\":1}\"}}]}}]}\n\n" +
				"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	defer server.Close()

	client, err := zaiapi.NewClient(server.URL, zaiSecuritySource{apiKey: "test-key"}, zaiapi.WithClient(server.Client()))
	if err != nil {
		t.Fatalf("create generated client: %v", err)
	}
	generator := NewZaiGenerator(client, "")

	var thinking, content string
	var toolCall string
	var usage map[string]int
	for chunk := range generator.Stream(t.Context(), GenerationRequest{
		Model:  "glm-5",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("calculate")}}},
	}) {
		if chunk.Err != nil {
			t.Fatalf("stream returned error: %v", chunk.Err)
		}
		switch chunk.Block.BlockType {
		case Thinking:
			thinking += chunk.Block.Content.String()
		case Content:
			content += chunk.Block.Content.String()
		case ToolCall:
			toolCall += chunk.Block.Content.String()
		case MetadataBlockType:
			if err := json.Unmarshal([]byte(chunk.Block.Content.String()), &usage); err != nil {
				t.Fatalf("decode usage metadata: %v", err)
			}
		}
	}

	if thinking != "thinking" || content != "answer" {
		t.Fatalf("thinking = %q, content = %q", thinking, content)
	}
	if toolCall != `calculate{"x":1}` {
		t.Fatalf("tool call chunks = %q", toolCall)
	}
	if usage[UsageMetricInputTokens] != 3 || usage[UsageMetricGenerationTokens] != 2 || usage[UsageMetricCacheReadTokens] != 1 {
		t.Fatalf("usage metadata = %v", usage)
	}
}

func TestZaiGeneratorUsesGeneratedJSONClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Stream bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil || request.Stream {
			http.Error(w, "expected non-streaming request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"completion_1","choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`))
	}))
	defer server.Close()

	client, err := zaiapi.NewClient(server.URL, zaiSecuritySource{apiKey: "test-key"}, zaiapi.WithClient(server.Client()))
	if err != nil {
		t.Fatalf("create generated client: %v", err)
	}
	response, err := NewZaiGenerator(client, "").Generate(t.Context(), GenerationRequest{
		Model:  "glm-5",
		Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("answer")}}},
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if response.FinishReason != EndTurn || len(response.Candidates) != 1 || len(response.Candidates[0].Blocks) != 1 {
		t.Fatalf("response = %+v", response)
	}
	if got := response.Candidates[0].Blocks[0].Content.String(); got != "answer" {
		t.Fatalf("content = %q, want answer", got)
	}
	if response.UsageMetadata[UsageMetricInputTokens] != 3 || response.UsageMetadata[UsageMetricGenerationTokens] != 2 {
		t.Fatalf("usage metadata = %v", response.UsageMetadata)
	}
}

func TestZaiGenerator(t *testing.T) {
	t.Run("Generate", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		dialog := Dialog{
			{
				Role:   User,
				Blocks: []Block{TextBlock("Hello!")},
			},
		}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) == 0 {
			t.Fatal("no candidates returned")
			return
		}
		if len(resp.Candidates[0].Blocks) == 0 {
			t.Fatal("no blocks in response")
			return
		}
	})

	t.Run("GenerateThinking", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		dialog := Dialog{
			{
				Role:   User,
				Blocks: []Block{TextBlock("What is the square root of 144?")},
			},
		}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant that explains your reasoning step by step.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
			t.Fatal("empty response")
			return
		}
		// Check for thinking block
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
				break
			}
		}
		if !hasThinking {
			t.Fatal("no thinking block found")
			return
		}
		// Check for correct answer in content
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				if strings.Contains(block.Content.String(), "12") {
					return
				}
				t.Fatalf("content = %q, want it to contain 12", block.Content.String())
			}
		}
		t.Fatal("no content block found")
	})

	t.Run("GenerateInterleavedThinking", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		// Preserve thinking across tool turns through a request option.
		gen := NewZaiGenerator(nil, apiKey)
		// Register a weather tool
		weatherTool := Tool{
			Name:        "get_weather",
			Description: "Get the current weather for a city",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					City string `json:"city" jsonschema:"required" jsonschema_description:"The city name"`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		// First turn: ask about weather
		dialog := Dialog{
			{
				Role:   User,
				Blocks: []Block{TextBlock("What's the weather like in Beijing?")},
			},
		}
		request := GenerationRequest{
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Tools:        []Tool{weatherTool},
			Options:      NewGenerationOptions(WithZaiClearThinking(false)),
		}
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		thinkingBlock := requireBlockType(t, resp, Thinking)
		if thinkingBlock.ExtraFields[ThinkingExtraFieldGeneratorKey] != ThinkingGeneratorZai {
			t.Fatalf("thinking generator = %v, want %s", thinkingBlock.ExtraFields[ThinkingExtraFieldGeneratorKey], ThinkingGeneratorZai)
		}
		toolCallBlock := requireBlockType(t, resp, ToolCall)
		var firstCall ToolCallInput
		if err := json.Unmarshal([]byte(toolCallBlock.Content.String()), &firstCall); err != nil {
			t.Fatalf("parse first tool call: %v", err)
		}
		if firstCall.Name != "get_weather" {
			t.Fatalf("tool call name = %q, want get_weather", firstCall.Name)
		}
		if city, _ := firstCall.Parameters["city"].(string); !strings.Contains(strings.ToLower(city), "beijing") {
			t.Fatalf("tool call city = %v, want Beijing", firstCall.Parameters["city"])
		}
		// Append assistant response and provide tool result
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           toolCallBlock.ID,
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(`{"weather": "Sunny", "temperature": "25°C", "humidity": "40%"}`),
				},
			},
		})
		// Second turn: model reasons about the tool result
		request.Dialog = dialog
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		requireContentContaining(t, resp, "Sunny")
	})

	t.Run("GenerateMultiTurn", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		// First turn
		dialog := Dialog{
			{
				Role:   User,
				Blocks: []Block{TextBlock("What is 5 + 3?")},
			},
		}
		request := GenerationRequest{
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock("You are a helpful math tutor.")),
			Dialog:       dialog,
		}
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		found := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content && strings.Contains(block.Content.String(), "8") {
				found = true
				break
			}
		}
		if !found {
			t.Fatal("Turn 1 expected '8' in response")
			return
		}
		// Second turn: continue conversation
		dialog = append(dialog, resp.Candidates[0], Message{
			Role:   User,
			Blocks: []Block{TextBlock("Now multiply that result by 2")},
		})
		request.Dialog = dialog
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		found = false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content && strings.Contains(block.Content.String(), "16") {
				found = true
				break
			}
		}
		if !found {
			t.Fatal("Turn 2 expected '16' in response")
			return
		}
		// Third turn
		dialog = append(dialog, resp.Candidates[0], Message{
			Role:   User,
			Blocks: []Block{TextBlock("Divide that by 4")},
		})
		request.Dialog = dialog
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		found = false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content && strings.Contains(block.Content.String(), "4") {
				found = true
				break
			}
		}
		if !found {
			t.Fatal("Turn 3 expected '4' in response")
			return
		}
	})

	t.Run("RequestTools", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		instructions := `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like:
<example>
435.56
</example>`
		// Register a stock price tool
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
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock(instructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options:      NewGenerationOptions(WithToolChoice("get_stock_price")),
		}
		// Force the tool call
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
			t.Fatal("empty response")
			return
		}
		// Find the tool call
		var toolCall Block
		for _, b := range resp.Candidates[0].Blocks {
			if b.BlockType == ToolCall {
				toolCall = b
				break
			}
		}
		if toolCall.BlockType != ToolCall {
			t.Fatal("no tool call found")
			return
		}
		var tc ToolCallInput
		if err := json.Unmarshal([]byte(toolCall.Content.String()), &tc); err != nil {
			t.Fatalf("parse tool call: %v", err)
		}
		if tc.Name != "get_stock_price" {
			t.Fatalf("tool name = %q, want get_stock_price", tc.Name)
		}
		// Append tool result and continue
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: ToolResult,
			Blocks: []Block{
				{ID: toolCall.ID, BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("189.45")},
			},
		})
		request.Dialog = dialog
		request.Options = NewGenerationOptions(WithToolChoice("none"))
		// Get final answer without calling tools
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Check if final response contains the price from tool result
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				if strings.Contains(block.Content.String(), "189.45") || strings.Contains(block.Content.String(), "189") {
					return
				}
				t.Fatalf("content = %q, want it to contain 189.45", block.Content.String())
			}
		}
		t.Fatal("no content block in final response")
	})

	t.Run("GenerateParallelToolCalls", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		instructions := `You are a tool-calling assistant.
When the user asks for weather and stock information together, call both tools in the same assistant response before answering.`
		weatherTool := Tool{
			Name:        "get_weather",
			Description: "Get the current weather for a city.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					City string `json:"city" jsonschema:"required" jsonschema_description:"The city name"`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		stockTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL"`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Call get_weather for Beijing and get_stock_price for AAPL now. Do not answer with prose until both tool calls have been made.")}}}
		resp, err := gen.Generate(t.Context(), GenerationRequest{
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock(instructions)),
			Dialog:       dialog,
			Tools:        []Tool{weatherTool, stockTool},
			Options:      NewGenerationOptions(WithToolChoice(ToolChoiceToolsRequired)),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		seen := map[string]bool{}
		var toolCalls int
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType != ToolCall {
				continue
			}
			toolCalls++
			var call ToolCallInput
			if err := json.Unmarshal([]byte(block.Content.String()), &call); err != nil {
				t.Fatalf("parse tool call %q: %v", block.Content.String(), err)
			}
			seen[call.Name] = true
		}
		if toolCalls < 2 || !seen["get_weather"] || !seen["get_stock_price"] {
			t.Fatalf("tool calls = %d, seen = %v, response: %+v", toolCalls, seen, resp)
		}
	})

	t.Run("Stream", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		dialog := Dialog{
			{
				Role:   User,
				Blocks: []Block{TextBlock("Count from 1 to 5")},
			},
		}
		var contentChunks int
		var thinkingChunks int
		for chunk := range gen.Stream(context.Background(), GenerationRequest{
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		}) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			switch chunk.Block.BlockType {
			case Content:
				contentChunks++
			case Thinking:
				thinkingChunks++
			case MetadataBlockType:
				// ignore usage metadata
			}
		}
		if contentChunks == 0 {
			t.Fatal("no content chunks received")
			return
		}
		if thinkingChunks == 0 {
			t.Fatal("no thinking chunks received")
			return
		}
	})

	t.Run("StreamDisableThinking", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Count from 1 to 3")}}}
		var contentChunks int
		for chunk := range gen.Stream(context.Background(), GenerationRequest{
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock("You are concise.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithZaiThinking(false)),
		}) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			switch chunk.Block.BlockType {
			case Thinking:
				t.Fatalf("thinking chunk found when thinking is disabled: %q", chunk.Block.Content.String())
			case Content:
				contentChunks++
			}
		}
		if contentChunks == 0 {
			t.Fatal("no content chunks received")
		}
	})

	t.Run("StreamToolCalling", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		// Register a calculator tool
		calcTool := Tool{
			Name:        "calculate",
			Description: "Perform a mathematical calculation",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Expression string `json:"expression" jsonschema:"required" jsonschema_description:"The mathematical expression to evaluate"`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		dialog := Dialog{
			{
				Role:   User,
				Blocks: []Block{TextBlock("What is 123 * 456? Use the calculator tool.")},
			},
		}
		var toolChunks []string
		for chunk := range gen.Stream(context.Background(), GenerationRequest{
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Tools:        []Tool{calcTool},
			Options:      NewGenerationOptions(WithToolChoice(ToolChoiceToolsRequired)),
		}) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			if chunk.Block.BlockType == ToolCall {
				toolChunks = append(toolChunks, chunk.Block.Content.String())
			}
		}
		if len(toolChunks) == 0 {
			t.Fatal("no tool call received in stream")
			return
		}
		toolPayload := strings.Join(toolChunks, "")
		if !strings.Contains(toolPayload, "calculate") && !strings.Contains(toolPayload, "123") {
			t.Fatalf("stream tool chunks = %q, want function name or arguments", toolPayload)
		}
	})

	t.Run("GenerateVisionImageURL", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		dialog := Dialog{{Role: User, Blocks: []Block{
			{BlockType: Content, ModalityType: Image, MimeType: "image/png", Content: Str("https://cdn.bigmodel.cn/static/logo/register.png")},
			TextBlock("Describe this image in one short sentence."),
		}}}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        "glm-5v-turbo",
			Instructions: SystemMessage(TextBlock("Answer briefly.")),
			Dialog:       dialog,
			Options: NewGenerationOptions(
				WithMaxGenerationTokens(512),
				WithZaiThinking(false),
			),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		requireBlockType(t, resp, Content)
	})

	t.Run("GenerateVisionVideoURL", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		dialog := Dialog{{Role: User, Blocks: []Block{
			{BlockType: Content, ModalityType: Video, MimeType: "video/quicktime", Content: Str("https://cdn.bigmodel.cn/agent-demos/lark/113123.mov")},
			TextBlock("Describe what happens in this video in one short sentence."),
		}}}
		resp, err := gen.Generate(t.Context(), GenerationRequest{
			Model:        "glm-5v-turbo",
			Instructions: SystemMessage(TextBlock("Answer briefly.")),
			Dialog:       dialog,
			Options: NewGenerationOptions(
				WithMaxGenerationTokens(512),
				WithZaiThinking(false),
			),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		requireBlockType(t, resp, Content)
	})

	t.Run("GenerateVisionPDFURL", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		pdf := PDFBlock([]byte("placeholder"), "demo1.pdf")
		pdf.ExtraFields[ZaiExtraFieldURL] = "https://cdn.bigmodel.cn/static/demo/demo1.pdf"
		dialog := Dialog{{Role: User, Blocks: []Block{
			pdf,
			TextBlock("What type of document is this? Answer in one sentence."),
		}}}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        "glm-5v-turbo",
			Instructions: SystemMessage(TextBlock("Answer briefly.")),
			Dialog:       dialog,
			Options: NewGenerationOptions(
				WithMaxGenerationTokens(512),
				WithZaiThinking(false),
			),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		requireBlockType(t, resp, Content)
	})

	t.Run("DisableThinking", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "Z_API_KEY")
		gen := NewZaiGenerator(nil, apiKey)
		dialog := Dialog{
			{
				Role:   User,
				Blocks: []Block{TextBlock("What is 2 + 2?")},
			},
		}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        "glm-5.1",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant. Be concise.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithZaiThinking(false)),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
			t.Fatal("empty response")
			return
		}
		// Verify no thinking blocks exist
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				t.Fatal("thinking block found when thinking is disabled")
				return
			}
		}
		// Verify we got a content block with the answer
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				if strings.Contains(block.Content.String(), "4") {
					return
				}
				t.Fatalf("content = %q, want it to contain 4", block.Content.String())
			}
		}
		t.Fatal("no content block found")
	})
}
