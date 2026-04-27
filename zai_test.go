package gai

import (
	"context"
	"encoding/json"
	"github.com/google/jsonschema-go/jsonschema"
	"strings"
	"testing"
)

func TestZaiGenerator_Generate(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "Z_API_KEY")
	gen := NewZaiGenerator(nil, "glm-5.1", "You are a helpful assistant.", apiKey)
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("Hello!")},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, nil)
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
}
func TestZaiGenerator_Generate_thinking(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "Z_API_KEY")
	gen := NewZaiGenerator(
		nil, "glm-5.1",
		"You are a helpful assistant that explains your reasoning step by step.",
		apiKey,
	)
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("What is the square root of 144?")},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, nil)
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
}
func TestZaiGenerator_Generate_interleavedThinking(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "Z_API_KEY")
	// Create generator with preserved thinking (clearThinking=false)
	gen := NewZaiGenerator(
		nil, "glm-5.1",
		"You are a helpful assistant.",
		apiKey,
		WithZaiClearThinking(false), // Enable preserved thinking
	)
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
	if err := gen.Register(weatherTool); err != nil {
		t.Fatalf("register tool: %v", err)
	}
	// First turn: ask about weather
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("What's the weather like in Beijing?")},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var firstTurnTypes []string
	var toolCallBlock Block
	for _, block := range resp.Candidates[0].Blocks {
		firstTurnTypes = append(firstTurnTypes, block.BlockType)
		if block.BlockType == ToolCall {
			toolCallBlock = block
		}
	}
	if len(firstTurnTypes) == 0 {
		t.Fatal("first turn returned no blocks")
	}
	if toolCallBlock.BlockType != ToolCall {
		t.Fatal("no tool call found")
		return
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
	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var secondTurnTypes []string
	for _, block := range resp.Candidates[0].Blocks {
		secondTurnTypes = append(secondTurnTypes, block.BlockType)
	}
	if len(secondTurnTypes) == 0 {
		t.Fatal("second turn returned no blocks")
	}
}
func TestZaiGenerator_Generate_multiTurn(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "Z_API_KEY")
	gen := NewZaiGenerator(nil, "glm-5.1", "You are a helpful math tutor.", apiKey)
	// First turn
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("What is 5 + 3?")},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, nil)
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
	resp, err = gen.Generate(context.Background(), dialog, nil)
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
	resp, err = gen.Generate(context.Background(), dialog, nil)
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
}
func TestZaiGenerator_Register(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "Z_API_KEY")
	gen := NewZaiGenerator(nil, "glm-5.1", `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like:
<example>
435.56
</example>`, apiKey)
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
	if err := gen.Register(tickerTool); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	dialog := Dialog{
		{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}},
	}
	// Force the tool call
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ToolChoice: "get_stock_price"})
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
	// Get final answer without calling tools
	resp, err = gen.Generate(context.Background(), dialog, &GenOpts{ToolChoice: "none"})
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
}
func TestZaiGenerator_Stream(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "Z_API_KEY")
	gen := NewZaiGenerator(nil, "glm-5.1", "You are a helpful assistant.", apiKey)
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("Count from 1 to 5")},
		},
	}
	var contentChunks int
	var thinkingChunks int
	for chunk, err := range gen.Stream(context.Background(), dialog, nil) {
		if err != nil {
			t.Fatalf("stream returned error: %v", err)
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
}
func TestZaiGenerator_Stream_toolCalling(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "Z_API_KEY")
	gen := NewZaiGenerator(nil, "glm-5.1", "You are a helpful assistant.", apiKey)
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
	if err := gen.Register(calcTool); err != nil {
		t.Fatalf("register tool: %v", err)
	}
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("What is 123 * 456? Use the calculator tool.")},
		},
	}
	var hasToolCall bool
	for chunk, err := range gen.Stream(context.Background(), dialog, &GenOpts{ToolChoice: ToolChoiceToolsRequired}) {
		if err != nil {
			t.Fatalf("stream returned error: %v", err)
		}
		if chunk.Block.BlockType == ToolCall {
			hasToolCall = true
		}
	}
	if !hasToolCall {
		t.Fatal("no tool call received in stream")
		return
	}
}
func TestZaiGenerator_disableThinking(t *testing.T) {
	apiKey := skipOnMissingEnv(t, "Z_API_KEY")
	// Create generator with thinking disabled
	gen := NewZaiGenerator(
		nil, "glm-5.1",
		"You are a helpful assistant. Be concise.",
		apiKey,
		WithZaiThinking(false),
	)
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("What is 2 + 2?")},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, nil)
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
}
