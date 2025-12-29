package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

func ExampleZaiGenerator_Generate() {
	apiKey := os.Getenv("Z_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set Z_API_KEY env]")
		return
	}
	gen := NewZaiGenerator(nil, "glm-4.7", "You are a helpful assistant.", apiKey)
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("Hello!")},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(resp.Candidates) == 0 {
		fmt.Println("Error: no candidates returned")
		return
	}
	if len(resp.Candidates[0].Blocks) == 0 {
		fmt.Println("Error: no blocks in response")
		return
	}
	fmt.Println("Response received")

	// Output: Response received
}

func ExampleZaiGenerator_Generate_thinking() {
	apiKey := os.Getenv("Z_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set Z_API_KEY env]")
		return
	}

	gen := NewZaiGenerator(
		nil, "glm-4.7",
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
		fmt.Println("Error:", err)
		return
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
		fmt.Println("Error: empty response")
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
		fmt.Println("Error: no thinking block found")
		return
	}
	fmt.Println("Thinking block found")

	// Check for correct answer in content
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == Content {
			if strings.Contains(block.Content.String(), "12") {
				fmt.Println("Correct answer found")
				return
			}
			fmt.Printf("Error: expected '12' in content, got: %s\n", block.Content.String())
			return
		}
	}
	fmt.Println("Error: no content block found")

	// Output: Thinking block found
	// Correct answer found
}

func ExampleZaiGenerator_Generate_interleavedThinking() {
	apiKey := os.Getenv("Z_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set Z_API_KEY env]")
		return
	}

	// Create generator with preserved thinking (clearThinking=false)
	gen := NewZaiGenerator(
		nil, "glm-4.7",
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
				panic(err)
			}
			return schema
		}(),
	}
	if err := gen.Register(weatherTool); err != nil {
		fmt.Println("Error registering tool:", err)
		return
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
		fmt.Println("Error:", err)
		return
	}

	// Print block types from first turn
	fmt.Print("First turn:")
	var toolCallBlock Block
	for _, block := range resp.Candidates[0].Blocks {
		fmt.Printf(" %s", block.BlockType)
		if block.BlockType == ToolCall {
			toolCallBlock = block
		}
	}
	fmt.Println()

	if toolCallBlock.BlockType != ToolCall {
		fmt.Println("Error: no tool call found")
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
		fmt.Println("Error:", err)
		return
	}

	// Print block types from second turn
	fmt.Print("Second turn:")
	for _, block := range resp.Candidates[0].Blocks {
		fmt.Printf(" %s", block.BlockType)
	}
	fmt.Println()

	// Output: First turn: thinking content tool_call
	// Second turn: thinking content
}

func ExampleZaiGenerator_Generate_multiTurn() {
	apiKey := os.Getenv("Z_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set Z_API_KEY env]")
		return
	}

	gen := NewZaiGenerator(nil, "glm-4.7", "You are a helpful math tutor.", apiKey)

	// First turn
	dialog := Dialog{
		{
			Role:   User,
			Blocks: []Block{TextBlock("What is 5 + 3?")},
		},
	}

	resp, err := gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	found := false
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == Content && strings.Contains(block.Content.String(), "8") {
			found = true
			break
		}
	}
	if !found {
		fmt.Println("Error: Turn 1 expected '8' in response")
		return
	}
	fmt.Println("Turn 1: correct")

	// Second turn: continue conversation
	dialog = append(dialog, resp.Candidates[0], Message{
		Role:   User,
		Blocks: []Block{TextBlock("Now multiply that result by 2")},
	})

	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	found = false
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == Content && strings.Contains(block.Content.String(), "16") {
			found = true
			break
		}
	}
	if !found {
		fmt.Println("Error: Turn 2 expected '16' in response")
		return
	}
	fmt.Println("Turn 2: correct")

	// Third turn
	dialog = append(dialog, resp.Candidates[0], Message{
		Role:   User,
		Blocks: []Block{TextBlock("Divide that by 4")},
	})

	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	found = false
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == Content && strings.Contains(block.Content.String(), "4") {
			found = true
			break
		}
	}
	if !found {
		fmt.Println("Error: Turn 3 expected '4' in response")
		return
	}
	fmt.Println("Turn 3: correct")

	// Output: Turn 1: correct
	// Turn 2: correct
	// Turn 3: correct
}

func ExampleZaiGenerator_Register() {
	apiKey := os.Getenv("Z_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set Z_API_KEY env]")
		return
	}

	gen := NewZaiGenerator(nil, "glm-4.7", `You are a helpful assistant that returns the price of a stock and nothing else.

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
				panic(err)
			}
			return schema
		}(),
	}
	if err := gen.Register(tickerTool); err != nil {
		fmt.Println("Error:", err)
		return
	}

	dialog := Dialog{
		{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}},
	}

	// Force the tool call
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ToolChoice: "get_stock_price"})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
		fmt.Println("Error: empty response")
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
		fmt.Println("Error: no tool call found")
		return
	}

	var tc ToolCallInput
	if err := json.Unmarshal([]byte(toolCall.Content.String()), &tc); err != nil {
		fmt.Println("Error parsing tool call:", err)
		return
	}

	if tc.Name != "get_stock_price" {
		fmt.Printf("Error: expected tool 'get_stock_price', got '%s'\n", tc.Name)
		return
	}
	fmt.Println("Tool call received")

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
		fmt.Println("Error:", err)
		return
	}

	// Check if final response contains the price from tool result
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == Content {
			if strings.Contains(block.Content.String(), "189.45") || strings.Contains(block.Content.String(), "189") {
				fmt.Println("Final answer contains tool result")
				return
			}
			fmt.Printf("Error: expected '189.45' in answer, got: %s\n", block.Content.String())
			return
		}
	}
	fmt.Println("Error: no content block in final response")

	// Output: Tool call received
	// Final answer contains tool result
}

func ExampleZaiGenerator_Stream() {
	apiKey := os.Getenv("Z_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set Z_API_KEY env]")
		return
	}

	gen := NewZaiGenerator(nil, "glm-4.7", "You are a helpful assistant.", apiKey)
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
			fmt.Println("Error:", err)
			return
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
		fmt.Println("Error: no content chunks received")
		return
	}
	fmt.Println("Content chunks received")

	if thinkingChunks == 0 {
		fmt.Println("Error: no thinking chunks received")
		return
	}
	fmt.Println("Thinking chunks received")

	// Output: Content chunks received
	// Thinking chunks received
}

func ExampleZaiGenerator_Stream_toolCalling() {
	apiKey := os.Getenv("Z_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set Z_API_KEY env]")
		return
	}

	gen := NewZaiGenerator(nil, "glm-4.7", "You are a helpful assistant.", apiKey)

	// Register a calculator tool
	calcTool := Tool{
		Name:        "calculate",
		Description: "Perform a mathematical calculation",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Expression string `json:"expression" jsonschema:"required" jsonschema_description:"The mathematical expression to evaluate"`
			}]()
			if err != nil {
				panic(err)
			}
			return schema
		}(),
	}
	if err := gen.Register(calcTool); err != nil {
		fmt.Println("Error registering tool:", err)
		return
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
			fmt.Println("Error:", err)
			return
		}
		if chunk.Block.BlockType == ToolCall {
			hasToolCall = true
		}
	}

	if !hasToolCall {
		fmt.Println("Error: no tool call received in stream")
		return
	}
	fmt.Println("Tool call streamed")

	// Output: Tool call streamed
}

func ExampleZaiGenerator_disableThinking() {
	apiKey := os.Getenv("Z_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set Z_API_KEY env]")
		return
	}

	// Create generator with thinking disabled
	gen := NewZaiGenerator(
		nil, "glm-4.7",
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
		fmt.Println("Error:", err)
		return
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
		fmt.Println("Error: empty response")
		return
	}

	// Verify no thinking blocks exist
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == Thinking {
			fmt.Println("Error: thinking block found when thinking is disabled")
			return
		}
	}
	fmt.Println("No thinking blocks")

	// Verify we got a content block with the answer
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == Content {
			if strings.Contains(block.Content.String(), "4") {
				fmt.Println("Direct answer received")
				return
			}
			fmt.Printf("Error: expected '4' in answer, got: %s\n", block.Content.String())
			return
		}
	}
	fmt.Println("Error: no content block found")

	// Output: No thinking blocks
	// Direct answer received
}
