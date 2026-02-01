package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"

	"github.com/openai/openai-go/v3"

	"github.com/google/jsonschema-go/jsonschema"
)

// ExampleStreamingAdapter demonstrates how to use StreamingAdapter to convert
// a StreamingGenerator to a regular Generator. This is useful when you want to
// use streaming internally but present a non-streaming interface to users.
func ExampleStreamingAdapter() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Create a generator with streaming support
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are a helpful assistant.",
	)

	// Wrap the generator with StreamingAdapter
	// This converts the streaming interface to a regular Generate interface
	adapter := StreamingAdapter{S: &gen}

	// Create a simple dialog
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("What is the capital of France?"),
			},
		},
	}

	// Use the adapter's Generate method - it will internally stream
	// and compress the chunks into a complete response
	response, err := adapter.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Print the response
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println("Assistant:", response.Candidates[0].Blocks[0].Content)
	}

	// Output:
	// Assistant: The capital of France is Paris.
}

// ExampleStreamingAdapter_withTools demonstrates using StreamingAdapter with tool calls.
// The adapter handles the compression of streaming tool call chunks into complete tool calls.
func ExampleStreamingAdapter_withTools() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Create a generator with streaming support
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are a helpful weather assistant.",
	)

	// Register a weather tool
	weatherTool := Tool{
		Name:        "get_weather",
		Description: "Get the current weather in a given location",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Location string `json:"location" jsonschema:"The city and state, e.g. San Francisco, CA"`
				Unit     string `json:"unit,omitempty" jsonschema:"The unit of temperature"`
			}]()
			if err != nil {
				panic(err)
			}
			return schema
		}(),
	}

	if err := gen.Register(weatherTool); err != nil {
		fmt.Printf("Error registering tool: %v\n", err)
		return
	}

	// Wrap with StreamingAdapter
	adapter := StreamingAdapter{S: &gen}

	// Create a dialog asking about weather
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("What's the weather like in New York?"),
			},
		},
	}

	// Generate with tool use enabled
	response, err := adapter.Generate(context.Background(), dialog, &GenOpts{
		ToolChoice: ToolChoiceAuto,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// The response should contain a tool call
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		block := response.Candidates[0].Blocks[0]
		if block.BlockType == ToolCall {
			// Parse the tool call
			var toolCall ToolCallInput
			if err := json.Unmarshal([]byte(block.Content.String()), &toolCall); err == nil {
				fmt.Printf("Tool called: %s\n", toolCall.Name)
				fmt.Printf("Location: %v\n", toolCall.Parameters["location"])
			}
		}
	}

	// Output:
	// Tool called: get_weather
	// Location: New York, NY
}

// ExampleStreamingAdapter_errorHandling demonstrates how StreamingAdapter handles
// errors that occur during streaming.
func ExampleStreamingAdapter_errorHandling() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Create a generator
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are a helpful assistant.",
	)

	// Wrap with StreamingAdapter
	adapter := StreamingAdapter{S: &gen}

	// Create an empty dialog (which should cause an error)
	dialog := Dialog{}

	// Try to generate - this should return an error
	_, err := adapter.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Printf("Got expected error: %v\n", err)
	}

	// Output:
	// Got expected error: empty dialog: at least one message required
}

// ExampleStreamingAdapter_multipleBlocks demonstrates how StreamingAdapter handles
// responses with multiple blocks of different types, showing the compression of
// consecutive blocks of the same type.
func ExampleStreamingAdapter_multipleBlocks() {
	// This example demonstrates the internal behavior of StreamingAdapter
	// by showing how it would handle a mock streaming generator

	// Create a mock streaming generator that yields multiple chunks
	mockGen := &mockStreamingGenerator{
		chunks: []StreamChunk{
			// First content chunk
			{
				Block: Block{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str("The weather in "),
				},
				CandidatesIndex: 0,
			},
			// Second content chunk (will be concatenated)
			{
				Block: Block{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str("Paris is "),
				},
				CandidatesIndex: 0,
			},
			// Third content chunk (will be concatenated)
			{
				Block: Block{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str("sunny today."),
				},
				CandidatesIndex: 0,
			},
		},
	}

	// Wrap with StreamingAdapter
	adapter := StreamingAdapter{S: mockGen}

	// Generate
	response, err := adapter.Generate(context.Background(), nil, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// The adapter should have compressed the three chunks into one block
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) == 1 {
		fmt.Println("Compressed content:", response.Candidates[0].Blocks[0].Content)
		fmt.Println("Finish reason:", response.FinishReason == EndTurn)
	}

	// Output:
	// Compressed content: The weather in Paris is sunny today.
	// Finish reason: true
}

// ExampleStreamingAdapter_parallelToolCalls demonstrates how StreamingAdapter handles
// parallel tool calls, showing the compression of multiple tool call chunks.
func ExampleStreamingAdapter_parallelToolCalls() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Create a generator
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are a helpful stock price assistant.",
	)

	// Register a stock price tool
	stockTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol",
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

	if err := gen.Register(stockTool); err != nil {
		fmt.Printf("Error registering tool: %v\n", err)
		return
	}

	// Wrap with StreamingAdapter
	adapter := StreamingAdapter{S: &gen}

	// Ask about multiple stocks
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("What are the current prices of Apple and Microsoft stocks?"),
			},
		},
	}

	// Generate with tool use
	response, err := adapter.Generate(context.Background(), dialog, &GenOpts{
		ToolChoice: ToolChoiceAuto,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Count the number of tool calls
	toolCallCount := 0
	for _, block := range response.Candidates[0].Blocks {
		if block.BlockType == ToolCall {
			toolCallCount++
			var toolCall ToolCallInput
			if err := json.Unmarshal([]byte(block.Content.String()), &toolCall); err == nil {
				fmt.Printf("Tool call %d: %s with ticker=%v\n",
					toolCallCount, toolCall.Name, toolCall.Parameters["ticker"])
			}
		}
	}

	fmt.Printf("Finish reason: %v\n", response.FinishReason == ToolUse)

	// Output:
	// Tool call 1: get_stock_price with ticker=AAPL
	// Tool call 2: get_stock_price with ticker=MSFT
	// Finish reason: true
}

// ExampleStreamingAdapter_customUsage shows how to create a custom generator
// that implements StreamingGenerator and use it with StreamingAdapter.
func ExampleStreamingAdapter_customUsage() {
	// This example shows how someone might implement their own StreamingGenerator
	// and use it with StreamingAdapter

	// Create a custom implementation
	customGen := &customStreamingGenerator{
		systemPrompt: "You are a helpful assistant.",
	}

	// Wrap with StreamingAdapter to get a regular Generator interface
	adapter := StreamingAdapter{S: customGen}

	// Now you can use it as a regular generator
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("Hello!"),
			},
		},
	}

	response, err := adapter.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		// Remove the "Mock response: " prefix for consistent output
		content := response.Candidates[0].Blocks[0].Content.String()
		content = strings.TrimPrefix(content, "Mock response: ")
		fmt.Println("Response:", content)
	}

	// Output:
	// Response: Hello! How can I help you today?
}

// customStreamingGenerator is an example of a custom StreamingGenerator implementation
type customStreamingGenerator struct {
	systemPrompt string
}

func (c *customStreamingGenerator) Stream(ctx context.Context, dialog Dialog, options *GenOpts) iter.Seq2[StreamChunk, error] {
	return func(yield func(StreamChunk, error) bool) {
		// Validate input
		if len(dialog) == 0 {
			yield(StreamChunk{}, EmptyDialogErr)
			return
		}

		// Simulate streaming response chunks
		responseChunks := []string{"Mock response: ", "Hello! ", "How can I ", "help you ", "today?"}

		for _, chunk := range responseChunks {
			if !yield(StreamChunk{
				Block: Block{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(chunk),
				},
				CandidatesIndex: 0,
			}, nil) {
				return // User stopped iteration
			}
		}
	}
}

// ExampleStreamingAdapter_withToolGenerator demonstrates using StreamingAdapter
// together with ToolGenerator to create a complete tool-using assistant that
// internally uses streaming but presents a non-streaming interface.
func ExampleStreamingAdapter_withToolGenerator() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Create a generator with streaming support
	baseGen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		"You are a helpful assistant that can check weather and time.",
	)

	// Create a ToolGenerator
	toolGen := &ToolGenerator{
		G: &baseGen,
	}

	// Register weather tool with callback
	weatherTool := Tool{
		Name:        "get_weather",
		Description: "Get the current weather in a location",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Location string `json:"location" jsonschema:"required" jsonschema_description:"The city and state, e.g. San Francisco, CA"`
			}]()
			if err != nil {
				panic(err)
			}
			return schema
		}(),
	}

	// Simple weather callback
	weatherCallback := ToolCallBackFunc[struct {
		Location string `json:"location"`
	}](func(ctx context.Context, params struct{ Location string }) (string, error) {
		// Mock weather data
		return fmt.Sprintf("The weather in %s is sunny and 72°F", params.Location), nil
	})

	if err := toolGen.Register(weatherTool, weatherCallback); err != nil {
		fmt.Printf("Error registering tool: %v\n", err)
		return
	}

	// Create a dialog
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("What's the weather in San Francisco?"),
			},
		},
	}

	// Use ToolGenerator's Generate method which will handle tool calls
	// The underlying OpenAI generator uses streaming internally
	completeDialog, err := toolGen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Print the final response (skipping tool calls and results)
	foundWeatherResponse := false
	for _, msg := range completeDialog {
		if msg.Role == Assistant && len(msg.Blocks) > 0 {
			block := msg.Blocks[0]
			if block.BlockType == Content {
				content := block.Content.String()
				// Check if the response mentions weather in San Francisco
				if strings.Contains(content, "San Francisco") &&
					strings.Contains(content, "sunny") &&
					strings.Contains(content, "72°F") {
					foundWeatherResponse = true
					fmt.Println("Found weather response for San Francisco")
				}
			}
		}
	}

	if !foundWeatherResponse {
		fmt.Println("No weather response found")
	}

	// Output:
	// Found weather response for San Francisco
}
