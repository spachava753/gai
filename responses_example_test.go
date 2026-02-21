package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"

	"github.com/google/jsonschema-go/jsonschema"
)

func ExampleResponsesGenerator_Generate_pdf() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}

	pdfBytes, err := os.ReadFile("sample.pdf")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.pdf]")
		return
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, "You are a helpful assistant.")
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("What is the title of this PDF? Just output the title and nothing else"),
				PDFBlock(pdfBytes, "sample.pdf"),
			},
		},
	}
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ThinkingBudget: "low"})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	// Find the first Content block (skip Thinking blocks from reasoning)
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			fmt.Println(blk.Content)
			break
		}
	}
	// Output: Attention Is All You Need
}

func ExampleResponsesGenerator_Generate_image() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.jpg]")
		return
	}
	imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))

	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, "You are a helpful assistant.")
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
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: Ptr(512), ThinkingBudget: "high"})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(resp.Candidates) != 1 {
		panic("Expected 1 candidate, got " + fmt.Sprint(len(resp.Candidates)))
	}
	if len(resp.Candidates[0].Blocks) == 0 {
		panic("Expected at least 1 block")
	}
	// Find the first Content block (skip Thinking blocks from reasoning)
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			fmt.Println(strings.Contains(blk.Content.String(), "Guy"))
			break
		}
	}
	// Output: true
}

func ExampleResponsesGenerator_Generate() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, "You are a helpful assistant")
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Hi!")}}}
	resp, err := gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println("Response received")
	fmt.Println(len(resp.Candidates))
	// Output: Response received
	// 1
}

func ExampleResponsesGenerator_Generate_thinking() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5, "You are a helpful assistant")
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Are LLMs conscious? Think it through and give a comprehensive answer")}}}
	opts := GenOpts{ThinkingBudget: "medium", Temperature: Ptr(1.0), ExtraArgs: map[string]any{
		ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
	}}
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println("Response received")
	// The generator is stateless: just append the assistant response and continue.
	// Reasoning blocks with encrypted content are automatically reconstructed as
	// input reasoning items on the next call.
	dialog = append(dialog, resp.Candidates[0], Message{Role: User, Blocks: []Block{TextBlock("What can you do?")}})
	resp, err = gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(len(resp.Candidates))
	// Output: Response received
	// 1
}

func ExampleResponsesGenerator_Register() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, `You are a helpful assistant that returns the price of a stock and nothing else.

Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`)
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
		panic(err.Error())
	}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}}}
	opts := GenOpts{ToolChoice: "get_stock_price"}
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	// Find the first ToolCall block (reasoning models may produce Thinking blocks before tool calls)
	var toolCallBlock Block
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == ToolCall {
			toolCallBlock = blk
			break
		}
	}
	fmt.Println(toolCallBlock.Content)
	// Append the assistant's response and the tool result. The generator is stateless
	// and manages conversation context through the dialog.
	dialog = append(dialog, resp.Candidates[0], Message{Role: ToolResult, Blocks: []Block{{ID: toolCallBlock.ID, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")}}})
	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		panic(err.Error())
	}
	// Find the first Content block in the final response
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			fmt.Println(blk.Content)
			break
		}
	}
	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// 123.45
}

func ExampleResponsesGenerator_Register_parallelToolUse() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}
	client := openai.NewClient(option.WithAPIKey(apiKey))
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, `You are a helpful assistant that compares the price of two stocks and returns the ticker of whichever is greater.
Only mentioned the ticker and nothing else.

Only output the price, like
<example>
User: Which one is more expensive? Apple or NVidia?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: Apple: 123.45; Nvidia: 345.65
Assistant: Nvidia
</example>
`)
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.\nYou can call this tool in parallel",
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
		panic(err.Error())
	}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Which stock, Apple vs. Microsoft, is more expensive?")}}}
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ThinkingBudget: "medium"})
	if err != nil {
		panic(err.Error())
	}
	// Collect ToolCall blocks (reasoning models may produce Thinking blocks before tool calls)
	var toolCallBlocks []Block
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == ToolCall {
			toolCallBlocks = append(toolCallBlocks, blk)
		}
	}
	fmt.Println(toolCallBlocks[0].Content)
	fmt.Println(toolCallBlocks[1].Content)
	// Append the assistant's response and tool results. The generator is stateless
	// and manages conversation context through the dialog.
	dialog = append(dialog, resp.Candidates[0], Message{Role: ToolResult, Blocks: []Block{{ID: toolCallBlocks[0].ID, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")}}}, Message{Role: ToolResult, Blocks: []Block{{ID: toolCallBlocks[1].ID, ModalityType: Text, MimeType: "text/plain", Content: Str("678.45")}}})
	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		panic(err.Error())
	}
	// Find the first Content block in the final response
	for _, blk := range resp.Candidates[0].Blocks {
		if blk.BlockType == Content {
			fmt.Println(blk.Content)
			break
		}
	}
	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// {"name":"get_stock_price","parameters":{"ticker":"MSFT"}}
	// MSFT
}

func ExampleResponsesGenerator_Stream_parallelToolUse() {
	// Create an OpenAI client
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	// Register tools
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

	// Instantiate a Responses Generator
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Mini, "You are a helpful assistant")

	// Register tools
	tickerTool.Description += "\nYou can call this tool in parallel"
	if err := gen.Register(tickerTool); err != nil {
		panic(err.Error())
	}

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

	// Stream a response
	var blocks []Block
	for chunk, err := range gen.Stream(context.Background(), dialog, nil) {
		if err != nil {
			fmt.Println(err.Error())
			return
		}
		blocks = append(blocks, chunk.Block)
	}

	if len(blocks) > 1 {
		fmt.Println("Response received")
	}

	// collect the blocks
	var prevToolCallId string
	var toolCalls []Block
	var toolcallArgs string
	var toolCallInput ToolCallInput
	for _, block := range blocks {
		// Skip metadata blocks
		if block.BlockType == MetadataBlockType {
			continue
		}
		if block.ID != "" && block.ID != prevToolCallId {
			if toolcallArgs != "" {
				// Parse the arguments string into a map
				if err := json.Unmarshal([]byte(toolcallArgs), &toolCallInput.Parameters); err != nil {
					panic(err.Error())
				}

				// Marshal back to JSON for consistent representation
				toolUseJSON, err := json.Marshal(toolCallInput)
				if err != nil {
					panic(err.Error())
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

	if toolcallArgs != "" {
		// Parse the arguments string into a map
		if err := json.Unmarshal([]byte(toolcallArgs), &toolCallInput.Parameters); err != nil {
			panic(err.Error())
		}

		// Marshal back to JSON for consistent representation
		toolUseJSON, err := json.Marshal(toolCallInput)
		if err != nil {
			panic(err.Error())
		}
		toolCalls[len(toolCalls)-1].Content = Str(toolUseJSON)
		toolCallInput = ToolCallInput{}
	}

	fmt.Println(len(toolCalls))

	dialog = append(dialog, Message{
		Role:   Assistant,
		Blocks: toolCalls,
	}, Message{
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
	for chunk, err := range gen.Stream(context.Background(), dialog, nil) {
		if err != nil {
			fmt.Println(err.Error())
			return
		}
		blocks = append(blocks, chunk.Block)
	}

	if len(blocks) > 1 {
		fmt.Println("Response received")
	}

	// Check if metadata block is present (emitted on response.completed)
	for _, blk := range blocks {
		if blk.BlockType == MetadataBlockType {
			fmt.Println("Received response_completed")
			break
		}
	}

	// Output: Response received
	// 2
	// Response received
	// Received response_completed
}

func ExampleResponsesGenerator_Stream_thinking() {
	// Create an OpenAI client
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set OPENAI_API_KEY env]")
		return
	}
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	// Instantiate a Responses Generator
	gen := NewResponsesGenerator(&client.Responses, openai.ChatModelGPT5Nano, "You are a helpful assistant")

	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Do LLMs have a soul, according to any definition of soul given by famous historical philosophers?"),
				},
			},
		},
	}

	// Stream a response
	var blocks []Block
	for chunk, err := range gen.Stream(context.Background(), dialog, &GenOpts{
		ThinkingBudget: "medium",
		ExtraArgs: map[string]any{
			ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
		},
	}) {
		if err != nil {
			fmt.Println(err.Error())
			return
		}
		blocks = append(blocks, chunk.Block)
	}

	for _, block := range blocks {
		if block.BlockType == Thinking {
			fmt.Println("Has thinking blocks")
			break
		}
	}

	if len(blocks) > 1 {
		fmt.Println("Response received")
	}

	// Check if metadata block is present (emitted on response.completed)
	if blocks[len(blocks)-1].BlockType == MetadataBlockType {
		fmt.Println("Received response_completed")
	}

	// Output: Has thinking blocks
	// Response received
	// Received response_completed
}
