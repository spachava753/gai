package gai

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"strings"
	"time"

	"google.golang.org/genai"
)

func ExampleGeminiGenerator_Generate() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-flash", "You are a helpful assistant. You respond to the user with plain text format.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}
	dialog := Dialog{
		{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("What is the capital of France?")}}},
	}
	response, err := g.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println(response.Candidates[0].Blocks[0].Content)
	}
	// Output: The capital of France is Paris.
}

func ExampleGeminiGenerator_Stream() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-flash", "You are a helpful assistant. You respond to the user with plain text format.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}
	dialog := Dialog{
		{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("What is the capital of France?")}}},
	}
	for chunk, err := range g.Stream(context.Background(), dialog, nil) {
		if err != nil {
			fmt.Println("Error:", err)
			return
		}
		// Skip metadata blocks
		if chunk.Block.BlockType == MetadataBlockType {
			continue
		}
		fmt.Println(chunk.Block.Content.String())
	}
	// Output: The capital of France is Paris.
}

func ExampleGeminiGenerator_Register() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)

	g, err := NewGeminiGenerator(
		client,
		"gemini-2.5-pro",
		`You are a helpful assistant. You can call tools in parallel. 
When a user asks for the server time, always call the server time tool, don't use previously returned results`,
	)
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}
	stockTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
	}
	getServerTimeTool := Tool{
		Name:        "get_server_time",
		Description: "Get the current server time in UTC.",
	}
	err = g.Register(stockTool)
	if err != nil {
		fmt.Println("Error registering tool:", err)
		return
	}
	err = g.Register(getServerTimeTool)
	if err != nil {
		fmt.Println("Error registering tool:", err)
		return
	}
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("What is the stock price for AAPL, and also tell me the server time?"),
			}},
		},
	}

	// Expect tool call for both tools
	response, err := g.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("tool calling response:")
	for _, block := range response.Candidates[0].Blocks {
		fmt.Printf("Block type: %s | ID: %s | Content: %s\n", block.BlockType, block.ID, block.Content)
	}

	dialog = append(dialog, response.Candidates[0])

	// Simulate tool result for tool calls
	dialog = append(dialog,
		Message{
			Role: ToolResult,
			Blocks: []Block{{
				ID:           response.Candidates[0].Blocks[0].ID,
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str("AAPL is $200.00"),
			}},
		},
		Message{
			Role: ToolResult,
			Blocks: []Block{{
				ID:           response.Candidates[0].Blocks[1].ID,
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str(time.Time{}.String()),
			}},
		},
	)

	response, err = g.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("Response has tool results:", strings.Contains(
		response.Candidates[0].Blocks[0].Content.String(),
		"AAPL is $200.00",
	) && strings.Contains(
		response.Candidates[0].Blocks[0].Content.String(),
		time.Time{}.String(),
	))

	dialog = append(dialog, response.Candidates[0], Message{
		Role: User,
		Blocks: []Block{{
			BlockType:    Content,
			ModalityType: Text,
			Content:      Str("What is the stock price for MSFT, and also tell me the server time again?"),
		}},
	})

	response, err = g.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("tool calling response:")
	for _, block := range response.Candidates[0].Blocks {
		fmt.Printf("Block type: %s | ID: %s | Content: %s\n", block.BlockType, block.ID, block.Content)
	}

	dialog = append(dialog, response.Candidates[0])

	// Simulate tool result for tool calls
	dialog = append(dialog,
		Message{
			Role: ToolResult,
			Blocks: []Block{{
				ID:           response.Candidates[0].Blocks[0].ID,
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str("MSFT is $300.00"),
			}},
		},
		Message{
			Role: ToolResult,
			Blocks: []Block{{
				ID:           response.Candidates[0].Blocks[1].ID,
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str(time.Time{}.Add(1 * time.Minute).String()),
			}},
		},
	)

	response, err = g.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("Response has tool results:", strings.Contains(
		response.Candidates[0].Blocks[0].Content.String(),
		"MSFT",
	) && strings.Contains(
		response.Candidates[0].Blocks[0].Content.String(),
		"300",
	) && strings.Contains(
		response.Candidates[0].Blocks[0].Content.String(),
		"UTC",
	))

	// Output: tool calling response:
	// Block type: tool_call | ID: toolcall-1 | Content: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// Block type: tool_call | ID: toolcall-2 | Content: {"name":"get_server_time","parameters":{}}
	// Response has tool results: true
	// tool calling response:
	// Block type: tool_call | ID: toolcall-3 | Content: {"name":"get_stock_price","parameters":{"ticker":"MSFT"}}
	// Block type: tool_call | ID: toolcall-4 | Content: {"name":"get_server_time","parameters":{}}
	// Response has tool results: true
}

func ExampleGeminiGenerator_Generate_image() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	// ---
	// This example assumes that sample.jpg is present in the current directory.
	// Place a JPEG image named sample.jpg in the same directory as this file (or adjust the path).
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.jpg]")
		return
	}
	// Encode as base64 for API usage
	imgBase64 := Str(
		// Use standard encoding, as required for image MIME input.
		// NOTE: the Blob part in Google Gemini Go SDK accepts raw bytes, but our gai.Block expects base64 encoded string.
		// The actual Gemini implementation will decode as needed, see gai.go.
		// This mirrors how other examples do it.
		base64.StdEncoding.EncodeToString(imgBytes),
	)

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-pro", "You are a helpful assistant.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}
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

	response, err := g.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(response.Candidates) != 1 {
		panic("Expected 1 candidate, got " + fmt.Sprint(len(response.Candidates)))
	}
	if len(response.Candidates[0].Blocks) != 1 {
		panic("Expected 1 block, got " + fmt.Sprint(len(response.Candidates[0].Blocks)))
	}
	fmt.Println(strings.Contains(response.Candidates[0].Blocks[0].Content.String(), "Crood"))

	// Output: true
}

func ExampleGeminiGenerator_Generate_audio() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	audioBytes, err := os.ReadFile("sample.wav")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.wav]")
		return
	}
	// Encode as base64 for inline audio usage
	audioBase64 := Str(base64.StdEncoding.EncodeToString(audioBytes))

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-pro", "You are a helpful assistant.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}

	// Using inline audio data
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Audio,
					MimeType:     "audio/wav",
					Content:      audioBase64,
				},
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is the name of person in the greeting in this audio? Return a one work response of the name"),
				},
			},
		},
	}

	response, err := g.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println(strings.ToLower(response.Candidates[0].Blocks[0].Content.String()))
	}

	// Output: friday
}

func ExampleGeminiGenerator_Register_parallelToolUse() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-flash", "You are a helpful assistant.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}

	// Register the get_stock_price tool
	stockTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
	}
	err = g.Register(stockTool)
	if err != nil {
		fmt.Println("Error registering tool:", err)
		return
	}

	dialog := Dialog{
		{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("Give me the current prices for AAPL, MSFT, and TSLA.")}}},
	}
	response, err := g.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	for _, cand := range response.Candidates {
		for _, block := range cand.Blocks {
			fmt.Printf("Block type: %s | ID: %s | Content: %s\n", block.BlockType, block.ID, block.Content)
		}
	}
	// Output: Block type: tool_call | ID: toolcall-1 | Content: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// Block type: tool_call | ID: toolcall-2 | Content: {"name":"get_stock_price","parameters":{"ticker":"MSFT"}}
	// Block type: tool_call | ID: toolcall-3 | Content: {"name":"get_stock_price","parameters":{"ticker":"TSLA"}}
}

func ExampleGeminiGenerator_Stream_parallelToolUse() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-flash", "You are a helpful assistant.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}

	// Register the get_stock_price tool
	stockTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
	}
	err = g.Register(stockTool)
	if err != nil {
		fmt.Println("Error registering tool:", err)
		return
	}

	dialog := Dialog{
		{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("Give me the current prices for AAPL, MSFT, and TSLA.")}}},
	}
	for chunk, err := range g.Stream(context.Background(), dialog, nil) {
		if err != nil {
			fmt.Println("Error:", err)
			return
		}
		// Skip metadata blocks
		if chunk.Block.BlockType == MetadataBlockType {
			continue
		}
		fmt.Printf("Block type: %s | ID: %s | Content: %s\n", chunk.Block.BlockType, chunk.Block.ID, chunk.Block.Content)
	}
	// Output: Block type: tool_call | ID: toolcall-1 | Content: get_stock_price
	// Block type: tool_call | ID:  | Content: {"ticker":"AAPL"}
	// Block type: tool_call | ID: toolcall-2 | Content: get_stock_price
	// Block type: tool_call | ID:  | Content: {"ticker":"MSFT"}
	// Block type: tool_call | ID: toolcall-3 | Content: get_stock_price
	// Block type: tool_call | ID:  | Content: {"ticker":"TSLA"}
}

func ExampleGeminiGenerator_Count() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		fmt.Println("Dialog contains approximately 25 tokens")
		fmt.Println("Dialog with image contains approximately 400 tokens")
		return
	}

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)

	// Create a generator
	g, err := NewGeminiGenerator(client, "gemini-2.5-pro", "You are a helpful assistant.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}

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

	// Count tokens in the dialog
	tokenCount, err := g.Count(context.Background(), dialog)
	if err != nil {
		fmt.Printf("Error counting tokens: %v\n", err)
		return
	}

	fmt.Printf("Dialog contains approximately %d tokens\n", tokenCount)

	// Try to load an image to add to the dialog
	imgPath := "sample.jpg"
	imgBytes, err := os.ReadFile(imgPath)
	if err != nil {
		fmt.Printf("Image file not found, skipping image token count example\n")
		return
	}

	// Add an image to the dialog
	dialog = Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Image,
					MimeType:     "image/jpeg",
					Content:      Str(base64.StdEncoding.EncodeToString(imgBytes)),
				},
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Describe this image."),
				},
			},
		},
	}

	// Count tokens with the image included
	tokenCount, err = g.Count(context.Background(), dialog)
	if err != nil {
		fmt.Printf("Error counting tokens: %v\n", err)
		return
	}

	fmt.Printf("Dialog with image contains approximately %d tokens\n", tokenCount)

	// Output: Dialog contains approximately 15 tokens
	// Dialog with image contains approximately 270 tokens
}

func ExampleGeminiGenerator_Generate_pdf() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-flash", "You are a helpful assistant.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}

	// This example assumes that sample.pdf is present in the current directory.
	pdfBytes, err := os.ReadFile("sample.pdf")
	if err != nil {
		fmt.Println("[Skipped: could not open sample.pdf]")
		return
	}

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
	response, err := g.Generate(ctx, dialog, &GenOpts{MaxGenerationTokens: 1024})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// The response would contain the model's analysis of the PDF
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println(response.Candidates[0].Blocks[0].Content)
	}
	// Output: Attention Is All You Need
}
