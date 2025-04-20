package gai

import (
	"context"
	"encoding/base64"
	"fmt"
	"github.com/google/generative-ai-go/genai"
	genaiopts "google.golang.org/api/option"
	"os"
	"strings"
	"time"
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
		genaiopts.WithAPIKey(apiKey),
	)

	g, err := NewGeminiGenerator(client, "gemini-1.5-flash", "You are a helpful assistant.")
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

func ExampleGeminiGenerator_Register() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		genaiopts.WithAPIKey(apiKey),
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-pro-preview-03-25", "You are a helpful assistant.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}
	stockTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: InputSchema{
			Type: Object,
			Properties: map[string]Property{
				"ticker": {Type: String, Description: "Stock ticker symbol (e.g. AAPL)"},
			},
			Required: []string{"ticker"},
		},
	}
	getServerTimeTool := Tool{
		Name:        "get_server_time",
		Description: "Get the current server time in UTC.",
	}
	_ = g.Register(stockTool)
	_ = g.Register(getServerTimeTool)
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

	dialog = append(dialog, Message{
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
		"MSFT is $300.00",
	) && strings.Contains(
		response.Candidates[0].Blocks[0].Content.String(),
		time.Time{}.Add(1*time.Minute).String(),
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
	// This example assumes that Guycrood.jpg is present in the current directory.
	// Place a JPEG image named Guycrood.jpg in the same directory as this file (or adjust the path).
	imgBytes, err := os.ReadFile("Guycrood.jpg")
	if err != nil {
		fmt.Println("[Skipped: could not open Guycrood.jpg]")
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
		genaiopts.WithAPIKey(apiKey),
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-pro-preview-03-25", "You are a helpful assistant.")
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

func ExampleGeminiGenerator_Register_parallelToolUse() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("[Skipped: set GEMINI_API_KEY env]")
		return
	}

	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		genaiopts.WithAPIKey(apiKey),
	)

	g, err := NewGeminiGenerator(client, "gemini-2.5-pro-preview-03-25", "You are a helpful assistant.")
	if err != nil {
		fmt.Println("Error creating GeminiGenerator:", err)
		return
	}

	// Register the get_stock_price tool
	stockTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: InputSchema{
			Type: Object,
			Properties: map[string]Property{
				"ticker": {Type: String, Description: "Stock ticker symbol (e.g. AAPL)"},
			},
			Required: []string{"ticker"},
		},
	}
	_ = g.Register(stockTool)
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
