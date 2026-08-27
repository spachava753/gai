package gai

import (
	"context"
	"encoding/base64"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"
	"os"
	"strings"
	"testing"
	"time"
)

func TestGoogleAdapterScenarios(t *testing.T) {
	t.Run("GeminiAPIErrorMapping", testGeminiAPIErrorMapping)
	t.Run("GeminiGenerator/Count", testGeminiGenerator_Count)
	t.Run("GeminiGenerator/Generate", testGeminiGenerator_Generate)
	t.Run("GeminiGenerator/Generate/audio", testGeminiGenerator_Generate_audio)
	t.Run("GeminiGenerator/Generate/image", testGeminiGenerator_Generate_image)
	t.Run("GeminiGenerator/Generate/pdf", testGeminiGenerator_Generate_pdf)
	t.Run("GeminiGenerator/RequestTools", testGeminiGenerator_RequestTools)
	t.Run("GeminiGenerator/RequestTools/parallelToolUse", testGeminiGenerator_RequestTools_parallelToolUse)
	t.Run("GeminiGenerator/RequestTools/parallelToolUse/multimedia", testGeminiGenerator_RequestTools_parallelToolUse_multimedia)
	t.Run("GeminiGenerator/Stream", testGeminiGenerator_Stream)
	t.Run("GeminiGenerator/Stream/parallelToolUse", testGeminiGenerator_Stream_parallelToolUse)
	t.Run("GeminiResponseError", testGeminiResponseError)
}

func testGeminiGenerator_Generate(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)
	g := NewGeminiGenerator(client)
	dialog := Dialog{
		{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("What is the blooms taxonomy, and how does it related to the psychology of child development?")}}},
	}
	response, err := g.Generate(context.Background(), GenerationRequest{
		Model:        "models/gemini-3-pro-preview",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant. You respond to the user with plain text format.")),
		Dialog:       dialog,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
	}
}
func testGeminiGenerator_Stream(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)
	if err != nil {
		t.Fatalf("create Gemini client: %v", err)
	}
	g := NewGeminiGenerator(client)
	dialog := Dialog{
		{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("What is the capital of France?")}}},
	}
	var content strings.Builder
	for chunk := range g.Stream(context.Background(), GenerationRequest{
		Model:        "models/gemini-3-pro-preview",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant. You respond to the user with plain text format.")),
		Dialog:       dialog,
	}) {
		if chunk.Err != nil {
			t.Fatalf("stream returned error: %v", chunk.Err)
		}
		if chunk.Block.BlockType == MetadataBlockType {
			continue
		}
		content.WriteString(chunk.Block.Content.String())
	}
	requireTextContains(t, content.String(), "Paris")
}
func testGeminiGenerator_RequestTools(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)
	g := NewGeminiGenerator(client)
	instructions := `You are a helpful assistant. You can call tools in parallel.
When a user asks for the server time, always call the server time tool, don't use previously returned results`
	stockTool := Tool{
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
	getServerTimeTool := Tool{
		Name:        "get_server_time",
		Description: "Get the current server time in UTC.",
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
	request := GenerationRequest{
		Model:        "models/gemini-3-pro-preview",
		Instructions: SystemMessage(TextBlock(instructions)),
		Dialog:       dialog,
		Tools:        []Tool{stockTool, getServerTimeTool},
	}
	// Expect tool call for both tools
	response, err := g.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	calls := collectToolCalls(t, requireCandidate(t, response).Blocks)
	if len(calls) < 2 {
		t.Fatalf("tool calls = %d, want at least 2", len(calls))
	}
	requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "AAPL")
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
	request.Dialog = dialog
	response, err = g.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	toolResult := requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0))
	requireTextContains(t, toolResult, "AAPL", "200.00", time.Time{}.String())
	dialog = append(dialog, response.Candidates[0], Message{
		Role: User,
		Blocks: []Block{{
			BlockType:    Content,
			ModalityType: Text,
			Content:      Str("What is the stock price for MSFT, and also tell me the server time again?"),
		}},
	})
	request.Dialog = dialog
	response, err = g.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	calls = collectToolCalls(t, requireCandidate(t, response).Blocks)
	if len(calls) < 2 {
		t.Fatalf("tool calls = %d, want at least 2", len(calls))
	}
	requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "MSFT")
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
	request.Dialog = dialog
	response, err = g.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	requireTextContains(t, requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0)), "MSFT", "300", "UTC")
}
func testGeminiGenerator_Generate_image(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	// ---
	// This example assumes that sample.jpg is present in the current directory.
	// Place a JPEG image named sample.jpg in the same directory as this file (or adjust the path).
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		t.Skip("could not open sample.jpg")
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
	g := NewGeminiGenerator(client)
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
	response, err := g.Generate(context.Background(), GenerationRequest{
		Model:        "gemini-2.5-pro",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(response.Candidates) != 1 {
		t.Fatalf("candidates = %d, want 1", len(response.Candidates))
	}
	if len(response.Candidates[0].Blocks) != 1 {
		t.Fatalf("blocks = %d, want 1", len(response.Candidates[0].Blocks))
	}
	if !strings.Contains(response.Candidates[0].Blocks[0].Content.String(), "Crood") {
		t.Fatalf("content does not contain Crood")
	}
}
func testGeminiGenerator_Generate_audio(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	audioBytes, err := os.ReadFile("sample.wav")
	if err != nil {
		t.Skip("could not open sample.wav")
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
	g := NewGeminiGenerator(client)
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
	response, err := g.Generate(context.Background(), GenerationRequest{
		Model:        "gemini-2.5-pro",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		if got := strings.ToLower(response.Candidates[0].Blocks[0].Content.String()); !strings.Contains(got, "friday") {
			t.Fatalf("content = %q, want it to contain friday", got)
		}
	}
}
func testGeminiGenerator_RequestTools_parallelToolUse(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)
	g := NewGeminiGenerator(client)
	// Define the request tool
	stockTool := Tool{
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
		{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("Give me the current prices for AAPL, MSFT, and TSLA.")}}},
	}
	response, err := g.Generate(context.Background(), GenerationRequest{
		Model:        "models/gemini-3-pro-preview",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
		Tools:        []Tool{stockTool},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	calls := collectToolCalls(t, requireCandidate(t, response).Blocks)
	if len(calls) < 3 {
		t.Fatalf("tool calls = %d, want at least 3", len(calls))
	}
	requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "AAPL")
	requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "MSFT")
	requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "TSLA")
}
func testGeminiGenerator_Stream_parallelToolUse(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)
	if err != nil {
		t.Fatalf("create Gemini client: %v", err)
	}
	g := NewGeminiGenerator(client)
	// Define the request tool
	stockTool := Tool{
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
		{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("Give me the current prices for AAPL, MSFT, and TSLA.")}}},
	}
	var toolCallCount int
	for chunk := range g.Stream(context.Background(), GenerationRequest{
		Model:        "models/gemini-3-pro-preview",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
		Tools:        []Tool{stockTool},
	}) {
		if chunk.Err != nil {
			t.Fatalf("stream returned error: %v", chunk.Err)
		}
		if chunk.Block.BlockType == MetadataBlockType {
			continue
		}
		if chunk.Block.BlockType == ToolCall {
			toolCallCount++
		}
	}
	if toolCallCount == 0 {
		t.Fatal("expected at least one streamed tool call")
	}
}
func testGeminiGenerator_Count(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)
	// Create a generator
	g := NewGeminiGenerator(client)
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
		Model:        "gemini-2.5-pro",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
	}
	// Count tokens in the dialog
	tokenCount, err := g.Count(context.Background(), request)
	if err != nil {
		t.Fatalf("count tokens: %v", err)
	}
	if tokenCount == 0 {
		t.Fatal("expected non-zero token count")
	}
	// Try to load an image to add to the dialog
	imgPath := "sample.jpg"
	imgBytes, err := os.ReadFile(imgPath)
	if err != nil {
		t.Skip("could not open sample.jpg")
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
	request.Dialog = dialog
	// Count tokens with the image included
	tokenCount, err = g.Count(context.Background(), request)
	if err != nil {
		t.Fatalf("count tokens: %v", err)
	}
	if tokenCount == 0 {
		t.Fatal("expected non-zero token count")
	}
}
func testGeminiGenerator_Generate_pdf(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)
	g := NewGeminiGenerator(client)
	// This example assumes that sample.pdf is present in the current directory.
	pdfBytes, err := os.ReadFile("sample.pdf")
	if err != nil {
		t.Skip("could not open sample.pdf")
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
	response, err := g.Generate(ctx, GenerationRequest{
		Model:        "models/gemini-3-pro-preview",
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
}
func testGeminiGenerator_RequestTools_parallelToolUse_multimedia(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
	ctx := context.Background()
	client, err := genai.NewClient(
		ctx,
		&genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		},
	)
	g := NewGeminiGenerator(client)
	// Define a request tool to view files
	viewFileTool := Tool{
		Name:        "view_file",
		Description: "View the contents of a file. Can handle text files, images, and other media types.",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				FilePath string `json:"file_path" jsonschema:"required" jsonschema_description:"The path to the file to view"`
			}]()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			return schema
		}(),
	}
	// User asks to view multiple files
	dialog := Dialog{
		{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("Please view sample.jpg and README.md, and tell me what character is in the image, and what is gai from the README")}}},
	}
	request := GenerationRequest{
		Model:        "models/gemini-3-pro-preview",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant that can view files.")),
		Dialog:       dialog,
		Tools:        []Tool{viewFileTool},
	}
	// Model makes parallel tool calls
	response, err := g.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, block := range response.Candidates[0].Blocks {
		if block.BlockType == Thinking {
			continue
		}
	}
	dialog = append(dialog, response.Candidates[0])
	// Find tool call blocks (skip thinking blocks)
	toolCallBlocks := []Block{}
	for _, block := range response.Candidates[0].Blocks {
		if block.BlockType == ToolCall {
			toolCallBlocks = append(toolCallBlocks, block)
		}
	}
	if len(toolCallBlocks) < 2 {
		t.Fatal("Expected at least 2 tool calls")
		return
	}
	// Simulate tool results - first for sample.jpg (image)
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		t.Skip("could not open sample.jpg")
		return
	}
	// Simulate tool results - for README.md (text)
	readmeBytes, err := os.ReadFile("README.md")
	if err != nil {
		t.Skip("could not open README.md")
		return
	}
	// Add both tool results in parallel
	dialog = append(dialog,
		Message{
			Role: ToolResult,
			Blocks: []Block{{
				ID:           toolCallBlocks[0].ID, // First tool call
				BlockType:    Content,
				ModalityType: Image,
				MimeType:     "image/jpeg",
				Content:      Str(base64.StdEncoding.EncodeToString(imgBytes)),
			}},
		},
		Message{
			Role: ToolResult,
			Blocks: []Block{{
				ID:           toolCallBlocks[1].ID, // Second tool call
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/markdown",
				Content:      Str(string(readmeBytes)),
			}},
		},
	)
	request.Dialog = dialog
	// Get final response with tool results
	response, err = g.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	content := requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0))
	requireTextContains(t, content, "Crood", "gai")
}
