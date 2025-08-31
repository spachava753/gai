package gai

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"strings"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/responses"
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
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		fmt.Println(resp.Candidates[0].Blocks[0].Content)
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
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{MaxGenerationTokens: 512, ThinkingBudget: "high"})
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
	fmt.Println(strings.Contains(resp.Candidates[0].Blocks[0].Content.String(), "Guy"))
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
	opts := GenOpts{ThinkingBudget: "medium", Temperature: 1.0, ExtraArgs: map[string]any{
		ResponsesThoughtSummaryDetailParam: responses.ReasoningSummaryDetailed,
	}}
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println("Response received")
	dialog = append(dialog, resp.Candidates[0], Message{Role: User, Blocks: []Block{TextBlock("What can you do?")}})
	opts.ExtraArgs[ResponsesPrevRespId] = resp.UsageMetrics[ResponsesPrevRespId]
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
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
	}
	if err := gen.Register(tickerTool); err != nil {
		panic(err.Error())
	}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}}}
	opts := GenOpts{ToolChoice: "get_stock_price", ExtraArgs: map[string]any{}}
	resp, err := gen.Generate(context.Background(), dialog, &opts)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)
	dialog = append(dialog, resp.Candidates[0], Message{Role: ToolResult, Blocks: []Block{{ID: resp.Candidates[0].Blocks[0].ID, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")}}})
	resp, err = gen.Generate(context.Background(), dialog, &GenOpts{ExtraArgs: map[string]any{
		ResponsesPrevRespId: resp.UsageMetrics[ResponsesPrevRespId],
	}})
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)
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
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
	}
	if err := gen.Register(tickerTool); err != nil {
		panic(err.Error())
	}
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("Which stock, Apple vs. Microsoft, is more expensive?")}}}
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{ThinkingBudget: "medium"})
	if err != nil {
		panic(err.Error())
	}
	blocks := resp.Candidates[0].Blocks
	fmt.Println(blocks[len(blocks)-2].Content)
	fmt.Println(blocks[len(blocks)-1].Content)
	dialog = append(dialog, resp.Candidates[0], Message{Role: ToolResult, Blocks: []Block{{ID: blocks[len(blocks)-2].ID, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")}}}, Message{Role: ToolResult, Blocks: []Block{{ID: blocks[len(blocks)-1].ID, ModalityType: Text, MimeType: "text/plain", Content: Str("678.45")}}})
	resp, err = gen.Generate(context.Background(), dialog, &GenOpts{ExtraArgs: map[string]any{
		ResponsesPrevRespId: resp.UsageMetrics[ResponsesPrevRespId],
	}})
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(resp.Candidates[0].Blocks[0].Content)
	// Output: {"name":"get_stock_price","parameters":{"ticker":"AAPL"}}
	// {"name":"get_stock_price","parameters":{"ticker":"MSFT"}}
	// MSFT
}
