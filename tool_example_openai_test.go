package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/openai/openai-go/v2"
)

type TickerTool struct {
	ticketPrices map[string]float64
}

func (t TickerTool) Call(ctx context.Context, parametersJSON json.RawMessage, toolCallID string) (Message, error) {
	// Parse parameters
	var params struct {
		Ticker string `json:"ticker"`
	}
	if err := json.Unmarshal(parametersJSON, &params); err != nil {
		return Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           toolCallID,
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(fmt.Sprintf("Error: invalid input format: %v", err)),
				},
			},
		}, nil
	}

	if params.Ticker == "" {
		return Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           toolCallID,
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(fmt.Sprintf("Error: ticker is required")),
				},
			},
		}, nil
	}

	price, ok := t.ticketPrices[params.Ticker]
	if !ok {
		return Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           toolCallID,
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(fmt.Sprintf("Error: ticker %s does not exist", params.Ticker)),
				},
			},
		}, nil
	}

	return Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           toolCallID,
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str(fmt.Sprintf("%v", price)),
			},
		},
	}, nil
}

var _ ToolCallback = (*TickerTool)(nil)

func ExampleToolGenerator_Generate() {
	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: GenerateSchema[struct {
			Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}](),
	}

	client := openai.NewClient()

	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4oMini,
		`You are a helpful assistant that returns the price of a stock and nothing else.

Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`,
	)

	tg := ToolGenerator{
		G: &gen,
	}

	// Register tools
	if err := tg.Register(
		tickerTool,
		&TickerTool{
			ticketPrices: map[string]float64{
				"AAPL": 435.56,
			},
		},
	); err != nil {
		panic(err.Error())
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

	// Generate a response
	newDialog, err := tg.Generate(context.Background(), dialog, func(d Dialog) *GenOpts {
		return nil
	})
	if err != nil {
		panic(err.Error())
	}
	fmt.Printf("len of the new dialog: %d\n", len(newDialog))
	fmt.Printf("%s\n", newDialog[len(newDialog)-1].Blocks[0].Content)

	// Output: len of the new dialog: 4
	// 435.56
}
