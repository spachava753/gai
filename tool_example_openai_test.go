package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/openai/openai-go/v3"

	"github.com/google/jsonschema-go/jsonschema"
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
					Content:      Str("Error: ticker is required"),
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

func TestToolGenerator_Generate_Example(t *testing.T) {
	skipOnMissingEnv(t, "OPENAI_API_KEY")

	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
			}]()
			if err != nil {
				t.Fatalf("generate ticker schema: %v", err)
			}
			return schema
		}(),
	}

	client := openai.NewClient()
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
	tg := ToolGenerator{G: &gen}
	if err := tg.Register(tickerTool, &TickerTool{ticketPrices: map[string]float64{"AAPL": 435.56}}); err != nil {
		t.Fatalf("register ticker tool: %v", err)
	}

	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}}}
	newDialog, err := tg.Generate(context.Background(), dialog, func(d Dialog) *GenOpts { return nil })
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if len(newDialog) != 4 {
		t.Fatalf("dialog length = %d, want 4", len(newDialog))
	}
	content := requireContentBlock(t, requireBlock(t, newDialog[len(newDialog)-1], 0))
	if got, want := content, "435.56"; got != want {
		t.Fatalf("final content = %q, want %q", got, want)
	}
}

func TestToolGenerator_Generate_responses_Example(t *testing.T) {
	skipOnMissingEnv(t, "OPENAI_API_KEY")

	tickerTool := Tool{
		Name:        "get_stock_price",
		Description: "Get the current stock price for a given ticker symbol.",
		InputSchema: func() *jsonschema.Schema {
			schema, err := GenerateSchema[struct {
				Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
			}]()
			if err != nil {
				t.Fatalf("generate ticker schema: %v", err)
			}
			return schema
		}(),
	}

	client := openai.NewClient()
	gen := NewResponsesGenerator(
		&client.Responses,
		openai.ChatModelGPT5Mini,
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
	tg := ToolGenerator{G: &gen}
	if err := tg.Register(tickerTool, &TickerTool{ticketPrices: map[string]float64{"AAPL": 435.56}}); err != nil {
		t.Fatalf("register ticker tool: %v", err)
	}

	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}}}
	newDialog, err := tg.Generate(context.Background(), dialog, func(d Dialog) *GenOpts { return nil })
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if len(newDialog) != 4 {
		t.Fatalf("dialog length = %d, want 4", len(newDialog))
	}

	lastMsg := newDialog[len(newDialog)-1]
	for _, blk := range lastMsg.Blocks {
		if blk.BlockType == Content {
			if got, want := blk.Content.String(), "435.56"; got != want {
				t.Fatalf("final content = %q, want %q", got, want)
			}
			return
		}
	}
	t.Fatalf("final response has no content block: %#v", lastMsg.Blocks)
}
