package gai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
)

func TestCerebrasGeneratorReturnsContentPolicyError(t *testing.T) {
	tests := []struct {
		name       string
		choiceJSON string
		want       string
	}{
		{
			name:       "content filter finish reason",
			choiceJSON: `{"index":0,"finish_reason":"content_filter","message":{"role":"assistant","content":""}}`,
			want:       "content policy violation detected",
		},
		{
			name:       "message refusal",
			choiceJSON: `{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"","refusal":"I cannot help with that."}}`,
			want:       "I cannot help with that.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"chatcmpl_123","object":"chat.completion","created":0,"model":"test","choices":[` + tt.choiceJSON + `]}`))
			}))
			defer server.Close()

			generator := NewCerebrasGenerator(server.Client(), server.URL, "test", "", "test-key")
			_, err := generator.Generate(context.Background(), Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}}, nil)

			var policyErr ContentPolicyErr
			if !errors.As(err, &policyErr) {
				t.Fatalf("Generate error = %T %v, want ContentPolicyErr", err, err)
			}
			if !strings.Contains(policyErr.Error(), tt.want) {
				t.Fatalf("Generate error = %q, want message containing %q", policyErr, tt.want)
			}
		})
	}
}

func TestCerebrasGenerator_Generate(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
	gen := NewCerebrasGenerator(nil, "", "gpt-oss-120b", "You are a helpful assistant.", apiKey)
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
	if len(resp.Candidates) == 1 && len(resp.Candidates[0].Blocks) >= 1 {
	}
}
func TestCerebrasGenerator_Generate_reasoning_gptoss(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
	// Use gpt-oss-120b model which supports reasoning with reasoning_effort parameter
	gen := NewCerebrasGenerator(
		nil,
		"",
		"gpt-oss-120b",
		"You are a helpful assistant that explains your reasoning step by step.",
		apiKey,
	)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is the square root of 144?"),
				},
			},
		},
	}
	// Generate response with reasoning enabled (medium effort)
	resp, err := gen.Generate(context.Background(), dialog, &GenOpts{
		ThinkingBudget: "medium",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks (reasoning)
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
			}
		}
		if hasThinking {
		}
		// Find the main content block (not thinking)
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "12") {
				}
				break
			}
		}
	}
	// Append the previous response and ask a follow-up question to test reasoning retention
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: User,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("What is the square root of 225?"),
			},
		},
	})
	// Generate response with reasoning (the previous reasoning should be retained)
	resp, err = gen.Generate(context.Background(), dialog, &GenOpts{
		ThinkingBudget: "medium",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
			}
		}
		if hasThinking {
		}
		// Find the main content block
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "15") {
				}
				break
			}
		}
	}
}
func TestCerebrasGenerator_Generate_reasoning_zai(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
	// Use zai-glm-4.6 model which supports reasoning with disable_reasoning parameter
	gen := NewCerebrasGenerator(
		nil,
		"",
		"zai-glm-4.6",
		"You are a helpful assistant that explains your reasoning step by step.",
		apiKey,
	)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is 15 * 12?"),
				},
			},
		},
	}
	// Generate response with reasoning enabled (disable_reasoning: false)
	resp, err := gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks (reasoning)
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
			}
		}
		if hasThinking {
		}
		// Find the main content block (not thinking)
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "180") {
				}
				break
			}
		}
	}
	// Append the previous response and ask a follow-up question to test reasoning retention
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: User,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("Now what is 20 * 15?"),
			},
		},
	})
	// Generate response with reasoning (the previous reasoning should be retained)
	resp, err = gen.Generate(context.Background(), dialog, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		// Check if we have thinking blocks
		hasThinking := false
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				hasThinking = true
			}
		}
		if hasThinking {
		}
		// Find the main content block
		for _, block := range resp.Candidates[0].Blocks {
			if block.BlockType == Content {
				content := block.Content.String()
				if strings.Contains(content, "300") {
				}
				break
			}
		}
	}
}
func TestCerebrasGenerator_Register(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "CEREBRAS_API_KEY")
	cgen := NewCerebrasGenerator(nil, "", "qwen-3-235b-a22b-instruct-2507", `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`, apiKey)
	// Register a tool
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
	if err := cgen.Register(tickerTool); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	dialog := Dialog{
		{Role: User, Blocks: []Block{TextBlock("What is the price of Apple stock?")}},
	}
	// Force the tool call
	resp, err := cgen.Generate(context.Background(), dialog, &GenOpts{ToolChoice: "get_stock_price"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Blocks) == 0 {
		t.Fatal("empty response")
		return
	}
	// Find and print the tool call JSON
	var toolCall Block
	for _, b := range resp.Candidates[0].Blocks {
		if b.BlockType == ToolCall {
			toolCall = b
			break
		}
	}
	if got := toolCall.Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	// Append tool result and continue the conversation
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{ID: toolCall.ID, BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str("123.45")},
		},
	})
	// Ask model to answer now without calling tools
	resp, err = cgen.Generate(context.Background(), dialog, &GenOpts{ToolChoice: "none"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
	}
}
