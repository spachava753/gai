package gai

import (
	"testing"

	"context"
	"encoding/base64"
	"encoding/json"
	"github.com/google/jsonschema-go/jsonschema"
	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"os"
	"strings"
)

const openAIStockInstructions = `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`

const openAIStockComparisonInstructions = `You are a helpful assistant that compares the price of two stocks and returns the ticker of whichever is greater.
Only mentioned the ticker and nothing else.
Only output the price, like
<example>
User: Which one is more expensive? Apple or NVidia?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: Apple: 123.45; Nvidia: 345.65
Assistant: Nvidia
</example>
`

func TestToOpenAIMessage(t *testing.T) {
	tests := []struct {
		name    string
		msg     Message
		want    oai.ChatCompletionMessageParamUnion
		wantErr bool
	}{
		{
			name: "error: empty blocks",
			msg: Message{
				Role:   User,
				Blocks: []Block{},
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "error: nil blocks",
			msg: Message{
				Role:   User,
				Blocks: nil,
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "user message",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hello, how are you?"),
					},
				},
			},
			want:    oai.UserMessage("Hello, how are you?"),
			wantErr: false,
		},
		{
			name: "assistant message",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("I'm doing well, thank you!"),
					},
				},
			},
			want:    oai.AssistantMessage("I'm doing well, thank you!"),
			wantErr: false,
		},
		{
			name: "tool call",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						ID:           "call_123",
						BlockType:    ToolCall,
						ModalityType: Text,
						Content:      Str(`{"name": "get_weather", "parameters": {"location": "London"}}`),
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					ToolCalls: []oai.ChatCompletionMessageToolCallUnionParam{
						{
							OfFunction: &oai.ChatCompletionMessageFunctionToolCallParam{
								ID: "call_123",
								Function: oai.ChatCompletionMessageFunctionToolCallFunctionParam{
									Name:      "get_weather",
									Arguments: `{"location":"London"}`,
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "tool result",
			msg: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "call_123",
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("The current temperature is 72°F"),
					},
				},
			},
			want:    oai.ToolMessage("The current temperature is 72°F", "call_123"),
			wantErr: false,
		},
		{
			name: "tool call with text",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str(`Let me get the weather for you:`),
					},
					{
						ID:           "call_123",
						BlockType:    ToolCall,
						ModalityType: Text,
						Content:      Str(`{"name": "get_weather", "parameters": {"location": "London"}}`),
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					Content: oai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: oai.String("Let me get the weather for you:"),
					},
					ToolCalls: []oai.ChatCompletionMessageToolCallUnionParam{
						{
							OfFunction: &oai.ChatCompletionMessageFunctionToolCallParam{
								ID: "call_123",
								Function: oai.ChatCompletionMessageFunctionToolCallFunctionParam{
									Name:      "get_weather",
									Arguments: `{"location":"London"}`,
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "error: video modality not supported",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Video,
						Content:      Str("fake-base64-data"),
					},
				},
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "error: invalid role",
			msg: Message{
				Role: 999,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hello"),
					},
				},
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "multimodal user message with text and image",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What's in this image?"),
					},
					{
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "image/jpeg",
						Content:      Str("fake-image-base64-data"),
					},
				},
			},
			want: oai.UserMessage([]oai.ChatCompletionContentPartUnionParam{
				{
					OfText: &oai.ChatCompletionContentPartTextParam{
						Text: "What's in this image?",
					},
				},
				{
					OfImageURL: &oai.ChatCompletionContentPartImageParam{
						ImageURL: oai.ChatCompletionContentPartImageImageURLParam{
							URL: "data:image/jpeg;base64,fake-image-base64-data",
						},
					},
				},
			}),
			wantErr: false,
		},
		{
			name: "multimodal user message with text and audio",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What's in this audio?"),
					},
					{
						BlockType:    Content,
						ModalityType: Audio,
						MimeType:     "audio/wav",
						Content:      Str("fake-audio-base64-data"),
					},
				},
			},
			want: oai.UserMessage([]oai.ChatCompletionContentPartUnionParam{
				{
					OfText: &oai.ChatCompletionContentPartTextParam{
						Text: "What's in this audio?",
					},
				},
				{
					OfInputAudio: &oai.ChatCompletionContentPartInputAudioParam{
						InputAudio: oai.ChatCompletionContentPartInputAudioInputAudioParam{
							Data:   "fake-audio-base64-data",
							Format: "audio/wav",
						},
					},
				},
			}),
			wantErr: false,
		},
		{
			name: "assistant message with audio ID",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						ID:           "audio_abc123",
						BlockType:    Content,
						ModalityType: Audio,
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					Audio: oai.ChatCompletionAssistantMessageParamAudio{
						ID: "audio_abc123",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "assistant message with text and audio ID",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Here's my response:"),
					},
					{
						ID:           "audio_abc123",
						BlockType:    Content,
						ModalityType: Audio,
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					Audio: oai.ChatCompletionAssistantMessageParamAudio{
						ID: "audio_abc123",
					},
					Content: oai.ChatCompletionAssistantMessageParamContentUnion{
						OfArrayOfContentParts: []oai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
							{
								OfText: &oai.ChatCompletionContentPartTextParam{
									Text: "Here's my response:",
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toOpenAIMessage(tt.msg)
			if (err != nil) != tt.wantErr {
				t.Errorf("toOpenAIMessage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Custom comparison that ignores unexported fields
				// This approach checks only the message types and key properties,
				// without attempting to compare the unexported fields of the complex
				// OpenAI SDK types such as the internal implementations of Opt[T]
				// and other generic structures.
				//
				// We're specifically checking that the message role (User, Assistant, Tool)
				// matches, and for specific message types, we verify relevant fields like
				// tool call IDs and tool function names.

				// Check if the role/message type matches
				if (got.OfUser != nil) != (tt.want.OfUser != nil) ||
					(got.OfAssistant != nil) != (tt.want.OfAssistant != nil) ||
					(got.OfTool != nil) != (tt.want.OfTool != nil) ||
					(got.OfSystem != nil) != (tt.want.OfSystem != nil) {
					t.Errorf("toOpenAIMessage() returned wrong message type")
					return
				}

				// For tool call messages, verify tool call ID matches
				if got.OfTool != nil && tt.want.OfTool != nil {
					if got.OfTool.ToolCallID != tt.want.OfTool.ToolCallID {
						t.Errorf("Tool call ID mismatch: got %v, want %v",
							got.OfTool.ToolCallID, tt.want.OfTool.ToolCallID)
					}
				}

				// For assistant messages with tool calls, verify tool call info
				if got.OfAssistant != nil && tt.want.OfAssistant != nil {
					// Check if both have tool calls
					if (len(got.OfAssistant.ToolCalls) > 0) != (len(tt.want.OfAssistant.ToolCalls) > 0) {
						t.Errorf("Tool calls presence mismatch")
						return
					}

					// If they have tool calls, verify basic properties
					if len(got.OfAssistant.ToolCalls) > 0 && len(tt.want.OfAssistant.ToolCalls) > 0 {
						if got.OfAssistant.ToolCalls[0].OfFunction.ID != tt.want.OfAssistant.ToolCalls[0].OfFunction.ID ||
							got.OfAssistant.ToolCalls[0].OfFunction.Function.Name != tt.want.OfAssistant.ToolCalls[0].OfFunction.Function.Name {
							t.Errorf("Tool call details mismatch")
						}
					}
				}
			}
		})
	}
}

func TestOpenAiGenerator_Generate_image(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	imgBytes, err := os.ReadFile("sample.jpg")
	if err != nil {
		t.Skip("could not open sample.jpg")
		return
	}
	imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))
	client := oai.NewClient(
		option.WithAPIKey(apiKey),
	)
	gen := NewOpenAiGenerator(&client.Chat.Completions)
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
	resp, err := gen.Generate(context.Background(), GenerationRequest{
		Model:        oai.ChatModelGPT4o,
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
		Options:      NewGenerationOptions(WithMaxGenerationTokens(512)),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) != 1 {
		t.Fatalf("candidates = %d, want 1", len(resp.Candidates))
	}
	if len(resp.Candidates[0].Blocks) != 1 {
		t.Fatalf("blocks = %d, want 1", len(resp.Candidates[0].Blocks))
	}
	if !strings.Contains(resp.Candidates[0].Blocks[0].Content.String(), "Crood") {
		t.Fatalf("content does not contain Crood")
	}
}
func TestOpenAiGenerator_Generate_audio(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	audioBytes, err := os.ReadFile("sample.wav")
	if err != nil {
		t.Skip("could not open sample.wav")
		return
	}
	// Encode as base64 for inline audio usage
	audioBase64 := Str(base64.StdEncoding.EncodeToString(audioBytes))
	client := oai.NewClient(
		option.WithAPIKey(apiKey),
	)
	gen := NewOpenAiGenerator(&client.Chat.Completions)
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
					Content:      Str("In this audio, a person is introducing themselves. What is the name of person in the greeting in this audio? Return a one word response of the name"),
				},
			},
		},
	}
	resp, err := gen.Generate(context.Background(), GenerationRequest{
		Model:        oai.ChatModelGPT4oAudioPreview,
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
		Options:      NewGenerationOptions(WithMaxGenerationTokens(128)),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		if got := strings.ToLower(resp.Candidates[0].Blocks[0].Content.String()); !strings.Contains(got, "friday") {
			t.Fatalf("content = %q, want it to contain friday", got)
		}
	}
}
func TestOpenAiGenerator_Generate(t *testing.T) {
	// Create an OpenAI client
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := oai.NewClient(
		option.WithAPIKey(apiKey),
	)
	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hi!"),
				},
			},
		},
	}
	request := GenerationRequest{
		Model:        oai.ChatModelGPT4oMini,
		Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
		Dialog:       dialog,
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The exact response text may vary, so we'll just print a placeholder
	// Customize generation parameters
	request.Options = NewGenerationOptions(
		WithTopK(10),
		WithCandidateCount(2),
		WithMaxGenerationTokens(1024),
	)
	resp, err = gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(resp.Candidates); got == 0 {
		t.Fatal("expected at least one item")
	}
}
func TestOpenAiGenerator_Stream(t *testing.T) {
	// Create an OpenAI client
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := oai.NewClient(
		option.WithAPIKey(apiKey),
	)
	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hi!"),
				},
			},
		},
	}
	// Stream a response
	blocks := make([][]Block, 2)
	for chunk := range gen.Stream(context.Background(), GenerationRequest{
		Model:        oai.ChatModelGPT4oMini,
		Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
		Dialog:       dialog,
		Options:      NewGenerationOptions(WithCandidateCount(2)),
	}) {
		if chunk.Err != nil {
			t.Fatalf("stream returned error: %v", chunk.Err)
		}
		blocks[chunk.CandidatesIndex] = append(blocks[chunk.CandidatesIndex], chunk.Block)
	}
	if len(blocks) == 2 && len(blocks[0]) > 1 && len(blocks[1]) > 1 {
	}
}
func TestOpenAiGenerator_Generate_openRouter(t *testing.T) {
	// Create an OpenAI client for open router
	client := oai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1/"),
		option.WithAPIKey(requireLiveAPIKey(t, "OPENROUTER_API_KEY")),
	)
	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hi!"),
				},
			},
		},
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), GenerationRequest{
		Model:        "google/gemini-2.5-pro-preview-03-25",
		Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
		Dialog:       dialog,
		Options:      NewGenerationOptions(WithMaxGenerationTokens(1024)),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The exact response text may vary, so we'll just print a placeholder
	if got := len(resp.Candidates); got == 0 {
		t.Fatal("expected at least one item")
	}
}
func TestOpenAiGenerator_Generate_thinking(t *testing.T) {
	requireLiveAPIKey(t, "OPENAI_API_KEY")
	// Create an OpenAI client
	client := oai.NewClient()
	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions)
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Hi!"),
				},
			},
		},
	}
	// Customize generation parameters
	request := GenerationRequest{
		Model:        oai.ChatModelO3Mini,
		Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
		Dialog:       dialog,
		Options: NewGenerationOptions(
			WithMaxGenerationTokens(4096),
			WithThinkingBudget("low"),
			WithTemperature(1.0),
		),
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The exact response text may vary, so we'll just print a placeholder
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: User,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("What can you do?"),
			},
		},
	})
	request.Dialog = dialog
	resp, err = gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(resp.Candidates); got == 0 {
		t.Fatal("expected at least one item")
	}
}
func TestOpenAiGenerator_RequestTools(t *testing.T) {
	requireLiveAPIKey(t, "OPENAI_API_KEY")
	// Create an OpenAI client
	client := oai.NewClient(option.WithBaseURL("https://gateway.ai.cloudflare.com/v1/4eee6dd2fdc8cebc7802c5a638f460fe/cpe/openai/"))
	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions)
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
	request := GenerationRequest{
		Model:        oai.ChatModelGPT4oMini,
		Instructions: SystemMessage(TextBlock(openAIStockInstructions)),
		Dialog:       dialog,
		Tools:        []Tool{tickerTool},
		Options:      NewGenerationOptions(WithToolChoice("get_stock_price")),
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[0].ID,
				ModalityType: Text,
				Content:      Str("123.45"),
			},
		},
	})
	request.Dialog = dialog
	request.Options = nil
	resp, err = gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
}
func TestOpenAiGenerator_Stream_parallelToolUse(t *testing.T) {
	// Create an OpenAI client
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	client := oai.NewClient(
		option.WithAPIKey(apiKey),
	)
	// Define request tools
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
	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions)
	tickerTool.Description += "\nYou can call this tool in parallel"
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
	request := GenerationRequest{
		Model:        oai.ChatModelGPT4oMini,
		Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
		Dialog:       dialog,
		Tools:        []Tool{tickerTool},
	}
	// Stream a response
	var blocks []Block
	for chunk := range gen.Stream(context.Background(), request) {
		if chunk.Err != nil {
			t.Fatalf("stream returned error: %v", chunk.Err)
		}
		blocks = append(blocks, chunk.Block)
	}
	if len(blocks) > 1 {
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
					t.Fatalf("unexpected error: %v", err)
				}
				// Marshal back to JSON for consistent representation
				toolUseJSON, err := json.Marshal(toolCallInput)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
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
			t.Fatalf("unexpected error: %v", err)
		}
		// Marshal back to JSON for consistent representation
		toolUseJSON, err := json.Marshal(toolCallInput)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		toolCalls[len(toolCalls)-1].Content = Str(toolUseJSON)
		toolCallInput = ToolCallInput{}
	}
	if got := len(toolCalls); got == 0 {
		t.Fatal("expected at least one item")
	}
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
	request.Dialog = dialog
	for chunk := range gen.Stream(context.Background(), request) {
		if chunk.Err != nil {
			t.Fatalf("stream returned error: %v", chunk.Err)
		}
		blocks = append(blocks, chunk.Block)
	}
	if len(blocks) > 1 {
	}
}
func TestOpenAiGenerator_RequestTools_parallelToolUse(t *testing.T) {
	requireLiveAPIKey(t, "OPENAI_API_KEY")
	// Create an OpenAI client
	client := oai.NewClient()
	// Define request tools
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
	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions)
	tickerTool.Description += "\nYou can call this tool in parallel"
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
	request := GenerationRequest{
		Model:        oai.ChatModelGPT4oMini,
		Instructions: SystemMessage(TextBlock(openAIStockComparisonInstructions)),
		Dialog:       dialog,
		Tools:        []Tool{tickerTool},
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	if got := resp.Candidates[0].Blocks[1].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[0].ID,
				ModalityType: Text,
				Content:      Str("123.45"),
			},
		},
	}, Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[1].ID,
				ModalityType: Text,
				Content:      Str("678.45"),
			},
		},
	})
	request.Dialog = dialog
	request.Options = nil
	resp, err = gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
}
func TestOpenAiGenerator_RequestTools_openRouter(t *testing.T) {
	// Define request tools
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
	// Create an OpenAI client for open router
	client := oai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1/"),
		option.WithAPIKey(requireLiveAPIKey(t, "OPENROUTER_API_KEY")),
	)
	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions)
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
	request := GenerationRequest{
		Model:        "google/gemini-2.5-pro-preview-03-25",
		Instructions: SystemMessage(TextBlock(openAIStockInstructions)),
		Dialog:       dialog,
		Tools:        []Tool{tickerTool},
		Options:      NewGenerationOptions(WithToolChoice("get_stock_price")),
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[0].ID,
				ModalityType: Text,
				Content:      Str("123.45"),
			},
		},
	})
	request.Dialog = dialog
	request.Options = nil
	resp, err = gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
}
func TestOpenAiGenerator_RequestTools_openRouterParallelToolUse(t *testing.T) {
	// Create an OpenAI client
	client := oai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1/"),
		option.WithAPIKey(requireLiveAPIKey(t, "OPENROUTER_API_KEY")),
	)
	// Define request tools
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
	// Instantiate a OpenAI Generator
	gen := NewOpenAiGenerator(&client.Chat.Completions)
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
	request := GenerationRequest{
		Model:        "google/gemini-2.5-pro-preview-03-25",
		Instructions: SystemMessage(TextBlock(openAIStockComparisonInstructions)),
		Dialog:       dialog,
		Tools:        []Tool{tickerTool},
	}
	// Generate a response
	resp, err := gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	if got := resp.Candidates[0].Blocks[1].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
	dialog = append(dialog, resp.Candidates[0], Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[0].ID,
				ModalityType: Text,
				Content:      Str("123.45"),
			},
		},
	}, Message{
		Role: ToolResult,
		Blocks: []Block{
			{
				ID:           resp.Candidates[0].Blocks[1].ID,
				ModalityType: Text,
				Content:      Str("678.45"),
			},
		},
	})
	request.Dialog = dialog
	request.Options = nil
	resp, err = gen.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
		t.Fatal("expected non-empty content")
	}
}
func TestOpenAiGenerator_Count_Example(t *testing.T) {
	// Create an OpenAI client
	client := oai.NewClient()
	// Create a generator
	generator := NewOpenAiGenerator(&client.Chat.Completions)
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
		Model:        oai.ChatModelGPT4o,
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
	}
	// Count tokens in the dialog
	tokenCount, err := generator.Count(context.Background(), request)
	if err != nil {
		t.Fatalf("count tokens: %v", err)
	}
	if tokenCount == 0 {
		t.Fatal("expected non-zero token count")
	}
	// Add a response to the dialog
	dialog = append(dialog, Message{
		Role: Assistant,
		Blocks: []Block{
			{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("The capital of France is Paris. It's known as the 'City of Light' and is famous for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."),
			},
		},
	})
	request.Dialog = dialog
	// Count tokens in the updated dialog
	tokenCount, err = generator.Count(context.Background(), request)
	if err != nil {
		t.Fatalf("count tokens: %v", err)
	}
	if tokenCount == 0 {
		t.Fatal("expected non-zero token count")
	}
}
func TestOpenAiGenerator_Generate_pdf(t *testing.T) {
	apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
	pdfBytes, err := os.ReadFile("sample.pdf")
	if err != nil {
		t.Skip("could not open sample.wav")
		return
	}
	client := oai.NewClient(
		option.WithAPIKey(apiKey),
	)
	gen := NewOpenAiGenerator(&client.Chat.Completions)
	// Create a dialog with PDF content
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				TextBlock("What is the title of this PDF? Just output the title and nothing else"),
				PDFBlock(pdfBytes, "sample.pdf"),
			},
		},
	}
	// Generate a response
	ctx := context.Background()
	response, err := gen.Generate(ctx, GenerationRequest{
		Model:        oai.ChatModelGPT4_1,
		Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
		Dialog:       dialog,
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
