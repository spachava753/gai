package gai

import (
	"context"
	"errors"
	"reflect"
	"testing"

	oai "github.com/openai/openai-go/v3"
)

func testNewGenerationOptions(t *testing.T) {
	stops := []string{"END", "STOP"}
	modalities := []Modality{Text, Audio}
	audio := AudioConfig{VoiceName: "alloy", Format: "wav"}

	options := NewGenerationOptions(
		WithTemperature(0.2),
		WithTopP(0.8),
		WithTopK(40),
		WithFrequencyPenalty(0.1),
		WithPresencePenalty(0.3),
		WithCandidateCount(2),
		WithMaxGenerationTokens(512),
		WithToolChoice(ToolChoiceAuto),
		WithStopSequences(stops...),
		WithOutputModalities(modalities...),
		WithAudioConfig(audio),
		WithThinkingBudget("medium"),
	)

	want := GenerationOptions{
		GenerationOptionTemperature:         0.2,
		GenerationOptionTopP:                0.8,
		GenerationOptionTopK:                uint(40),
		GenerationOptionFrequencyPenalty:    0.1,
		GenerationOptionPresencePenalty:     0.3,
		GenerationOptionCandidateCount:      uint(2),
		GenerationOptionMaxGenerationTokens: 512,
		GenerationOptionToolChoice:          ToolChoiceAuto,
		GenerationOptionStopSequences:       []string{"END", "STOP"},
		GenerationOptionOutputModalities:    []Modality{Text, Audio},
		GenerationOptionAudioConfig:         audio,
		GenerationOptionThinkingBudget:      "medium",
	}
	if !reflect.DeepEqual(options, want) {
		t.Fatalf("NewGenerationOptions() = %#v, want %#v", options, want)
	}

	stops[0] = "changed"
	modalities[0] = Video
	if !reflect.DeepEqual(options[GenerationOptionStopSequences], []string{"END", "STOP"}) {
		t.Fatalf("stop sequences changed after mutating input: %#v", options[GenerationOptionStopSequences])
	}
	if !reflect.DeepEqual(options[GenerationOptionOutputModalities], []Modality{Text, Audio}) {
		t.Fatalf("output modalities changed after mutating input: %#v", options[GenerationOptionOutputModalities])
	}
}

func TestGeneratedProviderConstructors(t *testing.T) {
	constructors := []struct {
		name      string
		construct func(string, string) error
	}{
		{
			name: "cerebras",
			construct: func(baseURL, apiKey string) error {
				_, err := NewCerebrasGenerator(nil, baseURL, apiKey)
				return err
			},
		},
		{
			name: "deepseek",
			construct: func(baseURL, apiKey string) error {
				_, err := NewDeepSeekGenerator(nil, baseURL, apiKey)
				return err
			},
		},
		{
			name: "openrouter",
			construct: func(baseURL, apiKey string) error {
				_, err := NewOpenRouterGenerator(nil, baseURL, apiKey)
				return err
			},
		},
		{
			name: "zai",
			construct: func(baseURL, apiKey string) error {
				_, err := NewZaiGenerator(nil, baseURL, apiKey)
				return err
			},
		},
	}

	for _, tt := range constructors {
		t.Run(tt.name+" default", func(t *testing.T) {
			if err := tt.construct("", "test-key"); err != nil {
				t.Fatalf("construct with default base URL: %v", err)
			}
		})
		t.Run(tt.name+" invalid URL", func(t *testing.T) {
			if err := tt.construct("https://example.com/%zz", "test-key"); err == nil {
				t.Fatal("construct with invalid base URL returned nil error")
			}
		})
		t.Run(tt.name+" missing API key", func(t *testing.T) {
			err := tt.construct("", "")
			if !errors.Is(err, ErrMissingAPIKey) {
				t.Fatalf("construct with empty API key returned %v, want ErrMissingAPIKey", err)
			}
		})
	}
}

func TestTextInstructions(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		parts, err := textInstructions(Message{})
		if err != nil || parts != nil {
			t.Fatalf("textInstructions() = %#v, %v; want nil, nil", parts, err)
		}
	})

	t.Run("empty system message", func(t *testing.T) {
		parts, err := textInstructions(SystemMessage())
		if err != nil || parts != nil {
			t.Fatalf("textInstructions() = %#v, %v; want nil, nil", parts, err)
		}
	})

	t.Run("ordered text blocks", func(t *testing.T) {
		instructions := SystemMessage(TextBlock("first"), TextBlock("second"))
		parts, err := textInstructions(instructions)
		if err != nil {
			t.Fatalf("textInstructions() error = %v", err)
		}
		if !reflect.DeepEqual(parts, []string{"first", "second"}) {
			t.Fatalf("textInstructions() = %#v", parts)
		}
		joined, err := joinedTextInstructions(instructions)
		if err != nil {
			t.Fatalf("joinedTextInstructions() error = %v", err)
		}
		if joined != "first\n\nsecond" {
			t.Fatalf("joinedTextInstructions() = %q, want %q", joined, "first\n\nsecond")
		}
	})

	tests := []struct {
		name         string
		instructions Message
		unsupported  bool
	}{
		{
			name:         "populated unknown role",
			instructions: Message{Blocks: []Block{TextBlock("missing role")}},
		},
		{
			name:         "unknown role with extra fields",
			instructions: Message{ExtraFields: map[string]any{"key": "value"}},
		},
		{
			name:         "empty non-system role",
			instructions: Message{Role: User},
		},
		{
			name:         "tool-result error marker",
			instructions: Message{Role: System, ToolResultError: true},
		},
		{
			name:         "non-system role",
			instructions: Message{Role: User, Blocks: []Block{TextBlock("wrong role")}},
		},
		{
			name: "unsupported block type",
			instructions: Message{Role: System, Blocks: []Block{{
				BlockType:    Thinking,
				ModalityType: Text,
				Content:      Str("thinking"),
			}}},
		},
		{
			name: "nil content",
			instructions: Message{Role: System, Blocks: []Block{{
				BlockType:    Content,
				ModalityType: Text,
			}}},
		},
		{
			name: "unsupported modality",
			instructions: Message{Role: System, Blocks: []Block{{
				BlockType:    Content,
				ModalityType: Image,
				Content:      Str("image"),
			}}},
			unsupported: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := textInstructions(tt.instructions)
			if err == nil {
				t.Fatal("textInstructions() error = nil")
			}
			if tt.unsupported {
				var target UnsupportedInputModalityErr
				if !errors.As(err, &target) {
					t.Fatalf("textInstructions() error = %T, want UnsupportedInputModalityErr", err)
				}
				return
			}
			var target *InvalidParameterErr
			if !errors.As(err, &target) {
				t.Fatalf("textInstructions() error = %T, want *InvalidParameterErr", err)
			}
		})
	}
}

func TestProviderOptionParsersIgnoreUnknownOptions(t *testing.T) {
	parsers := []struct {
		name  string
		parse func(GenerationOptions) error
	}{
		{name: "OpenAI", parse: func(options GenerationOptions) error { _, err := parseOpenAIGenerationOptions(options); return err }},
		{name: "OpenCode", parse: func(options GenerationOptions) error { _, err := parseOpenCodeGenerationOptions(options); return err }},
		{name: "Anthropic", parse: func(options GenerationOptions) error { _, err := parseAnthropicGenerationOptions(options); return err }},
		{name: "Gemini", parse: func(options GenerationOptions) error { _, err := parseGeminiGenerationOptions(options); return err }},
		{name: "Cerebras", parse: func(options GenerationOptions) error { _, err := parseCerebrasGenerationOptions(options); return err }},
		{name: "OpenRouter", parse: func(options GenerationOptions) error { _, err := parseOpenRouterGenerationOptions(options); return err }},
		{name: "Responses", parse: func(options GenerationOptions) error { _, err := parseResponsesGenerationOptions(options); return err }},
		{name: "ZAI", parse: func(options GenerationOptions) error { _, err := parseZaiGenerationOptions(options); return err }},
	}

	for _, tt := range parsers {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.parse(GenerationOptions{"unknown_provider_option": struct{}{}}); err != nil {
				t.Fatalf("unknown option returned error: %v", err)
			}
		})
	}
}

func testOpenAiGeneratorUsesRequestScopedState(t *testing.T) {
	client := &mockChatCompletionService{response: &oai.ChatCompletion{
		Choices: []oai.ChatCompletionChoice{{
			FinishReason: "stop",
			Message:      oai.ChatCompletionMessage{Role: "assistant", Content: "ok"},
		}},
	}}
	generator := NewOpenAiGenerator(client)
	dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}

	_, err := generator.Generate(context.Background(), GenerationRequest{
		Model:        "first-model",
		Instructions: SystemMessage(TextBlock("first instructions")),
		Dialog:       dialog,
		Tools:        []Tool{{Name: "lookup"}},
	})
	if err != nil {
		t.Fatalf("first Generate() error = %v", err)
	}
	_, err = generator.Generate(context.Background(), GenerationRequest{
		Model:  "second-model",
		Dialog: dialog,
	})
	if err != nil {
		t.Fatalf("second Generate() error = %v", err)
	}

	if len(client.requests) != 2 {
		t.Fatalf("captured requests = %d, want 2", len(client.requests))
	}
	if got := string(client.requests[0].Model); got != "first-model" {
		t.Fatalf("first model = %q", got)
	}
	if got := string(client.requests[1].Model); got != "second-model" {
		t.Fatalf("second model = %q", got)
	}
	if got := len(client.requests[0].Tools); got != 1 {
		t.Fatalf("first request tools = %d, want 1", got)
	}
	if got := len(client.requests[1].Tools); got != 0 {
		t.Fatalf("second request tools = %d, want 0", got)
	}
	if got := len(client.requests[0].Messages); got != 2 {
		t.Fatalf("first request messages = %d, want system + user", got)
	}
	if got := len(client.requests[1].Messages); got != 1 {
		t.Fatalf("second request messages = %d, want only user", got)
	}
}
