package gai

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	openai "github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
)

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

// Exercise actual HTTP serialization through public constructors, including
// preprocessing, streaming, and the providers' remote counting endpoints.
func TestMidConversationSystemMessageRequests(t *testing.T) {
	providers := []struct {
		name     string
		protocol string
		new      func(*http.Client, string) (Generator, error)
	}{
		{"OpenAI", "chat", func(c *http.Client, url string) (Generator, error) { return NewOpenAiGenerator(c, url, "test-key") }},
		{"Cerebras", "chat", func(c *http.Client, url string) (Generator, error) { return NewCerebrasGenerator(c, url, "test-key") }},
		{"OpenRouter", "chat", func(c *http.Client, url string) (Generator, error) { return NewOpenRouterGenerator(c, url, "test-key") }},
		{"DeepSeek", "chat", func(c *http.Client, url string) (Generator, error) { return NewDeepSeekGenerator(c, url, "test-key") }},
		{"ZAI", "chat", func(c *http.Client, url string) (Generator, error) { return NewZaiGenerator(c, url, "test-key") }},
		{"Moonshot", "chat", func(c *http.Client, url string) (Generator, error) { return NewMoonshotGenerator(c, url, "test-key") }},
		{"Anthropic", "anthropic", func(c *http.Client, url string) (Generator, error) {
			client := anthropic.NewClient(anthropicoption.WithoutEnvironmentDefaults(), anthropicoption.WithAPIKey("test-key"), anthropicoption.WithBaseURL(url), anthropicoption.WithHTTPClient(c), anthropicoption.WithMaxRetries(0))
			return NewAnthropicGenerator(&client.Messages), nil
		}},
		{"Responses", "responses", func(c *http.Client, url string) (Generator, error) {
			client := openai.NewClient(openaioption.WithAPIKey("test-key"), openaioption.WithBaseURL(url), openaioption.WithHTTPClient(c), openaioption.WithMaxRetries(0))
			return NewResponsesGenerator(&client.Responses), nil
		}},
	}
	for _, provider := range providers {
		t.Run(provider.name, func(t *testing.T) {
			captured := make(chan map[string]any, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				captured <- body
				w.Header().Set("Content-Type", "application/json")
				if strings.Contains(r.URL.Path, "token") {
					fmt.Fprint(w, `{"input_tokens":23,"usage":{"total_tokens":23},"data":{"total_tokens":23}}`)
					return
				}
				// A deliberate server rejection distinguishes successful request
				// conversion from local validation failures without live API calls.
				w.WriteHeader(http.StatusBadRequest)
				fmt.Fprint(w, `{"error":{"type":"invalid_request_error","message":"test rejection"}}`)
			}))
			defer server.Close()
			generator, err := provider.new(server.Client(), server.URL)
			if err != nil {
				t.Fatal(err)
			}
			for _, scenario := range []string{"SingleToolResult", "ParallelToolResults"} {
				t.Run(scenario, func(t *testing.T) {
					call, err := ToolCallBlock("call_1", "lookup", map[string]any{"item": "book"})
					if err != nil {
						t.Fatal(err)
					}
					calls := []Block{call}
					results := Dialog{ToolResultMessage("call_1", TextBlock("19.99"))}
					wantResults := map[string]string{"call_1": "19.99"}
					if scenario == "ParallelToolResults" {
						second, err := ToolCallBlock("call_2", "lookup", map[string]any{"item": "pen"})
						if err != nil {
							t.Fatal(err)
						}
						calls = append(calls, second)
						results = append(results, ToolResultMessage("call_2", TextBlock("2.50")))
						wantResults["call_2"] = "2.50"
					}
					request := GenerationRequest{
						Model: "caller-selected-model", Instructions: SystemMessage(TextBlock("base instructions")),
						Dialog: Dialog{
							{Role: User, Blocks: []Block{TextBlock("question")}},
							SystemMessage(TextBlock("new "), TextBlock("guidance")),
							{Role: Assistant, Blocks: []Block{TextBlock("acknowledged")}},
							{Role: User, Blocks: []Block{TextBlock("look up the price")}},
							{Role: Assistant, Blocks: calls},
						},
					}
					request.Dialog = append(request.Dialog, results...)
					request.Dialog = append(request.Dialog, SystemMessage(TextBlock("use the result")), SystemMessage(TextBlock("be brief")))
					before, err := json.Marshal(request)
					if err != nil {
						t.Fatal(err)
					}
					for _, operation := range []string{"Generate", "Stream", "Count"} {
						if operation == "Count" {
							if _, ok := generator.(TokenCounter); !ok || provider.name == "OpenAI" {
								continue // OpenAI counts locally rather than serializing HTTP.
							}
						}
						t.Run(operation, func(t *testing.T) {
							var err error
							switch operation {
							case "Generate":
								_, err = generator.Generate(t.Context(), request)
							case "Stream":
								for chunk := range generator.(StreamingGenerator).Stream(t.Context(), request) {
									if chunk.Err != nil {
										err = chunk.Err
									}
								}
							case "Count":
								var tokens uint
								tokens, err = generator.(TokenCounter).Count(t.Context(), request)
								if err != nil || tokens != 23 {
									t.Fatalf("Count() = %d, %v", tokens, err)
								}
							}
							if operation != "Count" {
								var apiErr *ApiErr
								if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusBadRequest {
									t.Fatalf("expected server rejection, got %v", err)
								}
							}
							select {
							case body := <-captured:
								consolidate := provider.protocol == "anthropic" && operation != "Count"
								assertInlineSystemHistory(t, provider.protocol, body, wantResults, consolidate)
							default:
								t.Fatal("request never reached the server")
							}
							after, err := json.Marshal(request)
							if err != nil || !reflect.DeepEqual(before, after) {
								t.Fatal("generator mutated the borrowed request")
							}
						})
					}
				})
			}
		})
	}
}

func TestInvalidInlineSystemMessages(t *testing.T) {
	converters := []struct {
		name    string
		convert func(Message) error
	}{
		{"Anthropic", func(message Message) error {
			_, err := toAnthropicMessage(message)
			return err
		}},
		{"Responses", func(message Message) error {
			_, err := (&ResponsesGenerator{}).buildInputItems(Dialog{message})
			return err
		}},
	}
	messages := []struct {
		name    string
		message Message
	}{
		{"empty", SystemMessage()},
		{"nil content", SystemMessage(Block{BlockType: Content})},
		{"thinking", SystemMessage(Block{BlockType: Thinking, Content: Str("reasoning")})},
		{"tool call", SystemMessage(Block{BlockType: ToolCall, Content: Str(`{"name":"lookup","parameters":{}}`)})},
		{"tool-result error", Message{Role: System, ToolResultError: true, Blocks: []Block{TextBlock("error")}}},
		{"unknown role", Message{Blocks: []Block{TextBlock("missing role")}}},
	}
	for _, converter := range converters {
		t.Run(converter.name, func(t *testing.T) {
			for _, test := range messages {
				t.Run(test.name, func(t *testing.T) {
					if err := converter.convert(test.message); err == nil {
						t.Fatal("invalid system message was accepted")
					}
				})
			}
		})
	}
	t.Run("Anthropic system media", func(t *testing.T) {
		for _, modality := range []Modality{Image, Audio, Video} {
			t.Run(modality.String(), func(t *testing.T) {
				_, err := toAnthropicMessage(SystemMessage(Block{BlockType: Content, ModalityType: modality, Content: Str("data")}))
				var unsupported UnsupportedInputModalityErr
				if !errors.As(err, &unsupported) {
					t.Fatalf("error = %v, want UnsupportedInputModalityErr", err)
				}
			})
		}
	})
}

func assertInlineSystemHistory(t *testing.T, protocol string, body map[string]any, wantResults map[string]string, consolidate bool) {
	t.Helper()
	key := "messages"
	if protocol == "responses" {
		key = "input"
	}
	messages, ok := body[key].([]any)
	if !ok {
		t.Fatalf("missing %s: %v", key, body)
	}
	if protocol == "chat" {
		if len(messages) == 0 {
			t.Fatal("missing initial instructions and dialog")
		}
		initial := messages[0].(map[string]any)
		if initial["role"] != "system" || initial["content"] != "base instructions" {
			t.Fatalf("initial instructions changed: %v", initial)
		}
		messages = messages[1:]
	} else if protocol == "responses" {
		if body["instructions"] != "base instructions" {
			t.Fatalf("initial instructions changed: %v", body["instructions"])
		}
	} else if got := systemRequestText(t, body["system"]); got != "base instructions" {
		t.Fatalf("initial instructions = %q", got)
	}
	roles := []string{"user", "system", "assistant", "user"}
	resultStart := 5
	if protocol == "responses" {
		for range len(wantResults) {
			roles = append(roles, "function_call")
		}
		resultStart = len(roles)
		for range len(wantResults) {
			roles = append(roles, "function_call_output")
		}
	} else {
		roles = append(roles, "assistant")
		resultRole, resultCount := "tool", len(wantResults)
		if protocol == "anthropic" {
			resultRole = "user"
		}
		if consolidate {
			resultCount = 1
		}
		for range resultCount {
			roles = append(roles, resultRole)
		}
	}
	roles = append(roles, "system", "system")
	if len(messages) != len(roles) {
		t.Fatalf("history = %v, want %d ordered items", messages, len(roles))
	}
	for i, want := range roles {
		message := messages[i].(map[string]any)
		role := message["role"]
		if role == nil {
			role = message["type"]
		}
		if role != want {
			t.Fatalf("item %d role = %v, want %s", i, role, want)
		}
	}
	for i, want := range map[int]string{1: "new guidance", len(roles) - 2: "use the result", len(roles) - 1: "be brief"} {
		message := messages[i].(map[string]any)
		if got := systemRequestText(t, message["content"]); got != want {
			t.Errorf("system message %d = %q, want %q", i, got, want)
		}
	}
	var calls []any
	callIDField := "id"
	switch protocol {
	case "chat":
		calls = messages[4].(map[string]any)["tool_calls"].([]any)
	case "anthropic":
		calls = messages[4].(map[string]any)["content"].([]any)
	case "responses":
		calls, callIDField = messages[4:resultStart], "call_id"
	}
	if len(calls) != len(wantResults) {
		t.Fatalf("tool calls = %v, want %d", calls, len(wantResults))
	}
	seenCalls := make(map[string]bool)
	for _, call := range calls {
		id := call.(map[string]any)[callIDField].(string)
		if _, known := wantResults[id]; !known || seenCalls[id] {
			t.Fatalf("unexpected or duplicate tool call %q", id)
		}
		seenCalls[id] = true
	}
	gotResults := make(map[string]string)
	for _, item := range messages[resultStart : len(roles)-2] {
		message := item.(map[string]any)
		if protocol == "anthropic" {
			for _, part := range message["content"].([]any) {
				block := part.(map[string]any)
				if block["type"] != "tool_result" {
					t.Fatalf("unexpected block in tool results: %v", block)
				}
				id := block["tool_use_id"].(string)
				if _, duplicate := gotResults[id]; duplicate {
					t.Fatalf("duplicate tool result %q", id)
				}
				gotResults[id] = systemRequestText(t, block["content"])
			}
			continue
		}
		idField, contentField := "tool_call_id", "content"
		if protocol == "responses" {
			idField, contentField = "call_id", "output"
		}
		id := message[idField].(string)
		if _, duplicate := gotResults[id]; duplicate {
			t.Fatalf("duplicate tool result %q", id)
		}
		gotResults[id] = systemRequestText(t, message[contentField])
	}
	if !reflect.DeepEqual(gotResults, wantResults) {
		t.Fatalf("tool results = %v, want %v", gotResults, wantResults)
	}
}

func systemRequestText(t *testing.T, content any) string {
	t.Helper()
	if text, ok := content.(string); ok {
		return text
	}
	parts, ok := content.([]any)
	if !ok {
		t.Fatalf("invalid system content: %v", content)
	}
	var text strings.Builder
	for _, part := range parts {
		block, ok := part.(map[string]any)
		if !ok || (block["type"] != "text" && block["type"] != "input_text") {
			t.Fatalf("invalid system block: %v", part)
		}
		value, ok := block["text"].(string)
		if !ok {
			t.Fatalf("missing system text: %v", block)
		}
		text.WriteString(value)
	}
	return text.String()
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
		params, err := (&ResponsesGenerator{}).buildParams(nil, GenerationRequest{Model: "test", Instructions: instructions})
		if err != nil {
			t.Fatal(err)
		}
		if params.Instructions.Value != "first\n\nsecond" {
			t.Fatalf("Responses instructions = %q, want %q", params.Instructions.Value, "first\n\nsecond")
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
		{name: "Anthropic", parse: func(options GenerationOptions) error { _, err := parseAnthropicGenerationOptions(options); return err }},
		{name: "Gemini", parse: func(options GenerationOptions) error { _, err := parseGeminiGenerationOptions(options); return err }},
		{name: "Cerebras", parse: func(options GenerationOptions) error {
			_, err := (&CerebrasGenerator{}).prepareRequest(GenerationRequest{Options: options})
			return err
		}},
		{name: "OpenRouter", parse: func(options GenerationOptions) error {
			_, err := (&OpenRouterGenerator{}).prepareRequest(GenerationRequest{Options: options})
			return err
		}},
		{name: "Responses", parse: func(options GenerationOptions) error { _, err := parseResponsesGenerationOptions(options); return err }},
		{name: "ZAI", parse: func(options GenerationOptions) error {
			_, err := (&ZaiGenerator{}).prepareRequest(GenerationRequest{Options: options})
			return err
		}},
	}

	for _, tt := range parsers {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.parse(GenerationOptions{"unknown_provider_option": struct{}{}}); err != nil {
				t.Fatalf("unknown option returned error: %v", err)
			}
		})
	}
}
