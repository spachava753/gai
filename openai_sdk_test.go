package gai_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"

	"github.com/spachava753/gai"
)

const sdkFixtureDirectory = "testdata/openai_sdk"

type sdkRequestMatrix struct {
	Version   int                        `json:"version"`
	Scenarios map[string]sdkConversation `json:"scenarios"`
	Cases     []sdkRequestCase           `json:"cases"`
}

type sdkConversation struct {
	Instructions string                `json:"instructions"`
	Tools        []sdkToolDefinition   `json:"tools"`
	Prices       map[string]int        `json:"prices"`
	Steps        []sdkConversationStep `json:"steps"`
}

type sdkToolDefinition struct {
	Type     string `json:"type"`
	Function struct {
		Name        string             `json:"name"`
		Description string             `json:"description"`
		Parameters  *jsonschema.Schema `json:"parameters"`
	} `json:"function"`
}

type sdkConversationStep struct {
	Name              string                     `json:"name"`
	Input             string                     `json:"input"`
	Text              string                     `json:"text,omitempty"`
	Images            []string                   `json:"images,omitempty"`
	PDFs              []string                   `json:"pdfs,omitempty"`
	SDKOptions        map[string]json.RawMessage `json:"sdk_options,omitempty"`
	ExpectedArguments map[string]json.RawMessage `json:"expected_arguments,omitempty"`
	ExpectedJSON      json.RawMessage            `json:"expected_json,omitempty"`
	ExpectedText      []string                   `json:"expected_text,omitempty"`
	ExpectedToolCalls int                        `json:"expected_tool_calls"`
}

type sdkRequestCase struct {
	Name       string                     `json:"name"`
	Scenario   string                     `json:"scenario"`
	Capture    string                     `json:"capture"`
	Provider   string                     `json:"provider"`
	SDK        string                     `json:"sdk"`
	SDKVersion string                     `json:"sdk_version"`
	BaseURL    string                     `json:"base_url"`
	Model      string                     `json:"model"`
	KeyEnv     string                     `json:"key_env"`
	Stream     bool                       `json:"stream"`
	SDKOptions map[string]json.RawMessage `json:"sdk_options"`
}

type sdkHTTPExchange struct {
	Request  string `json:"request"`
	Response string `json:"response"`
}

// sdkReplayTransport is a passive, in-memory HTTP boundary. It never calls a
// real transport and makes no assertions: the test owns request comparison.
type sdkReplayTransport struct {
	responses []string
	requests  []string
	stream    bool
}

func (s *sdkReplayTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	defer request.Body.Close()
	body, err := io.ReadAll(request.Body)
	if err != nil {
		return nil, err
	}
	index := len(s.requests)
	s.requests = append(s.requests, string(body))
	if index >= len(s.responses) {
		return nil, fmt.Errorf("unexpected HTTP call %d; no live fallback", index+1)
	}
	contentType := "application/json"
	if s.stream {
		contentType = "text/event-stream"
	}
	return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {contentType}}, Body: io.NopCloser(strings.NewReader(s.responses[index])), Request: request}, nil
}

// TestOpenAISDKRequestBodies exercises only the public generator API. Each
// named step has isolated state and replays the conversation prefix needed to
// construct its input. The assertion is the HTTP request, not an SDK-shaped
// reconstruction made by calling production conversion helpers from the test.
func TestOpenAISDKRequestBodies(t *testing.T) {
	matrix := loadSDKRequestMatrix(t)
	for _, test := range matrix.Cases {
		t.Run(test.Name, func(t *testing.T) {
			scenario := matrix.Scenarios[test.Scenario]
			exchanges := loadSDKExchanges(t, test, len(scenario.Steps))
			for target, step := range scenario.Steps {
				t.Run(step.Name, func(t *testing.T) {
					t.Parallel()
					runSDKRequestCase(t, scenario, test, exchanges, target)
				})
			}
		})
	}
}

// runSDKRequestCase drives the conversation up to one target send. Request
// mismatches take precedence over response errors. Every response, including
// the target, must complete successfully and satisfy the scenario's behavior.
func runSDKRequestCase(t *testing.T, scenario sdkConversation, test sdkRequestCase, exchanges []sdkHTTPExchange, target int) {
	t.Helper()
	transport := &sdkReplayTransport{stream: test.Stream}
	for _, exchange := range exchanges[:target+1] {
		transport.responses = append(transport.responses, exchange.Response)
	}
	generator, err := gai.NewOpenAiGenerator(&http.Client{Transport: transport}, test.BaseURL, "offline-fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	var executor gai.Generator = generator
	if test.Stream {
		executor = &gai.StreamingAdapter{S: generator}
	}
	var dialog gai.Dialog
	var previous gai.Message
	for index, step := range scenario.Steps[:target+1] {
		inputs, err := scenario.inputMessages(step, previous)
		if err != nil {
			t.Fatalf("prepare %s: %v", step.Name, err)
		}
		dialog = append(dialog, inputs...)
		request := scenario.generationRequest(t, test, step)
		request.Dialog = dialog
		response, generationErr := executor.Generate(t.Context(), request)
		if len(transport.requests) != index+1 {
			t.Fatalf("%s: got %d HTTP requests, want %d (generation error: %v)", step.Name, len(transport.requests), index+1, generationErr)
		}
		actual, want := transport.requests[index], exchanges[index].Request
		if difference := sdkRequestBodyDifference(want, actual); difference != "" {
			t.Fatalf("%s request body: %s", step.Name, difference)
		}
		if generationErr != nil {
			t.Fatalf("step %s: %v", step.Name, generationErr)
		}
		scenario.checkResponse(t, step, response, request.Options)
		previous = response.Candidates[0]
		dialog = append(dialog, previous)
	}
}

func (s sdkConversation) generationRequest(t *testing.T, test sdkRequestCase, step sdkConversationStep) gai.GenerationRequest {
	t.Helper()
	options := gai.NewGenerationOptions(gai.WithOpenAIStreamUsage(false))
	extra := map[string]json.RawMessage{}
	decode := func(raw json.RawMessage, into any) {
		t.Helper()
		if err := json.Unmarshal(raw, into); err != nil {
			t.Fatal(err)
		}
	}
	// Step options replace whole SDK parameters, leaving the case unmodified.
	sdkOptions := map[string]json.RawMessage{}
	for name, raw := range test.SDKOptions {
		sdkOptions[name] = raw
	}
	for name, raw := range step.SDKOptions {
		sdkOptions[name] = raw
	}
	// Translate declared SDK inputs, never expected captured requests.
	for name, raw := range sdkOptions {
		switch name {
		case "max_tokens", "max_completion_tokens":
			var limit int
			decode(raw, &limit)
			gai.WithMaxGenerationTokens(limit)(options)
			gai.WithOpenAITokenLimitField(name)(options)
		case "reasoning_effort":
			var effort string
			decode(raw, &effort)
			gai.WithThinkingBudget(effort)(options)
		case "stream":
			var stream bool
			decode(raw, &stream)
			if stream != test.Stream {
				t.Fatal("SDK stream option disagrees with case")
			}
		case "stream_options":
			var fields map[string]bool
			decode(raw, &fields)
			if len(fields) != 1 {
				t.Fatal("unsupported stream_options")
			}
			usage, ok := fields["include_usage"]
			if !ok {
				t.Fatal("unsupported stream_options")
			}
			gai.WithOpenAIStreamUsage(usage)(options)
		case "extra_body":
			var fields map[string]json.RawMessage
			decode(raw, &fields)
			for key, value := range fields {
				extra[key] = value
			}
		case "tool_choice":
			var choice string
			if err := json.Unmarshal(raw, &choice); err != nil {
				var named struct {
					Type     string `json:"type"`
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				}
				decode(raw, &named)
				if named.Type != "function" || named.Function.Name == "" {
					t.Fatal("invalid named tool choice")
				}
				choice = named.Function.Name
			}
			gai.WithToolChoice(choice)(options)
		case "thinking", "tool_stream", "response_format":
			extra[name] = raw
		default:
			t.Fatalf("unsupported SDK option %q", name)
		}
	}
	gai.WithOpenAIExtraBody(extra)(options)
	request := gai.GenerationRequest{
		Model: test.Model, Instructions: gai.SystemMessage(gai.TextBlock(s.Instructions)),
		Options: options,
	}
	for _, tool := range s.Tools {
		request.Tools = append(request.Tools, gai.Tool{Name: tool.Function.Name, Description: tool.Function.Description, InputSchema: tool.Function.Parameters})
	}
	return request
}

// inputMessages is application behavior, independent of both SDK serializers.
// Tool IDs and arguments come from GAI's preceding response, never from an
// expected SDK request or a fixture's precomputed tool-result messages.
func (s sdkConversation) inputMessages(step sdkConversationStep, previous gai.Message) (gai.Dialog, error) {
	var blocks []gai.Block
	if step.Text != "" {
		blocks = append(blocks, gai.TextBlock(step.Text))
	}
	for _, path := range step.Images {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, gai.ImageBlock(data, http.DetectContentType(data)))
	}
	for _, path := range step.PDFs {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, gai.PDFBlock(data, filepath.Base(path)))
	}
	if step.Input == "user" {
		return gai.Dialog{{Role: gai.User, Blocks: blocks}}, nil
	}
	var messages gai.Dialog
	for _, block := range previous.Blocks {
		if block.BlockType != gai.ToolCall {
			continue
		}
		var call gai.ToolCallInput
		if err := json.Unmarshal([]byte(block.Content.String()), &call); err != nil {
			return nil, err
		}
		if call.Name == "get_image" && len(call.Parameters) == 0 && block.ID != "" && len(step.Images) > 0 {
			result := append([]gai.Block(nil), blocks...)
			for i := range result {
				result[i].ID = block.ID
			}
			messages = append(messages, gai.Message{Role: gai.ToolResult, Blocks: result})
			continue
		}
		if block.ID == "" {
			return nil, fmt.Errorf("tool call has no ID")
		}
		var value any
		switch call.Name {
		case "quote_order":
			items, ok := call.Parameters["items"].([]any)
			if !ok {
				return nil, fmt.Errorf("quote_order requires items")
			}
			value = map[string]any{"quoted": true, "item_count": len(items)}
		case "lookup_shipping":
			region, ok := call.Parameters["region"].(string)
			fees := map[string]int{"EU": 250, "US": 500}
			fee, known := fees[region]
			if !ok || !known {
				return nil, fmt.Errorf("unexpected shipping region")
			}
			value = map[string]any{"region": region, "fee_cents": fee}
		case "lookup_price":
			code, ok := call.Parameters["product_code"].(string)
			if !ok || len(call.Parameters) != 1 || len(s.Prices) == 0 {
				return nil, fmt.Errorf("unexpected price lookup arguments")
			}
			price, known := s.Prices[code]
			if known {
				value = struct {
					ProductCode string `json:"product_code"`
					PriceCents  int    `json:"price_cents"`
				}{code, price}
			} else {
				codes := make([]string, 0, len(s.Prices))
				for code := range s.Prices {
					codes = append(codes, code)
				}
				sort.Strings(codes)
				value = map[string]any{"error": "unknown_product", "suggested_product_code": codes[0]}
			}
		default:
			return nil, fmt.Errorf("unexpected synthetic tool %q", call.Name)
		}
		result, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		messages = append(messages, gai.Message{Role: gai.ToolResult, Blocks: []gai.Block{{ID: block.ID, BlockType: gai.Content, ModalityType: gai.Text, MimeType: "text/plain", Content: gai.Str(result)}}})
	}
	if len(messages) == 0 {
		return nil, fmt.Errorf("tool_results step requires preceding function calls")
	}
	return messages, nil
}

// checkResponse asserts application behavior, not SDK-shaped output objects or
// exact prose. Expected JSON and arguments come from the scenario's known input.
func (s sdkConversation) checkResponse(t *testing.T, step sdkConversationStep, response gai.Response, options gai.GenerationOptions) {
	t.Helper()
	if len(response.Candidates) != 1 {
		t.Fatalf("%s: expected one assistant candidate", step.Name)
	}
	wantFinish := gai.EndTurn
	if step.ExpectedToolCalls > 0 {
		wantFinish = gai.ToolUse
	}
	if response.FinishReason != wantFinish {
		t.Fatalf("%s: finish reason %v, want %v", step.Name, response.FinishReason, wantFinish)
	}
	var text, reasoning strings.Builder
	ids := map[string]bool{}
	arguments := map[string]json.RawMessage{}
	for _, block := range response.Candidates[0].Blocks {
		switch block.BlockType {
		case gai.Content:
			if block.ModalityType == gai.Text {
				text.WriteString(block.Content.String())
			}
		case gai.Thinking:
			reasoning.WriteString(block.Content.String())
		case gai.ToolCall:
			if block.ID == "" || ids[block.ID] {
				t.Fatalf("%s: missing or duplicate tool ID", step.Name)
			}
			ids[block.ID] = true
			var call gai.ToolCallInput
			if err := json.Unmarshal([]byte(block.Content.String()), &call); err != nil {
				t.Fatal(err)
			}
			declared := false
			for _, tool := range s.Tools {
				if tool.Function.Name != call.Name {
					continue
				}
				declared = true
				resolved, err := tool.Function.Parameters.Resolve(nil)
				if err != nil {
					t.Fatal(err)
				}
				if err := resolved.Validate(call.Parameters); err != nil {
					t.Fatalf("%s: invalid %s arguments: %v", step.Name, call.Name, err)
				}
			}
			if !declared {
				t.Fatalf("%s: undeclared tool %q", step.Name, call.Name)
			}
			data, err := json.Marshal(call.Parameters)
			if err != nil {
				t.Fatal(err)
			}
			arguments[call.Name] = data
		}
	}
	if len(ids) != step.ExpectedToolCalls {
		t.Fatalf("%s: got %d calls, want %d", step.Name, len(ids), step.ExpectedToolCalls)
	}
	if step.ExpectedArguments != nil {
		want, _ := json.Marshal(step.ExpectedArguments)
		got, _ := json.Marshal(arguments)
		if difference := sdkRequestBodyDifference(string(want), string(got)); difference != "" {
			t.Fatalf("%s arguments: %s", step.Name, difference)
		}
	}
	if step.ExpectedToolCalls == 0 && strings.TrimSpace(text.String()) == "" {
		t.Fatalf("%s: empty final answer", step.Name)
	}
	if step.ExpectedJSON != nil {
		if difference := sdkRequestBodyDifference(string(step.ExpectedJSON), text.String()); difference != "" {
			t.Fatalf("%s JSON result: %s", step.Name, difference)
		}
	}
	for _, term := range step.ExpectedText {
		if !strings.Contains(text.String(), term) {
			t.Fatalf("%s: answer missing %q", step.Name, term)
		}
	}
	extra, _ := options[gai.OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)
	var thinking struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(extra["thinking"], &thinking)
	effort, _ := options[gai.GenerationOptionThinkingBudget].(string)
	if raw, ok := extra["reasoning_effort"]; ok {
		_ = json.Unmarshal(raw, &effort)
	}
	if (thinking.Type == "disabled" || effort == "none") && reasoning.Len() > 0 {
		t.Fatalf("%s: reasoning returned while disabled", step.Name)
	}
}

func loadSDKRequestMatrix(t *testing.T) sdkRequestMatrix {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(sdkFixtureDirectory, "matrix.json"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decodeSDKRequestJSON(string(data)); err != nil {
		t.Fatalf("invalid matrix JSON: %v", err)
	}
	var matrix sdkRequestMatrix
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&matrix); err != nil {
		t.Fatal(err)
	}
	if matrix.Version != 1 || len(matrix.Scenarios) == 0 || len(matrix.Cases) == 0 {
		t.Fatal("invalid SDK request matrix")
	}
	for name, scenario := range matrix.Scenarios {
		if name == "" || len(scenario.Steps) == 0 {
			t.Fatal("scenario needs a name and steps")
		}
		steps := map[string]bool{}
		for index, step := range scenario.Steps {
			if step.Name == "" || steps[step.Name] {
				t.Fatal("missing or duplicate step name")
			}
			steps[step.Name] = true
			switch step.Input {
			case "user":
				if step.Text == "" && len(step.Images) == 0 && len(step.PDFs) == 0 {
					t.Fatal("user step needs content")
				}
			case "tool_results":
				if index == 0 || scenario.Steps[index-1].ExpectedToolCalls <= 0 {
					t.Fatal("tool_results step must follow a tool-calling step")
				}
			default:
				t.Fatalf("unknown step input %q", step.Input)
			}
		}
	}
	captures, names := map[string]bool{}, map[string]bool{}
	for _, test := range matrix.Cases {
		if _, ok := matrix.Scenarios[test.Scenario]; !ok {
			t.Fatalf("unknown scenario %q", test.Scenario)
		}
		if test.Name == "" || names[test.Name] || captures[test.Capture] {
			t.Fatal("missing or duplicate request case")
		}
		names[test.Name] = true
		captures[test.Capture] = true
		if filepath.Base(test.Capture) != test.Capture || filepath.Ext(test.Capture) != ".json" {
			t.Fatal("capture must be a local JSON filename")
		}
		if test.SDKVersion == "" || test.Model == "" || test.BaseURL == "" || test.SDK != "openai" {
			t.Fatalf("invalid profile %s", test.Name)
		}
	}
	entries, err := os.ReadDir(filepath.Join(sdkFixtureDirectory, "captures"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if !captures[entry.Name()] {
			t.Fatalf("capture %s is not declared in matrix.json", entry.Name())
		}
	}
	return matrix
}

func loadSDKExchanges(t *testing.T, test sdkRequestCase, steps int) []sdkHTTPExchange {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(sdkFixtureDirectory, "captures", test.Capture))
	if err != nil {
		t.Fatal(err)
	}
	var exchanges []sdkHTTPExchange
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&exchanges); err != nil {
		t.Fatal(err)
	}
	if len(exchanges) != steps {
		t.Fatalf("got %d exchanges, want %d", len(exchanges), steps)
	}
	for index, exchange := range exchanges {
		if exchange.Response == "" {
			t.Fatalf("empty response fixture %d", index)
		}
		if _, err := decodeSDKRequestJSON(exchange.Request); err != nil {
			t.Fatalf("invalid request fixture %d: %v", index, err)
		}
	}
	return exchanges
}

func sdkRequestBodyDifference(want, got string) string {
	expected, err := decodeSDKRequestJSON(want)
	if err != nil {
		return "invalid SDK request JSON: " + err.Error()
	}
	actual, err := decodeSDKRequestJSON(got)
	if err != nil {
		return "invalid GAI request JSON: " + err.Error()
	}
	return sdkJSONDifference("$", expected, actual)
}

// decodeSDKRequestJSON rejects ambiguous fixtures (duplicate keys, trailing
// documents) and retains numbers without float64 rounding. Embedded JSON in
// tool arguments is interpreted by sdkJSONDifference; ordinary text is not.
func decodeSDKRequestJSON(text string) (any, error) {
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	value, err := readSDKJSONValue(decoder)
	if err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, fmt.Errorf("expected one JSON document, got trailing input")
	}
	return value, nil
}

// readSDKJSONValue reads one value while checking object member uniqueness.
func readSDKJSONValue(decoder *json.Decoder) (any, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	delimiter, container := token.(json.Delim)
	if !container {
		return token, nil
	}
	switch delimiter {
	case '{':
		object := map[string]any{}
		for decoder.More() {
			key, err := decoder.Token()
			if err != nil {
				return nil, err
			}
			name, ok := key.(string)
			if !ok {
				return nil, fmt.Errorf("expected object key")
			}
			if _, exists := object[name]; exists {
				return nil, fmt.Errorf("duplicate object key %q", name)
			}
			value, err := readSDKJSONValue(decoder)
			if err != nil {
				return nil, err
			}
			object[name] = value
		}
		if _, err := decoder.Token(); err != nil {
			return nil, err
		}
		return object, nil
	case '[':
		values := []any{}
		for decoder.More() {
			value, err := readSDKJSONValue(decoder)
			if err != nil {
				return nil, err
			}
			values = append(values, value)
		}
		if _, err := decoder.Token(); err != nil {
			return nil, err
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unexpected JSON delimiter %q", delimiter)
	}
}

// sdkJSONDifference produces a stable first-mismatch path using JSON
// equivalence. Objects ignore member order; numbers compare exactly by value.
// JSON-encoded argument strings compare as JSON but remain string-typed fields.
func sdkJSONDifference(path string, want, got any) string {
	if reflect.TypeOf(want) != reflect.TypeOf(got) {
		return fmt.Sprintf("%s: want %T (%v), got %T (%v)", path, want, want, got, got)
	}
	switch expected := want.(type) {
	case map[string]any:
		actual := got.(map[string]any)
		keys := map[string]bool{}
		for key := range expected {
			keys[key] = true
		}
		for key := range actual {
			keys[key] = true
		}
		ordered := make([]string, 0, len(keys))
		for key := range keys {
			ordered = append(ordered, key)
		}
		sort.Strings(ordered)
		for _, key := range ordered {
			a, wantPresent := expected[key]
			b, gotPresent := actual[key]
			if !wantPresent {
				return path + ": unexpected field " + key
			}
			if !gotPresent {
				return path + ": missing field " + key
			}
			if difference := sdkJSONDifference(path+"."+key, a, b); difference != "" {
				return difference
			}
		}
	case []any:
		actual := got.([]any)
		if len(expected) != len(actual) {
			return fmt.Sprintf("%s: want array length %d, got %d", path, len(expected), len(actual))
		}
		for i := range expected {
			if difference := sdkJSONDifference(fmt.Sprintf("%s[%d]", path, i), expected[i], actual[i]); difference != "" {
				return difference
			}
		}
	case json.Number:
		a, okA := new(big.Rat).SetString(string(expected))
		b, okB := new(big.Rat).SetString(string(got.(json.Number)))
		if !okA || !okB || a.Cmp(b) != 0 {
			return fmt.Sprintf("%s: want %s, got %s", path, expected, got)
		}
	case string:
		actual := got.(string)
		if expected == actual {
			return ""
		}
		if strings.HasSuffix(path, ".arguments") {
			a, errA := decodeSDKRequestJSON(expected)
			b, errB := decodeSDKRequestJSON(actual)
			if errA == nil && errB == nil {
				return sdkJSONDifference(path+" (argument JSON)", a, b)
			}
		}
		return fmt.Sprintf("%s: want %q, got %q", path, expected, actual)
	default:
		if !reflect.DeepEqual(want, got) {
			return fmt.Sprintf("%s: want %#v, got %#v", path, want, got)
		}
	}
	return ""
}

// These local checks cover image input construction before live SDK captures
// are approved. They do not claim provider acceptance of tool-role images.
func testSDKMediaInputs(t *testing.T) {
	t.Helper()
	matrix := loadSDKRequestMatrix(t)
	for _, name := range []string{"user_images", "tool_images", "pdf_inputs"} {
		t.Run(name, func(t *testing.T) {
			scenario := matrix.Scenarios[name]
			previous := gai.Message{Role: gai.Assistant, Blocks: []gai.Block{{
				ID: "actual-tool-id", BlockType: gai.ToolCall,
				Content: gai.Str(`{"name":"get_image","parameters":{}}`),
			}}}
			for _, step := range scenario.Steps {
				if len(step.Images) == 0 && len(step.PDFs) == 0 {
					continue
				}
				messages, err := scenario.inputMessages(step, previous)
				if err != nil {
					t.Fatal(err)
				}
				transport := &sdkReplayTransport{responses: []string{`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`}}
				generator, err := gai.NewOpenAiGenerator(&http.Client{Transport: transport}, "http://offline.invalid", "dummy")
				if err != nil {
					t.Fatal(err)
				}
				if step.Input == "tool_results" {
					messages = append(gai.Dialog{{Role: gai.User, Blocks: []gai.Block{gai.TextBlock("Retrieve an image.")}}, previous}, messages...)
				}
				imageMessageIndex := len(messages) - 1
				// Keep the image-bearing message in history before another user turn.
				messages = append(messages, gai.Message{Role: gai.User, Blocks: []gai.Block{gai.TextBlock("Follow up.")}})
				if _, err := generator.Generate(t.Context(), gai.GenerationRequest{Model: "offline", Dialog: messages}); err != nil {
					t.Fatal(err)
				}
				body, err := decodeSDKRequestJSON(transport.requests[0])
				if err != nil {
					t.Fatal(err)
				}
				message := body.(map[string]any)["messages"].([]any)[imageMessageIndex].(map[string]any)
				role := "user"
				if step.Input == "tool_results" {
					role = "tool"
					if message["tool_call_id"] != "actual-tool-id" {
						t.Fatal("image tool result lost its call ID")
					}
				}
				want := []any{map[string]any{"type": "text", "text": step.Text}}
				for _, path := range step.Images {
					data, err := os.ReadFile(path)
					if err != nil {
						t.Fatal(err)
					}
					mime := "image/png"
					if filepath.Ext(path) == ".jpg" {
						mime = "image/jpeg"
					}
					want = append(want, map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)}})
				}
				for _, path := range step.PDFs {
					data, err := os.ReadFile(path)
					if err != nil {
						t.Fatal(err)
					}
					want = append(want, map[string]any{"type": "file", "file": map[string]any{"filename": filepath.Base(path), "file_data": "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(data)}})
				}
				if message["role"] != role || !reflect.DeepEqual(message["content"], want) {
					t.Fatalf("%s: image role, content order, MIME type, or bytes changed", step.Name)
				}
			}
		})
	}
}

func TestSDKRequestHarness(t *testing.T) {
	t.Run("top-level reasoning effort", func(t *testing.T) {
		for _, effort := range []string{"low", "high", "max", "none"} {
			t.Run(effort, func(t *testing.T) {
				raw, _ := json.Marshal(effort)
				request := (sdkConversation{}).generationRequest(t, sdkRequestCase{
					SDKOptions: map[string]json.RawMessage{"reasoning_effort": raw},
				}, sdkConversationStep{})
				if got := request.Options[gai.GenerationOptionThinkingBudget]; got != effort {
					t.Fatalf("thinking budget = %v, want %s", got, effort)
				}
			})
		}
	})
	t.Run("media inputs", func(t *testing.T) {
		testSDKMediaInputs(t)
	})
	t.Run("body comparison", func(t *testing.T) {
		for _, test := range []struct {
			name, want, got string
			equal           bool
		}{
			{"outer whitespace and key order", `{"a":1,"b":[]}`, "{ \"b\": [], \"a\": 1 }", true},
			{"argument string whitespace", `{"arguments":"{\"x\": 1}"}`, `{"arguments":"{\"x\":1}"}`, true},
			{"argument key order", `{"arguments":"{\"x\":1,\"y\":2}"}`, `{"arguments":"{\"y\":2,\"x\":1}"}`, true},
			{"argument value change", `{"arguments":"{\"x\":1}"}`, `{"arguments":"{\"x\":2}"}`, false},
			{"equivalent numbers", `{"n":1}`, `{"n":1.0e0}`, true},
			{"precise unequal decimals", `{"n":1}`, `{"n":1.0000000000000001}`, false},
			{"argument string type", `{"arguments":"{}"}`, `{"arguments":{}}`, false},
			{"null versus omitted", `{"content":null}`, `{}`, false},
			{"empty versus null", `{"content":""}`, `{"content":null}`, false},
			{"array order", `["a","b"]`, `["b","a"]`, false},
			{"reasoning text", `{"reasoning_content":"keep"}`, `{"reasoning_content":"changed"}`, false},
			{"tool ID", `{"id":"call-a"}`, `{"id":"call-b"}`, false},
			{"large integer", `{"n":9007199254740993}`, `{"n":9007199254740992}`, false},
			{"trailing JSON", `{}`, `{} {}`, false},
			{"duplicate object key", `{"x":1}`, `{"x":0,"x":1}`, false},
			{"invalid JSON", `{}`, `{`, false},
		} {
			t.Run(test.name, func(t *testing.T) {
				if equal := sdkRequestBodyDifference(test.want, test.got) == ""; equal != test.equal {
					t.Fatalf("equality = %v, want %v", equal, test.equal)
				}
			})
		}
	})
	t.Run("unexpected calls cannot reach network", func(t *testing.T) {
		transport := &sdkReplayTransport{}
		request, err := http.NewRequestWithContext(context.Background(), http.MethodPost, "https://must-not-resolve.invalid/chat/completions", strings.NewReader(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := transport.RoundTrip(request); err == nil {
			t.Fatal("accepted unrecorded HTTP call")
		}
		if len(transport.requests) != 1 {
			t.Fatal("unexpected request was not captured")
		}
	})
}
