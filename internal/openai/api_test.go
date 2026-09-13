package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestGeneratedClientUpToDate keeps the checked-in client tied to the schema,
// overlay, configuration, and pinned tool version. Generation is offline and
// writes only to a temporary file, never over a developer's working copy.
func TestGeneratedClientUpToDate(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	target := filepath.Join(t.TempDir(), "client.gen.go")
	cmd := exec.CommandContext(ctx, filepath.Join(runtime.GOROOT(), "bin", "go"),
		"tool", "oapi-codegen", "-config", "config.yaml", "-o", target, "api.yaml")
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOPROXY=off", "GOSUMDB=off", "LIVE_TESTS=")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("generate client: %v\n%s", err, output)
	}
	generated, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	checkedIn, err := os.ReadFile("client.gen.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(generated, checkedIn) {
		t.Fatal("generated client is stale; run go generate ./internal/openai from the repository root")
	}
}

func roundTrip(t *testing.T, input string, value any) {
	t.Helper()
	if err := json.Unmarshal([]byte(input), value); err != nil {
		t.Fatal(err)
	}
	output, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	decode := func(s string) any {
		d := json.NewDecoder(strings.NewReader(s))
		d.UseNumber()
		var v any
		if err := d.Decode(&v); err != nil {
			t.Fatal(err)
		}
		return v
	}
	if !reflect.DeepEqual(decode(input), decode(string(output))) {
		t.Fatalf("wire value changed:\ninput: %s\noutput: %s", input, output)
	}
}
func TestRequestRoundTrips(t *testing.T) {
	cases := map[string]string{
		"nullable options":                `{"model":"m","messages":[],"temperature":null,"tools":null,"tool_choice":null,"stop":null,"stream_options":null,"max_tokens":null,"metadata":null}`,
		"no defaults":                     `{"model":"future-model","messages":[{"role":"user","content":"Hi"}]}`,
		"zero false null":                 `{"model":"m","messages":[],"temperature":0,"max_tokens":0,"stream":false,"store":false,"thinking":{"clear_thinking":false,"keep":null}}`,
		"both limit fields representable": `{"model":"m","messages":[],"max_tokens":8,"max_completion_tokens":32}`,
		"limits left to provider":         `{"model":"future-model","messages":[],"temperature":9,"top_p":2,"n":12,"user_id":"x","request_id":"x","reasoning_effort":"future-effort","stop":["a","b","c","d","e","f"]}`,
		"scalar stop":                     `{"model":"m","messages":[],"stop":"END"}`,
		"tool and reasoning history":      `{"model":"m","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Price?"},{"role":"assistant","content":"","reasoning_content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup_price","arguments":"{\"product_code\":\"ABC123\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"19.99"}],"thinking":{"type":"enabled","keep":"all","clear_thinking":false}}`,
		"null and absent content":         `{"model":"m","messages":[{"role":"assistant","content":null},{"role":"assistant"}]}`,
		"vision tool history":             `{"model":"glm-5.3-flash","messages":[{"role":"user","content":[{"type":"text","text":"Price?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]},{"role":"assistant","content":"Red","reasoning_content":"Look up RED.","tool_calls":[{"id":"call_1","index":0,"type":"function","function":{"name":"lookup_price","arguments":"{\"product_code\":\"RED\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"19.99"}]}`,
		"media across roles":              `{"model":"m","messages":[{"role":"developer","content":[{"type":"text","text":"Instruction"}]},{"role":"tool","tool_call_id":"c","content":[{"type":"image_url","image_url":{"url":"ms://image-id","detail":"original"}},{"type":"video_url","video_url":"data:video/mp4;base64,AA=="},{"type":"input_audio","input_audio":{"format":"wav","data":"AA=="}}]}]}`,
		"file layouts":                    `{"model":"m","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"openai-file"}},{"type":"file","file":{"file_url":"https://example.com/a.pdf"}},{"type":"file","file":{"file_data":"data:application/pdf;base64,AA==","filename":"a.pdf"}},{"type":"file","file_id":"deepseek-file","file_data":"AA==","filename":"image.png"},{"type":"file_url","file_url":{"url":"https://example.com/legacy.pdf"}}]}]}`,
		"positioned tools":                `{"model":"kimi-k3","messages":[{"role":"user","content":"Calculate"},{"role":"system","tools":[{"type":"function","function":{"name":"calc","parameters":{"type":"object","properties":{},"additionalProperties":false}}}]}],"reasoning_effort":"max"}`,
		"tool definitions":                `{"model":"m","messages":[],"tools":[{"type":"function","function":{"name":"lookup","strict":false,"parameters":{"type":"object","$defs":{"value":{"type":"integer","maximum":9007199254740993}},"properties":{"x":{"$ref":"#/$defs/value"}}}}},{"type":"builtin_function","function":{"name":"$web_search"}},{"type":"custom","custom":{"name":"shell","format":{"type":"grammar","grammar":{"syntax":"regex","definition":".*"}}}},{"type":"web_search","web_search":{"search_engine":"future-engine","count":100}},{"type":"retrieval","retrieval":{"knowledge_id":"k"}}]}`,
		"named choice":                    `{"model":"m","messages":[],"tool_choice":{"type":"function","function":{"name":"lookup"}}}`,
		"allowed tools":                   `{"model":"m","messages":[],"tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"required","tools":[{"type":"function","function":{"name":"lookup"}},{"type":"custom","custom":{"name":"shell"}}]}}}`,
		"open choice":                     `{"model":"m","messages":[],"tool_choice":"future-choice"}`,
		"output and cache":                `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"A","prompt_cache_breakpoint":{"mode":"explicit"},"cache_control":{"type":"ephemeral"}}]},{"role":"assistant","content":"prefix","partial":true,"prefix":true,"audio":{"id":"audio_1"}}],"response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","additionalProperties":false}}},"prompt_cache_options":{"mode":"explicit","ttl":"30m"},"prediction":{"type":"content","content":[{"type":"text","text":"expected"}]},"audio":{"voice":"new-voice","format":"wav"},"modalities":["text","audio"]}`,
		"unknown JSON survives":           `{"model":"m","messages":[{"role":"future-role","provider_field":{"opaque":[null,9007199254740993]},"content":[{"type":"future-part","opaque":{"signature":"abc"}}]}],"provider_option":{"x":9007199254740993}}`,
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) { roundTrip(t, input, &ChatCompletionRequest{}) })
	}
	// IDs are not silently truncated or rejected by one provider's length rule.
	input := `{"model":"m","messages":[],"prompt_cache_key":"` + strings.Repeat("x", 1024) + `","user_id":"` + strings.Repeat("y", 1024) + `"}`
	roundTrip(t, input, &ChatCompletionRequest{})
}

func TestResponseRoundTrips(t *testing.T) {
	cases := map[string]string{
		"minimal":               `{"choices":[]}`,
		"null content":          `{"choices":[{"message":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":"stop"}],"system_fingerprint":null,"usage":null}`,
		"string arguments":      `{"choices":[{"index":0,"message":{"role":"assistant","content":"","reasoning_content":"Lookup","tool_calls":[{"id":"c","type":"function","index":0,"function":{"name":"lookup","arguments":"{\"x\":9007199254740993}"}}]},"finish_reason":"tool_calls"}]}`,
		"object arguments":      `{"choices":[{"message":{"tool_calls":[{"id":"c","type":"function","function":{"name":"lookup","arguments":{"x":9007199254740993}}}]}}]}`,
		"opaque reasoning":      `{"choices":[{"message":{"reasoning_details":[{"type":"reasoning.text","index":0,"text":"Thinking","signature":null,"future":{"x":9007199254740993}},{"type":"reasoning.encrypted","id":"r","data":"ciphertext","signature":"sig","summary":[{"text":"summary"}]}]}}]}`,
		"metadata":              `{"id":"c","request_id":"r","model":"m","cost":"0","metadata":{"nested":[1,null]},"moderation":{"input":{"flagged":false}},"web_search":[{"title":"Source","link":"https://example.com","future":true}],"future_response":{"x":9007199254740993}}`,
		"audio and annotations": `{"choices":[{"message":{"audio":{"id":"a","data":"AA==","expires_at":123,"transcript":"Hi"},"refusal":null,"annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com"}}]}}]}`,
		"usage details":         `{"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"cached_tokens":64,"prompt_cache_hit_tokens":64,"prompt_cache_miss_tokens":36,"prompt_tokens_details":{"cached_tokens":64,"cache_write_tokens":36,"future":1},"completion_tokens_details":{"reasoning_tokens":10,"future":2}}}`,
		"logprobs":              `{"choices":[{"finish_reason":"future_reason","logprobs":{"content":[{"token":"a","logprob":-1,"bytes":[97],"top_logprobs":[{"token":"b","logprob":-2,"bytes":null}]}],"reasoning_content":null}}]}`,
		"HTTP success error":    `{"error":{"code":1234,"message":"quota"}}`,
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) { roundTrip(t, input, &ChatCompletionResponse{}) })
	}
}

func TestChunkRoundTrips(t *testing.T) {
	cases := []string{
		`{"choices":[],"usage":{"completion_tokens":8}}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"completion_tokens":8}}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":null,"content":"","refusal":null},"finish_reason":null}],"usage":null}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":"}},{"index":1,"function":{"arguments":"}"}}]}}]}`,
		`{"choices":[{"delta":{"reasoning_details":[{"index":0,"type":"reasoning.encrypted","signature":"signature-only"}]}}]}`,
		`{"cost":"0"}`,
		`{"error":{"code":"quota","message":"balance"}}`,
		`{"type":"error","message":"upstream failed","future":true}`,
	}
	for _, input := range cases {
		roundTrip(t, input, &ChatCompletionChunk{})
	}
}

func TestErrorRoundTrips(t *testing.T) {
	for _, input := range []string{
		`{"error":{"code":1234,"message":"quota"}}`,
		`{"error":{"code":"001234","param":null,"type":"rate_limit","message":"quota","metadata":{"upstream":{"code":9007199254740993}}}}`,
		`{"error":{"code":null,"message":"failed"}}`,
		`{"error":"upstream failed"}`,
		`{"message":"upstream failed","type":"error","code":400}`,
	} {
		roundTrip(t, input, &ErrorResponse{})
	}
}

func TestRawStreaming(t *testing.T) {
	var calls atomic.Int32
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/v1/chat/completions" || r.Method != "POST" {
			t.Errorf("bad request %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer fixture" {
			t.Error("missing request editor header")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("x-request-id", "request-1")
		io.WriteString(w, "data: {\"choices\":[]}\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		io.WriteString(w, "data: [DONE]\n\ndata: {\"cost\":\"0\"}\n\n")
	}))
	defer server.Close()
	defer close(release)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	client, err := NewClient(server.URL+"/v1", WithHTTPClient(server.Client()), WithRequestEditorFn(func(_ context.Context, r *http.Request) error {
		r.Header.Set("Authorization", "Bearer fixture")
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.CreateChatCompletion(ctx, nil, ChatCompletionRequest{Model: "m", Messages: []Message{}})
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	reader := bufio.NewReader(response.Body)
	line, err := reader.ReadString('\n')
	if err != nil || line != "data: {\"choices\":[]}\n" {
		t.Fatalf("first event before EOF: %q %v", line, err)
	}
	if response.Header.Get("x-request-id") != "request-1" {
		t.Fatal("lost header")
	}
	select {
	case release <- struct{}{}:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	rest, err := io.ReadAll(reader)
	if err != nil || !strings.Contains(string(rest), "[DONE]") || !strings.Contains(string(rest), `"cost":"0"`) {
		t.Fatalf("lost trailing events: %q %v", rest, err)
	}
	if calls.Load() != 1 {
		t.Fatalf("POST replayed %d times", calls.Load())
	}
}

func TestArbitraryErrorBodies(t *testing.T) {
	for _, contentType := range []string{"text/html", "application/x-provider-error", ""} {
		t.Run(contentType, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header()["Content-Type"] = []string{contentType}
				w.Header().Set("Retry-After", "120")
				w.WriteHeader(502)
				io.WriteString(w, "bad gateway")
			}))
			defer server.Close()
			client, err := NewClientWithResponses(server.URL, WithHTTPClient(server.Client()))
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.CreateChatCompletionWithResponse(t.Context(), nil, ChatCompletionRequest{Model: "m", Messages: []Message{}})
			if err != nil {
				t.Fatal(err)
			}
			if response.StatusCode() != 502 || string(response.Body) != "bad gateway" || response.HTTPResponse.Header.Get("Retry-After") != "120" {
				t.Fatalf("lost error: %+v", response)
			}
		})
	}
}

func TestHTTPHeadersAndRequestBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Opencode-Session") != "session" {
			t.Error("missing per-request session header")
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if string(body["max_tokens"]) != "8" || body["max_completion_tokens"] != nil || body["thinking"] != nil || body["stream"] != nil {
			t.Errorf("unwanted defaults or token field selection: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("x-request-id", "request-1")
		w.Header().Set("Msh-Request-Timestamp", "123")
		w.Header().Set("Msh-Request-Signature", "signature")
		_, _ = io.WriteString(w, `{"choices":[]}`)
	}))
	defer server.Close()
	client, err := NewClientWithResponses(server.URL, WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	var request ChatCompletionRequest
	if err := json.Unmarshal([]byte(`{"model":"m","messages":[],"max_tokens":8}`), &request); err != nil {
		t.Fatal(err)
	}
	session := "session"
	response, err := client.CreateChatCompletionWithResponse(t.Context(), &CreateChatCompletionParams{XOpencodeSession: &session}, request)
	if err != nil {
		t.Fatal(err)
	}
	if response.JSON200 == nil || response.Headers200 == nil {
		t.Fatalf("missing typed success response: %+v", response)
	}
	headers := response.Headers200
	if headers.XRequestId == nil || *headers.XRequestId != "request-1" ||
		headers.MshRequestTimestamp == nil || *headers.MshRequestTimestamp != "123" ||
		headers.MshRequestSignature == nil || *headers.MshRequestSignature != "signature" {
		t.Fatalf("lost declared success headers: %+v", headers)
	}
}

func TestTypedUnionPrecision(t *testing.T) {
	var args ToolArguments
	if err := json.Unmarshal([]byte(`{"x":9007199254740993}`), &args); err != nil {
		t.Fatal(err)
	}
	object, err := args.AsJSONObject()
	if err != nil {
		t.Fatal(err)
	}
	if err = args.FromJSONObject(object); err != nil {
		t.Fatal(err)
	}
	output, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	if string(output) != `{"x":9007199254740993}` {
		t.Fatalf("typed union access changed arguments: %s", output)
	}
}

func TestSummaryStates(t *testing.T) {
	for _, input := range []string{`{}`, `{"summary":null}`, `{"summary":{"x":9007199254740993}}`} {
		roundTrip(t, input, &ReasoningDetail{})
	}
}

func TestNumberPrecision(t *testing.T) {
	roundTrip(t, `{"model":"m","messages":[],"temperature":0.1234567890123456}`, &ChatCompletionRequest{})
}
