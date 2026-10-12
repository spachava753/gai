package perplexity

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

// TestGeneratedClientUpToDate checks regeneration offline without overwriting
// the checked-in client or refreshing fixtures from the provider.
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
		t.Fatal("generated client is stale; run go generate in this package")
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
	decode := func(data string) any {
		decoder := json.NewDecoder(strings.NewReader(data))
		decoder.UseNumber()
		var value any
		if err := decoder.Decode(&value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	if !reflect.DeepEqual(decode(input), decode(string(output))) {
		t.Fatalf("wire value changed:\ninput: %s\noutput: %s", input, output)
	}
}

func TestWireRoundTrips(t *testing.T) {
	for _, test := range []struct {
		name  string
		input string
		value any
	}{
		{"request", `{"model":"future-model","state":{"id":9007199254740993,"nested":[null,false]},"questions":{"p":{"type":"noul","instructions":["Check",{"n":0.123456789012345678901}],"criteria":{"true":"yes"}},"c":{"type":"choice","instructions":"Choose","criteria":{"billing":null,"other":{"examples":[]}}},"s":{"type":"score","instructions":"Rate","criteria":["low",{"label":"high"}]}},"future":{"n":9007199254740993}}`, &ClassificationRequest{}},
		{"response", `{"model":"resolved","answers":{"p":{"type":"noul","noul":0.123456789012345678901,"future":9007199254740993},"c":{"type":"choice","choice":"billing","confidence":0,"probabilities":{"billing":1,"other":0}},"s":{"type":"score","score":1.1000,"confidence":0.9876543210987654321,"probabilities":{"0":0.1,"1":0.7,"2":0.2},"legend":{"0":"low","1":{"label":"medium"},"2":null}}},"usage":{"input_tokens":12,"output_tokens":0,"future_cost":0.1234567890123456789}}`, &ClassificationResponse{}},
		{"image state", `{"model":"m","state":[{"record":{"id":9007199254740993}},{"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}}],"questions":{"p":{"type":"noul","criteria":{"true":"yes"}}}}`, &ClassificationRequest{}},
		{"null usage", `{"model":"resolved","answers":{},"usage":null}`, &ClassificationResponse{}},
		{"missing usage", `{"model":"resolved","answers":{}}`, &ClassificationResponse{}},
		{"null count", `{"model":"resolved","answers":{},"usage":{"input_tokens":null}}`, &ClassificationResponse{}},
		{"error", `{"error":{"message":"slow down","code":9007199254740993},"unknown":null}`, &ErrorResponse{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			roundTrip(t, test.input, test.value)
		})
	}
}

func TestRawHTTP(t *testing.T) {
	var calls atomic.Int32
	const payload = `{"error":{"message":"slow down","code":9007199254740993}}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.Method != http.MethodPost || r.URL.Path != "/prefix/decisions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer overridden" || r.Header.Get("X-Call") != "one" || r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("unexpected headers: %v", r.Header)
		}
		data, err := io.ReadAll(r.Body)
		if err != nil || !bytes.Contains(data, []byte(`9007199254740993`)) {
			t.Errorf("request body = %s, %v", data, err)
		}
		w.Header().Set("Retry-After", "2")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, payload)
	}))
	defer server.Close()
	client, err := NewClient(server.URL+"/prefix", WithHTTPClient(server.Client()), WithRequestEditorFn(func(_ context.Context, r *http.Request) error {
		r.Header.Set("Authorization", "Bearer default")
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	request := ClassificationRequest{
		Model: "future-model", State: json.RawMessage(`{"id":9007199254740993}`),
		Questions: map[string]json.RawMessage{"p": json.RawMessage(`{"type":"noul","instructions":"Check"}`)},
	}
	response, err := client.Classify(t.Context(), request, func(_ context.Context, r *http.Request) error {
		r.Header.Set("Authorization", "Bearer overridden")
		r.Header.Set("X-Call", "one")
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil || string(data) != payload || response.StatusCode != 429 || response.Header.Get("Retry-After") != "2" || calls.Load() != 1 {
		t.Fatalf("raw response changed: %s, %v, status=%d, calls=%d", data, err, response.StatusCode, calls.Load())
	}
}
