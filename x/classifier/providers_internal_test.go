package classifier

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/spachava753/gai"
)

func TestConstructorValidation(t *testing.T) {
	for _, test := range []struct {
		name string
		url  string
		key  string
	}{
		{"missing key", "", ""},
		{"relative URL", "/v1", "key"},
		{"unsupported scheme", "file:///tmp/api", "key"},
		{"query", "https://example.test/api?x=1", "key"},
		{"empty query", "https://example.test/api?", "key"},
		{"fragment", "https://example.test/api#x", "key"},
		{"userinfo", "https://user:password@example.test", "key"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := newClientConfig(nil, test.url, "https://example.test", test.key)
			if err == nil {
				t.Fatal("accepted invalid constructor arguments")
			}
			if test.key == "" && !errors.Is(err, gai.ErrMissingAPIKey) {
				t.Fatalf("missing key error = %v", err)
			}
		})
	}
	t.Run("defaults do not select a model", func(t *testing.T) {
		client, err := newClientConfig(nil, "", "https://example.test/v1", "key")
		if err != nil || client.client != http.DefaultClient || client.baseURL != "https://example.test/v1" {
			t.Fatalf("connection = %+v, %v", client, err)
		}
	})
}

func TestRetryMetadata(t *testing.T) {
	for _, test := range []struct {
		name    string
		headers http.Header
		delay   time.Duration
		present bool
	}{
		{"seconds", http.Header{"Retry-After": {"2"}}, 2 * time.Second, true},
		{"fractional", http.Header{"Retry-After": {"0.25"}}, 250 * time.Millisecond, true},
		{"milliseconds take priority", http.Header{"Retry-After": {"2"}, "Retry-After-Ms": {"25"}}, 25 * time.Millisecond, true},
		{"date", http.Header{"Date": {"Sun, 11 Oct 2026 20:00:00 GMT"}, "Retry-After": {"Sun, 11 Oct 2026 20:00:03 GMT"}}, 3 * time.Second, true},
		{"past", http.Header{"Date": {"Sun, 11 Oct 2026 20:00:00 GMT"}, "Retry-After": {"Sun, 11 Oct 2026 19:00:00 GMT"}}, 0, true},
		{"negative", http.Header{"Retry-After": {"-1"}}, 0, false},
		{"overflow", http.Header{"Retry-After": {"999999999999999999999"}}, 0, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			delay := retryAfter(test.headers)
			if (delay != nil) != test.present || (delay != nil && *delay != test.delay) {
				t.Fatalf("delay = %v, want %v (%v)", delay, test.delay, test.present)
			}
		})
	}
}

func TestStrictJSON(t *testing.T) {
	for _, data := range []string{`{"a":1,"a":2}`, `{"a":{"x":1,"x":2}}`, `[{"x":1,"x":2}]`, `{} {}`, ``, `   `, `{"a":`} {
		t.Run(data, func(t *testing.T) {
			if err := checkJSON(json.RawMessage(data)); err == nil {
				t.Fatal("accepted malformed/duplicate JSON")
			}
		})
	}
	t.Run("excessive nesting", func(t *testing.T) {
		data := strings.Repeat("[", 10002) + "0" + strings.Repeat("]", 10002)
		if err := checkJSON(json.RawMessage(data)); err == nil {
			t.Fatal("accepted unbounded token recursion")
		}
	})
	t.Run("empty object input is an error rather than panic", func(t *testing.T) {
		if _, err := object(json.RawMessage("   ")); err == nil {
			t.Fatal("accepted whitespace as object")
		}
	})
}

func TestStatusClassification(t *testing.T) {
	for _, test := range []struct {
		status int
		kind   gai.APIErrorKind
	}{
		{401, gai.APIErrorKindAuthentication}, {403, gai.APIErrorKindPermission},
		{404, gai.APIErrorKindNotFound}, {408, gai.APIErrorKindTimeout},
		{413, gai.APIErrorKindRequestTooLarge}, {422, gai.APIErrorKindInvalidRequest},
		{429, gai.APIErrorKindRateLimit}, {500, gai.APIErrorKindServer},
		{503, gai.APIErrorKindServiceUnavailable}, {504, gai.APIErrorKindTimeout},
		{529, gai.APIErrorKindOverloaded},
	} {
		t.Run(http.StatusText(test.status), func(t *testing.T) {
			if actual := httpErrorKind(test.status); actual != test.kind {
				t.Fatalf("kind = %v, want %v", actual, test.kind)
			}
		})
	}
}
