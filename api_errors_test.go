package gai

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	anthropicshared "github.com/anthropics/anthropic-sdk-go/shared"
	oai "github.com/openai/openai-go/v3"
	oairesponses "github.com/openai/openai-go/v3/responses"

	"github.com/spachava753/gai/internal/zai"
)

type trackingReadCloser struct {
	io.Reader
	closed bool
}

func (r *trackingReadCloser) Close() error {
	r.closed = true
	return nil
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, time.July, 30, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name  string
		value string
		want  time.Duration
		ok    bool
	}{
		{name: "delta seconds", value: "120", want: 2 * time.Minute, ok: true},
		{name: "fractional seconds", value: "0.1", want: 100 * time.Millisecond, ok: true},
		{name: "HTTP date", value: now.Add(90 * time.Second).Format(http.TimeFormat), want: 90 * time.Second, ok: true},
		{name: "past HTTP date", value: now.Add(-time.Minute).Format(http.TimeFormat), want: 0, ok: true},
		{name: "zero", value: "0", want: 0, ok: true},
		{name: "negative", value: "-1"},
		{name: "not a number", value: "NaN"},
		{name: "infinite", value: "+Inf"},
		{name: "overflow", value: "1e20"},
		{name: "invalid", value: "later"},
		{name: "missing"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := parseRetryAfter(tt.value, now)
			if got != tt.want || ok != tt.ok {
				t.Fatalf("parseRetryAfter(%q) = (%s, %t), want (%s, %t)", tt.value, got, ok, tt.want, tt.ok)
			}
		})
	}
}

func TestAPIErrorRetryAfter(t *testing.T) {
	var nilError *ApiErr
	if _, ok := nilError.RetryAfter(); ok {
		t.Fatal("nil ApiErr reports a retry delay")
	}
	if _, ok := (&ApiErr{}).RetryAfter(); ok {
		t.Fatal("ApiErr without a retry delay reports one")
	}

	delay := time.Duration(0)
	apiErr := &ApiErr{RetryAfterDuration: &delay}
	if got, ok := apiErr.RetryAfter(); !ok || got != 0 {
		t.Fatalf("RetryAfter() = (%s, %t), want (0s, true)", got, ok)
	}
}

func TestAPIErrorRetryable(t *testing.T) {
	tests := []struct {
		name      string
		apiErr    *ApiErr
		retryable bool
	}{
		{name: "nil"},
		{name: "internal server error", apiErr: &ApiErr{StatusCode: http.StatusInternalServerError}, retryable: true},
		{name: "not implemented", apiErr: &ApiErr{Kind: APIErrorKindServer, StatusCode: http.StatusNotImplemented}},
		{name: "bad gateway", apiErr: &ApiErr{StatusCode: http.StatusBadGateway}, retryable: true},
		{name: "service unavailable", apiErr: &ApiErr{StatusCode: http.StatusServiceUnavailable}, retryable: true},
		{name: "gateway timeout", apiErr: &ApiErr{StatusCode: http.StatusGatewayTimeout}, retryable: true},
		{name: "HTTP version not supported", apiErr: &ApiErr{Kind: APIErrorKindServer, StatusCode: http.StatusHTTPVersionNotSupported}},
		{name: "provider overload", apiErr: &ApiErr{StatusCode: 529}, retryable: true},
		{name: "unknown provider 5xx", apiErr: &ApiErr{StatusCode: 598}, retryable: true},
		{name: "server kind without status", apiErr: &ApiErr{Kind: APIErrorKindServer}, retryable: true},
		{name: "rate limit kind", apiErr: &ApiErr{Kind: APIErrorKindRateLimit}, retryable: true},
		{name: "invalid request kind", apiErr: &ApiErr{Kind: APIErrorKindInvalidRequest}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.apiErr.Retryable(); got != tt.retryable {
				t.Fatalf("Retryable() = %t, want %t", got, tt.retryable)
			}
		})
	}
}

func TestClassifyHTTPStatus(t *testing.T) {
	tests := []struct {
		statusCode int
		want       APIErrorKind
	}{
		{statusCode: http.StatusOK, want: APIErrorKindUnknown},
		{statusCode: http.StatusBadRequest, want: APIErrorKindInvalidRequest},
		{statusCode: http.StatusUnauthorized, want: APIErrorKindAuthentication},
		{statusCode: http.StatusForbidden, want: APIErrorKindPermission},
		{statusCode: http.StatusNotFound, want: APIErrorKindNotFound},
		{statusCode: http.StatusRequestTimeout, want: APIErrorKindTimeout},
		{statusCode: http.StatusRequestEntityTooLarge, want: APIErrorKindRequestTooLarge},
		{statusCode: http.StatusTooManyRequests, want: APIErrorKindRateLimit},
		{statusCode: http.StatusInternalServerError, want: APIErrorKindServer},
		{statusCode: http.StatusBadGateway, want: APIErrorKindServer},
		{statusCode: http.StatusServiceUnavailable, want: APIErrorKindServiceUnavailable},
		{statusCode: http.StatusGatewayTimeout, want: APIErrorKindTimeout},
		{statusCode: 529, want: APIErrorKindServer},
	}

	for _, tt := range tests {
		if got := classifyHTTPStatus(tt.statusCode); got != tt.want {
			t.Errorf("classifyHTTPStatus(%d) = %q, want %q", tt.statusCode, got, tt.want)
		}
	}
}

func TestHTTPAPIErrorNormalization(t *testing.T) {
	tests := []struct {
		name          string
		provider      Provider
		statusCode    int
		body          string
		wantKind      APIErrorKind
		wantMessage   string
		wantRetryable bool
	}{
		{
			name:        "response fields do not override status",
			provider:    ProviderOpenAI,
			statusCode:  http.StatusBadRequest,
			body:        `{"error":{"type":"authentication_error","code":"invalid_api_key","message":"Invalid API key"}}`,
			wantKind:    APIErrorKindInvalidRequest,
			wantMessage: "Invalid API key",
		},
		{
			name:          "plain text does not override status",
			provider:      ProviderOpenRouter,
			statusCode:    http.StatusServiceUnavailable,
			body:          "  Invalid API key  ",
			wantKind:      APIErrorKindServiceUnavailable,
			wantMessage:   "Invalid API key",
			wantRetryable: true,
		},
		{
			name:          "status fallback without body",
			provider:      ProviderOpenRouter,
			statusCode:    http.StatusGatewayTimeout,
			wantKind:      APIErrorKindTimeout,
			wantRetryable: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := &http.Response{
				StatusCode: tt.statusCode,
				Body:       io.NopCloser(strings.NewReader(tt.body)),
			}
			got := mapHTTPAPIError(tt.provider, response)
			if got.Provider != tt.provider {
				t.Fatalf("Provider = %q, want %q", got.Provider, tt.provider)
			}
			if got.StatusCode != tt.statusCode {
				t.Fatalf("StatusCode = %d, want %d", got.StatusCode, tt.statusCode)
			}
			if got.Kind != tt.wantKind {
				t.Fatalf("Kind = %q, want %q", got.Kind, tt.wantKind)
			}
			if got.Message != tt.wantMessage {
				t.Fatalf("Message = %q, want %q", got.Message, tt.wantMessage)
			}
			if got.RawBody != tt.body {
				t.Fatalf("RawBody = %q, want %q", got.RawBody, tt.body)
			}
			if got.Retryable() != tt.wantRetryable {
				t.Fatalf("Retryable() = %t, want %t", got.Retryable(), tt.wantRetryable)
			}
			if got.Cause != nil {
				t.Fatalf("Cause = %v, want nil without an underlying error", got.Cause)
			}
		})
	}
}

func TestHTTPAPIErrorRetryAfter(t *testing.T) {
	const body = `{"error":{"message":"rate limited"}}`
	responseBody := &trackingReadCloser{Reader: strings.NewReader(body)}
	response := &http.Response{
		StatusCode: http.StatusTooManyRequests,
		Header: http.Header{
			"Retry-After":    []string{"7"},
			"Retry-After-Ms": []string{"250"},
		},
		Body: responseBody,
	}
	got := mapHTTPAPIError(ProviderCerebras, response)

	if !responseBody.closed {
		t.Fatal("response body was not closed")
	}

	delay, ok := got.RetryAfter()
	if !ok || delay != 250*time.Millisecond {
		t.Fatalf("RetryAfter() = (%s, %t), want (250ms, true)", delay, ok)
	}
}

func TestRetryAfterFromResponseValidation(t *testing.T) {
	now := time.Date(2026, time.July, 30, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name   string
		header http.Header
		want   time.Duration
		ok     bool
	}{
		{
			name: "negative milliseconds falls back to seconds",
			header: http.Header{
				"Retry-After-Ms": []string{"-1"},
				"Retry-After":    []string{"7"},
			},
			want: 7 * time.Second,
			ok:   true,
		},
		{
			name: "past HTTP date",
			header: http.Header{
				"Date":        []string{now.Format(http.TimeFormat)},
				"Retry-After": []string{now.Add(-time.Minute).Format(http.TimeFormat)},
			},
			want: 0,
			ok:   true,
		},
		{
			name:   "negative seconds",
			header: http.Header{"Retry-After": []string{"-1"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := retryAfterFromResponse(&http.Response{Header: tt.header})
			if got == nil {
				if tt.ok {
					t.Fatalf("retryAfterFromResponse() = nil, want %s", tt.want)
				}
				return
			}
			if !tt.ok || *got != tt.want {
				t.Fatalf("retryAfterFromResponse() = %s, want (%s, %t)", *got, tt.want, tt.ok)
			}
		})
	}
}

func TestAnthropicErrorClassification(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		typeCode   anthropicshared.ErrorType
		want       APIErrorKind
	}{
		{
			name:     "SDK overload type without status",
			typeCode: anthropicshared.ErrorTypeOverloadedError,
			want:     APIErrorKindOverloaded,
		},
		{
			name:       "Anthropic overload status without type",
			statusCode: 529,
			want:       APIErrorKindOverloaded,
		},
		{
			name:       "HTTP fallback",
			statusCode: http.StatusUnauthorized,
			want:       APIErrorKindAuthentication,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := classifyAnthropicError(tt.statusCode, tt.typeCode); got != tt.want {
				t.Fatalf("classifyAnthropicError() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestOpenAIErrorMappingUsesHTTPStatus(t *testing.T) {
	var cause oai.Error
	if err := json.Unmarshal([]byte(`{"code":"invalid_api_key","message":"Invalid API key"}`), &cause); err != nil {
		t.Fatalf("unmarshal SDK error: %v", err)
	}
	cause.StatusCode = http.StatusServiceUnavailable
	cause.Response = &http.Response{Header: http.Header{"Retry-After": []string{"9"}}}

	got := mapOpenAISDKError(ProviderOpenAI, &cause)
	if got == nil {
		t.Fatal("mapOpenAISDKError() = nil, want *ApiErr")
	}
	if got.Kind != APIErrorKindServiceUnavailable {
		t.Fatalf("Kind = %q, want %q", got.Kind, APIErrorKindServiceUnavailable)
	}
	if delay, ok := got.RetryAfter(); !ok || delay != 9*time.Second {
		t.Fatalf("RetryAfter() = (%s, %t), want (9s, true)", delay, ok)
	}
	if !errors.Is(got, &cause) {
		t.Fatal("mapped error does not wrap the SDK error")
	}
}

func TestOpenRouterSDKOverloadMapping(t *testing.T) {
	var cause oai.Error
	if err := json.Unmarshal([]byte(`{"code":503,"message":"Upstream request failed","metadata":{"error_type":"provider_overloaded"}}`), &cause); err != nil {
		t.Fatalf("unmarshal SDK error: %v", err)
	}
	cause.StatusCode = http.StatusServiceUnavailable
	cause.Response = &http.Response{Header: http.Header{"Retry-After": []string{"11"}}}

	got := mapOpenRouterError(&cause)
	if got == nil {
		t.Fatal("mapOpenRouterError() = nil, want *ApiErr")
	}
	if got.Kind != APIErrorKindOverloaded {
		t.Fatalf("Kind = %q, want %q", got.Kind, APIErrorKindOverloaded)
	}
	if !got.Retryable() {
		t.Fatal("Retryable() = false, want true")
	}
	if delay, ok := got.RetryAfter(); !ok || delay != 11*time.Second {
		t.Fatalf("RetryAfter() = (%s, %t), want (11s, true)", delay, ok)
	}
	if !errors.Is(got, &cause) {
		t.Fatal("mapped error does not wrap the SDK error")
	}
}

func TestResponsesFailureClassification(t *testing.T) {
	tests := []struct {
		code oairesponses.ResponseErrorCode
		want APIErrorKind
	}{
		{code: oairesponses.ResponseErrorCodeServerError, want: APIErrorKindServer},
		{code: oairesponses.ResponseErrorCodeRateLimitExceeded, want: APIErrorKindRateLimit},
		{code: oairesponses.ResponseErrorCodeVectorStoreTimeout, want: APIErrorKindTimeout},
		{code: oairesponses.ResponseErrorCodeBioPolicy, want: APIErrorKindContentPolicy},
		{code: oairesponses.ResponseErrorCodeImageContentPolicyViolation, want: APIErrorKindContentPolicy},
		{code: oairesponses.ResponseErrorCodeImageFileNotFound, want: APIErrorKindNotFound},
		{code: oairesponses.ResponseErrorCodeImageFileTooLarge, want: APIErrorKindRequestTooLarge},
		{code: oairesponses.ResponseErrorCodeInvalidImageMode, want: APIErrorKindInvalidRequest},
		{code: oairesponses.ResponseErrorCode("unknown"), want: APIErrorKindUnknown},
	}

	for _, tt := range tests {
		t.Run(string(tt.code), func(t *testing.T) {
			got := mapResponsesFailure(oairesponses.ResponseError{Code: tt.code, Message: "request failed"}, "")
			if got.Kind != tt.want {
				t.Fatalf("Kind = %q, want %q", got.Kind, tt.want)
			}
		})
	}
}

func TestResponsesErrorEventClassification(t *testing.T) {
	tests := []struct {
		code string
		want APIErrorKind
	}{
		{code: "invalid_api_key", want: APIErrorKindAuthentication},
		{code: string(oairesponses.ResponseErrorCodeRateLimitExceeded), want: APIErrorKindRateLimit},
		{code: "content_policy", want: APIErrorKindContentPolicy},
		{code: "unknown", want: APIErrorKindUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.code, func(t *testing.T) {
			if got := mapResponsesErrorEvent(tt.code, "request failed", ""); got.Kind != tt.want {
				t.Fatalf("Kind = %q, want %q", got.Kind, tt.want)
			}
		})
	}
}

func TestZAIErrorMapping(t *testing.T) {
	cause := &zai.ErrorStatusCode{
		StatusCode: http.StatusTooManyRequests,
		Response: zai.Error{
			Code:    1113,
			Message: "Quota exceeded",
		},
	}

	mapped := mapZAIError(cause)
	var apiErr *ApiErr
	if !errors.As(mapped, &apiErr) {
		t.Fatalf("mapZAIError() returned %T, want *ApiErr", mapped)
	}
	if apiErr.Kind != APIErrorKindRateLimit {
		t.Fatalf("Kind = %q, want %q", apiErr.Kind, APIErrorKindRateLimit)
	}
	if apiErr.Message != cause.Response.Message {
		t.Fatalf("Message = %q, want %q", apiErr.Message, cause.Response.Message)
	}
	if !errors.Is(mapped, cause) {
		t.Fatalf("errors.Is(%v, %v) = false", mapped, cause)
	}

	transportErr := errors.New("dial 503.example: connection reset")
	if mapped := mapZAIError(transportErr); !errors.Is(mapped, transportErr) {
		t.Fatalf("mapZAIError() = %v, want original error", mapped)
	}
}
