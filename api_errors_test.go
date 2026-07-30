package gai

import (
	"errors"
	"net/http"
	"testing"

	"github.com/spachava753/gai/internal/zai"
)

func TestHTTPAPIErrorNormalization(t *testing.T) {
	tests := []struct {
		name        string
		statusCode  int
		body        string
		wantKind    APIErrorKind
		wantMessage string
	}{
		{
			name:        "nested OpenAI authentication error",
			statusCode:  http.StatusBadRequest,
			body:        `{"error":{"type":"invalid_request_error","code":"invalid_api_key","message":"Invalid API key"}}`,
			wantKind:    APIErrorKindAuthentication,
			wantMessage: "Invalid API key",
		},
		{
			name:        "plain text service unavailable",
			statusCode:  http.StatusServiceUnavailable,
			body:        "  temporarily unavailable  ",
			wantKind:    APIErrorKindServiceUnavailable,
			wantMessage: "temporarily unavailable",
		},
		{
			name:       "status fallback without body",
			statusCode: http.StatusGatewayTimeout,
			wantKind:   APIErrorKindTimeout,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := newHTTPAPIError(ProviderOpenRouter, tt.statusCode, tt.body)
			if got.Provider != ProviderOpenRouter {
				t.Fatalf("Provider = %q, want %q", got.Provider, ProviderOpenRouter)
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
			if tt.body != "" && got.Cause == nil {
				t.Fatal("Cause is nil for a non-empty response body")
			}
		})
	}
}

func TestResponsesStreamErrorClassification(t *testing.T) {
	tests := []struct {
		code string
		want APIErrorKind
	}{
		{code: "rate_limit_exceeded", want: APIErrorKindRateLimit},
		{code: "content_policy", want: APIErrorKindContentPolicy},
		{code: "invalid_image_mode", want: APIErrorKindInvalidRequest},
		{code: "unknown", want: APIErrorKindUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.code, func(t *testing.T) {
			if got := newResponsesStreamAPIError(tt.code, "request failed", ""); got.Kind != tt.want {
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
