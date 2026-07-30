package gai

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	anthropicshared "github.com/anthropics/anthropic-sdk-go/shared"
	oai "github.com/openai/openai-go/v3"
	oairesponses "github.com/openai/openai-go/v3/responses"

	"github.com/spachava753/gai/internal/zai"
)

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
			got := mapHTTPAPIError(tt.provider, tt.statusCode, tt.body)
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
			if got.Retryable() != tt.wantRetryable {
				t.Fatalf("Retryable() = %t, want %t", got.Retryable(), tt.wantRetryable)
			}
			if got.Cause != nil {
				t.Fatalf("Cause = %v, want nil without an underlying error", got.Cause)
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

	got := mapOpenAISDKError(ProviderOpenAI, &cause)
	if got == nil {
		t.Fatal("mapOpenAISDKError() = nil, want *ApiErr")
	}
	if got.Kind != APIErrorKindServiceUnavailable {
		t.Fatalf("Kind = %q, want %q", got.Kind, APIErrorKindServiceUnavailable)
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
