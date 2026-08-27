package gai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func testOpenRouterGeneratorSurfacesProviderOverload(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantPartial bool
	}{
		{
			name: "top-level error",
			body: `{"error":{"code":503,"message":"Invalid API key","metadata":{"error_type":"provider_overloaded"}}}`,
		},
		{
			name:        "choice error after partial output",
			body:        `{"id":"gen-123","object":"chat.completion","created":1,"model":"test/model","choices":[{"index":0,"message":{"role":"assistant","content":"partial output"},"finish_reason":"error","error":{"code":503,"message":"Upstream request failed","metadata":{"error_type":"provider_overloaded"}}}]}`,
			wantPartial: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			generator := newOpenRouterTestGenerator(t, server)
			response, err := generator.Generate(context.Background(), GenerationRequest{
				Model: "test/model",
				Dialog: Dialog{{
					Role:   User,
					Blocks: []Block{TextBlock("hello")},
				}},
			})
			if tt.wantPartial {
				if len(response.Candidates) != 1 || len(response.Candidates[0].Blocks) != 1 ||
					response.Candidates[0].Blocks[0].Content.String() != "partial output" {
					t.Fatalf("partial response = %+v", response)
				}
			} else if len(response.Candidates) != 0 {
				t.Fatalf("top-level error response = %+v", response)
			}

			var apiErr *ApiErr
			if !errors.As(err, &apiErr) {
				t.Fatalf("Generate() error = %v, want *ApiErr", err)
			}
			if apiErr.Provider != ProviderOpenRouter {
				t.Fatalf("Provider = %q, want %q", apiErr.Provider, ProviderOpenRouter)
			}
			if apiErr.StatusCode != http.StatusServiceUnavailable {
				t.Fatalf("StatusCode = %d, want %d", apiErr.StatusCode, http.StatusServiceUnavailable)
			}
			if apiErr.Kind != APIErrorKindOverloaded {
				t.Fatalf("Kind = %q, want %q", apiErr.Kind, APIErrorKindOverloaded)
			}
			if !apiErr.Retryable() {
				t.Fatal("Retryable() = false, want true")
			}
		})
	}
}

func testClassifyOpenRouterErrorCurrentTypedCodes(t *testing.T) {
	tests := []struct {
		errorType string
		want      APIErrorKind
	}{
		{errorType: "invalid_prompt", want: APIErrorKindInvalidRequest},
		{errorType: "image_too_large", want: APIErrorKindRequestTooLarge},
	}
	for _, tt := range tests {
		t.Run(tt.errorType, func(t *testing.T) {
			if got := classifyOpenRouterError(http.StatusBadRequest, tt.errorType); got != tt.want {
				t.Fatalf("classifyOpenRouterError() = %q, want %q", got, tt.want)
			}
		})
	}
}
