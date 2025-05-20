package gai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/genai"
)

func TestGeminiAPIErrorMapping(t *testing.T) {
	testCases := []struct {
		name       string
		statusCode int
		errorBody  string
		errChecker func(t *testing.T, err error)
	}{
		{
			name:       "500 Internal Server Error",
			statusCode: http.StatusInternalServerError,
			errorBody: `{
					"error": {
						"code": 500,
						"message": "An internal error has occurred. Please retry or report in https://developers.generativeai.google/guide/troubleshooting",
						"status": "INTERNAL"
					}
				}`,
			errChecker: func(t *testing.T, err error) {
				var apiErr ApiErr
				if !errors.As(err, &apiErr) {
					t.Fatalf("Expected error to be ApiErr, got %T: %v", err, err)
				}
				if apiErr.StatusCode != http.StatusInternalServerError {
					t.Errorf("Expected status code %d, got %d", http.StatusInternalServerError, apiErr.StatusCode)
				}
				if apiErr.Type != "api_error" {
					t.Errorf("Expected error type %q, got %q", "api_error", apiErr.Type)
				}
				if apiErr.Message == "" {
					t.Errorf("Expected non-empty error message")
				}
			},
		},
		{
			name:       "429 Rate Limit Error",
			statusCode: http.StatusTooManyRequests,
			errorBody: `{
				"error": {
					"code": 429,
					"message": "You exceeded your current quota. Go to https://aistudio.google.com/apikey to upgrade your quota tier, or submit a quota increase request in https://ai.google.dev/gemini-api/docs/rate-limits#request-rate-limit-increase",
					"status": "RESOURCE_EXHAUSTED",
					"details": [
						{
							"@type": "type.googleapis.com/google.rpc.QuotaFailure",
							"violations": [
								{
									"quotaMetric": "generativelanguage.googleapis.com/generate_content_paid_tier_input_token_count",
									"quotaId": "GenerateContentPaidTierInputTokensPerModelPerMinute",
									"quotaDimensions": {
										"model": "gemini-2.5-pro-exp",
										"location": "global"
									},
									"quotaValue": "2000000"
								}
							]
						},
						{
							"@type": "type.googleapis.com/google.rpc.Help",
							"links": [
								{
									"description": "Learn more about Gemini API quotas",
									"url": "https://ai.google.dev/gemini-api/docs/rate-limits"
								}
							]
						},
						{
							"@type": "type.googleapis.com/google.rpc.RetryInfo",
							"retryDelay": "43s"
						}
					]
				}
			}`,
			errChecker: func(t *testing.T, err error) {
				var rlErr RateLimitErr
				if !errors.As(err, &rlErr) {
					t.Fatalf("Expected error to be RateLimitErr, got %T: %v", err, err)
				}
				msg := rlErr.Error()
				if msg == "" || msg == "rate limit error: " {
					t.Errorf("Expected non-empty rate limit error message, got: %q", msg)
				}
			},
		},
		{
			name:       "400 Authentication Error",
			statusCode: http.StatusTooManyRequests,
			errorBody: `{
  "error": {
    "code": 400,
    "message": "API key not valid. Please pass a valid API key.",
    "status": "INVALID_ARGUMENT",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "API_KEY_INVALID",
        "domain": "googleapis.com",
        "metadata": {
          "service": "generativelanguage.googleapis.com"
        }
      },
      {
        "@type": "type.googleapis.com/google.rpc.LocalizedMessage",
        "locale": "en-US",
        "message": "API key not valid. Please pass a valid API key."
      }
    ]
  }
}`,
			errChecker: func(t *testing.T, err error) {
				var apiErr ApiErr
				if !errors.As(err, &apiErr) {
					t.Fatalf("Expected error to be ApiErr, got %T: %v", err, err)
				}
				if apiErr.StatusCode != http.StatusBadRequest {
					t.Errorf("Expected status code %d, got %d", http.StatusBadRequest, apiErr.StatusCode)
				}
				if apiErr.Type != "invalid_request_error" {
					t.Errorf("Expected error type %q, got %q", "api_error", apiErr.Type)
				}
				if apiErr.Message == "" {
					t.Errorf("Expected non-empty error message")
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.statusCode)
				_, _ = w.Write([]byte(tc.errorBody))
			}))
			defer server.Close()

			client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
				APIKey: "fake-api-key",
				HTTPOptions: genai.HTTPOptions{
					BaseURL: server.URL,
				},
			})
			if err != nil {
				t.Fatalf("Failed to create Gemini client: %v", err)
			}

			generator, err := NewGeminiGenerator(client, "test-model", "test instructions")
			if err != nil {
				t.Fatalf("Failed to create Gemini generator: %v", err)
			}

			dialog := Dialog{{
				Role: User,
				Blocks: []Block{{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str("Hello, world!"),
				}},
			}}

			_, err = generator.Generate(context.Background(), dialog, nil)

			tc.errChecker(t, err)
		})
	}
}
