package gai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"
)

func TestGeminiResponseError(t *testing.T) {
	tests := []struct {
		name       string
		response   *genai.GenerateContentResponse
		wantPolicy bool
		wantLimit  bool
		contains   string
	}{
		{name: "successful stop", response: geminiResponseWithFinishReason(genai.FinishReasonStop)},
		{name: "max tokens", response: geminiResponseWithFinishReason(genai.FinishReasonMaxTokens), wantLimit: true},
		{
			name: "blocked prompt",
			response: &genai.GenerateContentResponse{PromptFeedback: &genai.GenerateContentResponsePromptFeedback{
				BlockReason:        genai.BlockedReasonSafety,
				BlockReasonMessage: "unsafe prompt",
			}},
			wantPolicy: true,
			contains:   "unsafe prompt",
		},
		{name: "candidate safety", response: geminiResponseWithFinishReason(genai.FinishReasonSafety), wantPolicy: true, contains: "SAFETY"},
		{name: "candidate blocklist", response: geminiResponseWithFinishReason(genai.FinishReasonBlocklist), wantPolicy: true, contains: "BLOCKLIST"},
		{name: "candidate prohibited content", response: geminiResponseWithFinishReason(genai.FinishReasonProhibitedContent), wantPolicy: true, contains: "PROHIBITED_CONTENT"},
		{name: "candidate SPII", response: geminiResponseWithFinishReason(genai.FinishReasonSPII), wantPolicy: true, contains: "SPII"},
		{name: "candidate recitation", response: geminiResponseWithFinishReason(genai.FinishReasonRecitation), wantPolicy: true, contains: "RECITATION"},
		{name: "candidate image safety", response: geminiResponseWithFinishReason(genai.FinishReasonImageSafety), wantPolicy: true, contains: "IMAGE_SAFETY"},
		{name: "candidate image prohibited content", response: geminiResponseWithFinishReason(genai.FinishReasonImageProhibitedContent), wantPolicy: true, contains: "IMAGE_PROHIBITED_CONTENT"},
		{name: "candidate image recitation", response: geminiResponseWithFinishReason(genai.FinishReasonImageRecitation), wantPolicy: true, contains: "IMAGE_RECITATION"},
		{name: "malformed function call", response: geminiResponseWithFinishReason(genai.FinishReasonMalformedFunctionCall), contains: "MALFORMED_FUNCTION_CALL"},
		{name: "unexpected tool call", response: geminiResponseWithFinishReason(genai.FinishReasonUnexpectedToolCall), contains: "UNEXPECTED_TOOL_CALL"},
		{name: "unknown reason", response: geminiResponseWithFinishReason(genai.FinishReasonOther), contains: "OTHER"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := geminiResponseError(tt.response)
			wantErr := tt.wantPolicy || tt.wantLimit || tt.contains != ""
			if !wantErr {
				if err != nil {
					t.Fatalf("geminiResponseError() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatal("geminiResponseError() = nil, want error")
			}
			if tt.wantPolicy {
				var policyErr ContentPolicyErr
				if !errors.As(err, &policyErr) {
					t.Fatalf("geminiResponseError() = %T, want ContentPolicyErr", err)
				}
			}
			if tt.wantLimit && !errors.Is(err, ErrMaxGenerationLimit) {
				t.Fatalf("geminiResponseError() = %v, want ErrMaxGenerationLimit", err)
			}
			if tt.contains != "" && !strings.Contains(err.Error(), tt.contains) {
				t.Fatalf("geminiResponseError() = %q, want error containing %q", err, tt.contains)
			}
		})
	}
}

func geminiResponseWithFinishReason(reason genai.FinishReason) *genai.GenerateContentResponse {
	return &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{FinishReason: reason}}}
}

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
				var apiErr *ApiErr
				if !errors.As(err, &apiErr) {
					t.Fatalf("Expected error to be ApiErr, got %T: %v", err, err)
				}
				if apiErr.StatusCode != http.StatusInternalServerError {
					t.Errorf("Expected status code %d, got %d", http.StatusInternalServerError, apiErr.StatusCode)
				}
				if apiErr.Kind != APIErrorKindServer {
					t.Errorf("Expected error kind %q, got %q", APIErrorKindServer, apiErr.Kind)
				}
				if apiErr.Message == "" {
					t.Errorf("Expected non-empty error message")
				}
				var sdkErr genai.APIError
				if !errors.As(err, &sdkErr) {
					t.Errorf("Expected original Gemini APIError to be reachable in cause chain")
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
				var apiErr *ApiErr
				if !errors.As(err, &apiErr) {
					t.Fatalf("Expected error to be ApiErr, got %T: %v", err, err)
				}
				if apiErr.Kind != APIErrorKindRateLimit {
					t.Fatalf("Expected error kind %q, got %q", APIErrorKindRateLimit, apiErr.Kind)
				}
				if delay, ok := apiErr.RetryAfter(); !ok || delay != 43*time.Second {
					t.Fatalf("RetryAfter() = (%s, %t), want (43s, true)", delay, ok)
				}
				if apiErr.Message == "" {
					t.Errorf("Expected non-empty rate limit error message")
				}
			},
		},
		{
			name:       "400 Authentication Error",
			statusCode: http.StatusBadRequest,
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
				var apiErr *ApiErr
				if !errors.As(err, &apiErr) {
					t.Fatalf("Expected error to be ApiErr, got %T: %v", err, err)
				}
				if apiErr.StatusCode != http.StatusBadRequest {
					t.Errorf("Expected status code %d, got %d", http.StatusBadRequest, apiErr.StatusCode)
				}
				if apiErr.Kind != APIErrorKindAuthentication {
					t.Errorf("Expected error kind %q, got %q", APIErrorKindAuthentication, apiErr.Kind)
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
