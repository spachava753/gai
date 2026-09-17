package gai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"
)

func testGeminiNativeOutputAndThinking(t *testing.T) {
	for _, stream := range []bool{false, true} {
		calls := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
				return
			}
			calls++
			if calls == 2 {
				contents := body["contents"].([]any)
				model := contents[1].(map[string]any)
				parts := model["parts"].([]any)
				if len(parts) != 4 || model["role"] != "model" {
					t.Errorf("replayed content: %v", model)
					return
				}
				image := parts[1].(map[string]any)
				blob := image["inlineData"].(map[string]any)
				if blob["data"] != "AwQ=" || blob["mimeType"] != "image/png" || image["thoughtSignature"] != "BQY=" {
					t.Errorf("replayed image: %v", image)
				}
				thought := parts[3].(map[string]any)
				if thought["thought"] != true || thought["thoughtSignature"] != "Cww=" || thought["inlineData"].(map[string]any)["data"] != "CQo=" {
					t.Errorf("replayed thought image: %v", thought)
				}
			}
			config := body["generationConfig"].(map[string]any)
			modalities := config["responseModalities"].([]any)
			if len(modalities) != 3 || modalities[0] != "TEXT" || modalities[1] != "IMAGE" || modalities[2] != "AUDIO" {
				t.Errorf("modalities: %v", modalities)
			}
			thinking := config["thinkingConfig"].(map[string]any)
			if thinking["thinkingBudget"] != float64(1024) || thinking["thinkingLevel"] != nil {
				t.Errorf("thinking: %v", thinking)
			}
			response := `{"candidates":[{"content":{"role":"model","parts":[{"text":"hello","thoughtSignature":"AQI="},{"inlineData":{"mimeType":"image/png","data":"AwQ="},"thoughtSignature":"BQY="},{"inlineData":{"mimeType":"audio/L16;rate=24000","data":"Bwg="}},{"inlineData":{"mimeType":"image/png","data":"CQo="},"thought":true,"thoughtSignature":"Cww="}]},"finishReason":"STOP"}]}`
			if stream {
				w.Header().Set("Content-Type", "text/event-stream")
				response = "data: " + response + "\n\n"
			}
			_, _ = w.Write([]byte(response))
		}))
		client, err := genai.NewClient(t.Context(), &genai.ClientConfig{APIKey: "test", Backend: genai.BackendGeminiAPI, HTTPOptions: genai.HTTPOptions{BaseURL: server.URL}})
		if err != nil {
			server.Close()
			t.Fatal(err)
		}
		g := &GeminiGenerator{client: client}
		request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hi")}}}, Options: NewGenerationOptions(WithThinkingBudget(1024), WithGeminiOutputModalities(Text, Image, Audio))}
		var blocks []Block
		if stream {
			adapter := StreamingAdapter{S: g}
			result, err := adapter.Generate(t.Context(), request)
			if err != nil {
				t.Error(err)
			} else {
				blocks = result.Candidates[0].Blocks
			}
		} else {
			result, err := g.Generate(t.Context(), request)
			if err != nil {
				t.Error(err)
			} else {
				blocks = result.Candidates[0].Blocks
			}
		}
		if len(blocks) != 4 {
			server.Close()
			t.Fatalf("blocks: %#v", blocks)
		}
		for i, expected := range []struct {
			modality              Modality
			mime, data, signature string
		}{
			{Text, "text/plain", "hello", "AQI="}, {Image, "image/png", "AwQ=", "BQY="}, {Audio, "audio/L16;rate=24000", "Bwg=", ""}, {Image, "image/png", "CQo=", "Cww="},
		} {
			block := blocks[i]
			if block.ModalityType != expected.modality || block.MimeType != expected.mime || block.Content.String() != expected.data {
				t.Errorf("block %d: %#v", i, block)
			}
			if expected.signature != "" && block.ExtraFields[GeminiExtraFieldThoughtSignature] != expected.signature {
				t.Errorf("signature %d: %#v", i, block)
			}
		}
		if blocks[3].BlockType != Thinking {
			t.Errorf("thought image type: %#v", blocks[3])
		}
		request.Dialog = append(request.Dialog, Message{Role: Assistant, Blocks: blocks}, Message{Role: User, Blocks: []Block{TextBlock("continue")}})
		if stream {
			adapter := StreamingAdapter{S: g}
			_, err = adapter.Generate(t.Context(), request)
		} else {
			_, err = g.Generate(t.Context(), request)
		}
		server.Close()
		if err != nil {
			t.Error(err)
		}
		if calls != 2 {
			t.Errorf("calls = %d", calls)
		}
	}
	defaults, err := parseGeminiGenerationOptions(nil)
	if err != nil || defaults.ThinkingBudget != nil || defaults.OutputModalities != nil {
		t.Fatalf("defaults: %#v, %v", defaults, err)
	}
	for _, options := range []GenerationOptions{
		{GenerationOptionThinkingBudget: "1024"}, {GenerationOptionThinkingBudget: 0}, {GenerationOptionThinkingBudget: -1}, {GenerationOptionThinkingBudget: int(1 << 31)},
		{GenerationOptionThinkingBudget: 1024, GenerationOptionReasoningEffort: "low"},
		{GeminiGenerationOptionOutputModalities: []string{"TEXT"}}, {GeminiGenerationOptionOutputModalities: []Modality{}}, {GeminiGenerationOptionOutputModalities: []Modality{Video}}, {GeminiGenerationOptionOutputModalities: []Modality{Modality(99)}},
	} {
		if _, err := parseGeminiGenerationOptions(options); err == nil {
			t.Errorf("accepted invalid options %#v", options)
		}
	}
	for _, mimeType := range []string{"", "bad MIME", "application/octet-stream", "video/mp4"} {
		if _, err := geminiContentBlock(&genai.Part{InlineData: &genai.Blob{MIMEType: mimeType, Data: []byte{1}}}); err == nil {
			t.Errorf("accepted inline MIME %q", mimeType)
		}
	}
	part := &genai.Part{Thought: true, InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}, ThoughtSignature: []byte{2}}
	block, err := geminiContentBlock(part)
	if err != nil || block.BlockType != Thinking || block.ModalityType != Image || block.ExtraFields[GeminiExtraFieldThoughtSignature] != "Ag==" {
		t.Fatalf("thought image: %#v, %v", block, err)
	}
}

func testGeminiCommonLogprobOptions(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, enabled := range []bool{false, true} {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				config := body["generationConfig"].(map[string]any)
				if config["seed"] != float64(0) || config["logprobs"] != float64(0) || config["presencePenalty"] != float64(0) || config["frequencyPenalty"] != float64(0) {
					t.Errorf("lost zero: %v", config)
				}
				// The SDK omits its non-pointer false bool; this is the API default.
				if enabled && config["responseLogprobs"] != true || !enabled && config["responseLogprobs"] == true {
					t.Errorf("logprobs: %v", config)
				}
				if stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}],\"role\":\"model\"}}]}\n\ndata: {\"candidates\":[{\"logprobsResult\":{\"chosenCandidates\":[{\"token\":\"ok\",\"logProbability\":-0.1}]},\"finishReason\":\"STOP\"}]}\n\n"))
				} else {
					_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"logprobsResult":{"chosenCandidates":[{"token":"ok","logProbability":-0.1}]},"finishReason":"STOP"}]}`))
				}
			}))
			client, err := genai.NewClient(t.Context(), &genai.ClientConfig{APIKey: "test", Backend: genai.BackendGeminiAPI, HTTPOptions: genai.HTTPOptions{BaseURL: server.URL}})
			if err != nil {
				server.Close()
				t.Fatal(err)
			}
			g := &GeminiGenerator{client: client}
			request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hi")}}}, Options: NewGenerationOptions(WithSeed(0), WithTopLogprobs(0), WithLogprobs(enabled), WithPresencePenalty(0), WithFrequencyPenalty(0))}
			var result Response
			if stream {
				adapter := StreamingAdapter{S: g}
				result, err = adapter.Generate(t.Context(), request)
			} else {
				result, err = g.Generate(t.Context(), request)
			}
			if err != nil {
				t.Error(err)
			} else {
				reports, ok := result.Candidates[0].ExtraFields[GeminiExtraFieldLogprobs].([]*genai.LogprobsResult)
				if !ok || len(reports) != 1 || len(reports[0].ChosenCandidates) != 1 || reports[0].ChosenCandidates[0].Token != "ok" {
					t.Errorf("reports: %#v", reports)
				}
			}
			server.Close()
			if calls != 1 {
				t.Fatalf("calls = %d", calls)
			}
		}
	}
	for _, key := range []string{GenerationOptionSeed, GenerationOptionTopLogprobs, GenerationOptionLogprobs, GenerationOptionPresencePenalty, GenerationOptionFrequencyPenalty} {
		if _, err := parseGeminiGenerationOptions(GenerationOptions{key: "invalid"}); err == nil {
			t.Fatalf("accepted invalid %s", key)
		}
	}
	for _, key := range []string{GenerationOptionSeed, GenerationOptionTopLogprobs} {
		if _, err := parseGeminiGenerationOptions(GenerationOptions{key: int(1 << 31)}); err == nil {
			t.Fatalf("accepted overflowing %s", key)
		}
	}
}

func testGeminiResponseError(t *testing.T) {
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

func testGeminiAPIErrorMapping(t *testing.T) {
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

			generator := NewGeminiGenerator(client)

			dialog := Dialog{{
				Role: User,
				Blocks: []Block{{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str("Hello, world!"),
				}},
			}}

			_, err = generator.Generate(context.Background(), GenerationRequest{
				Model:        "test-model",
				Instructions: SystemMessage(TextBlock("test instructions")),
				Dialog:       dialog,
			})

			tc.errChecker(t, err)
		})
	}
}
