package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"
)

func TestGoogleAdapterScenarios(t *testing.T) {
	t.Run("InlineSystemMessagesRejected", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			t.Error("unsupported system message reached Gemini")
			w.WriteHeader(http.StatusBadRequest)
		}))
		defer server.Close()
		client, err := genai.NewClient(t.Context(), &genai.ClientConfig{APIKey: "test-key", Backend: genai.BackendGeminiAPI, HTTPClient: server.Client(), HTTPOptions: genai.HTTPOptions{BaseURL: server.URL}})
		if err != nil {
			t.Fatal(err)
		}
		generator := NewGeminiGenerator(client)
		request := GenerationRequest{Model: "model", Dialog: Dialog{
			{Role: User, Blocks: []Block{TextBlock("question")}},
			SystemMessage(TextBlock("new guidance")),
		}}
		for _, operation := range []string{"Generate", "Stream", "Count"} {
			t.Run(operation, func(t *testing.T) {
				var err error
				switch operation {
				case "Generate":
					_, err = generator.Generate(t.Context(), request)
				case "Stream":
					for chunk := range generator.Stream(t.Context(), request) {
						if chunk.Err != nil {
							err = chunk.Err
						}
					}
				case "Count":
					_, err = generator.Count(t.Context(), request)
				}
				if err == nil || !strings.Contains(err.Error(), "unsupported message role: system") {
					t.Fatalf("expected unsupported inline system message, got %v", err)
				}
			})
		}
	})
	t.Run("NativeOutputAndThinking", func(t *testing.T) {
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
	})
	t.Run("CommonLogprobOptions", func(t *testing.T) {
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
	})
	t.Run("GeminiAPIErrorMapping", func(t *testing.T) {
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
	})
	t.Run("GeminiGenerator/Count", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		// Create a generator
		g := NewGeminiGenerator(client)
		// Create a dialog with a user message
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is the capital of France?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        "gemini-2.5-pro",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		}
		// Count tokens in the dialog
		tokenCount, err := g.Count(context.Background(), request)
		if err != nil {
			t.Fatalf("count tokens: %v", err)
		}
		if tokenCount == 0 {
			t.Fatal("expected non-zero token count")
		}
		// Try to load an image to add to the dialog
		imgPath := "sample.jpg"
		imgBytes, err := os.ReadFile(imgPath)
		if err != nil {
			t.Skip("could not open sample.jpg")
			return
		}
		// Add an image to the dialog
		dialog = Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "image/jpeg",
						Content:      Str(base64.StdEncoding.EncodeToString(imgBytes)),
					},
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Describe this image."),
					},
				},
			},
		}
		request.Dialog = dialog
		// Count tokens with the image included
		tokenCount, err = g.Count(context.Background(), request)
		if err != nil {
			t.Fatalf("count tokens: %v", err)
		}
		if tokenCount == 0 {
			t.Fatal("expected non-zero token count")
		}
	})
	t.Run("GeminiGenerator/Generate", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		g := NewGeminiGenerator(client)
		dialog := Dialog{
			{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("What is the blooms taxonomy, and how does it related to the psychology of child development?")}}},
		}
		response, err := g.Generate(context.Background(), GenerationRequest{
			Model:        "models/gemini-3-pro-preview",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant. You respond to the user with plain text format.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		}
	})
	t.Run("GeminiGenerator/Generate/audio", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		audioBytes, err := os.ReadFile("sample.wav")
		if err != nil {
			t.Skip("could not open sample.wav")
			return
		}
		// Encode as base64 for inline audio usage
		audioBase64 := Str(base64.StdEncoding.EncodeToString(audioBytes))
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		g := NewGeminiGenerator(client)
		// Using inline audio data
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Audio,
						MimeType:     "audio/wav",
						Content:      audioBase64,
					},
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is the name of person in the greeting in this audio? Return a one work response of the name"),
					},
				},
			},
		}
		response, err := g.Generate(context.Background(), GenerationRequest{
			Model:        "gemini-2.5-pro",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
			if got := strings.ToLower(response.Candidates[0].Blocks[0].Content.String()); !strings.Contains(got, "friday") {
				t.Fatalf("content = %q, want it to contain friday", got)
			}
		}
	})
	t.Run("GeminiGenerator/Generate/image", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		// ---
		// This example assumes that sample.jpg is present in the current directory.
		// Place a JPEG image named sample.jpg in the same directory as this file (or adjust the path).
		imgBytes, err := os.ReadFile("sample.jpg")
		if err != nil {
			t.Skip("could not open sample.jpg")
			return
		}
		// Encode as base64 for API usage
		imgBase64 := Str(
			// Use standard encoding, as required for image MIME input.
			// NOTE: the Blob part in Google Gemini Go SDK accepts raw bytes, but our gai.Block expects base64 encoded string.
			// The actual Gemini implementation will decode as needed, see gai.go.
			// This mirrors how other examples do it.
			base64.StdEncoding.EncodeToString(imgBytes),
		)
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		g := NewGeminiGenerator(client)
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "image/jpeg",
						Content:      imgBase64,
					},
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is in this image? (Hint, it's a character from The Croods, a DreamWorks animated movie.)"),
					},
				},
			},
		}
		response, err := g.Generate(context.Background(), GenerationRequest{
			Model:        "gemini-2.5-pro",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(response.Candidates) != 1 {
			t.Fatalf("candidates = %d, want 1", len(response.Candidates))
		}
		if len(response.Candidates[0].Blocks) != 1 {
			t.Fatalf("blocks = %d, want 1", len(response.Candidates[0].Blocks))
		}
		if !strings.Contains(response.Candidates[0].Blocks[0].Content.String(), "Crood") {
			t.Fatalf("content does not contain Crood")
		}
	})
	t.Run("GeminiGenerator/Generate/pdf", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		g := NewGeminiGenerator(client)
		// This example assumes that sample.pdf is present in the current directory.
		pdfBytes, err := os.ReadFile("sample.pdf")
		if err != nil {
			t.Skip("could not open sample.pdf")
			return
		}
		// Create a dialog with PDF content
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					TextBlock("What is the title of this PDF? Just output the title and nothing else"),
					PDFBlock(pdfBytes, "paper.pdf"),
				},
			},
		}
		// Generate a response
		response, err := g.Generate(ctx, GenerationRequest{
			Model:        "models/gemini-3-pro-preview",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(1024)),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// The response would contain the model's analysis of the PDF
		if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
			if got := response.Candidates[0].Blocks[0].Content.String(); got == "" {
				t.Fatal("expected non-empty content")
			}
		}
	})
	t.Run("GeminiGenerator/RequestTools", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		g := NewGeminiGenerator(client)
		instructions := `You are a helpful assistant. You can call tools in parallel.
When a user asks for the server time, always call the server time tool, don't use previously returned results`
		stockTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		getServerTimeTool := Tool{
			Name:        "get_server_time",
			Description: "Get the current server time in UTC.",
		}
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What is the stock price for AAPL, and also tell me the server time?"),
				}},
			},
		}
		request := GenerationRequest{
			Model:        "models/gemini-3-pro-preview",
			Instructions: SystemMessage(TextBlock(instructions)),
			Dialog:       dialog,
			Tools:        []Tool{stockTool, getServerTimeTool},
		}
		// Expect tool call for both tools
		response, err := g.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		calls := collectToolCalls(t, requireCandidate(t, response).Blocks)
		if len(calls) < 2 {
			t.Fatalf("tool calls = %d, want at least 2", len(calls))
		}
		requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "AAPL")
		dialog = append(dialog, response.Candidates[0])
		// Simulate tool result for tool calls
		dialog = append(dialog,
			Message{
				Role: ToolResult,
				Blocks: []Block{{
					ID:           response.Candidates[0].Blocks[0].ID,
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str("AAPL is $200.00"),
				}},
			},
			Message{
				Role: ToolResult,
				Blocks: []Block{{
					ID:           response.Candidates[0].Blocks[1].ID,
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(time.Time{}.String()),
				}},
			},
		)
		request.Dialog = dialog
		response, err = g.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		toolResult := requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0))
		requireTextContains(t, toolResult, "AAPL", "200.00", time.Time{}.String())
		dialog = append(dialog, response.Candidates[0], Message{
			Role: User,
			Blocks: []Block{{
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str("What is the stock price for MSFT, and also tell me the server time again?"),
			}},
		})
		request.Dialog = dialog
		response, err = g.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		calls = collectToolCalls(t, requireCandidate(t, response).Blocks)
		if len(calls) < 2 {
			t.Fatalf("tool calls = %d, want at least 2", len(calls))
		}
		requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "MSFT")
		dialog = append(dialog, response.Candidates[0])
		// Simulate tool result for tool calls
		dialog = append(dialog,
			Message{
				Role: ToolResult,
				Blocks: []Block{{
					ID:           response.Candidates[0].Blocks[0].ID,
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str("MSFT is $300.00"),
				}},
			},
			Message{
				Role: ToolResult,
				Blocks: []Block{{
					ID:           response.Candidates[0].Blocks[1].ID,
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(time.Time{}.Add(1 * time.Minute).String()),
				}},
			},
		)
		request.Dialog = dialog
		response, err = g.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		requireTextContains(t, requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0)), "MSFT", "300", "UTC")
	})
	t.Run("GeminiGenerator/RequestTools/parallelToolUse", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		g := NewGeminiGenerator(client)
		// Define the request tool
		stockTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		dialog := Dialog{
			{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("Give me the current prices for AAPL, MSFT, and TSLA.")}}},
		}
		response, err := g.Generate(context.Background(), GenerationRequest{
			Model:        "models/gemini-3-pro-preview",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Tools:        []Tool{stockTool},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		calls := collectToolCalls(t, requireCandidate(t, response).Blocks)
		if len(calls) < 3 {
			t.Fatalf("tool calls = %d, want at least 3", len(calls))
		}
		requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "AAPL")
		requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "MSFT")
		requireToolCallWithParam(t, calls, "get_stock_price", "ticker", "TSLA")
	})
	t.Run("GeminiGenerator/RequestTools/parallelToolUse/multimedia", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		g := NewGeminiGenerator(client)
		// Define a request tool to view files
		viewFileTool := Tool{
			Name:        "view_file",
			Description: "View the contents of a file. Can handle text files, images, and other media types.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					FilePath string `json:"file_path" jsonschema:"required" jsonschema_description:"The path to the file to view"`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		// User asks to view multiple files
		dialog := Dialog{
			{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("Please view sample.jpg and README.md, and tell me what character is in the image, and what is gai from the README")}}},
		}
		request := GenerationRequest{
			Model:        "models/gemini-3-pro-preview",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant that can view files.")),
			Dialog:       dialog,
			Tools:        []Tool{viewFileTool},
		}
		// Model makes parallel tool calls
		response, err := g.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		for _, block := range response.Candidates[0].Blocks {
			if block.BlockType == Thinking {
				continue
			}
		}
		dialog = append(dialog, response.Candidates[0])
		// Find tool call blocks (skip thinking blocks)
		toolCallBlocks := []Block{}
		for _, block := range response.Candidates[0].Blocks {
			if block.BlockType == ToolCall {
				toolCallBlocks = append(toolCallBlocks, block)
			}
		}
		if len(toolCallBlocks) < 2 {
			t.Fatal("Expected at least 2 tool calls")
			return
		}
		// Simulate tool results - first for sample.jpg (image)
		imgBytes, err := os.ReadFile("sample.jpg")
		if err != nil {
			t.Skip("could not open sample.jpg")
			return
		}
		// Simulate tool results - for README.md (text)
		readmeBytes, err := os.ReadFile("README.md")
		if err != nil {
			t.Skip("could not open README.md")
			return
		}
		// Add both tool results in parallel
		dialog = append(dialog,
			Message{
				Role: ToolResult,
				Blocks: []Block{{
					ID:           toolCallBlocks[0].ID, // First tool call
					BlockType:    Content,
					ModalityType: Image,
					MimeType:     "image/jpeg",
					Content:      Str(base64.StdEncoding.EncodeToString(imgBytes)),
				}},
			},
			Message{
				Role: ToolResult,
				Blocks: []Block{{
					ID:           toolCallBlocks[1].ID, // Second tool call
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/markdown",
					Content:      Str(string(readmeBytes)),
				}},
			},
		)
		request.Dialog = dialog
		// Get final response with tool results
		response, err = g.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		content := requireContentBlock(t, requireBlock(t, requireCandidate(t, response), 0))
		requireTextContains(t, content, "Crood", "gai")
	})
	t.Run("GeminiGenerator/Stream", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		if err != nil {
			t.Fatalf("create Gemini client: %v", err)
		}
		g := NewGeminiGenerator(client)
		dialog := Dialog{
			{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("What is the capital of France?")}}},
		}
		var content strings.Builder
		for chunk := range g.Stream(context.Background(), GenerationRequest{
			Model:        "models/gemini-3-pro-preview",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant. You respond to the user with plain text format.")),
			Dialog:       dialog,
		}) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			if chunk.Block.BlockType == MetadataBlockType {
				continue
			}
			content.WriteString(chunk.Block.Content.String())
		}
		requireTextContains(t, content.String(), "Paris")
	})
	t.Run("GeminiGenerator/Stream/parallelToolUse", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "GEMINI_API_KEY")
		ctx := context.Background()
		client, err := genai.NewClient(
			ctx,
			&genai.ClientConfig{
				APIKey:  apiKey,
				Backend: genai.BackendGeminiAPI,
			},
		)
		if err != nil {
			t.Fatalf("create Gemini client: %v", err)
		}
		g := NewGeminiGenerator(client)
		// Define the request tool
		stockTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		dialog := Dialog{
			{Role: User, Blocks: []Block{{BlockType: Content, ModalityType: Text, Content: Str("Give me the current prices for AAPL, MSFT, and TSLA.")}}},
		}
		var toolCallCount int
		for chunk := range g.Stream(context.Background(), GenerationRequest{
			Model:        "models/gemini-3-pro-preview",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Tools:        []Tool{stockTool},
		}) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			if chunk.Block.BlockType == MetadataBlockType {
				continue
			}
			if chunk.Block.BlockType == ToolCall {
				toolCallCount++
			}
		}
		if toolCallCount == 0 {
			t.Fatal("expected at least one streamed tool call")
		}
	})
	t.Run("GeminiResponseError", func(t *testing.T) {
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
	})
}

func geminiResponseWithFinishReason(reason genai.FinishReason) *genai.GenerateContentResponse {
	return &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{FinishReason: reason}}}
}
