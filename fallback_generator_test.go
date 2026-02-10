package gai

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

// mockGenerator implements the Generator interface for testing
type mockGenerator struct {
	response Response
	err      error
}

// Generate implements the Generator interface for testing
func (m *mockGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	return m.response, m.err
}

func TestNewFallbackGenerator(t *testing.T) {
	tests := []struct {
		name       string
		generators []Generator
		config     *FallbackConfig
		wantErr    bool
	}{
		{
			name:       "too few generators",
			generators: []Generator{&mockGenerator{}},
			config:     nil,
			wantErr:    true,
		},
		{
			name:       "exactly two generators",
			generators: []Generator{&mockGenerator{}, &mockGenerator{}},
			config:     nil,
			wantErr:    false,
		},
		{
			name:       "more than two generators",
			generators: []Generator{&mockGenerator{}, &mockGenerator{}, &mockGenerator{}},
			config:     nil,
			wantErr:    false,
		},
		{
			name:       "with custom config",
			generators: []Generator{&mockGenerator{}, &mockGenerator{}},
			config:     &FallbackConfig{ShouldFallback: func(err error) bool { return true }},
			wantErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewFallbackGenerator(tt.generators, tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewFallbackGenerator() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestFallbackGenerator_Generate(t *testing.T) {
	// Create a successful response for testing
	successResponse := Response{
		Candidates: []Message{
			{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Successful response"),
					},
				},
			},
		},
		FinishReason: EndTurn,
		UsageMetadata: Metadata{
			UsageMetricInputTokens:      10,
			UsageMetricGenerationTokens: 5,
		},
	}

	// Create a fallback response for testing
	fallbackResponse := Response{
		Candidates: []Message{
			{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Fallback response"),
					},
				},
			},
		},
		FinishReason: EndTurn,
		UsageMetadata: Metadata{
			UsageMetricInputTokens:      10,
			UsageMetricGenerationTokens: 7,
		},
	}

	// Create a simple test dialog
	testDialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Test message"),
				},
			},
		},
	}

	// Create test options
	testOptions := &GenOpts{
		Temperature: Ptr(0.7),
	}

	tests := []struct {
		name       string
		generators []Generator
		config     *FallbackConfig
		dialog     Dialog
		options    *GenOpts
		want       Response
		wantErr    bool
	}{
		{
			name: "first generator succeeds",
			generators: []Generator{
				&mockGenerator{response: successResponse, err: nil},
				&mockGenerator{response: fallbackResponse, err: nil},
			},
			config:  nil,
			dialog:  testDialog,
			options: testOptions,
			want:    successResponse,
			wantErr: false,
		},
		{
			name: "first generator fails with rate limit, second succeeds",
			generators: []Generator{
				&mockGenerator{response: Response{}, err: RateLimitErr("rate limit exceeded")},
				&mockGenerator{response: fallbackResponse, err: nil},
			},
			config:  nil,
			dialog:  testDialog,
			options: testOptions,
			want:    fallbackResponse,
			wantErr: false,
		},
		{
			name: "first generator fails with 500 error, second succeeds",
			generators: []Generator{
				&mockGenerator{
					response: Response{},
					err:      ApiErr{StatusCode: 500, Type: "server_error", Message: "internal server error"},
				},
				&mockGenerator{response: fallbackResponse, err: nil},
			},
			config:  nil,
			dialog:  testDialog,
			options: testOptions,
			want:    fallbackResponse,
			wantErr: false,
		},
		{
			name: "first generator fails with 400 error, no fallback",
			generators: []Generator{
				&mockGenerator{
					response: Response{},
					err:      ApiErr{StatusCode: 400, Type: "invalid_request", Message: "bad request"},
				},
				&mockGenerator{response: fallbackResponse, err: nil},
			},
			config:  nil,
			dialog:  testDialog,
			options: testOptions,
			want:    Response{},
			wantErr: true,
		},
		{
			name: "first generator fails with 400 error, custom config fallbacks",
			generators: []Generator{
				&mockGenerator{
					response: Response{},
					err:      ApiErr{StatusCode: 400, Type: "invalid_request", Message: "bad request"},
				},
				&mockGenerator{response: fallbackResponse, err: nil},
			},
			config: &FallbackConfig{
				ShouldFallback: func(err error) bool {
					var apiErr ApiErr
					return errors.As(err, &apiErr) && apiErr.StatusCode == 400
				},
			},
			dialog:  testDialog,
			options: testOptions,
			want:    fallbackResponse,
			wantErr: false,
		},
		{
			name: "all generators fail with fallback errors",
			generators: []Generator{
				&mockGenerator{response: Response{}, err: RateLimitErr("rate limit exceeded")},
				&mockGenerator{
					response: Response{},
					err:      ApiErr{StatusCode: 500, Type: "server_error", Message: "internal server error"},
				},
			},
			config:  nil,
			dialog:  testDialog,
			options: testOptions,
			want:    Response{},
			wantErr: true,
		},
		{
			name: "HTTP status fallback config works",
			generators: []Generator{
				&mockGenerator{
					response: Response{},
					err:      ApiErr{StatusCode: 429, Type: "too_many_requests", Message: "too many requests"},
				},
				&mockGenerator{response: fallbackResponse, err: nil},
			},
			config:  &FallbackConfig{ShouldFallback: NewHTTPStatusFallbackConfig(429).ShouldFallback},
			dialog:  testDialog,
			options: testOptions,
			want:    fallbackResponse,
			wantErr: false,
		},
		{
			name: "Rate limit only config works",
			generators: []Generator{
				&mockGenerator{response: Response{}, err: RateLimitErr("rate limit exceeded")},
				&mockGenerator{response: fallbackResponse, err: nil},
			},
			config:  &FallbackConfig{ShouldFallback: NewRateLimitOnlyFallbackConfig().ShouldFallback},
			dialog:  testDialog,
			options: testOptions,
			want:    fallbackResponse,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fallbackGen, err := NewFallbackGenerator(tt.generators, tt.config)
			if err != nil {
				t.Fatalf("Failed to create fallback generator: %v", err)
			}

			got, err := fallbackGen.Generate(context.Background(), tt.dialog, tt.options)

			// Check error cases
			if (err != nil) != tt.wantErr {
				t.Errorf("FallbackGenerator.Generate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// For successful cases, check response
			if !tt.wantErr {
				// Check candidates
				if len(got.Candidates) != len(tt.want.Candidates) {
					t.Errorf("FallbackGenerator.Generate() candidates count = %d, want %d",
						len(got.Candidates), len(tt.want.Candidates))
					return
				}

				// Check the first candidate's content
				if len(got.Candidates) > 0 && len(got.Candidates[0].Blocks) > 0 {
					gotContent := got.Candidates[0].Blocks[0].Content.String()
					wantContent := tt.want.Candidates[0].Blocks[0].Content.String()
					if gotContent != wantContent {
						t.Errorf("FallbackGenerator.Generate() content = %s, want %s",
							gotContent, wantContent)
					}
				}

				// Check finish reason
				if got.FinishReason != tt.want.FinishReason {
					t.Errorf("FallbackGenerator.Generate() finish reason = %v, want %v",
						got.FinishReason, tt.want.FinishReason)
				}
			}
		})
	}
}

func TestNewHTTPStatusFallbackConfig(t *testing.T) {
	config := NewHTTPStatusFallbackConfig(400, 429)

	// Should fallback on rate limit errors
	if !config.ShouldFallback(RateLimitErr("rate limit exceeded")) {
		t.Error("Expected to fallback on rate limit errors")
	}

	// Should fallback on specified status codes
	if !config.ShouldFallback(ApiErr{StatusCode: 400}) {
		t.Error("Expected to fallback on status code 400")
	}

	if !config.ShouldFallback(ApiErr{StatusCode: 429}) {
		t.Error("Expected to fallback on status code 429")
	}

	// Should not fallback on other status codes
	if config.ShouldFallback(ApiErr{StatusCode: 404}) {
		t.Error("Expected not to fallback on status code 404")
	}

	if config.ShouldFallback(ApiErr{StatusCode: 500}) {
		t.Error("Expected not to fallback on status code 500 when not specified")
	}
}

func TestNewRateLimitOnlyFallbackConfig(t *testing.T) {
	config := NewRateLimitOnlyFallbackConfig()

	// Should fallback on rate limit errors
	if !config.ShouldFallback(RateLimitErr("rate limit exceeded")) {
		t.Error("Expected to fallback on rate limit errors")
	}

	// Should not fallback on other errors
	if config.ShouldFallback(errors.New("some other error")) {
		t.Error("Expected not to fallback on non-rate-limit errors")
	}

	// Should not fallback on API errors, even with 429 status code
	if config.ShouldFallback(ApiErr{StatusCode: 429}) {
		t.Error("Expected not to fallback on API errors with 429 status code")
	}
}

func ExampleFallbackGenerator() {
	// This is just an example, in a real case you would use actual generators
	openAIGen := &mockGenerator{
		response: Response{
			Candidates: []Message{
				{
					Role: Assistant,
					Blocks: []Block{
						{
							BlockType:    Content,
							ModalityType: Text,
							Content:      Str("Response from OpenAI"),
						},
					},
				},
			},
		},
	}

	anthropicGen := &mockGenerator{
		response: Response{
			Candidates: []Message{
				{
					Role: Assistant,
					Blocks: []Block{
						{
							BlockType:    Content,
							ModalityType: Text,
							Content:      Str("Response from Anthropic"),
						},
					},
				},
			},
		},
	}

	// Create a fallback generator that will try OpenAI first, then fallback to Anthropic
	// This example makes it fallback on 400 errors too, not just 500s
	fallbackGen, _ := NewFallbackGenerator(
		[]Generator{openAIGen, anthropicGen},
		&FallbackConfig{
			ShouldFallback: NewHTTPStatusFallbackConfig(400, 429, 500, 502, 503, 504).ShouldFallback,
		},
	)

	// Now we can use the fallback generator just like any other generator
	dialog := Dialog{
		{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("Tell me about AI fallback strategies"),
				},
			},
		},
	}

	resp, err := fallbackGen.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
		fmt.Println(resp.Candidates[0].Blocks[0].Content)
	}
	// Output: Response from OpenAI
}
