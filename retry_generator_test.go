package gai_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/spachava753/gai"
)

// mockGenerator implements generation, token counting, and streaming for tests.
type mockGenerator struct {
	GenerateFunc func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error)
	StreamFunc   func(ctx context.Context, request gai.GenerationRequest) iter.Seq[gai.StreamChunk]
	CountFunc    func(ctx context.Context, request gai.GenerationRequest) (uint, error)

	generateCallCount int
	streamCallCount   int
}

func (m *mockGenerator) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	m.generateCallCount++
	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, request)
	}
	return gai.Response{}, errors.New("GenerateFunc not implemented")
}

func (m *mockGenerator) Stream(ctx context.Context, request gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
	m.streamCallCount++
	if m.StreamFunc != nil {
		return m.StreamFunc(ctx, request)
	}
	return func(yield func(gai.StreamChunk) bool) {
		yield(gai.StreamChunk{Err: errors.New("StreamFunc not implemented")})
	}
}

func (m *mockGenerator) Count(ctx context.Context, request gai.GenerationRequest) (uint, error) {
	if m.CountFunc != nil {
		return m.CountFunc(ctx, request)
	}
	return 0, errors.New("CountFunc not implemented")
}

func (m *mockGenerator) ResetCallCount() {
	m.generateCallCount = 0
	m.streamCallCount = 0
}

func collectStream(seq iter.Seq[gai.StreamChunk]) ([]gai.StreamChunk, error) {
	var chunks []gai.StreamChunk
	for chunk := range seq {
		if chunk.Err != nil {
			return chunks, chunk.Err
		}
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}

type generatorFunc func(context.Context, gai.GenerationRequest) (gai.Response, error)

func (f generatorFunc) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	return f(ctx, request)
}

type retryAttemptsContextKey struct{}

func constantRetryConfig(delay time.Duration) gai.RetryConfig {
	return gai.RetryConfig{
		Backoff: func(uint) (time.Duration, bool) { return delay, true },
	}
}

func TestRetryPolicyScenarios(t *testing.T) {
	t.Run("RetryGenerator/Count/UnderlyingDoesNotImplementTokenCounter", func(t *testing.T) {
		type nonCountingGenerator struct{ gai.Generator }
		underlying := &nonCountingGenerator{Generator: &mockGenerator{}}
		rg := gai.NewRetryGenerator(underlying, gai.RetryConfig{})

		_, err := rg.Count(context.Background(), gai.GenerationRequest{})
		if err == nil {
			t.Fatal("Count() error = nil, want an error")
		}
		wantErrStr := fmt.Sprintf("inner generator of type %T does not implement TokenCounter", underlying)
		if err.Error() != wantErrStr {
			t.Errorf("Count() error = %q, want %q", err.Error(), wantErrStr)
		}
	})
	t.Run("RetryGenerator/Count/UnderlyingImplementsTokenCounter", func(t *testing.T) {
		expectedCount := uint(123)
		m := &mockGenerator{
			CountFunc: func(ctx context.Context, request gai.GenerationRequest) (uint, error) {
				return expectedCount, nil
			},
		}
		rg := gai.NewRetryGenerator(m, gai.RetryConfig{})

		count, err := rg.Count(context.Background(), gai.GenerationRequest{})
		if err != nil {
			t.Fatalf("Count() error = %v, wantErr false", err)
		}
		if count != expectedCount {
			t.Errorf("Count() = %d, want %d", count, expectedCount)
		}
	})
	t.Run("RetryGenerator/Generate/BackoffCanStopRetries", func(t *testing.T) {
		tests := []struct {
			name   string
			config gai.RetryConfig
		}{
			{name: "no backoff configured"},
			{
				name: "backoff stops",
				config: gai.RetryConfig{
					Backoff: func(uint) (time.Duration, bool) { return 0, false },
				},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				expectedErr := &gai.ApiErr{Kind: gai.APIErrorKindRateLimit}
				m := &mockGenerator{
					GenerateFunc: func(context.Context, gai.GenerationRequest) (gai.Response, error) {
						return gai.Response{}, expectedErr
					},
				}
				rg := gai.NewRetryGenerator(m, tt.config)

				_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
				if err != expectedErr {
					t.Fatalf("Generate() error = %T %v, want original ApiErr", err, err)
				}
				if m.generateCallCount != 1 {
					t.Fatalf("Generate() calls = %d, want 1", m.generateCallCount)
				}
			})
		}
	})
	t.Run("RetryGenerator/Generate/BackoffSequenceContinuesAcrossRetryAfter", func(t *testing.T) {
		retryAfter := time.Duration(0)
		fallbackErr := &gai.ApiErr{Kind: gai.APIErrorKindRateLimit}
		hintedErr := &gai.ApiErr{
			Kind:               gai.APIErrorKindRateLimit,
			RetryAfterDuration: &retryAfter,
		}
		m := &mockGenerator{}
		m.GenerateFunc = func(context.Context, gai.GenerationRequest) (gai.Response, error) {
			switch m.generateCallCount {
			case 1, 3:
				return gai.Response{}, fallbackErr
			case 2:
				return gai.Response{}, hintedErr
			default:
				return gai.Response{}, nil
			}
		}

		var retries []uint
		config := gai.RetryConfig{
			Backoff: func(retry uint) (time.Duration, bool) {
				retries = append(retries, retry)
				return 0, true
			},
			MaxAttempts: 4,
		}
		rg := gai.NewRetryGenerator(m, config)

		if _, err := rg.Generate(context.Background(), gai.GenerationRequest{}); err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if len(retries) != 3 || retries[0] != 1 || retries[1] != 2 || retries[2] != 3 {
			t.Fatalf("backoff retry numbers = %v, want [1 2 3]", retries)
		}
	})
	t.Run("RetryGenerator/Generate/ContextCancelled/BeforeFirstCall", func(t *testing.T) {
		m := &mockGenerator{}
		m.GenerateFunc = func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
			t.Error("Generate should not have been called")
			return gai.Response{}, nil
		}
		rg := gai.NewRetryGenerator(m, gai.RetryConfig{})

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := rg.Generate(ctx, gai.GenerationRequest{})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Generate() error = %v, want %v", err, context.Canceled)
		}
		if m.generateCallCount != 0 {
			t.Errorf("Expected Generate to be called 0 times, got %d", m.generateCallCount)
		}
	})
	t.Run("RetryGenerator/Generate/ContextCancelled/DuringBackoff", func(t *testing.T) {
		m := &mockGenerator{}
		m.GenerateFunc = func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
			return gai.Response{}, &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "rate limited"}
		}

		config := constantRetryConfig(100 * time.Millisecond)
		config.MaxElapsedTime = 5 * time.Second
		rg := gai.NewRetryGenerator(m, config)

		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			time.Sleep(50 * time.Millisecond)
			cancel()
		}()

		_, err := rg.Generate(ctx, gai.GenerationRequest{})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Generate() error = %v, want %v", err, context.Canceled)
		}
		if m.generateCallCount < 1 {
			t.Errorf("Expected Generate to be called at least 1 time, got %d", m.generateCallCount)
		}
	})
	t.Run("RetryGenerator/Generate/ContextCancelled/DuringOperation", func(t *testing.T) {
		opDuration := 100 * time.Millisecond
		cancelDelay := 20 * time.Millisecond

		m := &mockGenerator{
			GenerateFunc: func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
				select {
				case <-time.After(opDuration):
					return gai.Response{}, errors.New("operation should have been cancelled")
				case <-ctx.Done():
					return gai.Response{}, ctx.Err()
				}
			},
		}
		rg := gai.NewRetryGenerator(m, gai.RetryConfig{})

		ctx, cancel := context.WithTimeout(context.Background(), cancelDelay)
		defer cancel()

		_, err := rg.Generate(ctx, gai.GenerationRequest{})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("Generate() error = %v, want %v", err, context.DeadlineExceeded)
		}
		if m.generateCallCount != 1 {
			t.Errorf("Expected Generate to be called 1 time, got %d", m.generateCallCount)
		}
	})
	t.Run("RetryGenerator/Generate/DoesNotRetryCanceledOrNonRetryableAPIErrors", func(t *testing.T) {
		tests := []struct {
			name string
			err  *gai.ApiErr
		}{
			{
				name: "non-retryable API error wrapping deadline",
				err: &gai.ApiErr{
					Provider: gai.ProviderOpenAI,
					Kind:     gai.APIErrorKindInvalidRequest,
					Cause:    context.DeadlineExceeded,
				},
			},
			{
				name: "retryable API error wrapping cancellation",
				err: &gai.ApiErr{
					Provider: gai.ProviderOpenAI,
					Kind:     gai.APIErrorKindRateLimit,
					Cause:    context.Canceled,
				},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				m := &mockGenerator{
					GenerateFunc: func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
						return gai.Response{}, tt.err
					},
				}
				config := constantRetryConfig(0)
				config.MaxAttempts = 2
				rg := gai.NewRetryGenerator(m, config)

				_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
				if err != tt.err {
					t.Fatalf("Generate() error = %T %v, want original ApiErr", err, err)
				}
				if m.generateCallCount != 1 {
					t.Fatalf("Generate() calls = %d, want 1", m.generateCallCount)
				}
			})
		}
	})
	t.Run("RetryGenerator/Generate/HonorsRetryAfter", func(t *testing.T) {
		retryAfter := 5 * time.Millisecond
		retryableErr := &gai.ApiErr{
			Provider:           gai.ProviderOpenAI,
			Kind:               gai.APIErrorKindRateLimit,
			StatusCode:         http.StatusTooManyRequests,
			Message:            "too many requests",
			RetryAfterDuration: &retryAfter,
		}

		m := &mockGenerator{}
		m.GenerateFunc = func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
			if m.generateCallCount == 1 {
				return gai.Response{}, retryableErr
			}
			return gai.Response{}, nil
		}

		var notifiedDelay time.Duration
		var notifiedErr error
		config := constantRetryConfig(time.Millisecond)
		config.MaxAttempts = 2
		config.Notify = func(err error, delay time.Duration) {
			notifiedErr = err
			notifiedDelay = delay
		}
		rg := gai.NewRetryGenerator(m, config)

		_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if m.generateCallCount != 2 {
			t.Fatalf("Generate() calls = %d, want 2", m.generateCallCount)
		}
		if notifiedDelay != retryAfter {
			t.Fatalf("notified delay = %s, want %s", notifiedDelay, retryAfter)
		}
		if notifiedErr != retryableErr {
			t.Fatalf("notified error = %T %v, want original ApiErr", notifiedErr, notifiedErr)
		}
	})
	t.Run("RetryGenerator/Generate/MaxRetriesExceeded/WithMaxElapsedTime", func(t *testing.T) {
		expectedErr := &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "persistent rate limit"}
		m := &mockGenerator{}
		m.GenerateFunc = func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
			return gai.Response{}, expectedErr
		}

		config := gai.RetryConfig{
			Backoff: func(retry uint) (time.Duration, bool) {
				if retry == 1 {
					return time.Millisecond, true
				}
				return 2 * time.Millisecond, true
			},
			MaxElapsedTime: 4 * time.Millisecond,
		}

		rg := gai.NewRetryGenerator(m, config)

		_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if !errors.Is(err, expectedErr) {
			t.Fatalf("Generate() error = %v, want %v", err, expectedErr)
		}
		if m.generateCallCount < 2 || m.generateCallCount > 4 {
			t.Errorf("Expected Generate to be called a few times (e.g. 2-4), got %d", m.generateCallCount)
		}
	})
	t.Run("RetryGenerator/Generate/MaxRetriesExceeded/WithMaxTries", func(t *testing.T) {
		expectedErr := &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "persistent rate limit again"}
		m := &mockGenerator{}
		var attempts uint
		m.GenerateFunc = func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
			attempts++
			return gai.Response{}, expectedErr
		}

		maxAttempts := uint(3)
		config := constantRetryConfig(time.Millisecond)
		config.MaxAttempts = maxAttempts
		config.MaxElapsedTime = time.Second
		rg := gai.NewRetryGenerator(m, config)

		_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if !errors.Is(err, expectedErr) {
			t.Fatalf("Generate() error = %v, want %v", err, expectedErr)
		}
		if attempts != maxAttempts {
			t.Errorf("Expected Generate to be called %d times, got %d", maxAttempts, attempts)
		}
		if m.generateCallCount != int(maxAttempts) {
			t.Errorf("Expected mock generator call count to be %d, got %d", maxAttempts, m.generateCallCount)
		}
	})
	t.Run("RetryGenerator/Generate/PermanentError", func(t *testing.T) {
		permanentErr := errors.New("permanent error")
		m := &mockGenerator{
			GenerateFunc: func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
				return gai.Response{}, permanentErr
			},
		}

		config := constantRetryConfig(1 * time.Millisecond)
		rg := gai.NewRetryGenerator(m, config)

		_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if !errors.Is(err, permanentErr) {
			t.Fatalf("Generate() error = %v, want %v", err, permanentErr)
		}
		if m.generateCallCount != 1 { // Should only be called once
			t.Errorf("Expected Generate to be called 1 time, got %d", m.generateCallCount)
		}
	})
	t.Run("RetryGenerator/Generate/PermanentError/ContextCanceledByGenerator", func(t *testing.T) {
		genErr := context.Canceled
		m := &mockGenerator{
			GenerateFunc: func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
				return gai.Response{}, genErr
			},
		}
		rg := gai.NewRetryGenerator(m, gai.RetryConfig{})

		_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if !errors.Is(err, genErr) {
			t.Fatalf("Generate() error = %v, want %v", err, genErr)
		}
		if m.generateCallCount != 1 {
			t.Errorf("Expected Generate to be called 1 time, got %d", m.generateCallCount)
		}
	})
	t.Run("RetryGenerator/Generate/RetryAfterDoesNotOverrideStoppingBackoff", func(t *testing.T) {
		retryAfter := time.Duration(0)
		retryableErr := &gai.ApiErr{
			Kind:               gai.APIErrorKindRateLimit,
			RetryAfterDuration: &retryAfter,
		}
		m := &mockGenerator{
			GenerateFunc: func(context.Context, gai.GenerationRequest) (gai.Response, error) {
				return gai.Response{}, retryableErr
			},
		}
		var backoffCalls int
		config := gai.RetryConfig{
			Backoff: func(uint) (time.Duration, bool) {
				backoffCalls++
				return 0, false
			},
		}
		rg := gai.NewRetryGenerator(m, config)

		_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if err != retryableErr {
			t.Fatalf("Generate() error = %T %v, want original ApiErr", err, err)
		}
		if m.generateCallCount != 1 || backoffCalls != 1 {
			t.Fatalf("calls = (generate: %d, backoff: %d), want (1, 1)", m.generateCallCount, backoffCalls)
		}
	})
	t.Run("RetryGenerator/Generate/RetryAfterRespectsMaxElapsedTime", func(t *testing.T) {
		retryAfter := time.Hour
		expectedErr := &gai.ApiErr{
			Provider:           gai.ProviderOpenAI,
			Kind:               gai.APIErrorKindRateLimit,
			RetryAfterDuration: &retryAfter,
		}
		m := &mockGenerator{
			GenerateFunc: func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
				return gai.Response{}, expectedErr
			},
		}
		var notified bool
		config := constantRetryConfig(0)
		config.MaxElapsedTime = time.Second
		config.Notify = func(error, time.Duration) {
			notified = true
		}
		rg := gai.NewRetryGenerator(m, config)

		_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if err != expectedErr {
			t.Fatalf("Generate() error = %T %v, want original ApiErr", err, err)
		}
		if m.generateCallCount != 1 {
			t.Fatalf("Generate() calls = %d, want 1", m.generateCallCount)
		}
		if notified {
			t.Fatal("retry notification called even though Retry-After exceeded MaxElapsedTime")
		}
	})
	t.Run("RetryGenerator/Generate/RetryAfterReturnsOriginalError", func(t *testing.T) {
		retryAfter := time.Duration(0)
		expectedErr := &gai.ApiErr{
			Provider:           gai.ProviderOpenAI,
			Kind:               gai.APIErrorKindRateLimit,
			RetryAfterDuration: &retryAfter,
		}
		m := &mockGenerator{
			GenerateFunc: func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
				return gai.Response{}, expectedErr
			},
		}
		rg := gai.NewRetryGenerator(m, gai.RetryConfig{MaxAttempts: 1})

		_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if err != expectedErr {
			t.Fatalf("Generate() error = %T %v, want original ApiErr", err, err)
		}
	})
	t.Run("RetryGenerator/Generate/RetryAndSucceed", func(t *testing.T) {
		testCases := []struct {
			name          string
			retriableErr  error
			expectedCalls int
		}{
			{
				name:          "ApiErr rate limit",
				retriableErr:  &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "too many requests"},
				expectedCalls: 2,
			},
			{
				name:          "ApiErr 429",
				retriableErr:  &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "too many requests"},
				expectedCalls: 2,
			},
			{
				name:          "ApiErr 500",
				retriableErr:  &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindServer, StatusCode: http.StatusInternalServerError, Message: "internal server error"},
				expectedCalls: 2,
			},
			{
				name:          "ApiErr overloaded without status",
				retriableErr:  &gai.ApiErr{Provider: gai.ProviderAnthropic, Kind: gai.APIErrorKindOverloaded, Message: "temporarily overloaded"},
				expectedCalls: 2,
			},
			{
				name:          "ContextDeadlineExceeded",
				retriableErr:  context.DeadlineExceeded,
				expectedCalls: 2,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				m := &mockGenerator{}
				callCount := 0
				m.GenerateFunc = func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
					callCount++
					if callCount < tc.expectedCalls {
						return gai.Response{}, tc.retriableErr
					}
					return gai.Response{Candidates: []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{gai.TextBlock("Success")}}}}, nil
				}

				config := constantRetryConfig(1 * time.Millisecond)
				rg := gai.NewRetryGenerator(m, config)

				resp, err := rg.Generate(context.Background(), gai.GenerationRequest{})
				if err != nil {
					t.Fatalf("Generate() error = %v, wantErr %v", err, false)
				}
				if len(resp.Candidates) != 1 || resp.Candidates[0].Blocks[0].Content.String() != "Success" {
					t.Errorf("Generate() resp.Candidates[0].Blocks[0].Content.String() = %s, want %s", resp.Candidates[0].Blocks[0].Content.String(), "Success")
				}
				if m.generateCallCount != tc.expectedCalls {
					t.Errorf("Expected Generate to be called %d times, got %d", tc.expectedCalls, m.generateCallCount)
				}
				m.ResetCallCount()
			})
		}
	})
	t.Run("RetryGenerator/Generate/SuccessFirstAttempt", func(t *testing.T) {
		m := &mockGenerator{
			GenerateFunc: func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
				return gai.Response{Candidates: []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{gai.TextBlock("Hello")}}}}, nil
			},
		}
		rg := gai.NewRetryGenerator(m, gai.RetryConfig{})

		resp, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if err != nil {
			t.Fatalf("Generate() error = %v, wantErr %v", err, false)
		}
		if len(resp.Candidates) != 1 || resp.Candidates[0].Blocks[0].Content.String() != "Hello" {
			t.Errorf("Generate() resp.Candidates[0].Blocks[0].Content.String() = %s, want %s", resp.Candidates[0].Blocks[0].Content.String(), "Hello")
		}
		if m.generateCallCount != 1 {
			t.Errorf("Expected Generate to be called 1 time, got %d", m.generateCallCount)
		}
	})
	t.Run("RetryGenerator/Generate/UsesBackoffWithoutRetryAfter", func(t *testing.T) {
		fallbackDelay := 3 * time.Millisecond
		retryableErr := &gai.ApiErr{
			Provider:   gai.ProviderOpenAI,
			Kind:       gai.APIErrorKindRateLimit,
			StatusCode: http.StatusTooManyRequests,
			Message:    "too many requests",
		}

		m := &mockGenerator{}
		m.GenerateFunc = func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
			if m.generateCallCount == 1 {
				return gai.Response{}, retryableErr
			}
			return gai.Response{}, nil
		}

		var notifiedDelay time.Duration
		config := constantRetryConfig(fallbackDelay)
		config.MaxAttempts = 2
		config.Notify = func(_ error, delay time.Duration) {
			notifiedDelay = delay
		}
		rg := gai.NewRetryGenerator(m, config)

		_, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if notifiedDelay != fallbackDelay {
			t.Fatalf("notified delay = %s, want fallback %s", notifiedDelay, fallbackDelay)
		}
	})
	t.Run("RetryGenerator/Generate/UsesIndependentBackoffStatePerCall", func(t *testing.T) {
		retryableErr := &gai.ApiErr{
			Provider:   gai.ProviderOpenAI,
			Kind:       gai.APIErrorKindRateLimit,
			StatusCode: http.StatusTooManyRequests,
		}
		generator := generatorFunc(func(ctx context.Context, _ gai.GenerationRequest) (gai.Response, error) {
			attempts := ctx.Value(retryAttemptsContextKey{}).(*atomic.Int32)
			if attempts.Add(1) == 1 {
				return gai.Response{}, retryableErr
			}
			return gai.Response{}, nil
		})

		var backoffCalls atomic.Int32
		var unexpectedRetryNumbers atomic.Int32
		config := gai.RetryConfig{
			Backoff: func(retry uint) (time.Duration, bool) {
				backoffCalls.Add(1)
				if retry != 1 {
					unexpectedRetryNumbers.Add(1)
				}
				return 0, true
			},
			MaxAttempts: 2,
		}
		rg := gai.NewRetryGenerator(generator, config)

		const callCount = 32
		start := make(chan struct{})
		errs := make(chan error, callCount)
		var waitGroup sync.WaitGroup
		for range callCount {
			waitGroup.Go(func() {
				<-start

				attempts := &atomic.Int32{}
				ctx := context.WithValue(context.Background(), retryAttemptsContextKey{}, attempts)
				if _, err := rg.Generate(ctx, gai.GenerationRequest{}); err != nil {
					errs <- err
					return
				}
				if got := attempts.Load(); got != 2 {
					errs <- fmt.Errorf("Generate() attempts = %d, want 2", got)
				}
			})
		}
		close(start)
		waitGroup.Wait()
		close(errs)

		for err := range errs {
			t.Errorf("concurrent Generate(): %v", err)
		}
		if got := backoffCalls.Load(); got != callCount {
			t.Fatalf("backoff calls = %d, want %d", got, callCount)
		}
		if got := unexpectedRetryNumbers.Load(); got != 0 {
			t.Fatalf("backoff received %d retry numbers other than 1", got)
		}
	})
	t.Run("RetryGenerator/Generate/WithExplicitDefaultConfig", func(t *testing.T) {
		m := &mockGenerator{}
		callCount := 0
		m.GenerateFunc = func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
			callCount++
			if callCount < 2 {
				return gai.Response{}, &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "transient error"}
			}
			return gai.Response{Candidates: []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{gai.TextBlock("Success")}}}}, nil
		}

		rg := gai.NewRetryGenerator(m, gai.DefaultRetryConfig())
		resp, err := rg.Generate(context.Background(), gai.GenerationRequest{})
		if err != nil {
			t.Fatalf("Generate() error = %v, wantErr %v", err, false)
		}
		if len(resp.Candidates) != 1 || resp.Candidates[0].Blocks[0].Content.String() != "Success" {
			t.Errorf("Generate() resp.Candidates[0].Blocks[0].Content.String() = %s, want %s", resp.Candidates[0].Blocks[0].Content.String(), "Success")
		}
		if m.generateCallCount != 2 {
			t.Errorf("Expected Generate to be called 2 times, got %d", m.generateCallCount)
		}
	})
	t.Run("RetryGenerator/Generate/ZeroMaxElapsedTimeHasNoLimit", func(t *testing.T) {
		retryAfter := 2 * time.Minute
		expectedErr := &gai.ApiErr{
			Provider:           gai.ProviderOpenAI,
			Kind:               gai.APIErrorKindRateLimit,
			RetryAfterDuration: &retryAfter,
		}
		m := &mockGenerator{
			GenerateFunc: func(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
				return gai.Response{}, expectedErr
			},
		}
		var notified bool
		var notifiedDelay time.Duration
		config := constantRetryConfig(0)
		config.Notify = func(_ error, delay time.Duration) {
			notified = true
			notifiedDelay = delay
		}
		rg := gai.NewRetryGenerator(m, config)
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		_, err := rg.Generate(ctx, gai.GenerationRequest{})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("Generate() error = %T %v, want context deadline", err, err)
		}
		if m.generateCallCount != 1 {
			t.Fatalf("Generate() calls = %d, want 1", m.generateCallCount)
		}
		if !notified || notifiedDelay != retryAfter {
			t.Fatalf("retry notification = (%t, %s), want (true, %s)", notified, notifiedDelay, retryAfter)
		}
	})
	t.Run("RetryGenerator/Stream/ContextCancelled/BeforeFirstAttempt", func(t *testing.T) {
		m := &mockGenerator{}
		m.StreamFunc = func(ctx context.Context, request gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
			t.Fatal("Stream should not have been called")
			return nil
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		chunks, err := collectStream(gai.NewRetryGenerator(m, gai.RetryConfig{}).Stream(ctx, gai.GenerationRequest{}))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Stream() error = %v, want %v", err, context.Canceled)
		}
		if len(chunks) != 0 {
			t.Fatalf("expected no chunks, got %+v", chunks)
		}
		if m.streamCallCount != 0 {
			t.Fatalf("expected Stream to be called 0 times, got %d", m.streamCallCount)
		}
	})
	t.Run("RetryGenerator/Stream/DoesNotRetryAfterFirstChunk", func(t *testing.T) {
		retriableErr := &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindServer, StatusCode: http.StatusInternalServerError, Message: "temporary upstream failure"}
		m := &mockGenerator{}
		m.StreamFunc = func(ctx context.Context, request gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
			return func(yield func(gai.StreamChunk) bool) {
				if !yield(gai.StreamChunk{Block: gai.TextBlock("partial")}) {
					return
				}
				yield(gai.StreamChunk{Err: retriableErr})
			}
		}

		rg := gai.NewRetryGenerator(m, gai.RetryConfig{})
		chunks, err := collectStream(rg.Stream(context.Background(), gai.GenerationRequest{}))
		if !errors.Is(err, retriableErr) {
			t.Fatalf("Stream() error = %v, want %v", err, retriableErr)
		}
		if len(chunks) != 1 || chunks[0].Block.Content.String() != "partial" {
			t.Fatalf("unexpected streamed chunks: %+v", chunks)
		}
		if m.streamCallCount != 1 {
			t.Fatalf("expected Stream to be called once, got %d", m.streamCallCount)
		}
	})
	t.Run("RetryGenerator/Stream/RetryAndSucceedBeforeFirstChunk", func(t *testing.T) {
		retryAfter := time.Duration(0)
		retriableErr := &gai.ApiErr{
			Provider:           gai.ProviderOpenAI,
			Kind:               gai.APIErrorKindRateLimit,
			StatusCode:         http.StatusTooManyRequests,
			Message:            "too many requests",
			RetryAfterDuration: &retryAfter,
		}
		m := &mockGenerator{}
		m.StreamFunc = func(ctx context.Context, request gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
			attempt := m.streamCallCount
			return func(yield func(gai.StreamChunk) bool) {
				if attempt == 1 {
					yield(gai.StreamChunk{Err: retriableErr})
					return
				}
				yield(gai.StreamChunk{Block: gai.TextBlock("Success")})
			}
		}

		config := constantRetryConfig(time.Minute)
		config.MaxAttempts = 2
		rg := gai.NewRetryGenerator(m, config)
		chunks, err := collectStream(rg.Stream(context.Background(), gai.GenerationRequest{}))
		if err != nil {
			t.Fatalf("Stream() error = %v, want nil", err)
		}
		if len(chunks) != 1 || chunks[0].Block.Content.String() != "Success" {
			t.Fatalf("unexpected streamed chunks: %+v", chunks)
		}
		if m.streamCallCount != 2 {
			t.Fatalf("expected Stream to be called twice, got %d", m.streamCallCount)
		}
	})
	t.Run("RetryGenerator/Stream/SuccessFirstAttempt", func(t *testing.T) {
		m := &mockGenerator{
			StreamFunc: func(ctx context.Context, request gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
				return func(yield func(gai.StreamChunk) bool) {
					yield(gai.StreamChunk{Block: gai.TextBlock("Hello")})
				}
			},
		}
		chunks, err := collectStream(gai.NewRetryGenerator(m, gai.RetryConfig{}).Stream(context.Background(), gai.GenerationRequest{}))
		if err != nil {
			t.Fatalf("Stream() error = %v, want nil", err)
		}
		if len(chunks) != 1 || chunks[0].Block.Content.String() != "Hello" {
			t.Fatalf("unexpected streamed chunks: %+v", chunks)
		}
		if m.streamCallCount != 1 {
			t.Fatalf("expected Stream to be called once, got %d", m.streamCallCount)
		}
	})
	t.Run("RetryGenerator/Stream/UnderlyingDoesNotImplementStreamingGenerator", func(t *testing.T) {
		type nonStreamingGenerator struct{ gai.Generator }
		underlying := &nonStreamingGenerator{Generator: &mockGenerator{}}
		rg := gai.NewRetryGenerator(underlying, gai.RetryConfig{})

		chunks, err := collectStream(rg.Stream(context.Background(), gai.GenerationRequest{}))
		if err == nil {
			t.Fatal("Stream() error = nil, want an error")
		}
		if len(chunks) != 0 {
			t.Fatalf("expected no chunks, got %+v", chunks)
		}
		wantErrStr := fmt.Sprintf("inner generator of type %T does not implement StreamingGenerator", underlying)
		if err.Error() != wantErrStr {
			t.Fatalf("Stream() error = %q, want %q", err.Error(), wantErrStr)
		}
	})
}

var _ gai.StreamingGenerator = (*mockGenerator)(nil)
