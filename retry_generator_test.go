package gai_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/cenkalti/backoff/v5"
	"github.com/spachava753/gai"
)

// mockGenerator is a mock implementation of the gai.Generator and gai.TokenCounter interfaces for testing.
type mockGenerator struct {
	GenerateFunc      func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error)
	CountFunc         func(ctx context.Context, dialog gai.Dialog) (uint, error)
	generateCallCount int
}

func (m *mockGenerator) Generate(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
	m.generateCallCount++
	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, dialog, options)
	}
	return gai.Response{}, errors.New("GenerateFunc not implemented")
}

func (m *mockGenerator) Count(ctx context.Context, dialog gai.Dialog) (uint, error) {
	if m.CountFunc != nil {
		return m.CountFunc(ctx, dialog)
	}
	return 0, errors.New("CountFunc not implemented")
}

func (m *mockGenerator) ResetCallCount() {
	m.generateCallCount = 0
}

func TestRetryGenerator_Generate_SuccessFirstAttempt(t *testing.T) {
	m := &mockGenerator{
		GenerateFunc: func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
			return gai.Response{Candidates: []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{gai.TextBlock("Hello")}}}}, nil
		},
	}
	// Use StopBackOff as the base policy and no additional options for no retries.
	// The default MaxElapsedTime from RetryGenerator will be overridden by StopBackOff behavior.
	rg := gai.NewRetryGenerator(m, &backoff.StopBackOff{})

	resp, err := rg.Generate(context.Background(), gai.Dialog{}, nil)
	if err != nil {
		t.Fatalf("Generate() error = %v, wantErr %v", err, false)
	}
	if len(resp.Candidates) != 1 || resp.Candidates[0].Blocks[0].Content.String() != "Hello" {
		t.Errorf("Generate() resp.Candidates[0].Blocks[0].Content.String() = %s, want %s", resp.Candidates[0].Blocks[0].Content.String(), "Hello")
	}
	if m.generateCallCount != 1 {
		t.Errorf("Expected Generate to be called 1 time, got %d", m.generateCallCount)
	}
}

func TestRetryGenerator_Generate_RetryAndSucceed(t *testing.T) {
	testCases := []struct {
		name          string
		retriableErr  error
		expectedCalls int
	}{
		{
			name:          "RateLimitErr",
			retriableErr:  gai.RateLimitErr("test rate limit"),
			expectedCalls: 2,
		},
		{
			name:          "ApiErr 429",
			retriableErr:  gai.ApiErr{StatusCode: http.StatusTooManyRequests, Message: "too many requests"},
			expectedCalls: 2,
		},
		{
			name:          "ApiErr 500",
			retriableErr:  gai.ApiErr{StatusCode: http.StatusInternalServerError, Message: "internal server error"},
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
			m.GenerateFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
				callCount++
				if callCount < tc.expectedCalls {
					return gai.Response{}, tc.retriableErr
				}
				return gai.Response{Candidates: []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{gai.TextBlock("Success")}}}}, nil
			}

			constantBackoff := backoff.NewConstantBackOff(1 * time.Millisecond)
			// No specific retry options means it will use the default MaxElapsedTime from RetryGenerator.
			rg := gai.NewRetryGenerator(m, constantBackoff)

			resp, err := rg.Generate(context.Background(), gai.Dialog{}, nil)
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
}

func TestRetryGenerator_Generate_PermanentError(t *testing.T) {
	permanentErr := errors.New("permanent error")
	m := &mockGenerator{
		GenerateFunc: func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
			return gai.Response{}, permanentErr
		},
	}

	constantBackoff := backoff.NewConstantBackOff(1 * time.Millisecond)
	// No specific retry options, default MaxElapsedTime from RetryGenerator will apply.
	rg := gai.NewRetryGenerator(m, constantBackoff)

	_, err := rg.Generate(context.Background(), gai.Dialog{}, nil)
	if !errors.Is(err, permanentErr) {
		t.Fatalf("Generate() error = %v, want %v", err, permanentErr)
	}
	if m.generateCallCount != 1 { // Should only be called once
		t.Errorf("Expected Generate to be called 1 time, got %d", m.generateCallCount)
	}
}

func TestRetryGenerator_Generate_ContextCancelled_DuringBackoff(t *testing.T) {
	m := &mockGenerator{}
	callCount := 0
	m.GenerateFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
		callCount++
		return gai.Response{}, gai.RateLimitErr("rate limited") // Always return a retriable error
	}

	bo := backoff.NewExponentialBackOff()
	bo.InitialInterval = 100 * time.Millisecond
	// Pass a specific MaxElapsedTime option to ensure the test completes reasonably.
	rg := gai.NewRetryGenerator(m, bo, backoff.WithMaxElapsedTime(5*time.Second))

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		time.Sleep(50 * time.Millisecond) // Wait less than InitialInterval
		cancel()
	}()

	_, err := rg.Generate(ctx, gai.Dialog{}, nil)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate() error = %v, want %v", err, context.Canceled)
	}
	if m.generateCallCount < 1 {
		t.Errorf("Expected Generate to be called at least 1 time, got %d", m.generateCallCount)
	}
}

func TestRetryGenerator_Generate_ContextCancelled_BeforeFirstCall(t *testing.T) {
	m := &mockGenerator{}
	m.GenerateFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
		t.Error("Generate should not have been called")
		return gai.Response{}, nil
	}
	rg := gai.NewRetryGenerator(m, backoff.NewExponentialBackOff()) // Uses default RetryOptions from RG

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := rg.Generate(ctx, gai.Dialog{}, nil)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate() error = %v, want %v", err, context.Canceled)
	}
	if m.generateCallCount != 0 {
		t.Errorf("Expected Generate to be called 0 times, got %d", m.generateCallCount)
	}
}

func TestRetryGenerator_Generate_MaxRetriesExceeded_WithMaxElapsedTime(t *testing.T) {
	expectedErr := gai.RateLimitErr("persistent rate limit")
	m := &mockGenerator{}
	m.GenerateFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
		return gai.Response{}, expectedErr
	}

	bo := backoff.NewExponentialBackOff()
	bo.InitialInterval = 1 * time.Millisecond
	bo.MaxInterval = 2 * time.Millisecond

	rg := gai.NewRetryGenerator(m, bo, backoff.WithMaxElapsedTime(4*time.Millisecond))

	_, err := rg.Generate(context.Background(), gai.Dialog{}, nil)

	if !errors.Is(err, expectedErr) {
		t.Fatalf("Generate() error = %v, want %v", err, expectedErr)
	}
	if m.generateCallCount < 2 || m.generateCallCount > 4 {
		t.Errorf("Expected Generate to be called a few times (e.g. 2-4), got %d", m.generateCallCount)
	}
}

func TestRetryGenerator_Generate_MaxRetriesExceeded_WithMaxTries(t *testing.T) {
	expectedErr := gai.RateLimitErr("persistent rate limit again")
	m := &mockGenerator{}
	var attempts uint = 0
	m.GenerateFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
		attempts++
		return gai.Response{}, expectedErr
	}

	bo := backoff.NewConstantBackOff(1 * time.Millisecond) // Use constant backoff for predictable attempt counts
	maxAttempts := uint(3)

	// Pass WithMaxTries as a user option. Also pass a MaxElapsedTime to prevent test hanging if MaxTries is buggy.
	rg := gai.NewRetryGenerator(m, bo, backoff.WithMaxTries(maxAttempts), backoff.WithMaxElapsedTime(1*time.Second))

	_, err := rg.Generate(context.Background(), gai.Dialog{}, nil)

	if !errors.Is(err, expectedErr) {
		t.Fatalf("Generate() error = %v, want %v", err, expectedErr)
	}
	// WithMaxRetries(n) allows n attempts. The backoff package's Retry function counts the first attempt.
	if attempts != maxAttempts {
		t.Errorf("Expected Generate to be called %d times, got %d", maxAttempts, attempts)
	}
	if m.generateCallCount != int(maxAttempts) {
		t.Errorf("Expected mock generator call count to be %d, got %d", maxAttempts, m.generateCallCount)
	}
}

func TestRetryGenerator_Count_UnderlyingImplementsTokenCounter(t *testing.T) {
	expectedCount := uint(123)
	m := &mockGenerator{
		CountFunc: func(ctx context.Context, dialog gai.Dialog) (uint, error) {
			return expectedCount, nil
		},
	}
	rg := gai.NewRetryGenerator(m, nil) // Default base policy, default retry options

	count, err := rg.Count(context.Background(), gai.Dialog{})
	if err != nil {
		t.Fatalf("Count() error = %v, wantErr false", err)
	}
	if count != expectedCount {
		t.Errorf("Count() = %d, want %d", count, expectedCount)
	}
}

func TestRetryGenerator_Count_UnderlyingDoesNotImplementTokenCounter(t *testing.T) {
	type nonCountingGenerator struct{ gai.Generator }
	underlying := &nonCountingGenerator{Generator: &mockGenerator{}}
	rg := gai.NewRetryGenerator(underlying, nil) // Default base policy, default retry options

	_, err := rg.Count(context.Background(), gai.Dialog{})
	if err == nil {
		t.Fatal("Count() error = nil, want an error")
	}
	wantErrStr := fmt.Sprintf("underlying generator of type %T does not implement TokenCounter", underlying)
	if err.Error() != wantErrStr {
		t.Errorf("Count() error = %q, want %q", err.Error(), wantErrStr)
	}
}

func TestRetryGenerator_Generate_WithDefaultSettings(t *testing.T) {
	m := &mockGenerator{}
	callCount := 0
	m.GenerateFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
		callCount++
		if callCount < 2 {
			return gai.Response{}, gai.RateLimitErr("transient error")
		}
		return gai.Response{Candidates: []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{gai.TextBlock("Success")}}}}, nil
	}

	// Pass nil for base backoff policy and no retry options to use defaults for both.
	rg := gai.NewRetryGenerator(m, nil)

	resp, err := rg.Generate(context.Background(), gai.Dialog{}, nil)
	if err != nil {
		t.Fatalf("Generate() error = %v, wantErr %v", err, false)
	}
	if len(resp.Candidates) != 1 || resp.Candidates[0].Blocks[0].Content.String() != "Success" {
		t.Errorf("Generate() resp.Candidates[0].Blocks[0].Content.String() = %s, want %s", resp.Candidates[0].Blocks[0].Content.String(), "Success")
	}
	if m.generateCallCount != 2 {
		t.Errorf("Expected Generate to be called 2 times, got %d", m.generateCallCount)
	}
}

func TestRetryGenerator_Generate_ContextCancelled_DuringOperation(t *testing.T) {
	opDuration := 100 * time.Millisecond
	cancelDelay := 20 * time.Millisecond

	m := &mockGenerator{
		GenerateFunc: func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
			select {
			case <-time.After(opDuration):
				return gai.Response{}, errors.New("operation should have been cancelled")
			case <-ctx.Done():
				return gai.Response{}, ctx.Err()
			}
		},
	}
	rg := gai.NewRetryGenerator(m, &backoff.StopBackOff{}) // Ensure only one attempt

	ctx, cancel := context.WithTimeout(context.Background(), cancelDelay)
	defer cancel()

	_, err := rg.Generate(ctx, gai.Dialog{}, nil)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Generate() error = %v, want %v", err, context.DeadlineExceeded)
	}
	if m.generateCallCount != 1 {
		t.Errorf("Expected Generate to be called 1 time, got %d", m.generateCallCount)
	}
}

func TestRetryGenerator_Generate_PermanentError_ContextCanceledByGenerator(t *testing.T) {
	genErr := context.Canceled
	m := &mockGenerator{
		GenerateFunc: func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
			return gai.Response{}, genErr
		},
	}
	constantBackoff := backoff.NewConstantBackOff(1 * time.Millisecond)
	rg := gai.NewRetryGenerator(m, constantBackoff) // Default retry options (includes MaxElapsedTime)

	_, err := rg.Generate(context.Background(), gai.Dialog{}, nil)

	if !errors.Is(err, genErr) {
		t.Fatalf("Generate() error = %v, want %v", err, genErr)
	}
	if m.generateCallCount != 1 {
		t.Errorf("Expected Generate to be called 1 time, got %d", m.generateCallCount)
	}
}
