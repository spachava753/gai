package gai_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"testing"
	"time"

	"github.com/cenkalti/backoff/v5"
	"github.com/spachava753/gai"
)

// mockGenerator is a mock implementation of the gai.Generator, gai.TokenCounter,
// gai.ToolCapableGenerator, and gai.StreamingGenerator interfaces for testing.
type mockGenerator struct {
	GenerateFunc func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error)
	StreamFunc   func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) iter.Seq2[gai.StreamChunk, error]
	CountFunc    func(ctx context.Context, dialog gai.Dialog) (uint, error)
	RegisterFunc func(tool gai.Tool) error

	generateCallCount int
	streamCallCount   int
}

func (m *mockGenerator) Generate(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
	m.generateCallCount++
	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, dialog, options)
	}
	return gai.Response{}, errors.New("GenerateFunc not implemented")
}

func (m *mockGenerator) Stream(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) iter.Seq2[gai.StreamChunk, error] {
	m.streamCallCount++
	if m.StreamFunc != nil {
		return m.StreamFunc(ctx, dialog, options)
	}
	return func(yield func(gai.StreamChunk, error) bool) {
		yield(gai.StreamChunk{}, errors.New("StreamFunc not implemented"))
	}
}

func (m *mockGenerator) Count(ctx context.Context, dialog gai.Dialog) (uint, error) {
	if m.CountFunc != nil {
		return m.CountFunc(ctx, dialog)
	}
	return 0, errors.New("CountFunc not implemented")
}

// Register implements the gai.ToolRegister interface for the mock.
func (m *mockGenerator) Register(tool gai.Tool) error {
	if m.RegisterFunc != nil {
		return m.RegisterFunc(tool)
	}
	// This default error helps catch tests where Register is called unexpectedly.
	return errors.New("mockGenerator.RegisterFunc was not set")
}

func (m *mockGenerator) ResetCallCount() {
	m.generateCallCount = 0
	m.streamCallCount = 0
}

func collectStream(seq iter.Seq2[gai.StreamChunk, error]) ([]gai.StreamChunk, error) {
	var chunks []gai.StreamChunk
	for chunk, err := range seq {
		if err != nil {
			return chunks, err
		}
		chunks = append(chunks, chunk)
	}
	return chunks, nil
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
	m.GenerateFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
		return gai.Response{}, &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "rate limited"}
	}

	bo := backoff.NewExponentialBackOff()
	bo.InitialInterval = 100 * time.Millisecond
	rg := gai.NewRetryGenerator(m, bo, backoff.WithMaxElapsedTime(5*time.Second))

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
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
	rg := gai.NewRetryGenerator(m, backoff.NewExponentialBackOff())

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := rg.Generate(ctx, gai.Dialog{}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate() error = %v, want %v", err, context.Canceled)
	}
	if m.generateCallCount != 0 {
		t.Errorf("Expected Generate to be called 0 times, got %d", m.generateCallCount)
	}
}

func TestRetryGenerator_Generate_MaxRetriesExceeded_WithMaxElapsedTime(t *testing.T) {
	expectedErr := &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "persistent rate limit"}
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
	expectedErr := &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "persistent rate limit again"}
	m := &mockGenerator{}
	var attempts uint
	m.GenerateFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
		attempts++
		return gai.Response{}, expectedErr
	}

	bo := backoff.NewConstantBackOff(1 * time.Millisecond)
	maxAttempts := uint(3)
	rg := gai.NewRetryGenerator(m, bo, backoff.WithMaxTries(maxAttempts), backoff.WithMaxElapsedTime(1*time.Second))

	_, err := rg.Generate(context.Background(), gai.Dialog{}, nil)
	if !errors.Is(err, expectedErr) {
		t.Fatalf("Generate() error = %v, want %v", err, expectedErr)
	}
	if attempts != maxAttempts {
		t.Errorf("Expected Generate to be called %d times, got %d", maxAttempts, attempts)
	}
	if m.generateCallCount != int(maxAttempts) {
		t.Errorf("Expected mock generator call count to be %d, got %d", maxAttempts, m.generateCallCount)
	}
}

func TestRetryGenerator_Stream_SuccessFirstAttempt(t *testing.T) {
	m := &mockGenerator{
		StreamFunc: func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) iter.Seq2[gai.StreamChunk, error] {
			return func(yield func(gai.StreamChunk, error) bool) {
				yield(gai.StreamChunk{Block: gai.TextBlock("Hello")}, nil)
			}
		},
	}
	chunks, err := collectStream(gai.NewRetryGenerator(m, &backoff.StopBackOff{}).Stream(context.Background(), gai.Dialog{}, nil))
	if err != nil {
		t.Fatalf("Stream() error = %v, want nil", err)
	}
	if len(chunks) != 1 || chunks[0].Block.Content.String() != "Hello" {
		t.Fatalf("unexpected streamed chunks: %+v", chunks)
	}
	if m.streamCallCount != 1 {
		t.Fatalf("expected Stream to be called once, got %d", m.streamCallCount)
	}
}

func TestRetryGenerator_Stream_RetryAndSucceedBeforeFirstChunk(t *testing.T) {
	retriableErr := &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "too many requests"}
	m := &mockGenerator{}
	m.StreamFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) iter.Seq2[gai.StreamChunk, error] {
		attempt := m.streamCallCount
		return func(yield func(gai.StreamChunk, error) bool) {
			if attempt == 1 {
				yield(gai.StreamChunk{}, retriableErr)
				return
			}
			yield(gai.StreamChunk{Block: gai.TextBlock("Success")}, nil)
		}
	}

	rg := gai.NewRetryGenerator(m, backoff.NewConstantBackOff(1*time.Millisecond))
	chunks, err := collectStream(rg.Stream(context.Background(), gai.Dialog{}, nil))
	if err != nil {
		t.Fatalf("Stream() error = %v, want nil", err)
	}
	if len(chunks) != 1 || chunks[0].Block.Content.String() != "Success" {
		t.Fatalf("unexpected streamed chunks: %+v", chunks)
	}
	if m.streamCallCount != 2 {
		t.Fatalf("expected Stream to be called twice, got %d", m.streamCallCount)
	}
}

func TestRetryGenerator_Stream_DoesNotRetryAfterFirstChunk(t *testing.T) {
	retriableErr := &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindServer, StatusCode: http.StatusInternalServerError, Message: "temporary upstream failure"}
	m := &mockGenerator{}
	m.StreamFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) iter.Seq2[gai.StreamChunk, error] {
		return func(yield func(gai.StreamChunk, error) bool) {
			if !yield(gai.StreamChunk{Block: gai.TextBlock("partial")}, nil) {
				return
			}
			yield(gai.StreamChunk{}, retriableErr)
		}
	}

	rg := gai.NewRetryGenerator(m, backoff.NewConstantBackOff(1*time.Millisecond))
	chunks, err := collectStream(rg.Stream(context.Background(), gai.Dialog{}, nil))
	if !errors.Is(err, retriableErr) {
		t.Fatalf("Stream() error = %v, want %v", err, retriableErr)
	}
	if len(chunks) != 1 || chunks[0].Block.Content.String() != "partial" {
		t.Fatalf("unexpected streamed chunks: %+v", chunks)
	}
	if m.streamCallCount != 1 {
		t.Fatalf("expected Stream to be called once, got %d", m.streamCallCount)
	}
}

func TestRetryGenerator_Stream_ContextCancelled_BeforeFirstAttempt(t *testing.T) {
	m := &mockGenerator{}
	m.StreamFunc = func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) iter.Seq2[gai.StreamChunk, error] {
		t.Fatal("Stream should not have been called")
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	chunks, err := collectStream(gai.NewRetryGenerator(m, backoff.NewConstantBackOff(1*time.Millisecond)).Stream(ctx, gai.Dialog{}, nil))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Stream() error = %v, want %v", err, context.Canceled)
	}
	if len(chunks) != 0 {
		t.Fatalf("expected no chunks, got %+v", chunks)
	}
	if m.streamCallCount != 0 {
		t.Fatalf("expected Stream to be called 0 times, got %d", m.streamCallCount)
	}
}

func TestRetryGenerator_Stream_UnderlyingDoesNotImplementStreamingGenerator(t *testing.T) {
	type nonStreamingGenerator struct{ gai.Generator }
	underlying := &nonStreamingGenerator{Generator: &mockGenerator{}}
	rg := gai.NewRetryGenerator(underlying, nil)

	chunks, err := collectStream(rg.Stream(context.Background(), gai.Dialog{}, nil))
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
}

func TestRetryGenerator_Count_UnderlyingImplementsTokenCounter(t *testing.T) {
	expectedCount := uint(123)
	m := &mockGenerator{
		CountFunc: func(ctx context.Context, dialog gai.Dialog) (uint, error) {
			return expectedCount, nil
		},
	}
	rg := gai.NewRetryGenerator(m, nil)

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
	rg := gai.NewRetryGenerator(underlying, nil)

	_, err := rg.Count(context.Background(), gai.Dialog{})
	if err == nil {
		t.Fatal("Count() error = nil, want an error")
	}
	wantErrStr := fmt.Sprintf("inner generator of type %T does not implement TokenCounter", underlying)
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
			return gai.Response{}, &gai.ApiErr{Provider: gai.ProviderOpenAI, Kind: gai.APIErrorKindRateLimit, StatusCode: http.StatusTooManyRequests, Message: "transient error"}
		}
		return gai.Response{Candidates: []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{gai.TextBlock("Success")}}}}, nil
	}

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
	rg := gai.NewRetryGenerator(m, &backoff.StopBackOff{})

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
	rg := gai.NewRetryGenerator(m, constantBackoff)

	_, err := rg.Generate(context.Background(), gai.Dialog{}, nil)
	if !errors.Is(err, genErr) {
		t.Fatalf("Generate() error = %v, want %v", err, genErr)
	}
	if m.generateCallCount != 1 {
		t.Errorf("Expected Generate to be called 1 time, got %d", m.generateCallCount)
	}
}

// Ensure mockGenerator can satisfy the optional generator interfaces when its methods are implemented.
var _ gai.ToolCapableGenerator = (*mockGenerator)(nil)
var _ gai.StreamingGenerator = (*mockGenerator)(nil)

func TestRetryGenerator_Register_UnderlyingImplementsToolCapableGenerator(t *testing.T) {
	registeredTool := gai.Tool{Name: "test_tool"}
	var receivedTool gai.Tool
	var registerCalled bool

	m := &mockGenerator{
		RegisterFunc: func(tool gai.Tool) error {
			registerCalled = true
			receivedTool = tool
			if tool.Name == "error_tool" {
				return errors.New("registration failed")
			}
			return nil
		},
	}

	rg := gai.NewRetryGenerator(m, nil)

	err := rg.Register(registeredTool)
	if err != nil {
		t.Fatalf("Register() error = %v, wantErr false", err)
	}
	if !registerCalled {
		t.Error("Expected underlying Register to be called")
	}
	if receivedTool.Name != registeredTool.Name {
		t.Errorf("Register() tool.Name = %s, want %s", receivedTool.Name, registeredTool.Name)
	}

	registerCalled = false
	errTool := gai.Tool{Name: "error_tool"}
	err = rg.Register(errTool)
	if err == nil {
		t.Fatal("Register() error = nil, want an error")
	}
	if !registerCalled {
		t.Error("Expected underlying Register to be called even on error")
	}
	if err.Error() != "registration failed" {
		t.Errorf("Register() error = %q, want %q", err.Error(), "registration failed")
	}
}

func TestRetryGenerator_Register_UnderlyingDoesNotImplementToolCapableGenerator(t *testing.T) {
	type simpleGenerator struct{ gai.Generator }
	underlyingGen := &simpleGenerator{
		Generator: &mockGenerator{
			GenerateFunc: func(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
				return gai.Response{}, nil
			},
		},
	}

	rg := gai.NewRetryGenerator(underlyingGen, nil)
	toolToRegister := gai.Tool{Name: "test_tool"}

	err := rg.Register(toolToRegister)
	if err == nil {
		t.Fatal("Register() error = nil, want an error for non-ToolCapableGenerator")
	}

	wantErrStr := fmt.Sprintf("inner generator of type %T does not implement ToolCapableGenerator", underlyingGen)
	if err.Error() != wantErrStr {
		t.Errorf("Register() error = %q, want %q", err.Error(), wantErrStr)
	}
}
