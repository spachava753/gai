package gai

import (
	"context"
	"errors"
	"iter"
	"math/rand/v2"
	"time"
)

const (
	defaultGenRetryInitialInterval = 500 * time.Millisecond
	defaultGenRetryMaxInterval     = 15 * time.Second
)

// RetryConfig controls retry timing and limits.
type RetryConfig struct {
	// Backoff returns the fallback delay and whether to retry. It authorizes every
	// retry; a provider Retry-After hint replaces only the returned delay. The
	// retry argument starts at 1 and increments for each retry within one Generate
	// or Stream call. A nil Backoff disables retries. When retrying, a non-positive
	// delay retries immediately. The function may be called concurrently.
	Backoff func(retry uint) (time.Duration, bool)
	// MaxAttempts limits total calls, including the initial call. Zero has no limit.
	MaxAttempts uint
	// MaxElapsedTime prevents scheduling a retry whose selected delay would exceed
	// the total retry budget. It is checked only between attempts and does not
	// cancel an in-flight operation; use a context deadline for a hard time limit.
	// A non-positive duration has no retry-scheduling limit.
	MaxElapsedTime time.Duration
	// Notify runs before each retry with the original error and selected delay.
	// It may be called concurrently.
	Notify func(error, time.Duration)
}

// DefaultRetryConfig returns a config with exponential intervals starting at
// 500ms and doubling to a 15s cap, with up to 50% downward jitter and no attempt
// or retry-scheduling limit. Context cancellation governs the hard call limit.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{Backoff: exponentialBackoff(rand.Int64N)}
}

func exponentialBackoff(randomInt64N func(int64) int64) func(uint) (time.Duration, bool) {
	return func(retry uint) (time.Duration, bool) {
		interval := defaultGenRetryInitialInterval
		for current := uint(1); current < retry; current++ {
			if interval >= defaultGenRetryMaxInterval/2 {
				interval = defaultGenRetryMaxInterval
				break
			}
			interval *= 2
		}

		minimum := interval / 2
		jitter := time.Duration(randomInt64N(int64(interval-minimum) + 1))
		return minimum + jitter, true
	}
}

// RetryGenerator is a Generator that wraps another Generator and retries failed
// Generate calls and Stream startup failures according to its retry
// configuration.
//
// It retries on specific errors:
//   - context.DeadlineExceeded (from the Generate/Stream call itself, not the overall context)
//   - gai.ApiErr values classified as retryable (rate limits and transient upstream errors)
//
// Streaming retries are intentionally conservative: once a stream chunk has been
// emitted to the caller, subsequent errors are returned as-is rather than retried,
// because retrying after partial output would duplicate already-observed content.
// If the consumer stops iteration by returning false from yield, the stream ends
// successfully without surfacing an error, following the standard iter.Seq2 contract.
//
// RetryGenerator does not disable retries configured in the wrapped provider SDK.
// To avoid nested retry loops, disable provider retries when constructing OpenAI or
// Anthropic clients with the corresponding option.WithMaxRetries(0) client option.
type RetryGenerator struct {
	GeneratorWrapper             // Embed for default Count/Register delegation and unsupported Stream fallback.
	config           RetryConfig // Copied at construction; mutable retry state is scoped to each invocation.
}

type permanentRetryError struct {
	err error
}

func (e *permanentRetryError) Error() string {
	return e.err.Error()
}

func (e *permanentRetryError) Unwrap() error {
	return e.err
}

type retryDecision struct {
	retryAfter    time.Duration
	retry         bool
	hasRetryAfter bool
}

// NewRetryGenerator creates a RetryGenerator with the provided config.
func NewRetryGenerator(generator Generator, config RetryConfig) *RetryGenerator {
	return &RetryGenerator{
		GeneratorWrapper: GeneratorWrapper{Inner: generator},
		config:           config,
	}
}

func classifyRetryError(err error) retryDecision {
	if errors.Is(err, context.Canceled) {
		return retryDecision{}
	}

	var apiErr *ApiErr
	if errors.As(err, &apiErr) {
		if !apiErr.Retryable() {
			return retryDecision{}
		}
		delay, ok := apiErr.RetryAfter()
		return retryDecision{
			retryAfter:    delay,
			retry:         true,
			hasRetryAfter: ok,
		}
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return retryDecision{retry: true}
	}
	return retryDecision{}
}

func waitForRetry(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		select {
		case <-ctx.Done():
			return context.Cause(ctx)
		default:
			return nil
		}
	}

	select {
	case <-time.After(delay):
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

func retry[T any](ctx context.Context, operation func() (T, error), config RetryConfig) (T, error) {
	startedAt := time.Now()

	for attempt := uint(1); ; attempt++ {
		if cause := context.Cause(ctx); cause != nil {
			var zero T
			return zero, cause
		}

		result, err := operation()
		if err == nil {
			return result, nil
		}

		var permanent *permanentRetryError
		if errors.As(err, &permanent) {
			return result, permanent.err
		}

		decision := classifyRetryError(err)
		if !decision.retry {
			return result, err
		}
		if cause := context.Cause(ctx); cause != nil {
			return result, cause
		}
		if config.MaxAttempts > 0 && attempt >= config.MaxAttempts {
			return result, err
		}

		if config.Backoff == nil {
			return result, err
		}
		delay, shouldRetry := config.Backoff(attempt)
		if !shouldRetry {
			return result, err
		}
		delay = max(delay, 0)
		if decision.hasRetryAfter {
			delay = decision.retryAfter
		}

		if config.MaxElapsedTime > 0 {
			elapsed := time.Since(startedAt)
			if elapsed > config.MaxElapsedTime || delay > config.MaxElapsedTime-elapsed {
				return result, err
			}
		}
		if config.Notify != nil {
			config.Notify(err, delay)
		}
		if err := waitForRetry(ctx, delay); err != nil {
			return result, err
		}
	}
}

// Generate calls the underlying Generator's Generate method, retrying on
// specific errors according to the configured backoff and limits.
// The provided context stops retries and is passed to the underlying generator;
// use a context deadline to bound in-flight operations and total wall-clock time.
func (rg *RetryGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	return retry(ctx, func() (Response, error) {
		return rg.Inner.Generate(ctx, dialog, options)
	}, rg.config)
}

// Stream calls the underlying StreamingGenerator's Stream method, retrying only
// failures that occur before the first chunk is emitted. Once output has been
// observed by the caller, the stream becomes non-retriable to avoid duplicating
// partial content on a subsequent attempt. If yield returns false, Stream stops
// immediately and reports success rather than converting the early stop into an error.
func (rg *RetryGenerator) Stream(ctx context.Context, dialog Dialog, options *GenOpts) iter.Seq2[StreamChunk, error] {
	return func(yield func(StreamChunk, error) bool) {
		sg, ok := rg.Inner.(StreamingGenerator)
		if !ok {
			for chunk, err := range rg.GeneratorWrapper.Stream(ctx, dialog, options) {
				if !yield(chunk, err) {
					return
				}
			}
			return
		}

		operation := func() (struct{}, error) {
			emittedAny := false
			for chunk, err := range sg.Stream(ctx, dialog, options) {
				if err != nil {
					if emittedAny {
						return struct{}{}, &permanentRetryError{err: err}
					}
					return struct{}{}, err
				}

				emittedAny = true
				if !yield(chunk, nil) {
					return struct{}{}, nil
				}
			}

			return struct{}{}, nil
		}

		_, err := retry(ctx, operation, rg.config)
		if err != nil {
			yield(StreamChunk{}, err)
		}
	}
}

// Compile-time interface checks.
var (
	_ Generator            = (*RetryGenerator)(nil)
	_ TokenCounter         = (*RetryGenerator)(nil)
	_ ToolCallingGenerator = (*RetryGenerator)(nil)
	_ StreamingGenerator   = (*RetryGenerator)(nil)
)

// WithRetry returns a WrapperFunc that wraps a generator with retry logic.
// See NewRetryGenerator for parameter details.
func WithRetry(config RetryConfig) WrapperFunc {
	return func(g Generator) Generator {
		return NewRetryGenerator(g, config)
	}
}
