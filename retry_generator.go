package gai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/cenkalti/backoff/v5"
)

const (
	// Default parameters for the ExponentialBackOff if no base policy is provided by the user.
	defaultGenRetryInitialInterval = 500 * time.Millisecond
	defaultGenRetryMaxInterval     = 15 * time.Second
	// Default MaxElapsedTime if the user provides no specific RetryOptions that override it.
	defaultGenRetryMaxElapsedTime = 1 * time.Minute
)

// RetryGenerator is a Generator that wraps another Generator and retries the
// Generate call according to a specified base backoff policy and retry options.
//
// It retries on specific errors:
//   - context.DeadlineExceeded (from the Generate call itself, not the overall context)
//   - gai.RateLimitErr
//   - gai.ApiErr with HTTP status code 429 (Too Many Requests)
//   - gai.ApiErr with HTTP status codes 5xx (Server Errors)
type RetryGenerator struct {
	generator    Generator
	baseBackOff  backoff.BackOff       // The core backoff strategy (e.g., *ExponentialBackOff).
	retryOptions []backoff.RetryOption // User-provided options for the backoff.Retry call (e.g., MaxElapsedTime, Notify).
}

// NewRetryGenerator creates a new RetryGenerator.
//
// Parameters:
//   - generator: The underlying Generator to wrap.
//   - baseBo: The base backoff.BackOff policy to use (e.g., an instance of *ExponentialBackOff).
//     If nil, a default *ExponentialBackOff with standard intervals (Initial: 500ms, Max: 15s) is created.
//   - opts: Optional backoff.RetryOption(s) to apply to each Retry call. These can configure
//     aspects like max elapsed time, max retries, or notification functions.
//     If no opts are provided, a default MaxElapsedTime (1 minute) will be applied.
//     If opts are provided, they are used directly; ensure they are comprehensive for your needs
//     (e.g., if you provide WithMaxTries, consider if you also need WithMaxElapsedTime).
//     It is recommended NOT to include backoff.WithBackOff() in opts, as `baseBo` is
//     always applied as the primary backoff strategy.
func NewRetryGenerator(generator Generator, baseBo backoff.BackOff, opts ...backoff.RetryOption) *RetryGenerator {
	actualBaseBo := baseBo
	if actualBaseBo == nil {
		exp := backoff.NewExponentialBackOff()
		exp.InitialInterval = defaultGenRetryInitialInterval
		exp.MaxInterval = defaultGenRetryMaxInterval
		actualBaseBo = exp
	}

	// If user provides any options, use them as is.
	// Otherwise, apply a default MaxElapsedTime.
	finalOpts := opts
	if len(opts) == 0 {
		finalOpts = []backoff.RetryOption{
			backoff.WithMaxElapsedTime(defaultGenRetryMaxElapsedTime),
		}
	}

	return &RetryGenerator{
		generator:    generator,
		baseBackOff:  actualBaseBo,
		retryOptions: finalOpts,
	}
}

// Generate calls the underlying Generator's Generate method, retrying on
// specific errors according to the configured backoff policy and options.
// The provided context (ctx) is respected by the retry loop: if ctx is
// cancelled, retries will stop.
func (rg *RetryGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	operation := func() (Response, error) {
		// First, check if the overarching context (ctx) has been cancelled or has met its deadline.
		if err := ctx.Err(); err != nil {
			return Response{}, backoff.Permanent(err)
		}

		resp, err := rg.generator.Generate(ctx, dialog, options)
		if err != nil {
			// Analyze the error to determine if it's retriable.
			if errors.Is(err, context.DeadlineExceeded) {
				return resp, err // Retriable
			}
			var rateLimitErr RateLimitErr
			if errors.As(err, &rateLimitErr) {
				return resp, err // Retriable
			}
			var apiErr ApiErr
			if errors.As(err, &apiErr) {
				if apiErr.StatusCode == http.StatusTooManyRequests || // 429
					(apiErr.StatusCode >= 500 && apiErr.StatusCode <= 599) { // 5xx
					return resp, err // Retriable
				}
			}
			// context.Canceled from the operation itself is treated as permanent.
			if errors.Is(err, context.Canceled) {
				return resp, backoff.Permanent(err)
			}
			// All other errors are considered permanent for the retry mechanism.
			return resp, backoff.Permanent(err)
		}
		// Successful call
		return resp, nil
	}

	// Reset the state of the base backoff policy (e.g., for ExponentialBackOff).
	rg.baseBackOff.Reset()

	// Combine the base backoff policy (via WithBackOff) with other configured retry options.
	// The user-supplied rg.retryOptions might include WithMaxElapsedTime, WithMaxTries, etc.
	// We always prepend WithBackOff using rg.baseBackOff.
	callOpts := make([]backoff.RetryOption, 0, 1+len(rg.retryOptions))
	callOpts = append(callOpts, backoff.WithBackOff(rg.baseBackOff))
	callOpts = append(callOpts, rg.retryOptions...)

	resp, err := backoff.Retry(ctx, operation, callOpts...)

	if err != nil {
		var permanent *backoff.PermanentError
		if errors.As(err, &permanent) {
			return resp, permanent.Err
		}
		return resp, err
	}
	return resp, nil
}

// ensure RetryGenerator implements Generator
var _ Generator = (*RetryGenerator)(nil)

// Count implements the TokenCounter interface if the underlying generator also implements it.
// Retries are not applied to the Count method by this generator.
func (rg *RetryGenerator) Count(ctx context.Context, dialog Dialog) (uint, error) {
	if tc, ok := rg.generator.(TokenCounter); ok {
		return tc.Count(ctx, dialog)
	}
	return 0, fmt.Errorf("underlying generator of type %T does not implement TokenCounter", rg.generator)
}

// Compile-time check to ensure RetryGenerator implements TokenCounter.
var _ TokenCounter = (*RetryGenerator)(nil)

// Register attempts to register a tool with the underlying generator, if it supports ToolCapableGenerator.
// Retries are not applied to this method.
func (rg *RetryGenerator) Register(tool Tool) error {
	if tcg, ok := rg.generator.(ToolCapableGenerator); ok {
		return tcg.Register(tool)
	}
	return fmt.Errorf("underlying generator of type %T does not implement ToolCapableGenerator", rg.generator)
}

// Compile-time check to ensure RetryGenerator implements ToolCapableGenerator.
// This will only compile if RetryGenerator has the methods of ToolCapableGenerator.
var _ ToolCapableGenerator = (*RetryGenerator)(nil)
