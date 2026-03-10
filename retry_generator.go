package gai

import (
	"context"
	"errors"
	"iter"
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

// RetryGenerator is a Generator that wraps another Generator and retries failed
// Generate calls and Stream startup failures according to a specified base backoff
// policy and retry options.
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
type RetryGenerator struct {
	GeneratorWrapper                       // Embed for default Count/Register delegation and unsupported Stream fallback.
	baseBackOff      backoff.BackOff       // The core backoff strategy (e.g., *ExponentialBackOff).
	retryOptions     []backoff.RetryOption // User-provided options for the backoff.Retry call (e.g., MaxElapsedTime, Notify).
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
		GeneratorWrapper: GeneratorWrapper{Inner: generator},
		baseBackOff:      actualBaseBo,
		retryOptions:     finalOpts,
	}
}

func (rg *RetryGenerator) wrapRetryError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	var apiErr *ApiErr
	if errors.As(err, &apiErr) && apiErr.Retryable() {
		return err
	}
	if errors.Is(err, context.Canceled) {
		return backoff.Permanent(err)
	}
	return backoff.Permanent(err)
}

func (rg *RetryGenerator) retryCallOptions() []backoff.RetryOption {
	rg.baseBackOff.Reset()

	callOpts := make([]backoff.RetryOption, 0, 1+len(rg.retryOptions))
	callOpts = append(callOpts, backoff.WithBackOff(rg.baseBackOff))
	callOpts = append(callOpts, rg.retryOptions...)
	return callOpts
}

func unwrapPermanentRetryError(err error) error {
	var permanent *backoff.PermanentError
	if errors.As(err, &permanent) {
		return permanent.Err
	}
	return err
}

// Generate calls the underlying Generator's Generate method, retrying on
// specific errors according to the configured backoff policy and options.
// The provided context (ctx) is respected by the retry loop: if ctx is
// cancelled, retries will stop.
func (rg *RetryGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	operation := func() (Response, error) {
		if err := ctx.Err(); err != nil {
			return Response{}, backoff.Permanent(err)
		}

		resp, err := rg.Inner.Generate(ctx, dialog, options)
		if err != nil {
			return resp, rg.wrapRetryError(err)
		}
		return resp, nil
	}

	resp, err := backoff.Retry(ctx, operation, rg.retryCallOptions()...)
	if err != nil {
		return resp, unwrapPermanentRetryError(err)
	}
	return resp, nil
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
			if err := ctx.Err(); err != nil {
				return struct{}{}, backoff.Permanent(err)
			}

			emittedAny := false
			for chunk, err := range sg.Stream(ctx, dialog, options) {
				if err != nil {
					if emittedAny {
						return struct{}{}, backoff.Permanent(err)
					}
					return struct{}{}, rg.wrapRetryError(err)
				}

				emittedAny = true
				if !yield(chunk, nil) {
					return struct{}{}, nil
				}
			}

			return struct{}{}, nil
		}

		_, err := backoff.Retry(ctx, operation, rg.retryCallOptions()...)
		if err == nil {
			return
		}

		yield(StreamChunk{}, unwrapPermanentRetryError(err))
	}
}

// Compile-time interface checks.
var (
	_ Generator            = (*RetryGenerator)(nil)
	_ TokenCounter         = (*RetryGenerator)(nil)
	_ ToolCapableGenerator = (*RetryGenerator)(nil)
	_ StreamingGenerator   = (*RetryGenerator)(nil)
)

// WithRetry returns a WrapperFunc that wraps a generator with retry logic.
// See NewRetryGenerator for parameter details.
func WithRetry(baseBo backoff.BackOff, opts ...backoff.RetryOption) WrapperFunc {
	return func(g Generator) Generator {
		return NewRetryGenerator(g, baseBo, opts...)
	}
}
