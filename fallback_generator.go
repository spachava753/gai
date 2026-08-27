package gai

import (
	"context"
	"errors"
	"fmt"
)

// FallbackConfig controls which [Generator.Generate] errors advance to the next
// generator in a [FallbackGenerator].
type FallbackConfig struct {
	// ShouldFallback returns true to try the next generator. A nil function uses
	// [ApiErr.Retryable]. It can be called concurrently when the fallback
	// generator is shared.
	ShouldFallback func(err error) bool
}

// FallbackGenerator implements [Generator] by trying an ordered set of
// generators. Every generator receives the same [GenerationRequest], including
// its model identifier. Callers must choose compatible generators or wrap them
// with an explicit request transformation.
type FallbackGenerator struct {
	generators []Generator
	config     FallbackConfig
}

// NewFallbackGenerator constructs an ordered fallback chain. It requires at
// least two generators. A nil config, or a config with nil ShouldFallback, uses
// [ApiErr.Retryable].
func NewFallbackGenerator(generators []Generator, config *FallbackConfig) (*FallbackGenerator, error) {
	if len(generators) < 2 {
		return nil, errors.New("fallback generator requires at least 2 generators")
	}

	// Initialize with default config if not provided
	actualConfig := FallbackConfig{}
	if config != nil {
		actualConfig = *config
	}

	// Use default fallback logic if not specified
	if actualConfig.ShouldFallback == nil {
		actualConfig.ShouldFallback = defaultShouldFallback
	}

	return &FallbackGenerator{
		generators: generators,
		config:     actualConfig,
	}, nil
}

// defaultShouldFallback is the default logic for determining when to fallback to another generator.
// It fallbacks on rate limit errors and API errors with 5xx status codes.
func defaultShouldFallback(err error) bool {
	var apiErr *ApiErr
	return errors.As(err, &apiErr) && apiErr.Retryable()
}

// Generate tries generators in order while [FallbackConfig.ShouldFallback]
// accepts each error. A rejected error returns immediately. If every generator
// fails, Generate wraps the final error.
func (f *FallbackGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	var lastErr error

	// Try each generator in sequence
	for _, generator := range f.generators {
		response, err := generator.Generate(ctx, request)

		// If no error, return the successful response
		if err == nil {
			return response, nil
		}

		// Store the error for potential return if all generators fail
		lastErr = err

		// Check if we should fallback based on the error
		if !f.config.ShouldFallback(err) {
			// If this is not a fallback-worthy error, return it immediately
			return Response{}, err
		}

		// Otherwise, continue to the next generator
	}

	// This point should only be reached if all generators failed
	// and the last generator's error was not a fallback error
	return Response{}, fmt.Errorf("all generators failed: %w", lastErr)
}

// NewHTTPStatusFallbackConfig returns a policy that accepts [ApiErr] values
// whose status code matches one of statusCodes.
func NewHTTPStatusFallbackConfig(statusCodes ...int) FallbackConfig {
	return FallbackConfig{
		ShouldFallback: func(err error) bool {
			var apiErr *ApiErr
			if errors.As(err, &apiErr) {
				for _, code := range statusCodes {
					if apiErr.StatusCode == code {
						return true
					}
				}
			}
			return false
		},
	}
}

// NewRateLimitOnlyFallbackConfig returns a policy that accepts only [ApiErr]
// values classified as [APIErrorKindRateLimit].
func NewRateLimitOnlyFallbackConfig() FallbackConfig {
	return FallbackConfig{
		ShouldFallback: func(err error) bool {
			var apiErr *ApiErr
			return errors.As(err, &apiErr) && apiErr.Kind == APIErrorKindRateLimit
		},
	}
}

// Ensure FallbackGenerator implements the Generator interface
var _ Generator = (*FallbackGenerator)(nil)
