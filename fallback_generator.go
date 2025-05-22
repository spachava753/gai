package gai

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// FallbackConfig represents the configuration for when to fallback to another generator
type FallbackConfig struct {
	// ShouldFallback is a function that determines whether to fallback to another generator
	// based on the error returned by the current generator.
	// If nil, the default behavior is used, which fallbacks on rate limit errors and 5xx status codes.
	ShouldFallback func(err error) bool
}

// FallbackGenerator implements the Generator interface by composing multiple generators.
// If one generator returns an error that meets the fallback criteria, it tries the next generator.
type FallbackGenerator struct {
	generators []Generator
	config     FallbackConfig
}

// NewFallbackGenerator creates a new FallbackGenerator with the provided generators and configuration.
// It returns an error if fewer than 2 generators are provided.
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
	// Check for rate limit errors
	var rateLimitErr RateLimitErr
	if errors.As(err, &rateLimitErr) {
		return true
	}

	// Check for API errors with 5xx status codes
	var apiErr ApiErr
	if errors.As(err, &apiErr) && apiErr.StatusCode >= 500 && apiErr.StatusCode < 600 {
		return true
	}

	// Check if the error is related to rate limits by examining the error message
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "rate limit") {
		return true
	}

	return false
}

// Generate implements the Generator interface.
// It tries each generator in order, falling back to the next one if the current returns an error
// that meets the fallback criteria.
func (f *FallbackGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	var lastErr error

	// Try each generator in sequence
	for _, generator := range f.generators {
		response, err := generator.Generate(ctx, dialog, options)

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

// NewHTTPStatusFallbackConfig creates a FallbackConfig that fallbacks on specific HTTP status codes.
// It will fallback on rate limit errors and the specified status codes.
func NewHTTPStatusFallbackConfig(statusCodes ...int) FallbackConfig {
	return FallbackConfig{
		ShouldFallback: func(err error) bool {
			// Check for rate limit errors first (always fallback on these)
			var rateLimitErr RateLimitErr
			if errors.As(err, &rateLimitErr) {
				return true
			}

			// Check for API errors with specific status codes
			var apiErr ApiErr
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

// NewRateLimitOnlyFallbackConfig creates a FallbackConfig that only fallbacks on rate limit errors.
func NewRateLimitOnlyFallbackConfig() FallbackConfig {
	return FallbackConfig{
		ShouldFallback: func(err error) bool {
			var rateLimitErr RateLimitErr
			return errors.As(err, &rateLimitErr)
		},
	}
}

// Ensure FallbackGenerator implements the Generator interface
var _ Generator = (*FallbackGenerator)(nil)
