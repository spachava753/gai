package gai_test

import (
	"bytes"
	"context"
	"log/slog"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/spachava753/gai"
)

// -----------------------------------------------------------------------------
// Example 1: A simple wrapper that only overrides Generate
// -----------------------------------------------------------------------------

// LoggingGenerator logs Generate calls. It does NOT override Count or Stream, so those methods pass through to Inner automatically via GeneratorWrapper.
type LoggingGenerator struct {
	gai.GeneratorWrapper // Embed for automatic delegation of non-overridden methods
	Logger               *slog.Logger
}

// Generate logs before and after delegating to the next generator in the chain.
func (l *LoggingGenerator) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	l.Logger.Info("generate: starting", "messages", len(request.Dialog))
	start := time.Now()

	// Delegate to Inner (the next wrapper or base generator)
	resp, err := l.GeneratorWrapper.Generate(ctx, request)

	l.Logger.Info("generate: finished", "duration", time.Since(start), "error", err)
	return resp, err
}

// WithLogging returns a WrapperFunc for use with gai.Wrap.
func WithLogging(logger *slog.Logger) gai.WrapperFunc {
	return func(g gai.Generator) gai.Generator {
		return &LoggingGenerator{
			GeneratorWrapper: gai.GeneratorWrapper{Inner: g},
			Logger:           logger,
		}
	}
}

// -----------------------------------------------------------------------------
// Example 2: A wrapper that overrides MULTIPLE methods (Generate AND Count)
// -----------------------------------------------------------------------------

// MetricsGenerator collects timing metrics for both Generate and Count operations.
// This demonstrates how a single wrapper can intercept multiple interface methods.
type MetricsGenerator struct {
	gai.GeneratorWrapper
	RecordMetric func(operation string, duration time.Duration, err error)
}

// Generate records metrics for generation calls.
func (m *MetricsGenerator) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	start := time.Now()
	resp, err := m.GeneratorWrapper.Generate(ctx, request)
	m.RecordMetric("generate", time.Since(start), err)
	return resp, err
}

// Count records metrics for token counting calls.
// By overriding this, MetricsGenerator participates in the Count call chain.
func (m *MetricsGenerator) Count(ctx context.Context, request gai.GenerationRequest) (uint, error) {
	start := time.Now()
	count, err := m.GeneratorWrapper.Count(ctx, request)
	m.RecordMetric("count", time.Since(start), err)
	return count, err
}

// WithMetrics returns a WrapperFunc for use with gai.Wrap.
func WithMetrics(record func(string, time.Duration, error)) gai.WrapperFunc {
	return func(g gai.Generator) gai.Generator {
		return &MetricsGenerator{
			GeneratorWrapper: gai.GeneratorWrapper{Inner: g},
			RecordMetric:     record,
		}
	}
}

// -----------------------------------------------------------------------------
// Example 3: Mock generator for demonstrating the middleware stack
// -----------------------------------------------------------------------------

// trackingMockGen is a simple generator for examples that records calls via a callback.
type trackingMockGen struct {
	record     func(string)
	tokenCount uint
}

func (m *trackingMockGen) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	m.record("base:Generate")
	return gai.Response{
		Candidates:   []gai.Message{{Role: gai.Assistant}},
		FinishReason: gai.EndTurn,
	}, nil
}

func (m *trackingMockGen) Count(ctx context.Context, request gai.GenerationRequest) (uint, error) {
	m.record("base:Count")
	return m.tokenCount, nil
}

// simpleMockGen is a minimal generator for examples that don't need call tracking.
type simpleMockGen struct {
	tokenCount uint
}

func (m *simpleMockGen) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	return gai.Response{
		Candidates:   []gai.Message{{Role: gai.Assistant}},
		FinishReason: gai.EndTurn,
	}, nil
}

func (m *simpleMockGen) Count(ctx context.Context, request gai.GenerationRequest) (uint, error) {
	return m.tokenCount, nil
}

// -----------------------------------------------------------------------------
// Runnable Examples
// -----------------------------------------------------------------------------

// This example demonstrates how wrappers that override different methods
// create independent call chains for each method.
func Test_selectiveOverride(t *testing.T) {
	var logs bytes.Buffer
	var metrics []string

	base := &simpleMockGen{tokenCount: 100}
	gen := gai.Wrap(base,
		WithLogging(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{}))),
		WithMetrics(func(op string, d time.Duration, err error) {
			metrics = append(metrics, op)
		}),
	)

	resp, err := gen.Generate(context.Background(), gai.GenerationRequest{})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if resp.FinishReason != gai.EndTurn {
		t.Fatalf("FinishReason = %v, want %v", resp.FinishReason, gai.EndTurn)
	}
	if !slices.Equal(metrics, []string{"generate"}) {
		t.Fatalf("metrics after Generate = %v, want [generate]", metrics)
	}
	if got := logs.String(); !strings.Contains(got, "generate: starting") || !strings.Contains(got, "generate: finished") {
		t.Fatalf("logs did not contain Generate start and finish entries: %q", got)
	}

	count, err := gen.(gai.TokenCounter).Count(context.Background(), gai.GenerationRequest{})
	if err != nil {
		t.Fatalf("Count returned error: %v", err)
	}
	if count != 100 {
		t.Fatalf("Count = %d, want 100", count)
	}
	if !slices.Equal(metrics, []string{"generate", "count"}) {
		t.Fatalf("metrics after Count = %v, want [generate count]", metrics)
	}
}

// This example shows the complete call flow through a middleware stack,
// demonstrating the "onion" pattern where calls flow in and responses flow out.
func Test_middlewareCallFlow(t *testing.T) {
	var calls []string
	record := func(s string) { calls = append(calls, s) }

	withAlpha := func(g gai.Generator) gai.Generator {
		return &alphaWrapper{
			GeneratorWrapper: gai.GeneratorWrapper{Inner: g},
			record:           record,
		}
	}
	withBeta := func(g gai.Generator) gai.Generator {
		return &betaWrapper{
			GeneratorWrapper: gai.GeneratorWrapper{Inner: g},
			record:           record,
		}
	}

	base := &trackingMockGen{record: record, tokenCount: 42}
	gen := gai.Wrap(base, withAlpha, withBeta)

	_, err := gen.Generate(context.Background(), gai.GenerationRequest{})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	wantGenerate := []string{"alpha:before", "beta:before", "base:Generate", "beta:after", "alpha:after"}
	if !slices.Equal(calls, wantGenerate) {
		t.Fatalf("Generate call flow = %v, want %v", calls, wantGenerate)
	}

	calls = nil
	count, err := gen.(gai.TokenCounter).Count(context.Background(), gai.GenerationRequest{})
	if err != nil {
		t.Fatalf("Count returned error: %v", err)
	}
	if count != 42 {
		t.Fatalf("Count = %d, want 42", count)
	}
	wantCount := []string{"alpha:before", "beta:before", "base:Count", "beta:after", "alpha:after"}
	if !slices.Equal(calls, wantCount) {
		t.Fatalf("Count call flow = %v, want %v", calls, wantCount)
	}
}

// alphaWrapper and betaWrapper are helpers for Example_middlewareCallFlow
type alphaWrapper struct {
	gai.GeneratorWrapper
	record func(string)
}

func (a *alphaWrapper) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	a.record("alpha:before")
	resp, err := a.GeneratorWrapper.Generate(ctx, request)
	a.record("alpha:after")
	return resp, err
}

func (a *alphaWrapper) Count(ctx context.Context, request gai.GenerationRequest) (uint, error) {
	a.record("alpha:before")
	count, err := a.GeneratorWrapper.Count(ctx, request)
	a.record("alpha:after")
	return count, err
}

type betaWrapper struct {
	gai.GeneratorWrapper
	record func(string)
}

func (b *betaWrapper) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	b.record("beta:before")
	resp, err := b.GeneratorWrapper.Generate(ctx, request)
	b.record("beta:after")
	return resp, err
}

func (b *betaWrapper) Count(ctx context.Context, request gai.GenerationRequest) (uint, error) {
	b.record("beta:before")
	count, err := b.GeneratorWrapper.Count(ctx, request)
	b.record("beta:after")
	return count, err
}

// This example shows the recommended pattern for creating a reusable wrapper.
func Test_creatingAWrapper(t *testing.T) {
	var metrics []string
	base := &simpleMockGen{tokenCount: 7}
	gen := gai.Wrap(base, WithMetrics(func(op string, d time.Duration, err error) {
		metrics = append(metrics, op)
	}))

	if _, err := gen.Generate(context.Background(), gai.GenerationRequest{}); err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	count, err := gen.(gai.TokenCounter).Count(context.Background(), gai.GenerationRequest{})
	if err != nil {
		t.Fatalf("Count returned error: %v", err)
	}
	if count != 7 {
		t.Fatalf("Count = %d, want 7", count)
	}
	if !slices.Equal(metrics, []string{"generate", "count"}) {
		t.Fatalf("metrics = %v, want [generate count]", metrics)
	}
}
