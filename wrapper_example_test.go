package gai_test

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/spachava753/gai"
)

// -----------------------------------------------------------------------------
// Example 1: A simple wrapper that only overrides Generate
// -----------------------------------------------------------------------------

// LoggingGenerator logs Generate calls. It does NOT override Count, Stream, or
// Register, so those methods pass through to Inner automatically via GeneratorWrapper.
type LoggingGenerator struct {
	gai.GeneratorWrapper // Embed for automatic delegation of non-overridden methods
	Logger               *slog.Logger
}

// Generate logs before and after delegating to the next generator in the chain.
func (l *LoggingGenerator) Generate(ctx context.Context, dialog gai.Dialog, opts *gai.GenOpts) (gai.Response, error) {
	l.Logger.Info("generate: starting", "messages", len(dialog))
	start := time.Now()

	// Delegate to Inner (the next wrapper or base generator)
	resp, err := l.GeneratorWrapper.Generate(ctx, dialog, opts)

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
func (m *MetricsGenerator) Generate(ctx context.Context, dialog gai.Dialog, opts *gai.GenOpts) (gai.Response, error) {
	start := time.Now()
	resp, err := m.GeneratorWrapper.Generate(ctx, dialog, opts)
	m.RecordMetric("generate", time.Since(start), err)
	return resp, err
}

// Count records metrics for token counting calls.
// By overriding this, MetricsGenerator participates in the Count call chain.
func (m *MetricsGenerator) Count(ctx context.Context, dialog gai.Dialog) (uint, error) {
	start := time.Now()
	count, err := m.GeneratorWrapper.Count(ctx, dialog)
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

func (m *trackingMockGen) Generate(ctx context.Context, dialog gai.Dialog, opts *gai.GenOpts) (gai.Response, error) {
	m.record("base:Generate")
	return gai.Response{
		Candidates:   []gai.Message{{Role: gai.Assistant}},
		FinishReason: gai.EndTurn,
	}, nil
}

func (m *trackingMockGen) Count(ctx context.Context, dialog gai.Dialog) (uint, error) {
	m.record("base:Count")
	return m.tokenCount, nil
}

// simpleMockGen is a minimal generator for examples that don't need call tracking.
type simpleMockGen struct {
	tokenCount uint
}

func (m *simpleMockGen) Generate(ctx context.Context, dialog gai.Dialog, opts *gai.GenOpts) (gai.Response, error) {
	return gai.Response{
		Candidates:   []gai.Message{{Role: gai.Assistant}},
		FinishReason: gai.EndTurn,
	}, nil
}

func (m *simpleMockGen) Count(ctx context.Context, dialog gai.Dialog) (uint, error) {
	return m.tokenCount, nil
}

// -----------------------------------------------------------------------------
// Runnable Examples
// -----------------------------------------------------------------------------

// This example demonstrates how wrappers that override different methods
// create independent call chains for each method.
func Example_selectiveOverride() {
	// LoggingGenerator only overrides Generate
	// MetricsGenerator overrides both Generate AND Count

	base := &simpleMockGen{tokenCount: 100}

	// Stack: Logging (outer) → Metrics (inner) → base
	gen := gai.Wrap(base,
		WithLogging(slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
				// Remove time for reproducible output
				if a.Key == slog.TimeKey {
					return slog.Attr{}
				}
				// Simplify duration
				if a.Key == "duration" {
					return slog.String("duration", "Xms")
				}
				return a
			},
		}))),
		WithMetrics(func(op string, d time.Duration, err error) {
			fmt.Printf("metric: %s took some time\n", op)
		}),
	)

	fmt.Println("=== Calling Generate ===")
	fmt.Println("Flow: Logging.Generate → Metrics.Generate → base.Generate")
	_, _ = gen.Generate(context.Background(), gai.Dialog{}, nil)

	fmt.Println("\n=== Calling Count ===")
	fmt.Println("Flow: Metrics.Count → base.Count (Logging has no Count override)")
	_, _ = gen.(gai.TokenCounter).Count(context.Background(), gai.Dialog{})

	// Output:
	// === Calling Generate ===
	// Flow: Logging.Generate → Metrics.Generate → base.Generate
	// level=INFO msg="generate: starting" messages=0
	// metric: generate took some time
	// level=INFO msg="generate: finished" duration=Xms error=<nil>
	//
	// === Calling Count ===
	// Flow: Metrics.Count → base.Count (Logging has no Count override)
	// metric: count took some time
}

// This example shows the complete call flow through a middleware stack,
// demonstrating the "onion" pattern where calls flow in and responses flow out.
func Example_middlewareCallFlow() {
	// CallTracker records the order of calls to visualize the flow
	var calls []string
	record := func(s string) { calls = append(calls, s) }

	// Create wrappers that record before/after
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

	// Base generator also uses the same record function
	base := &trackingMockGen{record: record, tokenCount: 42}

	// Stack: Alpha (outer) → Beta (inner) → base
	gen := gai.Wrap(base, withAlpha, withBeta)

	// Call Generate
	_, _ = gen.Generate(context.Background(), gai.Dialog{}, nil)

	fmt.Println("Generate call flow:")
	fmt.Println("  " + strings.Join(calls, " → "))

	// Reset and call Count
	calls = nil
	_, _ = gen.(gai.TokenCounter).Count(context.Background(), gai.Dialog{})

	fmt.Println("\nCount call flow:")
	fmt.Println("  " + strings.Join(calls, " → "))

	// Output:
	// Generate call flow:
	//   alpha:before → beta:before → base:Generate → beta:after → alpha:after
	//
	// Count call flow:
	//   alpha:before → beta:before → base:Count → beta:after → alpha:after
}

// alphaWrapper and betaWrapper are helpers for Example_middlewareCallFlow
type alphaWrapper struct {
	gai.GeneratorWrapper
	record func(string)
}

func (a *alphaWrapper) Generate(ctx context.Context, d gai.Dialog, o *gai.GenOpts) (gai.Response, error) {
	a.record("alpha:before")
	resp, err := a.GeneratorWrapper.Generate(ctx, d, o)
	a.record("alpha:after")
	return resp, err
}

func (a *alphaWrapper) Count(ctx context.Context, d gai.Dialog) (uint, error) {
	a.record("alpha:before")
	count, err := a.GeneratorWrapper.Count(ctx, d)
	a.record("alpha:after")
	return count, err
}

type betaWrapper struct {
	gai.GeneratorWrapper
	record func(string)
}

func (b *betaWrapper) Generate(ctx context.Context, d gai.Dialog, o *gai.GenOpts) (gai.Response, error) {
	b.record("beta:before")
	resp, err := b.GeneratorWrapper.Generate(ctx, d, o)
	b.record("beta:after")
	return resp, err
}

func (b *betaWrapper) Count(ctx context.Context, d gai.Dialog) (uint, error) {
	b.record("beta:before")
	count, err := b.GeneratorWrapper.Count(ctx, d)
	b.record("beta:after")
	return count, err
}

// This example shows the recommended pattern for creating a reusable wrapper.
func Example_creatingAWrapper() {
	fmt.Println("To create a middleware wrapper:")
	fmt.Println("")
	fmt.Println("1. Define a struct that embeds gai.GeneratorWrapper")
	fmt.Println("2. Override only the methods you want to intercept")
	fmt.Println("3. Call GeneratorWrapper.Method() to delegate to the next in chain")
	fmt.Println("4. Create a WithXxx() function that returns gai.WrapperFunc")
	fmt.Println("")
	fmt.Println("Methods you DON'T override pass through automatically.")

	// Output:
	// To create a middleware wrapper:
	//
	// 1. Define a struct that embeds gai.GeneratorWrapper
	// 2. Override only the methods you want to intercept
	// 3. Call GeneratorWrapper.Method() to delegate to the next in chain
	// 4. Create a WithXxx() function that returns gai.WrapperFunc
	//
	// Methods you DON'T override pass through automatically.
}
