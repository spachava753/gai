package gai

import (
	"context"
	"strings"
	"sync"
	"testing"
)

// --- Test middlewares that intercept both Generate and Count ---

// CallTracker records the order of middleware calls
type CallTracker struct {
	mu    sync.Mutex
	calls []string
}

func (t *CallTracker) Record(s string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.calls = append(t.calls, s)
}

func (t *CallTracker) Calls() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string{}, t.calls...)
}

// AlphaMiddleware - records "alpha:before" and "alpha:after" for each call
type AlphaMiddleware struct {
	GeneratorWrapper
	tracker *CallTracker
}

func (a *AlphaMiddleware) Generate(ctx context.Context, dialog Dialog, opts *GenOpts) (Response, error) {
	a.tracker.Record("alpha:generate:before")
	resp, err := a.GeneratorWrapper.Generate(ctx, dialog, opts)
	a.tracker.Record("alpha:generate:after")
	return resp, err
}

func (a *AlphaMiddleware) Count(ctx context.Context, dialog Dialog) (uint, error) {
	a.tracker.Record("alpha:count:before")
	count, err := a.GeneratorWrapper.Count(ctx, dialog)
	a.tracker.Record("alpha:count:after")
	return count, err
}

func WithAlpha(tracker *CallTracker) WrapperFunc {
	return func(g Generator) Generator {
		return &AlphaMiddleware{
			GeneratorWrapper: GeneratorWrapper{Inner: g},
			tracker:          tracker,
		}
	}
}

// BetaMiddleware - records "beta:before" and "beta:after" for each call
type BetaMiddleware struct {
	GeneratorWrapper
	tracker *CallTracker
}

func (b *BetaMiddleware) Generate(ctx context.Context, dialog Dialog, opts *GenOpts) (Response, error) {
	b.tracker.Record("beta:generate:before")
	resp, err := b.GeneratorWrapper.Generate(ctx, dialog, opts)
	b.tracker.Record("beta:generate:after")
	return resp, err
}

func (b *BetaMiddleware) Count(ctx context.Context, dialog Dialog) (uint, error) {
	b.tracker.Record("beta:count:before")
	count, err := b.GeneratorWrapper.Count(ctx, dialog)
	b.tracker.Record("beta:count:after")
	return count, err
}

func WithBeta(tracker *CallTracker) WrapperFunc {
	return func(g Generator) Generator {
		return &BetaMiddleware{
			GeneratorWrapper: GeneratorWrapper{Inner: g},
			tracker:          tracker,
		}
	}
}

// GammaMiddleware - only intercepts Generate, NOT Count (to show selective override)
type GammaMiddleware struct {
	GeneratorWrapper
	tracker *CallTracker
}

func (g *GammaMiddleware) Generate(ctx context.Context, dialog Dialog, opts *GenOpts) (Response, error) {
	g.tracker.Record("gamma:generate:before")
	resp, err := g.GeneratorWrapper.Generate(ctx, dialog, opts)
	g.tracker.Record("gamma:generate:after")
	return resp, err
}

// Note: NO Count override - uses GeneratorWrapper.Count (pass-through)

func WithGamma(tracker *CallTracker) WrapperFunc {
	return func(gen Generator) Generator {
		return &GammaMiddleware{
			GeneratorWrapper: GeneratorWrapper{Inner: gen},
			tracker:          tracker,
		}
	}
}

// MockBaseGenerator - the innermost generator
type MockBaseGenerator struct {
	tracker    *CallTracker
	tokenCount uint
}

func (m *MockBaseGenerator) Generate(ctx context.Context, dialog Dialog, opts *GenOpts) (Response, error) {
	m.tracker.Record("base:generate")
	return Response{
		Candidates:   []Message{{Role: Assistant, Blocks: []Block{{Content: Str("hello")}}}},
		FinishReason: EndTurn,
	}, nil
}

func (m *MockBaseGenerator) Count(ctx context.Context, dialog Dialog) (uint, error) {
	m.tracker.Record("base:count")
	return m.tokenCount, nil
}

// --- Tests ---

func TestMiddlewareStack_Generate(t *testing.T) {
	tracker := &CallTracker{}
	base := &MockBaseGenerator{tracker: tracker, tokenCount: 100}

	// Stack: Alpha (outermost) -> Beta -> Gamma -> Base (innermost)
	gen := Wrap(base,
		WithAlpha(tracker),
		WithBeta(tracker),
		WithGamma(tracker),
	)

	// Call Generate
	_, err := gen.Generate(context.Background(), Dialog{}, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// Expected order: alpha:before -> beta:before -> gamma:before -> base -> gamma:after -> beta:after -> alpha:after
	expected := []string{
		"alpha:generate:before",
		"beta:generate:before",
		"gamma:generate:before",
		"base:generate",
		"gamma:generate:after",
		"beta:generate:after",
		"alpha:generate:after",
	}

	calls := tracker.Calls()
	if len(calls) != len(expected) {
		t.Fatalf("got %d calls, want %d\ngot:  %v\nwant: %v", len(calls), len(expected), calls, expected)
	}

	for i, want := range expected {
		if calls[i] != want {
			t.Errorf("call[%d] = %q, want %q", i, calls[i], want)
		}
	}

	t.Logf("Generate call order: %s", strings.Join(calls, " -> "))
}

func TestMiddlewareStack_Count(t *testing.T) {
	tracker := &CallTracker{}
	base := &MockBaseGenerator{tracker: tracker, tokenCount: 42}

	// Stack: Alpha (outermost) -> Beta -> Gamma -> Base (innermost)
	// Note: Gamma does NOT override Count, so it passes through
	gen := Wrap(base,
		WithAlpha(tracker),
		WithBeta(tracker),
		WithGamma(tracker),
	)

	// Call Count
	count, err := gen.(TokenCounter).Count(context.Background(), Dialog{})
	if err != nil {
		t.Fatalf("Count failed: %v", err)
	}

	if count != 42 {
		t.Errorf("count = %d, want 42", count)
	}

	// Expected order: alpha:before -> beta:before -> (gamma passes through) -> base -> beta:after -> alpha:after
	expected := []string{
		"alpha:count:before",
		"beta:count:before",
		// gamma has no Count override, so GeneratorWrapper.Count just calls Inner.Count
		"base:count",
		"beta:count:after",
		"alpha:count:after",
	}

	calls := tracker.Calls()
	if len(calls) != len(expected) {
		t.Fatalf("got %d calls, want %d\ngot:  %v\nwant: %v", len(calls), len(expected), calls, expected)
	}

	for i, want := range expected {
		if calls[i] != want {
			t.Errorf("call[%d] = %q, want %q", i, calls[i], want)
		}
	}

	t.Logf("Count call order: %s", strings.Join(calls, " -> "))
}

func TestMiddlewareStack_BothMethods(t *testing.T) {
	tracker := &CallTracker{}
	base := &MockBaseGenerator{tracker: tracker, tokenCount: 99}

	gen := Wrap(base,
		WithAlpha(tracker),
		WithBeta(tracker),
	)

	// Call Generate first
	_, _ = gen.Generate(context.Background(), Dialog{}, nil)

	// Then call Count
	_, _ = gen.(TokenCounter).Count(context.Background(), Dialog{})

	calls := tracker.Calls()
	t.Logf("All calls in order:\n")
	for i, c := range calls {
		t.Logf("  %d: %s", i+1, c)
	}

	// Verify Generate calls came first, then Count calls
	generateCalls := 0
	countCalls := 0
	for _, c := range calls {
		if strings.Contains(c, "generate") {
			generateCalls++
		}
		if strings.Contains(c, "count") {
			countCalls++
		}
	}

	if generateCalls != 5 { // alpha, beta, base, beta, alpha
		t.Errorf("expected 5 generate-related calls, got %d", generateCalls)
	}
	if countCalls != 5 { // alpha, beta, base, beta, alpha
		t.Errorf("expected 5 count-related calls, got %d", countCalls)
	}
}
