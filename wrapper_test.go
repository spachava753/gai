package gai

import (
	"context"
	"iter"
	"reflect"
	"strings"
	"sync"
	"testing"
)

type wrapperMockGenerator struct {
	generateFunc func(context.Context, GenerationRequest) (Response, error)
	countFunc    func(context.Context, GenerationRequest) (uint, error)
	streamFunc   func(context.Context, GenerationRequest) iter.Seq[StreamChunk]
}

func (m *wrapperMockGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, request)
	}
	return Response{}, nil
}

func (m *wrapperMockGenerator) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	if m.countFunc != nil {
		return m.countFunc(ctx, request)
	}
	return 0, nil
}

func (m *wrapperMockGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	if m.streamFunc != nil {
		return m.streamFunc(ctx, request)
	}
	return func(yield func(StreamChunk) bool) {}
}

type wrapperBasicGenerator struct{}

func (b *wrapperBasicGenerator) Generate(context.Context, GenerationRequest) (Response, error) {
	return Response{}, nil
}

func TestWrapperCompositionScenarios(t *testing.T) {
	t.Run("GeneratorWrapper/Count/NotSupported", func(t *testing.T) {
		wrapper := &GeneratorWrapper{Inner: &wrapperBasicGenerator{}}
		_, err := wrapper.Count(context.Background(), GenerationRequest{})
		if err == nil {
			t.Error("expected error for unsupported TokenCounter")
		}
	})
	t.Run("GeneratorWrapper/Count/Supported", func(t *testing.T) {
		mock := &wrapperMockGenerator{
			countFunc: func(context.Context, GenerationRequest) (uint, error) { return 42, nil },
		}

		wrapper := &GeneratorWrapper{Inner: mock}
		count, err := wrapper.Count(context.Background(), GenerationRequest{})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if count != 42 {
			t.Errorf("expected 42, got %d", count)
		}
	})
	t.Run("GeneratorWrapper/Generate", func(t *testing.T) {
		request := GenerationRequest{Model: "test-model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
		var received GenerationRequest
		mock := &wrapperMockGenerator{
			generateFunc: func(_ context.Context, request GenerationRequest) (Response, error) {
				received = request
				return Response{FinishReason: EndTurn}, nil
			},
		}

		wrapper := &GeneratorWrapper{Inner: mock}
		resp, err := wrapper.Generate(context.Background(), request)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(received, request) {
			t.Errorf("request changed during delegation: got %#v", received)
		}
		if resp.FinishReason != EndTurn {
			t.Error("response not passed through")
		}
	})
	t.Run("GeneratorWrapper/Stream/NotSupported", func(t *testing.T) {
		wrapper := &GeneratorWrapper{Inner: &wrapperBasicGenerator{}}

		var streamErr error
		for chunk := range wrapper.Stream(context.Background(), GenerationRequest{}) {
			streamErr = chunk.Err
		}
		if streamErr == nil {
			t.Error("expected error for unsupported StreamingGenerator")
		}
	})
	t.Run("GeneratorWrapper/Stream/Supported", func(t *testing.T) {
		mock := &wrapperMockGenerator{
			streamFunc: func(context.Context, GenerationRequest) iter.Seq[StreamChunk] {
				return func(yield func(StreamChunk) bool) {
					yield(StreamChunk{Block: Block{BlockType: Content}})
				}
			},
		}

		wrapper := &GeneratorWrapper{Inner: mock}
		chunks := 0
		for range wrapper.Stream(context.Background(), GenerationRequest{}) {
			chunks++
		}
		if chunks != 1 {
			t.Errorf("expected 1 chunk, got %d", chunks)
		}
	})
	t.Run("MiddlewareStack/BothMethods", func(t *testing.T) {
		tracker := &CallTracker{}
		base := &MockBaseGenerator{tracker: tracker, tokenCount: 99}

		gen := Wrap(base,
			WithAlpha(tracker),
			WithBeta(tracker),
		)

		// Call Generate first
		_, _ = gen.Generate(context.Background(), GenerationRequest{})

		// Then call Count
		_, _ = gen.(TokenCounter).Count(context.Background(), GenerationRequest{})

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
	})
	t.Run("MiddlewareStack/Count", func(t *testing.T) {
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
		count, err := gen.(TokenCounter).Count(context.Background(), GenerationRequest{})
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
	})
	t.Run("MiddlewareStack/Generate", func(t *testing.T) {
		tracker := &CallTracker{}
		base := &MockBaseGenerator{tracker: tracker, tokenCount: 100}

		// Stack: Alpha (outermost) -> Beta -> Gamma -> Base (innermost)
		gen := Wrap(base,
			WithAlpha(tracker),
			WithBeta(tracker),
			WithGamma(tracker),
		)

		// Call Generate
		_, err := gen.Generate(context.Background(), GenerationRequest{})
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
	})
	t.Run("WithPreprocessing", func(t *testing.T) {
		wrapped := WithPreprocessing()(&wrapperBasicGenerator{})
		if _, ok := wrapped.(*PreprocessingGenerator); !ok {
			t.Errorf("expected *PreprocessingGenerator, got %T", wrapped)
		}
	})
	t.Run("WithRetry", func(t *testing.T) {
		wrapped := WithRetry(DefaultRetryConfig())(&wrapperMockGenerator{})
		if _, ok := wrapped.(*RetryGenerator); !ok {
			t.Errorf("expected *RetryGenerator, got %T", wrapped)
		}
	})
	t.Run("Wrap/Empty", func(t *testing.T) {
		base := &wrapperBasicGenerator{}
		if gen := Wrap(base); gen != base {
			t.Error("Wrap with no wrappers should return base unchanged")
		}
	})
	t.Run("Wrap/Order", func(t *testing.T) {
		var order []string
		makeWrapper := func(name string) WrapperFunc {
			return func(inner Generator) Generator {
				return &wrapperOrderTrackingWrapper{
					GeneratorWrapper: GeneratorWrapper{Inner: inner},
					name:             name,
					order:            &order,
				}
			}
		}

		gen := Wrap(&wrapperBasicGenerator{}, makeWrapper("first"), makeWrapper("second"), makeWrapper("third"))
		_, _ = gen.Generate(context.Background(), GenerationRequest{})

		expected := []string{"first", "second", "third"}
		if !reflect.DeepEqual(order, expected) {
			t.Fatalf("expected %v, got %v", expected, order)
		}
	})
}

type wrapperOrderTrackingWrapper struct {
	GeneratorWrapper
	name  string
	order *[]string
}

func (w *wrapperOrderTrackingWrapper) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	*w.order = append(*w.order, w.name)
	return w.GeneratorWrapper.Generate(ctx, request)
}

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

func (a *AlphaMiddleware) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	a.tracker.Record("alpha:generate:before")
	resp, err := a.GeneratorWrapper.Generate(ctx, request)
	a.tracker.Record("alpha:generate:after")
	return resp, err
}

func (a *AlphaMiddleware) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	a.tracker.Record("alpha:count:before")
	count, err := a.GeneratorWrapper.Count(ctx, request)
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

func (b *BetaMiddleware) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	b.tracker.Record("beta:generate:before")
	resp, err := b.GeneratorWrapper.Generate(ctx, request)
	b.tracker.Record("beta:generate:after")
	return resp, err
}

func (b *BetaMiddleware) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	b.tracker.Record("beta:count:before")
	count, err := b.GeneratorWrapper.Count(ctx, request)
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

func (g *GammaMiddleware) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	g.tracker.Record("gamma:generate:before")
	resp, err := g.GeneratorWrapper.Generate(ctx, request)
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

func (m *MockBaseGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	m.tracker.Record("base:generate")
	return Response{
		Candidates:   []Message{{Role: Assistant, Blocks: []Block{{Content: Str("hello")}}}},
		FinishReason: EndTurn,
	}, nil
}

func (m *MockBaseGenerator) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	m.tracker.Record("base:count")
	return m.tokenCount, nil
}
