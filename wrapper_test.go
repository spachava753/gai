package gai

import (
	"context"
	"iter"
	"testing"
)

// wrapperMockGenerator implements all generator interfaces for testing wrapper functionality
type wrapperMockGenerator struct {
	generateFunc func(ctx context.Context, d Dialog, o *GenOpts) (Response, error)
	countFunc    func(ctx context.Context, d Dialog) (uint, error)
	registerFunc func(tool Tool) error
	streamFunc   func(ctx context.Context, d Dialog, o *GenOpts) iter.Seq2[StreamChunk, error]
}

func (m *wrapperMockGenerator) Generate(ctx context.Context, d Dialog, o *GenOpts) (Response, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, d, o)
	}
	return Response{}, nil
}

func (m *wrapperMockGenerator) Count(ctx context.Context, d Dialog) (uint, error) {
	if m.countFunc != nil {
		return m.countFunc(ctx, d)
	}
	return 0, nil
}

func (m *wrapperMockGenerator) Register(tool Tool) error {
	if m.registerFunc != nil {
		return m.registerFunc(tool)
	}
	return nil
}

func (m *wrapperMockGenerator) Stream(ctx context.Context, d Dialog, o *GenOpts) iter.Seq2[StreamChunk, error] {
	if m.streamFunc != nil {
		return m.streamFunc(ctx, d, o)
	}
	return func(yield func(StreamChunk, error) bool) {}
}

// wrapperBasicGenerator only implements Generator (not TokenCounter, ToolCapableGenerator, etc.)
type wrapperBasicGenerator struct{}

func (b *wrapperBasicGenerator) Generate(ctx context.Context, d Dialog, o *GenOpts) (Response, error) {
	return Response{}, nil
}

func TestGeneratorWrapper_Generate(t *testing.T) {
	called := false
	mock := &wrapperMockGenerator{
		generateFunc: func(ctx context.Context, d Dialog, o *GenOpts) (Response, error) {
			called = true
			return Response{FinishReason: EndTurn}, nil
		},
	}

	wrapper := &GeneratorWrapper{Inner: mock}
	resp, err := wrapper.Generate(context.Background(), nil, nil)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !called {
		t.Error("inner Generate was not called")
	}
	if resp.FinishReason != EndTurn {
		t.Error("response not passed through")
	}
}

func TestGeneratorWrapper_Count_Supported(t *testing.T) {
	mock := &wrapperMockGenerator{
		countFunc: func(ctx context.Context, d Dialog) (uint, error) {
			return 42, nil
		},
	}

	wrapper := &GeneratorWrapper{Inner: mock}
	count, err := wrapper.Count(context.Background(), nil)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if count != 42 {
		t.Errorf("expected 42, got %d", count)
	}
}

func TestGeneratorWrapper_Count_NotSupported(t *testing.T) {
	wrapper := &GeneratorWrapper{Inner: &wrapperBasicGenerator{}}
	_, err := wrapper.Count(context.Background(), nil)

	if err == nil {
		t.Error("expected error for unsupported TokenCounter")
	}
}

func TestGeneratorWrapper_Register_Supported(t *testing.T) {
	registered := ""
	mock := &wrapperMockGenerator{
		registerFunc: func(tool Tool) error {
			registered = tool.Name
			return nil
		},
	}

	wrapper := &GeneratorWrapper{Inner: mock}
	err := wrapper.Register(Tool{Name: "test_tool"})

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if registered != "test_tool" {
		t.Errorf("expected test_tool, got %s", registered)
	}
}

func TestGeneratorWrapper_Register_NotSupported(t *testing.T) {
	wrapper := &GeneratorWrapper{Inner: &wrapperBasicGenerator{}}
	err := wrapper.Register(Tool{Name: "test"})

	if err == nil {
		t.Error("expected error for unsupported ToolCapableGenerator")
	}
}

func TestGeneratorWrapper_Stream_Supported(t *testing.T) {
	mock := &wrapperMockGenerator{
		streamFunc: func(ctx context.Context, d Dialog, o *GenOpts) iter.Seq2[StreamChunk, error] {
			return func(yield func(StreamChunk, error) bool) {
				yield(StreamChunk{Block: Block{BlockType: Content}}, nil)
			}
		},
	}

	wrapper := &GeneratorWrapper{Inner: mock}
	chunks := 0
	for range wrapper.Stream(context.Background(), nil, nil) {
		chunks++
	}

	if chunks != 1 {
		t.Errorf("expected 1 chunk, got %d", chunks)
	}
}

func TestGeneratorWrapper_Stream_NotSupported(t *testing.T) {
	wrapper := &GeneratorWrapper{Inner: &wrapperBasicGenerator{}}

	var streamErr error
	for _, err := range wrapper.Stream(context.Background(), nil, nil) {
		streamErr = err
	}

	if streamErr == nil {
		t.Error("expected error for unsupported StreamingGenerator")
	}
}

// wrapperOrderTrackingWrapper tracks order of Generate calls for TestWrap_Order
type wrapperOrderTrackingWrapper struct {
	GeneratorWrapper
	name  string
	order *[]string
}

func (w *wrapperOrderTrackingWrapper) Generate(ctx context.Context, d Dialog, o *GenOpts) (Response, error) {
	*w.order = append(*w.order, w.name)
	return w.GeneratorWrapper.Generate(ctx, d, o)
}

func TestWrap_Order(t *testing.T) {
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

	base := &wrapperBasicGenerator{}
	gen := Wrap(base,
		makeWrapper("first"),
		makeWrapper("second"),
		makeWrapper("third"),
	)

	_, _ = gen.Generate(context.Background(), nil, nil)

	// First wrapper is outermost, so it logs first
	expected := []string{"first", "second", "third"}
	if len(order) != len(expected) {
		t.Fatalf("expected %d entries, got %d", len(expected), len(order))
	}
	for i, v := range expected {
		if order[i] != v {
			t.Errorf("position %d: expected %s, got %s", i, v, order[i])
		}
	}
}

func TestWrap_Empty(t *testing.T) {
	base := &wrapperBasicGenerator{}
	gen := Wrap(base)

	if gen != base {
		t.Error("Wrap with no wrappers should return base unchanged")
	}
}

func TestWithRetry(t *testing.T) {
	base := &wrapperMockGenerator{}
	wrapperFn := WithRetry(nil)
	wrapped := wrapperFn(base)

	if _, ok := wrapped.(*RetryGenerator); !ok {
		t.Errorf("expected *RetryGenerator, got %T", wrapped)
	}
}

func TestWithPreprocessing(t *testing.T) {
	base := &wrapperMockGenerator{}
	wrapperFn := WithPreprocessing()
	wrapped := wrapperFn(base)

	if _, ok := wrapped.(*PreprocessingGenerator); !ok {
		t.Errorf("expected *PreprocessingGenerator, got %T", wrapped)
	}
}

func TestWithPreprocessing_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for non-ToolCapableGenerator")
		}
	}()

	base := &wrapperBasicGenerator{}
	wrapperFn := WithPreprocessing()
	wrapperFn(base) // Should panic
}
