package gai

import (
	"context"
	"iter"
	"reflect"
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
	t.Run("GeneratorWrapper/Count/NotSupported", testGeneratorWrapper_Count_NotSupported)
	t.Run("GeneratorWrapper/Count/Supported", testGeneratorWrapper_Count_Supported)
	t.Run("GeneratorWrapper/Generate", testGeneratorWrapper_Generate)
	t.Run("GeneratorWrapper/Stream/NotSupported", testGeneratorWrapper_Stream_NotSupported)
	t.Run("GeneratorWrapper/Stream/Supported", testGeneratorWrapper_Stream_Supported)
	t.Run("MiddlewareStack/BothMethods", testMiddlewareStack_BothMethods)
	t.Run("MiddlewareStack/Count", testMiddlewareStack_Count)
	t.Run("MiddlewareStack/Generate", testMiddlewareStack_Generate)
	t.Run("WithPreprocessing", testWithPreprocessing)
	t.Run("WithRetry", testWithRetry)
	t.Run("Wrap/Empty", testWrap_Empty)
	t.Run("Wrap/Order", testWrap_Order)
}

func testGeneratorWrapper_Generate(t *testing.T) {
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
}

func testGeneratorWrapper_Count_Supported(t *testing.T) {
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
}

func testGeneratorWrapper_Count_NotSupported(t *testing.T) {
	wrapper := &GeneratorWrapper{Inner: &wrapperBasicGenerator{}}
	_, err := wrapper.Count(context.Background(), GenerationRequest{})
	if err == nil {
		t.Error("expected error for unsupported TokenCounter")
	}
}

func testGeneratorWrapper_Stream_Supported(t *testing.T) {
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
}

func testGeneratorWrapper_Stream_NotSupported(t *testing.T) {
	wrapper := &GeneratorWrapper{Inner: &wrapperBasicGenerator{}}

	var streamErr error
	for chunk := range wrapper.Stream(context.Background(), GenerationRequest{}) {
		streamErr = chunk.Err
	}
	if streamErr == nil {
		t.Error("expected error for unsupported StreamingGenerator")
	}
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

func testWrap_Order(t *testing.T) {
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
}

func testWrap_Empty(t *testing.T) {
	base := &wrapperBasicGenerator{}
	if gen := Wrap(base); gen != base {
		t.Error("Wrap with no wrappers should return base unchanged")
	}
}

func testWithRetry(t *testing.T) {
	wrapped := WithRetry(DefaultRetryConfig())(&wrapperMockGenerator{})
	if _, ok := wrapped.(*RetryGenerator); !ok {
		t.Errorf("expected *RetryGenerator, got %T", wrapped)
	}
}

func testWithPreprocessing(t *testing.T) {
	wrapped := WithPreprocessing()(&wrapperBasicGenerator{})
	if _, ok := wrapped.(*PreprocessingGenerator); !ok {
		t.Errorf("expected *PreprocessingGenerator, got %T", wrapped)
	}
}
