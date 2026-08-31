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
	t.Run("MiddlewareStack/BothMethods", func(t *testing.T) { testMiddlewareStack_BothMethods(t) })
	t.Run("MiddlewareStack/Count", func(t *testing.T) { testMiddlewareStack_Count(t) })
	t.Run("MiddlewareStack/Generate", func(t *testing.T) { testMiddlewareStack_Generate(t) })
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
