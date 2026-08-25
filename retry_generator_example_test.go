package gai_test

import (
	"context"
	"fmt"
	"iter"
	"sync/atomic"
	"time"

	"github.com/spachava753/gai"
)

type retryExampleGenerator struct {
	generateCalls atomic.Uint32
	streamCalls   atomic.Uint32
}

func (g *retryExampleGenerator) Generate(context.Context, gai.GenerationRequest) (gai.Response, error) {
	if g.generateCalls.Add(1) == 1 {
		return gai.Response{}, retryExampleRateLimitError()
	}
	return gai.Response{Candidates: []gai.Message{{
		Role:   gai.Assistant,
		Blocks: []gai.Block{gai.TextBlock("generated")},
	}}}, nil
}

func (g *retryExampleGenerator) Stream(context.Context, gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
	attempt := g.streamCalls.Add(1)
	return func(yield func(gai.StreamChunk) bool) {
		if attempt == 1 {
			yield(gai.StreamChunk{Err: retryExampleRateLimitError()})
			return
		}
		yield(gai.StreamChunk{Block: gai.TextBlock("streamed")})
	}
}

func retryExampleRateLimitError() error {
	// A valid zero Retry-After overrides only the delay, not caller authorization.
	retryAfter := time.Duration(0)
	return &gai.ApiErr{
		Provider:           gai.ProviderOpenAI,
		Kind:               gai.APIErrorKindRateLimit,
		RetryAfterDuration: &retryAfter,
	}
}

func ExampleRetryGenerator() {
	generator := &retryExampleGenerator{}

	// Bound retry scheduling separately from the hard operation deadline below.
	config := gai.DefaultRetryConfig()
	config.MaxAttempts = 4
	config.MaxElapsedTime = 2 * time.Second

	// RetryGenerator can be shared, so callbacks must be concurrency-safe.
	var notifications atomic.Uint32
	config.Notify = func(error, time.Duration) {
		notifications.Add(1)
	}

	retrying := gai.NewRetryGenerator(generator, config)

	// MaxElapsedTime cannot interrupt a provider call; the context is the hard bound.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	response, err := retrying.Generate(ctx, gai.GenerationRequest{})
	if err != nil {
		fmt.Println("generate error:", err)
		return
	}
	fmt.Println(response.Candidates[0].Blocks[0].Content.String())

	// Startup failures can be retried. After a chunk, errors are returned without replay.
	for chunk := range retrying.Stream(ctx, gai.GenerationRequest{}) {
		if chunk.Err != nil {
			fmt.Println("stream error:", chunk.Err)
			return
		}
		fmt.Println(chunk.Block.Content.String())
	}
	fmt.Println("notifications:", notifications.Load())

	// Output:
	// generated
	// streamed
	// notifications: 2
}
