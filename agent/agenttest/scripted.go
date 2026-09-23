// Package agenttest provides deterministic Agent test fixtures.
package agenttest

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"

	"github.com/spachava753/gai"
)

// GenerateStep defines one expected ScriptedGenerator call.
type GenerateStep struct {
	// Check validates the received request before Response is returned.
	Check func(gai.GenerationRequest) error
	// Response is returned when Check succeeds.
	Response gai.Response
	// Err is returned with Response when Check succeeds.
	Err error
}

// ScriptedGenerator returns configured steps in call order and records requests.
type ScriptedGenerator struct {
	mu       sync.Mutex
	steps    []GenerateStep
	requests []gai.GenerationRequest
	next     int
}

// NewScriptedGenerator returns a generator that borrows steps for its lifetime.
func NewScriptedGenerator(steps ...GenerateStep) *ScriptedGenerator {
	return &ScriptedGenerator{steps: steps}
}

// Generate records request and returns the next scripted step.
func (g *ScriptedGenerator) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	if err := ctx.Err(); err != nil {
		return gai.Response{}, err
	}
	g.mu.Lock()
	if g.next >= len(g.steps) {
		g.mu.Unlock()
		return gai.Response{}, errors.New("agenttest: unexpected generation call")
	}
	step := g.steps[g.next]
	g.next++
	g.requests = append(g.requests, request)
	g.mu.Unlock()

	if step.Check != nil {
		if err := step.Check(request); err != nil {
			return gai.Response{}, fmt.Errorf("agenttest: request check: %w", err)
		}
	}
	return step.Response, step.Err
}

// Requests returns the borrowed recorded request slice. Call it after generation
// is complete and do not mutate the returned values.
func (g *ScriptedGenerator) Requests() []gai.GenerationRequest {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.requests
}

// StreamStep defines one expected ScriptedStreamingGenerator call.
type StreamStep struct {
	// Check validates the received request before chunks are yielded.
	Check func(gai.GenerationRequest) error
	// Chunks are yielded in order after Check succeeds.
	Chunks []gai.StreamChunk
}

// ScriptedStreamingGenerator returns configured streams in call order and
// records requests.
type ScriptedStreamingGenerator struct {
	mu       sync.Mutex
	steps    []StreamStep
	requests []gai.GenerationRequest
	next     int
}

// NewScriptedStreamingGenerator returns a streaming generator that borrows
// steps and their chunks for its lifetime.
func NewScriptedStreamingGenerator(steps ...StreamStep) *ScriptedStreamingGenerator {
	return &ScriptedStreamingGenerator{steps: steps}
}

// Generate collects the next scripted stream through gai.StreamingAdapter.
func (g *ScriptedStreamingGenerator) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	return (&gai.StreamingAdapter{S: g}).Generate(ctx, request)
}

// Stream records request and lazily yields the next scripted stream.
func (g *ScriptedStreamingGenerator) Stream(ctx context.Context, request gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
	return func(yield func(gai.StreamChunk) bool) {
		if err := ctx.Err(); err != nil {
			yield(gai.StreamChunk{Err: err})
			return
		}
		g.mu.Lock()
		if g.next >= len(g.steps) {
			g.mu.Unlock()
			yield(gai.StreamChunk{Err: errors.New("agenttest: unexpected streaming call")})
			return
		}
		step := g.steps[g.next]
		g.next++
		g.requests = append(g.requests, request)
		g.mu.Unlock()

		if step.Check != nil {
			if err := step.Check(request); err != nil {
				yield(gai.StreamChunk{Err: fmt.Errorf("agenttest: request check: %w", err)})
				return
			}
		}
		for _, chunk := range step.Chunks {
			if !yield(chunk) {
				return
			}
		}
	}
}

// Requests returns the borrowed recorded request slice. Call it after streaming
// is complete and do not mutate the returned values.
func (g *ScriptedStreamingGenerator) Requests() []gai.GenerationRequest {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.requests
}
