package gai

import (
	"context"
	"fmt"
	"iter"
)

// GeneratorWrapper is a base type for creating middleware-style generator wrappers.
// Embed it in your custom wrapper struct to get automatic delegation for all generator
// interfaces, then override only the methods where you need custom behavior.
//
// # The Middleware Pattern
//
// When you stack multiple wrappers using [Wrap], calls flow through them like an onion:
//
//	gen := Wrap(base, WithA(), WithB(), WithC())
//
//	// Structure: A wraps B wraps C wraps base
//	//
//	// Call flow for gen.Generate():
//	//   A.Generate (before) →
//	//     B.Generate (before) →
//	//       C.Generate (before) →
//	//         base.Generate
//	//       C.Generate (after) ←
//	//     B.Generate (after) ←
//	//   A.Generate (after) ←
//
// Each interface method (Generate, Count, Stream, Register) flows through the stack
// independently. If a wrapper doesn't override a method, GeneratorWrapper passes
// the call straight through to Inner.
//
// # Selective Override
//
// You choose which methods each wrapper intercepts by overriding them:
//
//   - Override a method → your wrapper participates in that method's call chain
//   - Don't override → GeneratorWrapper delegates directly to Inner (transparent pass-through)
//
// For example, a logging wrapper might override both Generate and Count to log both
// operations, while a retry wrapper only overrides Generate (retrying Count doesn't
// make sense for most use cases).
//
// # Supported Interfaces
//
// GeneratorWrapper implements all standard generator interfaces:
//   - [Generator]: Generate() delegates to Inner.Generate()
//   - [TokenCounter]: Count() delegates to Inner if it implements TokenCounter
//   - [ToolCapableGenerator]: Register() delegates to Inner if it implements ToolCapableGenerator
//   - [StreamingGenerator]: Stream() delegates to Inner if it implements StreamingGenerator
//
// If Inner doesn't implement an optional interface (TokenCounter, ToolCapableGenerator,
// StreamingGenerator), the corresponding method returns an appropriate error.
//
// # Example: Creating a Wrapper
//
//	// TimingGenerator measures how long Generate and Count take.
//	type TimingGenerator struct {
//	    gai.GeneratorWrapper  // Embed for automatic delegation
//	    Observer func(method string, duration time.Duration)
//	}
//
//	// Override Generate to add timing
//	func (t *TimingGenerator) Generate(ctx context.Context, d Dialog, o *GenOpts) (Response, error) {
//	    start := time.Now()
//	    resp, err := t.GeneratorWrapper.Generate(ctx, d, o)  // Delegate to next in chain
//	    t.Observer("Generate", time.Since(start))
//	    return resp, err
//	}
//
//	// Override Count to add timing
//	func (t *TimingGenerator) Count(ctx context.Context, d Dialog) (uint, error) {
//	    start := time.Now()
//	    count, err := t.GeneratorWrapper.Count(ctx, d)  // Delegate to next in chain
//	    t.Observer("Count", time.Since(start))
//	    return count, err
//	}
//
//	// Stream is NOT overridden - calls pass through to Inner automatically
//
//	// WrapperFunc for use with Wrap()
//	func WithTiming(observer func(string, time.Duration)) gai.WrapperFunc {
//	    return func(g gai.Generator) gai.Generator {
//	        return &TimingGenerator{
//	            GeneratorWrapper: gai.GeneratorWrapper{Inner: g},
//	            Observer:         observer,
//	        }
//	    }
//	}
//
// # Example: Stacking Multiple Wrappers
//
//	gen := gai.Wrap(baseGenerator,
//	    WithLogging(logger),     // Outermost: logs all calls
//	    WithMetrics(collector),  // Middle: collects metrics
//	    WithRetry(nil),          // Innermost: retries failed Generate calls
//	)
//
//	// Now gen.Generate() flows: Logging → Metrics → Retry → base
//	// And gen.Count() flows:    Logging → Metrics → base (Retry doesn't override Count)
type GeneratorWrapper struct {
	Inner Generator
}

// Generate delegates to Inner.Generate.
// Override this method in your wrapper to intercept Generate calls.
func (w *GeneratorWrapper) Generate(ctx context.Context, dialog Dialog, opts *GenOpts) (Response, error) {
	return w.Inner.Generate(ctx, dialog, opts)
}

// Count delegates to Inner.Count if Inner implements [TokenCounter].
// Override this method in your wrapper to intercept Count calls.
// Returns an error if Inner does not implement TokenCounter.
func (w *GeneratorWrapper) Count(ctx context.Context, dialog Dialog) (uint, error) {
	if tc, ok := w.Inner.(TokenCounter); ok {
		return tc.Count(ctx, dialog)
	}
	return 0, fmt.Errorf("inner generator of type %T does not implement TokenCounter", w.Inner)
}

// Register delegates to Inner.Register if Inner implements [ToolCapableGenerator].
// Override this method in your wrapper to intercept Register calls.
// Returns an error if Inner does not implement ToolCapableGenerator.
func (w *GeneratorWrapper) Register(tool Tool) error {
	if tcg, ok := w.Inner.(ToolCapableGenerator); ok {
		return tcg.Register(tool)
	}
	return fmt.Errorf("inner generator of type %T does not implement ToolCapableGenerator", w.Inner)
}

// Stream delegates to Inner.Stream if Inner implements [StreamingGenerator].
// Override this method in your wrapper to intercept Stream calls.
// Returns an error-yielding iterator if Inner does not implement StreamingGenerator.
func (w *GeneratorWrapper) Stream(ctx context.Context, dialog Dialog, opts *GenOpts) iter.Seq2[StreamChunk, error] {
	if sg, ok := w.Inner.(StreamingGenerator); ok {
		return sg.Stream(ctx, dialog, opts)
	}
	return func(yield func(StreamChunk, error) bool) {
		yield(StreamChunk{}, fmt.Errorf("inner generator of type %T does not implement StreamingGenerator", w.Inner))
	}
}

// Compile-time interface assertions
var (
	_ Generator            = (*GeneratorWrapper)(nil)
	_ TokenCounter         = (*GeneratorWrapper)(nil)
	_ ToolCapableGenerator = (*GeneratorWrapper)(nil)
	_ StreamingGenerator   = (*GeneratorWrapper)(nil)
)

// WrapperFunc is a function that wraps a [Generator], returning a new Generator.
// Use with [Wrap] to compose multiple wrappers into a middleware stack.
//
// Convention: define a WithXxx function that returns a WrapperFunc for your wrapper:
//
//	func WithLogging(logger *slog.Logger) gai.WrapperFunc {
//	    return func(g gai.Generator) gai.Generator {
//	        return &LoggingGenerator{
//	            GeneratorWrapper: gai.GeneratorWrapper{Inner: g},
//	            Logger:           logger,
//	        }
//	    }
//	}
type WrapperFunc func(Generator) Generator

// Wrap applies wrappers to a generator, creating a middleware stack.
// Wrappers are applied in order: the first wrapper becomes the outermost layer
// (first to receive calls, last to return).
//
// Example:
//
//	gen := Wrap(baseGen,
//	    WithLogging(logger),   // 1st: outermost - receives call first
//	    WithMetrics(collector),// 2nd: middle
//	    WithRetry(nil),        // 3rd: innermost - closest to baseGen
//	)
//
// This creates the structure: Logging{Metrics{Retry{baseGen}}}
//
// When gen.Generate() is called:
//  1. Logging.Generate runs (before logic)
//  2. Metrics.Generate runs (before logic)
//  3. Retry.Generate runs (with retry loop calling...)
//  4. baseGen.Generate runs
//  5. Retry.Generate returns
//  6. Metrics.Generate runs (after logic)
//  7. Logging.Generate runs (after logic)
func Wrap(gen Generator, wrappers ...WrapperFunc) Generator {
	for i := len(wrappers) - 1; i >= 0; i-- {
		gen = wrappers[i](gen)
	}
	return gen
}
