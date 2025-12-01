package gai

import (
	"context"
	"strings"

	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

// AnthropicServiceParamModifierFunc is a function type that modifies Anthropic API parameters
// before they are sent to the API. This allows for intercepting and modifying request parameters
// such as enabling caching, adding headers, or transforming the content.
//
// The function receives a context and a pointer to the message parameters, and returns an error
// if the modification cannot be completed successfully. Multiple modifier functions can be chained
// together in a middleware-like pattern.
//
// Example:
//
//	// Create a custom modifier that adds system context
//	addWeatherContext := func(_ context.Context, params *a.MessageNewParams) error {
//	    params.System = append(params.System, a.SystemParam{Text: "Current weather: 72°F and sunny"})
//	    return nil
//	}
//
//	// Wrap the Anthropic client with multiple modifiers
//	wrappedClient := NewAnthropicServiceWrapper(
//	    client.Messages,
//	    EnableSystemCaching,
//	    EnableMultiTurnCaching,
//	    addWeatherContext,
//	)
type AnthropicServiceParamModifierFunc func(ctx context.Context, params *a.MessageNewParams) error

// AnthropicServiceWrapper wraps an Anthropic API client with parameter modifier functions.
// This allows for intercepting and modifying requests before they are sent to the Anthropic API,
// enabling features like caching, request transformation, and dynamic context management.
//
// The wrapper implements the AnthropicSvc interface, making it a drop-in
// replacement for the standard Anthropic client in the context of this library.
//
// Common use cases include:
// - Enabling API response caching for reduced latency and costs
// - Adding dynamic system prompts based on runtime conditions
// - Transforming or filtering message content
// - Adding consistent metadata or parameters across all requests
type AnthropicServiceWrapper struct {
	// funcs contains the parameter modifier functions to apply to each request
	funcs []AnthropicServiceParamModifierFunc

	// wrapped is the underlying Anthropic client service being wrapped
	wrapped AnthropicSvc
}

// New implements the AnthropicSvc interface by applying all registered
// parameter modifier functions to the request parameters before passing them to the
// wrapped service.
//
// Each modifier function is called in the order they were registered. If any modifier
// returns an error, the request is aborted and the error is returned.
//
// After all modifiers have been successfully applied, the modified parameters are
// passed to the wrapped service's New method.
func (svc AnthropicServiceWrapper) New(ctx context.Context, params a.MessageNewParams, opts ...option.RequestOption) (res *a.Message, err error) {
	for _, f := range svc.funcs {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if err := f(ctx, &params); err != nil {
			return nil, err
		}
	}
	return svc.wrapped.New(ctx, params, opts...)
}

// NewStreaming implements the AnthropicSvc interface by applying all registered
// parameter modifier functions to the request parameters before passing them to the
// wrapped service.
//
// Each modifier function is called in the order they were registered. If any modifier
// returns an error, the request is aborted and the error is returned.
//
// After all modifiers have been successfully applied, the modified parameters are
// passed to the wrapped service's New method.
func (svc AnthropicServiceWrapper) NewStreaming(ctx context.Context, params a.MessageNewParams, opts ...option.RequestOption) (stream *ssestream.Stream[a.MessageStreamEventUnion]) {
	for _, f := range svc.funcs {
		if ctx.Err() != nil {
			return ssestream.NewStream[a.MessageStreamEventUnion](nil, ctx.Err())
		}
		if err := f(ctx, &params); err != nil {
			return ssestream.NewStream[a.MessageStreamEventUnion](nil, err)
		}
	}
	return svc.wrapped.NewStreaming(ctx, params, opts...)
}

// CountTokens forwards token counting requests to the wrapped service.
// This method simply passes the request through without applying any modifiers.
func (svc AnthropicServiceWrapper) CountTokens(ctx context.Context, params a.MessageCountTokensParams, opts ...option.RequestOption) (res *a.MessageTokensCount, err error) {
	// Forward the request to the wrapped service
	return svc.wrapped.CountTokens(ctx, params, opts...)
}

// NewAnthropicServiceWrapper creates a new wrapper around an Anthropic API client
// with the provided parameter modifier functions.
//
// The wrapper intercepts API calls, applies the modifier functions in sequence,
// and then forwards the modified parameters to the actual Anthropic API client.
//
// This pattern is useful for consistently applying transformations or middleware-like
// functionality to all Anthropic API calls without modifying client code.
//
// Example:
//
//	// Create a wrapped client with caching enabled
//	wrappedClient := NewAnthropicServiceWrapper(
//	    client.Messages,
//	    EnableSystemCaching,
//	    EnableMultiTurnCaching,
//	)
//
//	// Use the wrapped client with AnthropicGenerator
//	generator := NewAnthropicGenerator(
//	    wrappedClient,
//	    "claude-3-opus-20240229",
//	    "You are a helpful assistant.",
//	)
func NewAnthropicServiceWrapper(wrapped AnthropicSvc, funcs ...AnthropicServiceParamModifierFunc) *AnthropicServiceWrapper {
	return &AnthropicServiceWrapper{
		wrapped: wrapped,
		funcs:   funcs,
	}
}

// isOpus45OrLater checks if the model is Claude Opus 4.5 or a later version.
// Claude Opus 4.5 introduced thinking block preservation, which means thinking blocks
// from previous assistant turns are preserved in model context by default.
// This enables cache optimization when using extended thinking with tool use.
func isOpus45OrLater(model string) bool {
	return strings.HasPrefix(model, "claude-opus-4-5")
}

// EnableSystemCaching modifies Anthropic API parameters to enable caching of system instructions.
// This can improve performance and reduce costs when making multiple requests with the same
// system instructions.
//
// When applied, this modifier adds an "ephemeral" cache control directive to the last system
// instruction block, indicating to Anthropic's API that the system instruction can be cached.
//
// Example:
//
//	// Create a wrapped client with system instruction caching
//	wrappedClient := NewAnthropicServiceWrapper(
//	    client.Messages,
//	    EnableSystemCaching,
//	)
//
//	// Use the wrapped client with your generator
//	generator := NewAnthropicGenerator(
//	    wrappedClient,
//	    "claude-3-opus-20240229",
//	    "You are a helpful assistant.",
//	)
//
// Note: This has no effect if the request doesn't include system instructions.
// System prompts remain cached even with extended thinking enabled.
func EnableSystemCaching(_ context.Context, params *a.MessageNewParams) error {
	if len(params.System) == 0 {
		return nil
	}

	params.System[len(params.System)-1].CacheControl = a.CacheControlEphemeralParam{Type: "ephemeral"}
	return nil
}

// EnableMultiTurnCaching modifies Anthropic API parameters to enable caching for multi-turn
// conversations. This can significantly improve response time and reduce costs when having
// extended conversations with an Anthropic model.
//
// When applied, this modifier adds an "ephemeral" cache control directive to the last content
// block of the last message in the conversation, enabling caching for various types of content
// including text, images, tool use, tool results, and documents.
//
// Example:
//
//	// Create a wrapped client with multi-turn conversation caching
//	wrappedClient := NewAnthropicServiceWrapper(
//	    client.Messages,
//	    EnableMultiTurnCaching,
//	)
//
//	// Use the wrapped client with your generator
//	generator := NewAnthropicGenerator(
//	    wrappedClient,
//	    "claude-3-opus-20240229",
//	    "You are a helpful assistant.",
//	)
//
// Note: This has no effect if the request doesn't include any messages.
// For models prior to Claude Opus 4.5, caching is skipped when extended thinking is enabled
// because thinking blocks are stripped from prior turns, invalidating the cache.
// For Claude Opus 4.5 and later, thinking blocks are preserved by default, so caching
// works normally even with extended thinking.
//
// It is particularly useful for applications with interactive, multi-turn conversations.
func EnableMultiTurnCaching(_ context.Context, params *a.MessageNewParams) error {
	if len(params.Messages) == 0 {
		return nil
	}

	// For pre-Opus 4.5 models with extended thinking, skip caching because thinking blocks
	// are stripped from prior turns, which invalidates the cache.
	// For Opus 4.5+, thinking blocks are preserved, so caching works normally.
	thinkingEnabled := params.Thinking.OfEnabled != nil && params.Thinking.OfEnabled.BudgetTokens > 0
	if thinkingEnabled && !isOpus45OrLater(string(params.Model)) {
		return nil
	}

	lastMsg := params.Messages[len(params.Messages)-1]
	lastContentBlock := lastMsg.Content[len(lastMsg.Content)-1]

	if vt := lastContentBlock.OfText; vt != nil {
		vt.CacheControl = a.CacheControlEphemeralParam{Type: "ephemeral"}
	}
	if vt := lastContentBlock.OfImage; vt != nil {
		vt.CacheControl = a.CacheControlEphemeralParam{Type: "ephemeral"}
	}
	if vt := lastContentBlock.OfToolUse; vt != nil {
		vt.CacheControl = a.CacheControlEphemeralParam{Type: "ephemeral"}
	}
	if vt := lastContentBlock.OfToolResult; vt != nil {
		vt.CacheControl = a.CacheControlEphemeralParam{Type: "ephemeral"}
	}
	if vt := lastContentBlock.OfDocument; vt != nil {
		vt.CacheControl = a.CacheControlEphemeralParam{Type: "ephemeral"}
	}
	if vt := lastContentBlock.OfServerToolUse; vt != nil {
		vt.CacheControl = a.CacheControlEphemeralParam{Type: "ephemeral"}
	}
	return nil
}
