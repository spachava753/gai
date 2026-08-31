// Package gai defines provider-neutral requests, responses, and composition
// utilities for large language model generation.
//
// A [Generator] accepts one self-contained [GenerationRequest]. The request
// carries the model, system instructions, conversation, tools, and generation
// options for that invocation. Generator implementations retain only execution
// dependencies such as API clients, credentials, and endpoints.
//
// # Requests and responses
//
// A [GenerationRequest] separates system [Message] content in Instructions from
// the conversational [Dialog]. A zero Message means instructions are absent;
// any populated instruction message must have the [System] role. Use
// [SystemMessage] to construct one. A dialog contains user, assistant, and
// [ToolResult] messages, each represented as an ordered sequence of [Block]
// values.
//
// The block helpers [TextBlock], [ImageBlock], [AudioBlock], [PDFBlock], and
// [ToolCallBlock] construct the common block forms. Provider-specific replay
// data remains attached to [Block.ExtraFields] or [Message.ExtraFields].
// Invocation-level provider data belongs to [Response.ExtraFields], while
// measurements belong to [Response.UsageMetadata].
//
// [Response.Candidates] contains generated assistant messages. The
// [Response.FinishReason] distinguishes a completed turn from tool use, a stop
// sequence, a generation limit, or a content-policy stop.
//
// # Options
//
// Build request options with [NewGenerationOptions] and [GenerationOption]
// helpers such as [WithTemperature], [WithToolChoice], and
// [WithMaxGenerationTokens]. Each helper documents its matching
// GenerationOption constant. Provider adapters ignore option keys they do not
// support and return [InvalidParameterErr] for recognized values with invalid
// types or ranges.
//
// # Streaming and counting
//
// Generators with incremental output implement [StreamingGenerator]. A stream
// yields [StreamChunk] values and at most one terminal error chunk.
// [StreamingAdapter] collects a single-candidate stream into a [Response].
//
// Generators that can estimate or query input usage implement [TokenCounter].
// Token counting receives the same [GenerationRequest] used for generation so
// it includes the selected model, instructions, dialog, and tools.
//
// # Tools
//
// A [Tool] declares a caller-owned function with a JSON Schema input. Models
// return calls as [ToolCall] blocks containing [ToolCallInput]. Applications
// authorize and execute those calls, then append a [ToolResultMessage] before
// continuing generation. [ToolCallback] and [ToolCallBackFunc] provide optional
// helpers for application-side dispatch.
//
// # Composition
//
// [Wrap] composes [WrapperFunc] values around a generator. [WithRetry] retries
// transient provider failures according to [RetryConfig], and
// [WithPreprocessing] normalizes parallel tool results for providers that
// require one combined tool-result message. [FallbackGenerator] tries an
// ordered set of generators according to [FallbackConfig].
//
// # Providers
//
// Provider adapters share the core request and response types:
//
//   - OpenAI Chat Completions: [NewOpenAiGenerator]
//   - OpenAI Responses: [NewResponsesGenerator]
//   - Anthropic: [NewAnthropicGenerator]
//   - Google Gemini: [NewGeminiGenerator]
//   - Cerebras: [NewCerebrasGenerator]
//   - OpenRouter: [NewOpenRouterGenerator]
//   - DeepSeek: [NewDeepSeekGenerator]
//   - Z.AI: [NewZaiGenerator]
//
// Each provider type documents supported content, common options,
// provider-specific options, replay metadata, and optional interfaces.
//
// # Errors
//
// Provider failures are returned as [ApiErr], which retains provider details
// and classifies failures with [APIErrorKind]. Use errors.Is for sentinel errors
// and errors.As for structured errors. [RetryGenerator] and [FallbackGenerator] use these classifications without discarding the
// underlying provider error.
package gai
