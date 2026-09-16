# gai - Go for AI

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Go Version](https://img.shields.io/badge/Go-1.26.6+-00ADD8.svg)

`gai` provides provider-neutral request, response, streaming, tool, retry, and fallback types for large language model generation in Go.

Each generation call receives a self-contained `GenerationRequest`. Provider objects retain only clients, credentials, endpoints, and other execution dependencies, so one generator can safely serve requests with different models, instructions, tools, and options.

## Installation

```bash
go get github.com/spachava753/gai
```

GAI requires Go 1.26.6 or later.

## Quick start

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/spachava753/gai"
)

func main() {
	generator, err := gai.NewOpenAiGenerator(nil, "", os.Getenv("OPENAI_API_KEY"))
	if err != nil {
		panic(err)
	}

	response, err := generator.Generate(context.Background(), gai.GenerationRequest{
		Model: "gpt-5-mini",
		Instructions: gai.SystemMessage(
			gai.TextBlock("Answer clearly and concisely."),
		),
		Dialog: gai.Dialog{{
			Role:   gai.User,
			Blocks: []gai.Block{gai.TextBlock("Why is the sky blue?")},
		}},
		Options: gai.NewGenerationOptions(
			gai.WithTemperature(0.2),
			gai.WithMaxGenerationTokens(300),
		),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(response.Candidates[0].Blocks[0].Content)
}
```

Applications load credentials and pass them explicitly to provider constructors. Constructors do not read environment variables.

## Core API

[`Generator`](https://pkg.go.dev/github.com/spachava753/gai#Generator) is the base interface:

```go
type Generator interface {
	Generate(context.Context, GenerationRequest) (Response, error)
}
```

A request contains all semantic state for one invocation:

- `Model` selects the provider model.
- `Instructions` is either the zero message for no instructions or a `System` message outside the conversation.
- `Dialog` contains user, assistant, and tool-result messages.
- `Tools` contains the complete caller-defined function set for this invocation.
- `Options` contains common and provider-specific generation controls.

Messages contain ordered blocks. Use `TextBlock`, `ImageBlock`, `AudioBlock`, `PDFBlock`, and `ToolCallBlock` rather than constructing common blocks manually.

A response contains generated candidate messages, a normalized finish reason, usage measurements, and provider-specific invocation details. Replay-critical metadata stays on the message or block that requires it in a later request.

## Providers

| Provider | Constructor | Optional interfaces |
| --- | --- | --- |
| OpenAI Chat Completions | `NewOpenAiGenerator` | `StreamingGenerator`, `TokenCounter` |
| OpenAI Responses | `NewResponsesGenerator` | `StreamingGenerator` |
| OpenCode | `NewOpenCodeGenerator` | `StreamingGenerator` |
| Anthropic | `NewAnthropicGenerator` | `StreamingGenerator`, `TokenCounter` |
| Google Gemini | `NewGeminiGenerator` | `StreamingGenerator`, `TokenCounter` |
| Cerebras | `NewCerebrasGenerator` | `StreamingGenerator` |
| OpenRouter | `NewOpenRouterGenerator` | `StreamingGenerator` |
| DeepSeek | `NewDeepSeekGenerator` | `StreamingGenerator` |
| Z.AI | `NewZaiGenerator` | `StreamingGenerator`, `TokenCounter` |
| Moonshot (Kimi) | `NewMoonshotGenerator` | `StreamingGenerator`, `TokenCounter` |

Provider type documentation lists supported content, common options, native options, response metadata, and replay requirements. See the [package documentation](https://pkg.go.dev/github.com/spachava753/gai).

`OpenAiGenerator` takes an HTTP client, a base URL (including any API prefix), and an explicit API key. It no longer accepts an OpenAI SDK completion service. An empty base URL selects OpenAI; a custom endpoint does not change option behavior. `WithMaxGenerationTokens` uses `max_completion_tokens` by default; add `WithOpenAITokenLimitField("max_tokens")` for endpoints that require that field. There are no automatic request retries or model-name capability rules.

`WithOpenAIExtraBody` supplies explicit native JSON fields such as `thinking` without adding model-name rules to the generator. `WithOpenAIStreamUsage(false)` omits the streamed-usage request switch for APIs that do not need it. Images use data URLs, including images in tool results and history. Complete provider usage breakdowns are retained under `OpenAIResponseExtraFieldUsage`, in addition to common usage metrics.

Chat Completions streaming consumes through EOF to retain metadata after `[DONE]`; use a context deadline. Parallel tool calls are assembled by index and emitted as complete calls at stream completion. Replay metadata stays in message/block `ExtraFields` under the documented OpenAI keys, not on the generator. If serializing arbitrary metadata through untyped JSON maps, use `json.Decoder.UseNumber` to avoid rounding large integers.

`OpenCodeGenerator` uses the OpenCode Go subscription Chat Completions endpoint. It passes model IDs and `WithReasoningEffort` effort labels through to OpenCode, preserves both `reasoning_content` and structured `reasoning_details` for tool-call replay, and sends supported `ImageBlock` values as `image_url` data URLs. Reuse one `WithOpenCodeSessionID` value across a dialog so OpenCode keeps multi-turn tool reasoning on the same upstream provider. OpenCode or the selected model rejects unsupported capabilities.

`ZaiGenerator`, `DeepSeekGenerator`, and `MoonshotGenerator` delegate to a private `OpenAiGenerator`, sharing its multimodal conversion, streaming assembly, tool validation, reasoning replay, and OpenAI-named metadata keys. They do not embed a generic wrapper or inherit its optional capabilities. Z.AI and DeepSeek default to `max_tokens`; Moonshot keeps `max_completion_tokens`. Z.AI omits `stream_options` by default; Moonshot and DeepSeek request usage. Explicit options override these defaults. Z.AI's default URL is `https://api.z.ai/api/paas/v4`; pass the Coding Plan base URL explicitly when needed.

Thinking controls are provider-specific: `WithZaiThinking`, `WithZaiClearThinking`, `WithMoonshotThinking`, `WithMoonshotKeepThinking`, and `WithDeepSeekThinking`. Omitting them preserves server defaults; none removes caller history. Kimi K3 should use `WithReasoningEffort` without the older `thinking` controls. Always-thinking models, fixed sampling settings, and tool-choice restrictions remain the caller's responsibility. `WithZaiDoSample` and `WithZaiToolStream` expose Z.AI's separate sampling and tool-stream switches.

Z.AI counting calls `/tokenizer` with model, messages, and tools. Moonshot counting calls `/tokenizers/estimate-token-count` and rejects tool-bearing requests because the endpoint does not document tool accounting. DeepSeek does **not** implement `TokenCounter`; it has no native counting endpoint. Counters honor cancellation and do not retry or fall back to an OpenAI tokenizer.

The shared adapters send image data URLs and inline PDF bodies where the backend accepts them. The former Z.AI-only remote image/video/PDF URL conversion and provider-specific response metadata constants have been removed. Use the shared OpenAI metadata keys for returned native fields. PDF acceptance on Z.AI remains backend-dependent; no upload, extraction, or preprocessing workaround is added.

## Options

Use typed helpers to construct `GenerationOptions`:

```go
options := gai.NewGenerationOptions(
	gai.WithTemperature(0.3),
	gai.WithTopP(0.9),
	gai.WithStopSequences("END"),
	gai.WithToolChoice(gai.ToolChoiceAuto),
)
```

Provider-specific helpers compose with common helpers:

```go
options := gai.NewGenerationOptions(
	gai.WithTemperature(0.2),
	gai.WithOpenRouterFallbackModels("anthropic/claude-sonnet-4.5"),
	gai.WithOpenRouterProviderPreferences(map[string]any{
		"sort": "throughput",
	}),
)
```

`WithReasoningEffort("high")` sets a qualitative effort label. `WithThinkingBudget(4096)` sets an actual thinking-token budget on Anthropic or OpenRouter; those providers reject combining it with effort. Other adapters ignore numeric thinking budgets rather than serializing them as effort. Anthropic also accepts `"adaptive"` and `"disabled"` through the effort control.

Chat Completions adapters share `WithOpenAIResponseFormat` (native JSON-object/schema configuration), `WithOpenAILogprobs`, `WithOpenAITopLogprobs`, and `WithOpenAIStreamUsage`. OpenAI cache policies use `WithOpenAIPromptCacheRetention` or `WithOpenAIPromptCacheOptions`; the server validates model support. These helpers compose with `WithOpenAIExtraBody` in option order. Explicit extra-body fields override corresponding common and provider-specific settings when building the request.

`GenerationRequest` has two independent optional fields:

| Field | Mapping and meaning |
| --- | --- |
| `SafetyIdentifier` | An opaque stable end-user identity, preferably a hash or UUID—not personal data. OpenAI Chat Completions/Responses and Moonshot send `safety_identifier`; Anthropic sends `metadata.user_id`; Z.AI and DeepSeek send `user_id`. DeepSeek also uses it for cache and scheduling isolation. |
| `PromptCacheKey` | A cache grouping/routing hint sent as `prompt_cache_key` by OpenAI Chat Completions/Responses and Moonshot. It does not enable caching, set a TTL, or name a stored cache resource. |

Unsupported providers ignore these fields. Native Gemini ignores both; neither is translated to safety settings, labels, or `cachedContent`. Anthropic's prompt cache controls remain separate from end-user identity. Values are never derived from each other or inserted into prompt content. The former `WithResponsesPromptCacheKey` option is replaced by the request field.

Providers ignore unknown option keys. Recognized values with an invalid type, range, or combination return `InvalidParameterErr`; native fields and model-dependent constraints may instead be rejected by the provider.

## Streaming

A `StreamingGenerator` yields ordered `StreamChunk` values:

```go
streaming := generator // a value that implements gai.StreamingGenerator
for chunk := range streaming.Stream(ctx, request) {
	if chunk.Err != nil {
		return chunk.Err
	}
	if chunk.Block.BlockType == gai.Content {
		fmt.Print(chunk.Block.Content)
	}
}
```

`StreamingAdapter` collects a single-candidate stream into a normal `Response` when an application wants one code path for streaming-only generators.

## Tools

Tools are request data. GAI converts their JSON Schemas to each provider's function-tool representation, but the application owns authorization and execution.

```go
weather := gai.Tool{
	Name:        "get_weather",
	Description: "Return the current weather for a location.",
	InputSchema: schema,
}

request.Tools = []gai.Tool{weather}
request.Options = gai.NewGenerationOptions(
	gai.WithToolChoice(gai.ToolChoiceAuto),
)
```

When a response finishes with `ToolUse`, inspect its `ToolCall` blocks, execute approved calls, append `ToolResultMessage` values to the dialog, and generate again.

`ToolCallback` and `ToolCallBackFunc` are optional application-side dispatch helpers. Generators never execute tools automatically.

## Agents

Package [`agent`](https://pkg.go.dev/github.com/spachava753/gai/agent) runs a complete model and tool loop over any `gai.Generator`. An Agent has fixed instructions and executable tools, while each run supplies a prior dialog, one new user message, and generation options.

```go
runner, err := agent.New(agent.Config{
	Generator: provider,
	Model:     "provider-model",
	Tools:     executableTools,
})
if err != nil {
	return err
}

result, err := runner.Run(ctx, agent.RunRequest{
	Dialog: previousDialog,
	Input:  gai.Message{Role: gai.User, Blocks: []gai.Block{gai.TextBlock("Continue")}},
}, observer)
```

The loop prefers streaming when the generator supports it, validates model responses, executes tool calls one at a time, and returns the active dialog plus standard usage. `PrepareDialog`, generation, and tool hooks can make explicit changes or request a normal stop. Configuration and call inputs are borrowed read-only values. Hook and handler outputs belong to the loop after return. Observers receive ordered borrowed events, valid only during `Observe`, and can stop the run by returning an error.

Use [`agent/agenttest`](https://pkg.go.dev/github.com/spachava753/gai/agent/agenttest) for scripted generators and recording observers. See [`agent/design.md`](agent/design.md) for exact ordering, error, ownership, and persistence behavior.

## Composition

`Wrap` applies middleware-style wrappers in order:

```go
generator := gai.Wrap(
	base,
	gai.WithRetry(gai.DefaultRetryConfig()),
	gai.WithPreprocessing(),
)
```

`RetryGenerator` retries transient failures and stream startup failures. It never restarts a stream after emitting output.

`FallbackGenerator` tries an ordered set of generators when `FallbackConfig` accepts the preceding error. Every fallback receives the same request, including its model name, so callers must choose generators that understand that model or wrap them with an explicit request transformation.

## Errors and metrics

Provider failures use `ApiErr`, which retains the provider, normalized `APIErrorKind`, HTTP status, raw response body, retry timing, and underlying cause when available.

Use `errors.Is` for sentinel errors and `errors.As` for structured errors:

```go
var apiErr *gai.ApiErr
switch {
case errors.Is(err, gai.ErrEmptyDialog):
	// Fix the request.
case errors.As(err, &apiErr) && apiErr.Retryable():
	// Apply application retry policy.
}
```

Common usage values have typed accessors:

```go
input, hasInput := gai.InputTokens(response.UsageMetadata)
output, hasOutput := gai.OutputTokens(response.UsageMetadata)
```

Use `GetMetric` with provider-specific metric constants for native cost, timing, cache, and routing details.

## Development

```bash
go test ./...
go vet ./...
go test -race ./...
go tool laas -exclude-packages='^github\.com/spachava753/gai/internal/(cerebras|deepseek|openai|opencode|openrouter|zai)$' ./...
```

The shared Chat Completions wire client in `internal/openai` uses oapi-codegen. Regenerate it after changing its schema, configuration, or lossless-JSON overlay:

```bash
go generate ./internal/openai
```

Its tests check that the generated file is current and exercise JSON round trips and local HTTP behavior. `OpenAiGenerator` uses this client; the other generated provider packages still use OGEN.

The OpenAI adapter has [SDK request contract tests](testdata/openai_sdk/README.md). An explicit scenario matrix drives public GAI APIs against conversations recorded through the OpenAI Python SDK for OpenAI, Z.AI, Kimi, and DeepSeek. Captures store only request/response body pairs; options are declared once in the matrix and translated into public GAI options by the tests. Each named provider/mode/step case checks JSON-equivalent outgoing HTTP bodies, with later history built from GAI output. Replay also checks successful completion, usable function calls, and known scenario arguments or JSON results—not exact generated prose. New captures come from a small SDK conversation script running through mitmproxy. Building the corpus requires explicit approval for live calls; ordinary tests never refresh it.

The tracked pre-commit hook runs LAAS against hand-written packages. Activate it after cloning:

```bash
git config --local core.hooksPath .githooks
```

## License

MIT
