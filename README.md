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
- `Dialog` contains user, assistant, and tool-result messages, plus inline `System` messages on supporting providers.
- `Tools` contains the complete caller-defined function set for this invocation.
- `Options` contains common and provider-specific generation controls.

Messages contain ordered blocks. Use `TextBlock`, `ImageBlock`, `AudioBlock`, `PDFBlock`, and `ToolCallBlock` rather than constructing common blocks manually.

To inject instructions during a conversation, put a `SystemMessage` at the desired position in `Dialog`:

```go
request.Dialog = append(request.Dialog,
	gai.SystemMessage(gai.TextBlock("The remaining budget is small; finish concisely.")),
)
```

OpenAI Chat Completions and Responses preserve inline system messages in order, separately from `Instructions`. The compatible Chat Completions adapters forward them unchanged; acceptance depends on the endpoint and model.

[Anthropic supports mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages#limitations) on supporting models. GAI's adapter accepts only text content in them; ordinary text messages require no beta header. In GAI's user/function-tool history, a system section cannot be first: it must follow a user turn (including the complete tool-result batch) and precede an assistant turn or end the dialog. Consecutive system messages form one section. Do not insert one between a tool call and its result. Model support and placement are server-validated; GAI does not infer capabilities from model names. Initial instructions still belong in `Instructions`.

The native Anthropic API also permits text after an assistant turn ending in a server tool result, which GAI cannot currently represent. It supports native tool-change blocks with separate beta requirements; this feature does not expose those blocks.

Native Gemini does not support inline system messages and returns a conversion error. GAI never silently moves them into top-level instructions or changes them to user messages.

A response contains generated candidate messages, a normalized finish reason, usage measurements, and provider-specific invocation details. Replay-critical metadata stays on the message or block that requires it in a later request.

## Providers

| Provider | Constructor | Optional interfaces |
| --- | --- | --- |
| OpenAI Chat Completions | `NewOpenAiGenerator` | `StreamingGenerator`, `TokenCounter` |
| OpenAI Responses | `NewResponsesGenerator` | `StreamingGenerator` |
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

For OpenCode Go, use the protocol generator required by the model: `OpenAiGenerator` for Chat Completions, `ResponsesGenerator` for Responses, or `AnthropicGenerator` for Messages. There is no separate OpenCode adapter. Configure the Chat Completions/Responses base URL as `https://opencode.ai/zen/go/v1`; configure the Anthropic SDK base URL as `https://opencode.ai/zen/go/`. Pass an OpenCode API key explicitly. Set `GenerationRequest.Headers` to `http.Header{"X-Opencode-Session": {sessionID}}` and reuse the session ID across tool follow-ups and retries. Response metadata and errors use the selected protocol adapter's keys and provider label.

`ZaiGenerator`, `DeepSeekGenerator`, and `MoonshotGenerator` delegate to a private `OpenAiGenerator`, sharing its multimodal conversion, streaming assembly, tool validation, reasoning replay, and OpenAI-named metadata keys. They do not embed a generic wrapper or inherit its optional capabilities. Z.AI and DeepSeek default to `max_tokens`; Moonshot keeps `max_completion_tokens`. Z.AI omits `stream_options` by default; Moonshot and DeepSeek request usage. Explicit options override these defaults. Z.AI's default URL is `https://api.z.ai/api/paas/v4`; pass the Coding Plan base URL explicitly when needed.

`OpenRouterGenerator` delegates to the shared Chat Completions implementation with `max_tokens` and streamed usage enabled by default. Use common helpers for sampling, log probabilities, seeds, and parallel tool calls. OpenRouter-specific helpers remain for min-p, top-a, repetition penalty, fallback models, routing, session IDs, service tiers, prediction, and response formats; explicit `WithOpenAIExtraBody` fields override translated settings. Capability and range checks are left to the server. Effort maps to `reasoning.effort`, numeric budgets to `reasoning.max_tokens`; combining them is rejected. Set `GenerationRequest.SafetyIdentifier` to send OpenRouter's `user` identity and `GenerationRequest.PromptCacheKey` to send `prompt_cache_key`. The redundant `WithOpenRouterUser` and `WithOpenRouterPromptCacheKey` helpers and their option constants have been removed. Explicit extra-body values still take precedence. OpenRouter uses an explicit session ID ahead of the cache key for sticky routing.

**OpenRouter metadata migration (breaking, pre-v1):** the former `OpenRouterExtraField…`, `OpenRouterResponseExtraField…`, `OpenRouterMessageExtraFieldLogprobs`, and `OpenRouterUsageMetric…` constants are removed.

Preserve returned candidate `ExtraFields` when replaying reasoning: `OpenAIExtraFieldWireFields` retains the complete native `reasoning_details`, including signatures and encrypted data, while `OpenAIExtraFieldReasoningSource` marks readable thinking blocks. Text and summaries appear as `Thinking` blocks; detail IDs, signatures, and encrypted content stay in message metadata for the next request, rather than on the displayed blocks.

Use `OpenAIExtraFieldChoice` for logprobs, `OpenAIResponseExtraFieldWireFields` for IDs, service tiers, and `openrouter_metadata`, and `OpenAIResponseExtraFieldUsage` for cost, BYOK, cache, and server-tool usage details. These native objects use `map[string]json.RawMessage`; common token metrics remain in `UsageMetadata`. No OpenAI local token counter is exposed.

Embedded OpenRouter errors, including errors on individual choices, return `ApiErr` alongside any partial generated response, headers, and usage. Numeric error codes become the error status when the HTTP response was successful; router `metadata.error_type` takes precedence over `error_type` for classification.

`CerebrasGenerator` also delegates to the shared Chat Completions implementation, with `max_completion_tokens` and streamed usage enabled by default. Use common helpers for logit bias, log probabilities, seeds, and parallel tool calls. Prediction, response format, and service tier keep their Cerebras-specific helpers. `GenerationRequest.SafetyIdentifier` and `GenerationRequest.PromptCacheKey` replace the former native user/cache helpers; explicit `WithOpenAIExtraBody` fields take precedence. Model capabilities and native option ranges are validated by the server. Cerebras-specific response and usage constants have been removed: use `OpenAIExtraFieldChoice` for logprobs, `OpenAIResponseExtraFieldUsage` for native usage (including image tokens), and `OpenAIResponseExtraFieldWireFields` for timing and service-tier fields. Cerebras does not expose OpenAI's local token counter.

Thinking controls are provider-specific: `WithZaiThinking`, `WithZaiClearThinking`, `WithMoonshotThinking`, `WithMoonshotKeepThinking`, and `WithDeepSeekThinking`. Omitting them preserves server defaults; none removes caller history. Kimi K3 should use `WithReasoningEffort` without the older `thinking` controls. Always-thinking models, fixed sampling settings, and tool-choice restrictions remain the caller's responsibility. `WithZaiDoSample` and `WithZaiToolStream` expose Z.AI's separate sampling and tool-stream switches.

Z.AI counting calls `/tokenizer` with model, messages, and tools. Moonshot counting calls `/tokenizers/estimate-token-count` and rejects tool-bearing requests because the endpoint does not document tool accounting. DeepSeek does **not** implement `TokenCounter`; it has no native counting endpoint. Counters honor cancellation and do not retry or fall back to an OpenAI tokenizer.

The shared adapters send image data URLs and inline PDF bodies where the backend accepts them. The former Z.AI-only remote image/video/PDF URL conversion and provider-specific response metadata constants have been removed. Use the shared OpenAI metadata keys for returned native fields. PDF acceptance on Z.AI remains backend-dependent; no upload, extraction, or preprocessing workaround is added.

`GenerationRequest.Headers` adds per-invocation HTTP headers to generation, streaming, and remote token-count requests. Supplied values replace matching ordinary client/provider headers without modifying shared clients. SDK-managed authentication can take precedence: Gemini sets `X-Goog-Api-Key` from its client credentials. Multiple values are preserved. Do not mutate the map or its slices while the invocation is running, and avoid storing secrets in serialized requests. The field is HTTP metadata, not part of the provider JSON body.

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

`WithReasoningEffort("high")` sets a qualitative effort label. `WithThinkingBudget(4096)` sets a positive thinking-token budget on Anthropic, Gemini, or OpenRouter; those providers reject combining it with effort. Other adapters ignore numeric thinking budgets rather than serializing them as effort. Anthropic also accepts `"adaptive"` and `"disabled"` through the effort control. Gemini's native zero/automatic budget sentinels are not part of the shared positive-budget option.

Use one common helper for settings with the same meaning across providers:

| Helpers | Documented provider controls wired through GAI |
| --- | --- |
| `WithLogitBias` | OpenAI Chat Completions, Cerebras, OpenRouter |
| `WithLogprobs`, `WithTopLogprobs` | OpenAI Chat Completions, Cerebras, OpenRouter, DeepSeek, Moonshot, Gemini, Responses |
| `WithParallelToolCalls` | OpenAI Chat Completions, Cerebras, OpenRouter, Anthropic, Responses |
| `WithSeed` | OpenAI Chat Completions, Cerebras, OpenRouter, Gemini |
| `WithFrequencyPenalty`, `WithPresencePenalty` | OpenAI Chat Completions, Cerebras, OpenRouter, Gemini |
| `WithTopK` | Anthropic (supported models only; deprecated on newer models), Gemini, OpenRouter |
| `WithThinkingBudget` | Anthropic, Gemini, OpenRouter |

Chat Completions includes OpenAI, Cerebras, OpenRouter, Z.AI, DeepSeek, and Moonshot. These wrappers can forward additional shared OpenAI fields, but forwarding is not evidence of provider support. Z.AI does not document logit bias, logprob controls, seed, parallel-tool control, or frequency/presence penalties. DeepSeek documents logprob controls, but explicitly deprecates and ignores frequency/presence penalties; seed, logit bias, and parallel-tool control are not documented. Moonshot documents logprob controls; seed, logit bias, and parallel-tool control are not documented, and current Kimi models fix temperature, top-p, candidate count, and frequency/presence penalties. Explicit settings are preserved rather than filtered using model-name guesses; the server may reject them. OpenRouter support depends on the routed model/provider. Other adapters ignore options they do not implement. `WithTopLogprobs` selects how many token alternatives to return; pair it with `WithLogprobs(true)` to enable reporting. A seed requests repeatable sampling but does not guarantee identical answers. `WithCandidateCount` requests multiple candidates on OpenAI Chat Completions or Gemini; Cerebras and current Kimi models accept only one, and `StreamingAdapter` only accepts candidate zero. Log probabilities are available in Chat Completions choice metadata (`OpenAIExtraFieldChoice`), Gemini message metadata (`GeminiExtraFieldLogprobs`), or Responses text-block metadata (`ResponsesExtraFieldLogprobs`); see those keys for their value types.

**Breaking change:** the duplicated OpenAI logprob helpers and Cerebras/OpenRouter helpers for these common settings, along with their provider-specific option constants, have been removed. Replace them with the common helpers above; there are no compatibility aliases.

Min-p, top-a, and repetition penalty are OpenRouter-specific: use `WithOpenRouterMinP`, `WithOpenRouterTopA`, and `WithOpenRouterRepetitionPenalty`. Repetition penalty is distinct from frequency and presence penalties.

Output controls are provider-specific, with no aliases for the former shared APIs. Use `WithOpenAIOutputModalities` and `WithOpenAIAudioConfig(OpenAIAudioConfig{...})` for OpenAI Chat Completions audio. Use `WithGeminiOutputModalities` for Gemini text, image, or audio output on supporting models. Generated media blocks preserve base64 content, MIME types, and thought signatures; streaming media chunks remain separate blocks rather than being joined as text. Anthropic and Responses no longer parse the former shared text-only modality setting.

Prediction and response formats remain provider-specific. Use helpers such as `WithOpenAIResponseFormat`, `WithCerebrasPrediction`, or `WithOpenRouterResponseFormat` for their native contracts. `WithOpenAIStreamUsage` controls Chat Completions usage requests. OpenAI cache policies use `WithOpenAIPromptCacheRetention` or `WithOpenAIPromptCacheOptions`; the server validates model support. Native-body helpers compose with `WithOpenAIExtraBody` in option order. Explicit extra-body fields override corresponding common and provider-specific settings when building the request.

`GenerationRequest` has two independent optional fields:

| Field | Mapping and meaning |
| --- | --- |
| `SafetyIdentifier` | An opaque stable end-user identity, preferably a hash or UUID—not personal data. OpenAI Chat Completions/Responses and Moonshot send `safety_identifier`; Anthropic sends `metadata.user_id`; Z.AI and DeepSeek send `user_id`; Cerebras and OpenRouter send `user`. DeepSeek also uses it for cache and scheduling isolation. |
| `PromptCacheKey` | A cache grouping/routing hint sent as `prompt_cache_key` by OpenAI Chat Completions/Responses, Moonshot, Cerebras, and OpenRouter. An explicit OpenRouter session ID takes routing precedence. It does not enable caching, set a TTL, or name a stored cache resource. |

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

Package [`agent`](https://pkg.go.dev/github.com/spachava753/gai/agent) runs a complete model and tool loop over any `gai.Generator`. An Agent has fixed instructions and executable tools, while each run supplies a complete dialog and generation options.

```go
runner, err := agent.New(agent.Config{
	Generator: provider,
	Model:     "provider-model",
	Tools:     executableTools,
})
if err != nil {
	return err
}

dialog := append(previousDialog, gai.Message{
    Role: gai.User, Blocks: []gai.Block{gai.TextBlock("Continue")},
})
result, err := runner.Run(ctx, agent.RunRequest{Dialog: dialog})
```

The loop prefers streaming when available, validates responses, executes tools sequentially, and returns the active dialog plus standard usage. `BeforeGeneration` is the single request-change hook for compaction, retrieval, instructions, and generation settings. Its validated replacement dialog becomes the conversation that receives responses and tool results, appears in later requests, and is returned in `RunResult.Dialog`, including on a subsequent generation failure. Applications decide what to store or reuse outside the run. Other base request fields come from the Agent configuration and run options each round.

Generation and tool hooks can replace data or request a normal stop. Nil replacements preserve existing values. The optional `GenerationChunk` hook receives provisional stream chunks; all hooks use interfaces with function adapters.

Configuration and call inputs are borrowed read-only values. Struct copies share maps, slices, and pointers; callers copy the parts they want to change. Hook and handler outputs belong to the loop after return. Failures return accepted state and an `*agent.RunError` with a phase and an unwrap-able cause.

`Run` accepts a complete dialog without requiring a new user message. Historical tool calls must already have results; unresolved calls are rejected before execution. Applications own persistence and reconciliation, then call `Run` with the reconciled dialog to continue generation. The loop never repeats historical tools or restores interrupted hook state.

Use [`agent/agenttest`](https://pkg.go.dev/github.com/spachava753/gai/agent/agenttest) for scripted generators. See [`agent/design.md`](agent/design.md) for exact ordering, errors, ownership, and continuation behavior.

## Classifiers

Experimental package [`x/classifier`](x/classifier/design.md) asks AI models to classify or score input. Each request contains input and one or more questions about it. A question can ask for the probability of yes, a choice from a list, or a rating on a scale. Answers contain probabilities and scores, not generated messages or tool calls.

`classifier.Input` accepts text, structured JSON data, and media attachments. Each question can include instructions and examples. Probabilities and scores use `json.Number`; optional confidence values also use `json.Number`, but their meaning depends on the implementation. Token counts use the existing `gai.InputTokens` and `gai.OutputTokens` accessors.

Use `classifier.NewTypeSafe` for Jev, `classifier.NewOpenAI` for OpenAI Decisions, `classifier.NewCloudflare` for hosted Clef/Clef-Omni, or `classifier.NewPerplexity` for Perplexity Decisions. Constructors take an HTTP client, an optional API base URL, and credentials; Cloudflare also takes an account ID. Each request specifies its model. Clients reject unsupported input and do not add retries.

See the [classifier design and provider comparison](x/classifier/design.md) and [compiled examples](x/classifier/example_test.go) for supported inputs, usage, and validation. Tests use local HTTP servers; they have not verified live-service compatibility.

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
go tool laas -exclude-packages='^github\.com/spachava753/gai/internal/(openai|cloudflare|perplexity|typesafe)$' ./...
```

Provider HTTP clients are generated with the pinned oapi-codegen tool; do not write them by hand. `internal/openai` contains Chat Completions and OpenAI Decisions. Classifiers also use `internal/typesafe`, `internal/perplexity`, and `internal/cloudflare`. Add endpoints to these schemas or create a generated package for a new protocol. Public adapters translate inputs and convert generated response models; generated clients construct and send HTTP requests.

Regenerate clients after changing schemas, configurations, or overlays:

```bash
go generate ./internal/...
```

Each package checks that its generated file is current and tests JSON round trips and local HTTP behavior. Public classifier numbers use `json.Number`. Generated classifier models use a strict internal number type that preserves numeric spelling and rejects quoted numbers. Arbitrary JSON uses `json.RawMessage`; Chat Completions retains its existing numeric types. All Chat Completions adapters still share `internal/openai`, rather than separate compatible-provider clients.

The OpenAI adapter has [SDK request contract tests](testdata/openai_sdk/README.md). An explicit scenario matrix drives public GAI APIs against conversations recorded through the OpenAI Python SDK for OpenAI, Z.AI, Kimi, and DeepSeek. Captures store only request/response body pairs; options are declared once in the matrix and translated into public GAI options by the tests. Each named provider/mode/step case checks JSON-equivalent outgoing HTTP bodies, with later history built from GAI output. Replay also checks successful completion, usable function calls, and known scenario arguments or JSON results—not exact generated prose. New captures come from a small SDK conversation script running through mitmproxy. Building the corpus requires explicit approval for live calls; ordinary tests never refresh it.

The tracked pre-commit hook runs LAAS against hand-written packages. Activate it after cloning:

```bash
git config --local core.hooksPath .githooks
```

## License

MIT
