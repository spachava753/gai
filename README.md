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

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/spachava753/gai"
)

func main() {
	client := openai.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	generator := gai.NewOpenAiGenerator(&client.Chat.Completions)

	response, err := generator.Generate(context.Background(), gai.GenerationRequest{
		Model: openai.ChatModelGPT5Mini,
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
- `Instructions` contains a `System` message outside the conversation.
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
| Anthropic | `NewAnthropicGenerator` | `StreamingGenerator`, `TokenCounter` |
| Google Gemini | `NewGeminiGenerator` | `StreamingGenerator`, `TokenCounter` |
| Cerebras | `NewCerebrasGenerator` | `StreamingGenerator` |
| OpenRouter | `NewOpenRouterGenerator` | `StreamingGenerator` |
| DeepSeek | `NewDeepSeekGenerator` | `StreamingGenerator` |
| Z.AI | `NewZaiGenerator` | `StreamingGenerator`, `TokenCounter` |

Provider type documentation lists supported content, common options, native options, response metadata, and replay requirements. See the [package documentation](https://pkg.go.dev/github.com/spachava753/gai).

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

Providers ignore unknown option keys. Recognized values with an invalid type, range, or combination return `InvalidParameterErr`.

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
go tool laas -exclude-packages='^github\.com/spachava753/gai/internal/(cerebras|deepseek|openrouter|zai)$' ./...
```

The tracked pre-commit hook runs LAAS against hand-written packages. Activate it after cloning:

```bash
git config --local core.hooksPath .githooks
```

## License

MIT
