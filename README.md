# GAI — Go for AI

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Go Version](https://img.shields.io/badge/Go-1.26.6+-00ADD8.svg)

GAI is a Go library for building applications with large language models. It provides a provider-neutral API for generation, streaming, multimodal content, and tool calls, along with utilities for retries, fallbacks, and agent loops.

Each call uses a self-contained request containing the model, instructions, conversation, tools, and options. Provider generators hold only execution dependencies such as clients and credentials, so the same generator can serve different requests safely.

## Capabilities

- OpenAI Chat Completions and Responses, Anthropic, Gemini, Cerebras, OpenRouter, DeepSeek, Z.AI, and Moonshot (Kimi)
- Text, image, audio, and PDF inputs, with provider-dependent media output
- Incremental streaming and token counting on supporting providers
- Function calling with application-controlled tool execution
- Common generation options plus provider-specific options and metadata
- Retry, fallback, and preprocessing middleware
- Reusable agent loops and experimental structured classifiers

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

## Streaming

Providers that support incremental output implement `StreamingGenerator`:

```go
for chunk := range generator.Stream(ctx, request) {
	if chunk.Err != nil {
		return chunk.Err
	}
	if chunk.Block.BlockType == gai.Content {
		fmt.Print(chunk.Block.Content)
	}
}
```

Use `StreamingAdapter` to collect a single-candidate stream into a regular response.

## Tools and agents

A `Tool` declares a function and its JSON Schema. GAI translates tool declarations and returned calls for each provider, but your application remains responsible for authorizing and executing calls.

Package [`agent`](https://pkg.go.dev/github.com/spachava753/gai/agent) builds on this API to run a complete generation and tool loop. Applications own dialog persistence and can customize runs with generation and tool hooks.

## Classifiers

The experimental [`x/classifier`](https://pkg.go.dev/github.com/spachava753/gai/x/classifier) package provides structured classification, scoring, extraction, estimation, and forecasting. It accepts text, JSON, and supported media inputs and returns typed answers rather than conversational messages.

## Documentation and examples

See the [Go package documentation](https://pkg.go.dev/github.com/spachava753/gai) for request and response types, provider support, generation options, streaming, tools, composition, errors, and provider-specific behavior.

Compiled examples are available alongside the source:

- [Core generation, streaming, tools, options, and errors](example_test.go)
- [Retries](retry_generator_example_test.go)
- [Agent loops and hooks](agent/example_test.go)
- [Classifiers](x/classifier/example_test.go)

## License

MIT
