# gai - Go for AI

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8.svg)

Package gai provides a unified interface for interacting with various large language model (LLM) providers.

The package abstracts away provider-specific implementations, allowing you to write code that works with multiple AI providers (OpenAI, Anthropic, Google Gemini) without changing your core logic. It supports text, image, audio, and PDF modalities (provider dependent), tool integration with JSON Schema-based parameters, callback-based tool execution, automatic fallback strategies for reliability, standardized error types for better error handling, and detailed usage metrics.

## Features

- Unified API across different LLM providers
- Support for text, image, audio, and PDF modalities (provider dependent)
- Tool integration with JSON Schema-based parameters
- Callback-based tool execution
- Automatic fallback strategies for reliability
- Standardized error types for better error handling
- Detailed usage metrics
- Model Context Protocol (MCP) client support

## Installation

```bash
go get github.com/spachava753/gai
```

## Core Concepts

Generator: The core interface that all providers implement. It takes a Dialog and generates a Response.

```go
type Generator interface {
	Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error)
}
```

Each LLM provider (OpenAI, Anthropic, Gemini) has its own implementation of the Generator interface.

Dialog: A conversation with a language model, represented as a slice of Message objects.

```go
type Dialog []Message
```

Message: A single exchange in the conversation, with a Role (User, Assistant, or ToolResult) and a collection of Blocks.

```go
type Message struct {
	Role   Role
	Blocks []Block
	ToolResultError bool
}
```

Block: A self-contained piece of content within a message, which can be text, image, audio, or a tool call.

```go
type Block struct {
	ID           string
	BlockType    string
	ModalityType Modality
	MimeType     string
	Content      fmt.Stringer
	ExtraFields  map[string]interface{}
}
```

Common block types include:
- Content - Regular content like text or images
- Thinking - Reasoning/thinking from the model
- ToolCall - A request to call a tool

Modalities: gai supports multiple modalities for input and output.

```go
type Modality uint

const (
	Text Modality = iota
	Image
	Audio
	Video
)
```

Support for specific modalities depends on the underlying model provider.

Tool: A function that can be called by the language model during generation.

```go
type Tool struct {
	Name        string
	Description string
	InputSchema InputSchema
}
```

The InputSchema defines the parameters the tool accepts using JSON Schema conventions:

```go
type InputSchema struct {
	Type       PropertyType
	Properties map[string]Property
	Required   []string
}
```

## Basic Usage Examples

Basic usage with OpenAI:

```go
package main

import (
	"context"
	"fmt"
	"github.com/openai/openai-go"
	"github.com/spachava753/gai"
)

func main() {
	// Create an OpenAI client
	client := openai.NewClient()

	// Create a generator with a specific model
	generator := gai.NewOpenAiGenerator(
		client.Chat.Completions,
		openai.ChatModelGPT4,
		"You are a helpful assistant.",
	)

	// Create a dialog with a user message
	dialog := gai.Dialog{
		{
			Role: gai.User,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      gai.Str("What is the capital of France?"),
				},
			},
		},
	}

	// Generate a response
	response, err := generator.Generate(context.Background(), dialog, &gai.GenOpts{
		Temperature: 0.7,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Print the response
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println(response.Candidates[0].Blocks[0].Content)
	}

	// Get usage metrics
	if inputTokens, ok := gai.InputTokens(response.UsageMetrics); ok {
		fmt.Printf("Input tokens: %d\n", inputTokens)
	}
	if outputTokens, ok := gai.OutputTokens(response.UsageMetrics); ok {
		fmt.Printf("Output tokens: %d\n", outputTokens)
	}
}
```

## Tool Usage Example

Using tools with a language model:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
	"github.com/openai/openai-go"
	"github.com/spachava753/gai"
)

// Define a tool callback for getting the current time
type TimeToolCallback struct{}

func (t TimeToolCallback) Call(ctx context.Context, parametersJSON json.RawMessage, toolCallID string) (gai.Message, error) {
	return gai.TextToolResultMessage(toolCallID, time.Now().Format(time.RFC1123)), nil
}

func main() {
	client := openai.NewClient()

	// Create an OpenAI generator
	baseGen := gai.NewOpenAiGenerator(
		client.Chat.Completions,
		openai.ChatModelGPT4,
		"You are a helpful assistant.",
	)

	// Create a tool generator that wraps the base generator
	toolGen := &gai.ToolGenerator{
		G: &baseGen,
	}

	// Define a time tool
	timeTool := gai.Tool{
		Name:        "get_current_time",
		Description: "Get the current server time",
	}

	// Register the tool with its callback
	if err := toolGen.Register(timeTool, &TimeToolCallback{}); err != nil {
		fmt.Printf("Error registering tool: %v\n", err)
		return
	}

	// Create a dialog
	dialog := gai.Dialog{
		{
			Role: gai.User,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      gai.Str("What time is it now?"),
				},
			},
		},
	}

	// Generate a response with tool usage
	completeDialog, err := toolGen.Generate(context.Background(), dialog, func(d gai.Dialog) *gai.GenOpts {
		return &gai.GenOpts{
			ToolChoice: gai.ToolChoiceAuto,
		}
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Print the final result
	finalMsg := completeDialog[len(completeDialog)-1]
	if len(finalMsg.Blocks) > 0 {
		fmt.Println(finalMsg.Blocks[0].Content)
	}
}
```

## Fallback Strategy Example

Implementing a fallback strategy between providers:

```go
package main

import (
	"context"
	"fmt"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/spachava753/gai"
)

func main() {
	// Create clients for both providers
	openaiClient := openai.NewClient()
	anthropicClient := anthropic.NewClient()

	// Create generators for each provider
	openaiGen := gai.NewOpenAiGenerator(
		openaiClient.Chat.Completions,
		openai.ChatModelGPT4,
		"You are a helpful assistant.",
	)

	anthropicGen := gai.NewAnthropicGenerator(
		anthropicClient.Messages,
		"claude-3-opus-20240229",
		"You are a helpful assistant.",
	)

	// Create a fallback generator that tries OpenAI first, then falls back to Anthropic
	fallbackGen, err := gai.NewFallbackGenerator(
		[]gai.Generator{&openaiGen, &anthropicGen},
		&gai.FallbackConfig{
			// Custom fallback condition: fall back on rate limits and 5xx errors
			ShouldFallback: gai.NewHTTPStatusFallbackConfig(429, 500, 502, 503, 504).ShouldFallback,
		},
	)
	if err != nil {
		fmt.Printf("Error creating fallback generator: %v\n", err)
		return
	}

	// Create a dialog
	dialog := gai.Dialog{
		{
			Role: gai.User,
			Blocks: []gai.Block{
				{
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					Content:      gai.Str("What is the meaning of life?"),
				},
			},
		},
	}

	// Generate a response using the fallback strategy
	response, err := fallbackGen.Generate(context.Background(), dialog, &gai.GenOpts{
		Temperature: 0.7,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Print the response
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println(response.Candidates[0].Blocks[0].Content)
	}
}
```

## Working with PDFs

gai supports PDF documents as a special case of the Image modality. PDFs are automatically converted to images at the model provider's API level:

```go
package main

import (
	"context"
	"fmt"
	"os"
	"github.com/openai/openai-go"
	"github.com/spachava753/gai"
)

func main() {
	// Read a PDF file
	pdfData, err := os.ReadFile("document.pdf")
	if err != nil {
		fmt.Printf("Error reading PDF: %v\n", err)
		return
	}

	// Create an OpenAI client and generator
	client := openai.NewClient()
	generator := gai.NewOpenAiGenerator(
		&client.Chat.Completions,
		openai.ChatModelGPT4o,
		"You are a helpful document analyst.",
	)

	// Create a dialog with PDF content
	dialog := gai.Dialog{
		{
			Role: gai.User,
			Blocks: []gai.Block{
				gai.TextBlock("Please summarize this PDF document:"),
				gai.PDFBlock(pdfData, "document.pdf"),
			},
		},
	}

	// Generate a response
	response, err := generator.Generate(context.Background(), dialog, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Print the response
	if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
		fmt.Println(response.Candidates[0].Blocks[0].Content)
	}
}
```

PDF support notes:
- OpenAI Token counting: PDF token counting is not supported and will return an error when using the TokenCounter interface
- When creating a PDF block, you must provide both the PDF data and a filename, e.g. PDFBlock(data, "paper.pdf")
- All providers: PDFs are converted to images server-side, so exact page dimensions are not known

## Provider Support

The package supports multiple LLM providers with varying capabilities:

OpenAI: The OpenAI implementation supports text generation, image inputs (including PDFs), audio inputs, and tool calling.

```go
import (
	"github.com/openai/openai-go"
	"github.com/spachava753/gai"
)

client := openai.NewClient()
generator := gai.NewOpenAiGenerator(
	&client.Chat.Completions,
	openai.ChatModelGPT4,
	"System instructions here.",
)
```

Anthropic: The Anthropic implementation supports text generation, image inputs (including PDFs with special handling), and tool calling.

```go
import (
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/spachava753/gai"
)

client := anthropic.NewClient()
generator := gai.NewAnthropicGenerator(
	&client.Messages,
	"claude-3-opus-20240229",
	"System instructions here.",
)
```

Gemini: The Gemini implementation supports text generation, image inputs (including PDFs), audio inputs, and tool calling.

```go
import (
	"google.golang.org/genai"
	"github.com/spachava753/gai"
)

client, err := genai.NewClient(ctx, &genai.ClientConfig{
	APIKey: "your-api-key",
})
generator, err := gai.NewGeminiGenerator(
	client,
	"gemini-1.5-pro",
	"System instructions here.",
)
```

## Error Handling

The package provides standardized error types for consistent error handling across providers:

- MaxGenerationLimitErr - Maximum token generation limit reached
- UnsupportedInputModalityErr - Model doesn't support the requested input modality
- UnsupportedOutputModalityErr - Model doesn't support the requested output modality
- InvalidToolChoiceErr - Invalid tool choice specified
- InvalidParameterErr - Invalid generation parameter
- ContextLengthExceededErr - Input dialog exceeds model's context length
- ContentPolicyErr - Content violates usage policies
- EmptyDialogErr - No messages provided
- AuthenticationErr - Authentication/authorization issues
- RateLimitErr - API request rate limits exceeded
- ApiErr - Other API errors with status code, type, and message

Example error handling:

```go
response, err := generator.Generate(ctx, dialog, options)
if err != nil {
	switch {
	case errors.Is(err, gai.MaxGenerationLimitErr):
		fmt.Println("Maximum generation limit reached")
	case errors.Is(err, gai.ContextLengthExceededErr):
		fmt.Println("Context length exceeded")
	case errors.Is(err, gai.EmptyDialogErr):
		fmt.Println("Empty dialog provided")

	// Type-specific errors
	case errors.As(err, &gai.RateLimitErr{}):
		fmt.Println("Rate limit exceeded:", err)
	case errors.As(err, &gai.ContentPolicyErr{}):
		fmt.Println("Content policy violation:", err)
	case errors.As(err, &gai.ApiErr{}):
		apiErr := err.(gai.ApiErr)
		fmt.Printf("API error: %d %s - %s\n", apiErr.StatusCode, apiErr.Type, apiErr.Message)
	default:
		fmt.Println("Unexpected error:", err)
	}
	return
}
```

## Advanced Usage

Tool Generator: The ToolGenerator provides advanced functionality for working with tools. It automatically handles registering tools with the underlying generator, executing tool callbacks when tools are called, managing the conversation flow during tool use, and handling parallel tool calls.

```go
type ToolGenerator struct {
	G             ToolCapableGenerator
	toolCallbacks map[string]ToolCallback
}
```

Example:

```go
// Create a base generator (OpenAI or Anthropic)
baseGen := gai.NewOpenAiGenerator(...)

// Create a tool generator
toolGen := &gai.ToolGenerator{
	G: &baseGen,
}

// Register tools with callbacks
toolGen.Register(weatherTool, &WeatherAPI{})
toolGen.Register(stockPriceTool, &StockAPI{})

// Generate with tool support
completeDialog, err := toolGen.Generate(ctx, dialog, func(d gai.Dialog) *gai.GenOpts {
	return &gai.GenOpts{
		ToolChoice: gai.ToolChoiceAuto,
		Temperature: 0.7,
	}
})
```

Fallback Generator: The FallbackGenerator provides automatic fallback between different providers. It automatically tries each generator in sequence, falls back based on configurable conditions, and preserves the original error if all generators fail.

```go
type FallbackGenerator struct {
	generators []Generator
	config     FallbackConfig
}
```

Configuration options:
- NewHTTPStatusFallbackConfig() - Fallback on specific HTTP status codes
- NewRateLimitOnlyFallbackConfig() - Fallback only on rate limit errors
- Custom fallback logic via ShouldFallback function

Example:

```go
primaryGen := gai.NewOpenAiGenerator(...)
backupGen := gai.NewAnthropicGenerator(...)

fallbackGen, err := gai.NewFallbackGenerator(
	[]gai.Generator{primaryGen, backupGen},
	&gai.FallbackConfig{
		ShouldFallback: func(err error) bool {
			// Custom fallback logic
			return gai.IsRateLimitError(err) || gai.IsServerError(err)
		},
	},
)
```

## Model Context Protocol (MCP)

The package includes MCP (Model Context Protocol) client support for connecting to external tools and data sources. The MCP client allows you to connect to MCP servers via stdio, HTTP, or other transports and use their tools within the gai framework.

Example MCP usage:

```go
import "github.com/spachava753/gai/mcp"

// Create MCP client
transport := mcp.NewStdio(mcp.StdioConfig{
	Command: "python",
	Args:    []string{"mcp_server.py"},
})

client, err := mcp.NewClient(ctx, transport, mcp.ClientInfo{
	Name:    "gai-client",
	Version: "1.0.0",
}, mcp.ClientCapabilities{}, mcp.DefaultOptions())

// Register MCP tools with a tool generator
err = mcp.RegisterMCPToolsWithGenerator(ctx, client, toolGen)
```

For more information and examples, see the README and example files in the repository.

## License

This project is licensed under the MIT License.

