// Package gai provides a unified interface for interacting with various large language model (LLM) providers.
//
// The package abstracts away provider-specific implementations, allowing you to write code that works
// with OpenAI, Anthropic, Google Gemini, Cerebras, OpenRouter, ZAI, and DeepSeek without changing
// your core logic. It supports text, image, audio, and PDF modalities (provider dependent), tool integration with
// JSON Schema-based parameters, callback-based tool execution, automatic fallback strategies for
// reliability, standardized error types for better error handling, and detailed usage metrics.
//
// # Features
//
//   - Unified API across different LLM providers
//   - Support for text, image, audio, and PDF modalities (provider dependent)
//   - Tool integration with JSON Schema-based parameters
//   - Callback-based tool execution
//   - Automatic fallback strategies for reliability
//   - Standardized error types for better error handling
//   - Detailed usage metrics
//   - Model Context Protocol (MCP) client support
//
// # Installation
//
//	go get github.com/spachava753/gai
//
// # Core Concepts
//
// Generator: The core interface that all providers implement. Each call receives a
// self-contained GenerationRequest with its model, instructions, dialog, tools, and options.
//
//	type Generator interface {
//		Generate(ctx context.Context, request GenerationRequest) (Response, error)
//	}
//
//	type GenerationRequest struct {
//		Model        string
//		Instructions Message
//		Dialog       Dialog
//		Tools        []Tool
//		Options      GenerationOptions
//	}
//
// Provider generators store only execution dependencies such as clients and endpoints.
//
// Dialog: A conversation with a language model, represented as a slice of Message objects.
//
//	type Dialog []Message
//
// Message: A single exchange or instruction, with a Role (User, Assistant,
// ToolResult, or System) and a collection of Blocks.
//
//	type Message struct {
//		Role   Role
//		Blocks []Block
//		ToolResultError bool
//		ExtraFields  map[string]interface{}
//	}
//
// Block: A self-contained piece of content within a message, which can be text, image, audio,
// or a tool call.
//
//	type Block struct {
//		ID           string
//		BlockType    string
//		ModalityType Modality
//		MimeType     string
//		Content      fmt.Stringer
//		ExtraFields  map[string]interface{}
//	}
//
// Common block types include:
//   - Content - Regular content like text or images
//   - Thinking - Reasoning/thinking from the model
//   - ToolCall - A request to call a tool
//
// Modalities: gai supports multiple modalities for input and output.
//
//	type Modality uint
//
//	const (
//		Text Modality = iota
//		Image
//		Audio
//		Video
//	)
//
// Support for specific modalities depends on the underlying model provider.
//
// Tool: A function that can be called by the language model during generation.
//
//	type Tool struct {
//		Name        string
//		Description string
//		InputSchema *jsonschema.Schema
//	}
//
// The InputSchema defines the parameters the tool accepts using JSON Schema conventions:
//
//	&jsonschema.Schema{
//		Type:       "object",
//		Properties: map[string]*jsonschema.Schema{...},
//		Required:   []string{...},
//	}
//
// # Basic Usage Examples
//
// Basic usage with OpenAI:
//
//	package main
//
//	import (
//		"context"
//		"fmt"
//		"github.com/openai/openai-go/v3"
//		"github.com/spachava753/gai"
//	)
//
//	func main() {
//		// Create an OpenAI client
//		client := openai.NewClient()
//
//		// Create a stateless generator. Model configuration belongs to each request.
//		generator := gai.NewOpenAiGenerator(&client.Chat.Completions)
//
//		// Create a dialog with a user message
//		dialog := gai.Dialog{
//			{
//				Role: gai.User,
//				Blocks: []gai.Block{
//					{
//						BlockType:    gai.Content,
//						ModalityType: gai.Text,
//						Content:      gai.Str("What is the capital of France?"),
//					},
//				},
//			},
//		}
//
//		// Generate a response
//		response, err := generator.Generate(context.Background(), gai.GenerationRequest{
//			Model:        openai.ChatModelGPT4,
//			Instructions: gai.SystemMessage(gai.TextBlock("You are a helpful assistant.")),
//			Dialog:       dialog,
//			Options:      gai.NewGenerationOptions(gai.WithTemperature(0.7)),
//		})
//		if err != nil {
//			fmt.Printf("Error: %v\n", err)
//			return
//		}
//
//		// Print the response
//		if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
//			fmt.Println(response.Candidates[0].Blocks[0].Content)
//		}
//
//		// Get usage metrics
//		if inputTokens, ok := gai.InputTokens(response.UsageMetadata); ok {
//			fmt.Printf("Input tokens: %d\n", inputTokens)
//		}
//		if outputTokens, ok := gai.OutputTokens(response.UsageMetadata); ok {
//			fmt.Printf("Output tokens: %d\n", outputTokens)
//		}
//	}
//
// # Provider-specific controls and response details
//
// Provider option functions such as [WithCerebrasServiceTier],
// [WithOpenRouterProviderPreferences], and [WithResponsesPromptCacheKey] return
// GenerationOption values and compose with the common helpers in [NewGenerationOptions].
// The underlying keys remain exported for inspection and experimental direct assignment.
//
// Providers put measurements in Response.UsageMetadata. Invocation details such as
// completion IDs, model names, timestamps, fingerprints, and service tiers go in
// Response.ExtraFields. Candidate-level details go in Message.ExtraFields, while
// replay-critical content metadata stays with the corresponding Block.ExtraFields.
// [StreamingAdapter] preserves response and message extra fields from stream chunks.
//
// # Prompt Caching with ResponsesGenerator
//
// The OpenAI Responses generator supports explicit prompt cache routing through
// GenerationOptions. Use [WithResponsesPromptCacheKey] with a stable key for requests
// that share the same long static prompt prefix. Keep repeated instructions, schemas,
// and tool definitions at the beginning of the prompt, and put request-specific content
// near the end.
//
//	client := openai.NewClient()
//	gen := gai.NewResponsesGenerator(&client.Responses)
//	options := gai.NewGenerationOptions(
//		gai.WithResponsesPromptCacheKey("support-incident-summary:v1"),
//	)
//	resp, err := gen.Generate(ctx, gai.GenerationRequest{
//		Model:        openai.ChatModelGPT5Mini,
//		Instructions: gai.SystemMessage(gai.TextBlock("You are a helpful assistant that summarizes support incidents.")),
//		Dialog:       dialog,
//		Options:      options,
//	})
//	if err != nil {
//		fmt.Printf("Error: %v\n", err)
//		return
//	}
//	if cached, ok := gai.CacheReadTokens(resp.UsageMetadata); ok {
//		fmt.Printf("cached tokens: %d\n", cached)
//	}
//
// # Responses API Service Tiers
//
// Use [WithResponsesServiceTier] to choose how OpenAI processes a Responses API
// request. Supported values are "auto", "default", "flex", "scale", "priority",
// "fast", and "ultrafast". If omitted, OpenAI uses its default "auto" behavior.
// The option applies to both Generate and Stream calls.
//
//	request := gai.GenerationRequest{
//		Model:  openai.ChatModelGPT5Mini,
//		Dialog: dialog,
//		Options: gai.NewGenerationOptions(
//			gai.WithResponsesServiceTier("priority"),
//		),
//	}
//	resp, err := gen.Generate(ctx, request)
//	if err != nil {
//		fmt.Printf("Error: %v\n", err)
//		return
//	}
//
// # Tool Usage Example
//
// Tools are request data. The generator exposes tool calls in its response;
// applications own execution so they can apply authorization, validation, retries,
// tracing, and persistence before continuing the dialog.
//
//	client := openai.NewClient()
//	gen := gai.NewOpenAiGenerator(&client.Chat.Completions)
//	currentTimeTool := gai.Tool{
//		Name:        "get_current_time",
//		Description: "Get the current server time",
//	}
//	dialog := gai.Dialog{{Role: gai.User, Blocks: []gai.Block{gai.TextBlock("What time is it now?")}}}
//	request := gai.GenerationRequest{
//		Model:        openai.ChatModelGPT4,
//		Instructions: gai.SystemMessage(gai.TextBlock("You are a helpful assistant.")),
//		Dialog:       dialog,
//		Tools:        []gai.Tool{currentTimeTool},
//		Options:      gai.NewGenerationOptions(gai.WithToolChoice(gai.ToolChoiceAuto)),
//	}
//	resp, err := gen.Generate(context.Background(), request)
//	if err != nil {
//		fmt.Printf("Error: %v\n", err)
//		return
//	}
//
//	if resp.FinishReason == gai.ToolUse {
//		// Inspect ToolCall blocks, execute approved tools, append ToolResultMessage
//		// entries to the dialog, and call Generate again to produce the final answer.
//	}
//
// # Fallback Strategy Example
//
// Fallback forwards one request unchanged, so every target must understand the
// request's model identifier. This example uses two OpenAI-compatible endpoints:
//
//	primaryClient := openai.NewClient()
//	backupClient := openai.NewClient() // Configure a separate endpoint or credential.
//	primaryGen := gai.NewOpenAiGenerator(&primaryClient.Chat.Completions)
//	backupGen := gai.NewOpenAiGenerator(&backupClient.Chat.Completions)
//
//	fallbackGen, err := gai.NewFallbackGenerator(
//		[]gai.Generator{primaryGen, backupGen},
//		nil,
//	)
//	if err != nil {
//		fmt.Printf("Error creating fallback generator: %v\n", err)
//		return
//	}
//
//	request := gai.GenerationRequest{
//		Model:        openai.ChatModelGPT4,
//		Instructions: gai.SystemMessage(gai.TextBlock("You are a helpful assistant.")),
//		Dialog: gai.Dialog{{
//			Role:   gai.User,
//			Blocks: []gai.Block{gai.TextBlock("What is the meaning of life?")},
//		}},
//		Options: gai.NewGenerationOptions(gai.WithTemperature(0.7)),
//	}
//	response, err := fallbackGen.Generate(context.Background(), request)
//	if err != nil {
//		fmt.Printf("Error: %v\n", err)
//		return
//	}
//	fmt.Println(response.Candidates[0].Blocks[0].Content)
//
// # Retrying Transient Failures
//
// RetryGenerator retries standalone context deadline errors and ApiErr values classified as
// retryable. The caller authorizes every retry through RetryConfig.Backoff. A provider
// Retry-After hint can replace the callback's delay, but cannot override a false decision.
// Start from DefaultRetryConfig to opt into exponential backoff with jitter, then add explicit
// retry-scheduling limits:
//
//	config := gai.DefaultRetryConfig()
//	config.MaxAttempts = 4
//	config.MaxElapsedTime = 30 * time.Second
//
//	var retries atomic.Uint64
//	config.Notify = func(err error, delay time.Duration) {
//		retries.Add(1)
//	}
//
//	retryingGenerator := gai.NewRetryGenerator(baseGenerator, config)
//	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
//	defer cancel()
//	response, err := retryingGenerator.Generate(ctx, request)
//
// MaxElapsedTime is checked only between attempts; it cannot interrupt an in-flight provider
// call. Use a context deadline, as above, for a hard wall-clock bound. Backoff and Notify can
// run concurrently when the generator is shared, so callbacks must be concurrency-safe. Stream
// retries only failures before its first emitted chunk; errors after output are returned without
// replaying the stream. RetryGenerator does not disable retries in provider SDKs; disable SDK
// retries separately to avoid nested retry loops.
//
// # Working with Thinking Blocks
//
// Many LLM providers support "thinking" or "reasoning" output, where the model shows its
// internal reasoning process. gai normalizes these into Thinking blocks (BlockType == Thinking).
//
// Some providers can produce multiple logical thinking blocks in one streamed response.
// Streaming generators may emit SeparatorBlock chunks between those provider blocks so
// StreamingAdapter can preserve the same boundaries in the final Message. Separator blocks
// are internal streaming markers and do not appear in non-streaming Response messages.
//
// For OpenAI Responses reasoning summaries, each summary_index is represented as a
// separate Thinking block. Those blocks include ResponsesExtraFieldReasoningID and
// ResponsesExtraFieldSummaryIndex in ExtraFields, and may include encrypted replay
// content under ResponsesExtraFieldEncryptedContent.
//
// To identify which generator produced a thinking block, check the ThinkingExtraFieldGeneratorKey
// in the block's ExtraFields. This allows you to handle provider-specific features:
//
//	for _, block := range message.Blocks {
//	    if block.BlockType == gai.Thinking {
//	        generator := block.ExtraFields[gai.ThinkingExtraFieldGeneratorKey]
//	        fmt.Printf("Thinking from %s: %s\n", generator, block.Content)
//
//	        // Handle provider-specific fields
//	        switch generator {
//	        case gai.ThinkingGeneratorAnthropic:
//	            // Anthropic requires signatures for extended thinking
//	            if sig, ok := block.ExtraFields[gai.AnthropicExtraFieldThinkingSignature]; ok {
//	                fmt.Printf("Signature: %s\n", sig)
//	            }
//	        case gai.ThinkingGeneratorGemini:
//	            // Gemini may include thought signatures
//	            if sig, ok := block.ExtraFields[gai.GeminiExtraFieldThoughtSignature]; ok {
//	                fmt.Printf("Thought signature: %s\n", sig)
//	            }
//	        case gai.ThinkingGeneratorOpenRouter:
//	            // OpenRouter includes reasoning metadata
//	            reasonType := block.ExtraFields[gai.OpenRouterExtraFieldReasoningType]
//	            fmt.Printf("Reasoning type: %s\n", reasonType)
//	        }
//	    }
//	}
//
// Available generator constants:
//   - ThinkingGeneratorAnthropic - Anthropic Claude models with extended thinking
//   - ThinkingGeneratorCerebras - Cerebras models with reasoning
//   - ThinkingGeneratorDeepSeek - DeepSeek models with reasoning
//   - ThinkingGeneratorGemini - Google Gemini models with thinking
//   - ThinkingGeneratorOpenRouter - OpenRouter with reasoning models
//   - ThinkingGeneratorResponses - OpenAI Responses API with reasoning
//   - ThinkingGeneratorZai - Zai generator with reasoning
//
// Note: The OpenAI Chat Completions generator (OpenAiGenerator) does not support thinking blocks.
//
// # Working with PDFs
//
// gai supports PDF documents as a special case of the Image modality. PDFs are automatically
// converted to images at the model provider's API level:
//
//	package main
//
//	import (
//		"context"
//		"fmt"
//		"os"
//		"github.com/openai/openai-go/v3"
//		"github.com/spachava753/gai"
//	)
//
//	func main() {
//		// Read a PDF file
//		pdfData, err := os.ReadFile("document.pdf")
//		if err != nil {
//			fmt.Printf("Error reading PDF: %v\n", err)
//			return
//		}
//
//		// Create an OpenAI client and generator
//		client := openai.NewClient()
//		generator := gai.NewOpenAiGenerator(&client.Chat.Completions)
//
//		// Create a dialog with PDF content
//		dialog := gai.Dialog{
//			{
//				Role: gai.User,
//				Blocks: []gai.Block{
//					gai.TextBlock("Please summarize this PDF document:"),
//					gai.PDFBlock(pdfData, "document.pdf"),
//				},
//			},
//		}
//
//		// Generate a response
//		response, err := generator.Generate(context.Background(), gai.GenerationRequest{
//			Model:        openai.ChatModelGPT4o,
//			Instructions: gai.SystemMessage(gai.TextBlock("You are a helpful document analyst.")),
//			Dialog:       dialog,
//		})
//		if err != nil {
//			fmt.Printf("Error: %v\n", err)
//			return
//		}
//
//		// Print the response
//		if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
//			fmt.Println(response.Candidates[0].Blocks[0].Content)
//		}
//	}
//
// PDF support notes:
//   - OpenAI Token counting: PDF token counting is not supported and will return an error when using the TokenCounter interface
//   - When creating a PDF block, you must provide both the PDF data and a filename, e.g. PDFBlock(data, "paper.pdf")
//   - All providers: PDFs are converted to images server-side, so exact page dimensions are not known
//
// # Provider Support
//
// The package supports multiple LLM providers with varying capabilities:
//
// OpenAI: The OpenAI implementation supports text generation, image inputs (including PDFs),
// audio inputs, and tool calling.
//
//	import (
//		"github.com/openai/openai-go/v3"
//		"github.com/spachava753/gai"
//	)
//
//	client := openai.NewClient()
//	generator := gai.NewOpenAiGenerator(&client.Chat.Completions)
//
// Anthropic: The Anthropic implementation supports text generation, image inputs
// (including PDFs with special handling), and tool calling.
//
//	import (
//		"github.com/anthropics/anthropic-sdk-go"
//		"github.com/spachava753/gai"
//	)
//
//	client := anthropic.NewClient()
//	generator := gai.NewAnthropicGenerator(&client.Messages)
//
// Gemini: The Gemini implementation supports text generation, image inputs (including PDFs),
// audio inputs, and tool calling.
//
//	import (
//		"google.golang.org/genai"
//		"github.com/spachava753/gai"
//	)
//
//	client, err := genai.NewClient(ctx, &genai.ClientConfig{
//		APIKey: "your-api-key",
//	})
//	generator := gai.NewGeminiGenerator(client)
//
// Cerebras: The Cerebras implementation supports streaming text generation, PNG/JPEG image input,
// function tools, and replayable reasoning content. A nil HTTP client and empty base URL use the
// defaults. The API key is required.
//
//	generator, err := gai.NewCerebrasGenerator(nil, "", "your-api-key")
//
// OpenRouter: The OpenRouter implementation supports streaming text generation with multimodal input,
// function tools, and replayable reasoning details. A nil HTTP client and empty base URL use the
// defaults. The API key is required.
//
//	generator, err := gai.NewOpenRouterGenerator(nil, "", "your-api-key")
//
// Z.AI: The Z.AI implementation supports multimodal generation, streaming, function tools,
// and replayable reasoning content. A nil HTTP client and empty base URL use the defaults.
// The API key is required.
//
//	generator, err := gai.NewZaiGenerator(nil, "", "your-api-key")
//
// DeepSeek: The DeepSeek implementation supports text generation, streaming, function tools,
// and replayable reasoning content. A nil HTTP client and empty base URL use the defaults.
// The API key is required.
//
//	generator, err := gai.NewDeepSeekGenerator(nil, "", "your-api-key")
//
// For every provider, set the model and system instructions on GenerationRequest.
// Constructors retain only clients, credentials, endpoints, and other execution dependencies.
//
// # Error Handling
//
// The package provides standardized error types for consistent error handling across providers:
//
//   - ErrMaxGenerationLimit - Maximum token generation limit reached
//   - UnsupportedInputModalityErr - Model doesn't support the requested input modality
//   - UnsupportedOutputModalityErr - Model doesn't support the requested output modality
//   - InvalidToolChoiceErr - Invalid tool choice specified
//   - InvalidParameterErr - Invalid generation parameter
//   - ErrContextLengthExceeded - Input dialog exceeds model's context length
//   - ContentPolicyErr - Content violates usage policies
//   - ErrEmptyDialog - No messages provided
//   - ApiErr - Provider/server errors with normalized provider, kind, status, and message fields
//
// Example error handling:
//
//	response, err := generator.Generate(ctx, request)
//	if err != nil {
//		switch {
//		case errors.Is(err, gai.ErrMaxGenerationLimit):
//			fmt.Println("Maximum generation limit reached")
//		case errors.Is(err, gai.ErrContextLengthExceeded):
//			fmt.Println("Context length exceeded")
//		case errors.Is(err, gai.ErrEmptyDialog):
//			fmt.Println("Empty dialog provided")
//
//		case errors.As(err, &gai.ContentPolicyErr{}):
//			fmt.Println("Content policy violation:", err)
//		default:
//			var apiErr *gai.ApiErr
//			if errors.As(err, &apiErr) {
//				fmt.Printf("API error: provider=%s kind=%s status=%d message=%s\n", apiErr.Provider, apiErr.Kind, apiErr.StatusCode, apiErr.Message)
//			} else {
//				fmt.Println("Unexpected error:", err)
//			}
//		}
//		return
//	}
//
// # Advanced Usage
//
// Tool Calling: Put the complete set of tools available to one invocation in
// GenerationRequest.Tools. Provider adapters validate and convert those definitions.
// Applications execute approved calls and decide how results re-enter the dialog.
//
// Example:
//
//	gen := gai.NewOpenAiGenerator(&client.Chat.Completions)
//	request := gai.GenerationRequest{
//		Model:        openai.ChatModelGPT4,
//		Instructions: gai.SystemMessage(gai.TextBlock("You are a helpful assistant.")),
//		Dialog:       dialog,
//		Tools:        []gai.Tool{weatherTool, stockPriceTool},
//		Options: gai.NewGenerationOptions(
//			gai.WithToolChoice(gai.ToolChoiceAuto),
//			gai.WithTemperature(0.7),
//		),
//	}
//	resp, err := gen.Generate(ctx, request)
//
//	// When resp.FinishReason is ToolUse, inspect ToolCall blocks, execute the
//	// corresponding application callbacks, append ToolResultMessage values, and
//	// invoke Generate again to continue the conversation.
//
// Fallback Generator: The FallbackGenerator provides automatic fallback between different providers.
// It automatically tries each generator in sequence, falls back based on configurable conditions,
// and preserves the original error if all generators fail.
//
//	type FallbackGenerator struct {
//		generators []Generator
//		config     FallbackConfig
//	}
//
// Configuration options:
//   - NewHTTPStatusFallbackConfig() - Fallback on specific HTTP status codes
//   - NewRateLimitOnlyFallbackConfig() - Fallback only on rate limit errors
//   - Custom fallback logic via ShouldFallback function
//
// Example:
//
//	// primaryGen and backupGen must both understand request.Model.
//	fallbackGen, err := gai.NewFallbackGenerator(
//		[]gai.Generator{primaryGen, backupGen},
//		&gai.FallbackConfig{
//			ShouldFallback: func(err error) bool {
//				// Custom fallback logic
//				var apiErr *gai.ApiErr
//				return errors.As(err, &apiErr) && apiErr.Retryable()
//			},
//		},
//	)
//
// # License
//
// This project is licensed under the MIT License.
package gai
