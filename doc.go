// Package gai provides a unified interface for interacting with various large language model (LLM) providers.
//
// The package abstracts away provider-specific implementations, allowing you to write code that works
// with multiple AI providers (OpenAI, Anthropic, Google Gemini) without changing your core logic.
// It supports text, image, and audio modalities, tool integration with JSON schema-based parameters,
// callback-based tool execution, automatic fallback strategies, and standardized error handling.
//
// # Core Concepts
//
// Generator: The core interface that all providers implement. It takes a Dialog and generates a Response.
//
//	type Generator interface {
//		Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error)
//	}
//
// Dialog: A conversation with a language model, represented as a slice of Message objects.
//
//	type Dialog []Message
//
// Message: A single exchange in the conversation, with a Role (User, Assistant, or ToolResult)
// and a collection of Blocks.
//
//	type Message struct {
//		Role   Role
//		Blocks []Block
//		ToolResultError bool
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
// Tool: A function that can be called by the language model during generation.
//
//	type Tool struct {
//		Name        string
//		Description string
//		InputSchema InputSchema
//	}
//
// # Examples
//
// Basic usage with OpenAI:
//
//	client := openai.NewClient()
//	generator := gai.NewOpenAiGenerator(
//		client.Chat.Completions,
//		openai.ChatModelGPT4,
//		"You are a helpful assistant.",
//	)
//
//	dialog := gai.Dialog{
//		{
//			Role: gai.User,
//			Blocks: []gai.Block{
//				{
//					BlockType:    gai.Content,
//					ModalityType: gai.Text,
//					Content:      gai.Str("What is the capital of France?"),
//				},
//			},
//		},
//	}
//
//	response, err := generator.Generate(context.Background(), dialog, nil)
//
// For more examples, see the README and example files.
package gai
