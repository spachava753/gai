package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"os"
	"strings"

	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	oaissestream "github.com/openai/openai-go/v3/packages/ssestream"
)

// ZaiGenerator implements the Generator and StreamingGenerator interfaces for Z.AI API.
// Z.AI provides OpenAI-compatible endpoints with extended thinking/reasoning capabilities.
//
// Key features:
//   - OpenAI-compatible chat completions API
//   - Interleaved thinking: the model can reason between tool calls
//   - Preserved thinking: reasoning context can be retained across turns
//   - Streaming with Server-Sent Events (SSE)
//
// Supported models include glm-4.7, glm-4.6, glm-4.5, and variants.
type ZaiGenerator struct {
	client             ZaiCompletionService
	model              string
	systemInstructions string
	tools              map[string]oai.ChatCompletionToolUnionParam

	// thinkingEnabled controls whether thinking/reasoning is enabled
	thinkingEnabled bool
	// clearThinking controls whether to clear reasoning_content from previous turns
	clearThinking bool
}

// ZaiCompletionService defines the interface for Z.AI chat completions
type ZaiCompletionService interface {
	New(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) (*oai.ChatCompletion, error)
	NewStreaming(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) *oaissestream.Stream[oai.ChatCompletionChunk]
}

// ZaiGeneratorOption is a functional option for configuring the ZaiGenerator.
type ZaiGeneratorOption func(*ZaiGenerator)

// WithZaiThinking enables or disables thinking mode.
// When enabled, the model will perform chain-of-thought reasoning.
func WithZaiThinking(enabled bool) ZaiGeneratorOption {
	return func(g *ZaiGenerator) {
		g.thinkingEnabled = enabled
	}
}

// WithZaiClearThinking controls whether to clear reasoning_content from previous turns.
// Set to false to enable preserved thinking (retain reasoning across turns).
func WithZaiClearThinking(clear bool) ZaiGeneratorOption {
	return func(g *ZaiGenerator) {
		g.clearThinking = clear
	}
}

// NewZaiGenerator creates a new Z.AI generator using the OpenAI SDK.
// If client is nil, a new client is created with the Z.AI base URL.
// apiKey is read from Z_API_KEY environment variable if empty.
//
// By default, thinking is enabled and clearThinking is true.
func NewZaiGenerator(client ZaiCompletionService, model, systemInstructions, apiKey string, opts ...ZaiGeneratorOption) *ZaiGenerator {
	if apiKey == "" {
		apiKey = os.Getenv("Z_API_KEY")
	}
	if client == nil {
		oaiClient := oai.NewClient(
			option.WithAPIKey(apiKey),
			option.WithBaseURL("https://api.z.ai/api/paas/v4/"),
		)
		client = &oaiClient.Chat.Completions
	}

	g := &ZaiGenerator{
		client:             client,
		model:              model,
		systemInstructions: systemInstructions,
		tools:              make(map[string]oai.ChatCompletionToolUnionParam),
		thinkingEnabled:    true,
		clearThinking:      true,
	}
	for _, opt := range opts {
		opt(g)
	}
	return g
}

// Register implements ToolRegister
func (g *ZaiGenerator) Register(tool Tool) error {
	if tool.Name == "" {
		return &ToolRegistrationErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be empty")}
	}
	if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
		return &ToolRegistrationErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be %s", tool.Name)}
	}
	if _, exists := g.tools[tool.Name]; exists {
		return &ToolRegistrationErr{Tool: tool.Name, Cause: fmt.Errorf("tool already registered")}
	}

	oaiTool, err := convertToolToOpenAI(tool)
	if err != nil {
		return &ToolRegistrationErr{Tool: tool.Name, Cause: err}
	}
	g.tools[tool.Name] = oaiTool
	return nil
}

// zaiAssistantMessage extends the message with reasoning_content for Z.AI
type zaiAssistantMessage struct {
	Role             string                 `json:"role"`
	Content          string                 `json:"content,omitempty"`
	ReasoningContent string                 `json:"reasoning_content,omitempty"`
	ToolCalls        []zaiToolCallForParams `json:"tool_calls,omitempty"`
}

type zaiToolCallForParams struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (g *ZaiGenerator) buildMessages(dialog Dialog) ([]any, error) {
	var messages []any

	// Add system instructions if present
	if g.systemInstructions != "" {
		messages = append(messages, oai.SystemMessage(g.systemInstructions))
	}

	for i, msg := range dialog {
		switch msg.Role {
		case User:
			// Handle user messages - only text supported for now
			if len(msg.Blocks) == 1 && msg.Blocks[0].ModalityType == Text {
				messages = append(messages, oai.UserMessage(msg.Blocks[0].Content.String()))
				continue
			}
			
			var textContent strings.Builder
			for _, blk := range msg.Blocks {
				if blk.BlockType != Content {
					return nil, fmt.Errorf("unsupported block type for user: %v", blk.BlockType)
				}
				if blk.ModalityType != Text {
					return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
				}
				textContent.WriteString(blk.Content.String())
			}
			messages = append(messages, oai.UserMessage(textContent.String()))

		case Assistant:
			var content string
			var reasoningContent string
			var toolCalls []zaiToolCallForParams

			for _, blk := range msg.Blocks {
				switch blk.BlockType {
				case Content:
					if blk.ModalityType != Text {
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
					content = blk.Content.String()
				case Thinking:
					if blk.ModalityType != Text {
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
					reasoningContent = blk.Content.String()
				case ToolCall:
					var toolUse ToolCallInput
					if err := json.Unmarshal([]byte(blk.Content.String()), &toolUse); err != nil {
						return nil, fmt.Errorf("invalid tool call content: %w", err)
					}
					argsJSON, err := json.Marshal(toolUse.Parameters)
					if err != nil {
						return nil, fmt.Errorf("failed to marshal tool parameters: %w", err)
					}
					tc := zaiToolCallForParams{
						ID:   blk.ID,
						Type: "function",
					}
					tc.Function.Name = toolUse.Name
					tc.Function.Arguments = string(argsJSON)
					toolCalls = append(toolCalls, tc)
				default:
					return nil, fmt.Errorf("unsupported block type for assistant: %v", blk.BlockType)
				}
			}

			// Use custom struct to include reasoning_content for preserved thinking
			assistantMsg := zaiAssistantMessage{
				Role:             "assistant",
				Content:          content,
				ReasoningContent: reasoningContent,
			}
			if len(toolCalls) > 0 {
				assistantMsg.ToolCalls = toolCalls
			}
			messages = append(messages, assistantMsg)

		case ToolResult:
			if len(msg.Blocks) == 0 {
				return nil, fmt.Errorf("tool result message must have at least one block")
			}
			for _, blk := range msg.Blocks {
				if blk.ModalityType != Text {
					return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
				}
				if blk.ID == "" {
					return nil, fmt.Errorf("tool result message block must have the tool_call_id as ID")
				}
				messages = append(messages, oai.ToolMessage(blk.Content.String(), blk.ID))
			}

		default:
			return nil, fmt.Errorf("unsupported role at index %d: %v", i, msg.Role)
		}
	}
	return messages, nil
}

func (g *ZaiGenerator) getRequestOptions() []option.RequestOption {
	var opts []option.RequestOption
	
	// Add thinking configuration
	thinkingType := "enabled"
	if !g.thinkingEnabled {
		thinkingType = "disabled"
	}
	opts = append(opts, option.WithJSONSet("thinking", map[string]any{
		"type":           thinkingType,
		"clear_thinking": g.clearThinking,
	}))
	
	// Add Accept-Language header for Z.AI
	opts = append(opts, option.WithHeader("Accept-Language", "en-US,en"))
	
	return opts
}

// extractReasoningContent extracts reasoning_content from the response's extra fields
func extractReasoningContent(msg oai.ChatCompletionMessage) string {
	if msg.JSON.ExtraFields == nil {
		return ""
	}
	if rc, ok := msg.JSON.ExtraFields["reasoning_content"]; ok {
		// The field's raw value should be a JSON string
		var s string
		if err := json.Unmarshal([]byte(rc.Raw()), &s); err == nil {
			return s
		}
	}
	return ""
}

// extractStreamReasoningContent extracts reasoning_content from streaming delta
func extractStreamReasoningContent(delta oai.ChatCompletionChunkChoiceDelta) string {
	if delta.JSON.ExtraFields == nil {
		return ""
	}
	if rc, ok := delta.JSON.ExtraFields["reasoning_content"]; ok {
		var s string
		if err := json.Unmarshal([]byte(rc.Raw()), &s); err == nil {
			return s
		}
	}
	return ""
}

// Generate implements Generator
func (g *ZaiGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("zai: client not initialized")
	}
	if len(dialog) == 0 {
		return Response{}, EmptyDialogErr
	}

	messages, err := g.buildMessages(dialog)
	if err != nil {
		return Response{}, err
	}

	// Build params - we'll use WithJSONSet to override messages since we need custom fields
	params := oai.ChatCompletionNewParams{
		Model: g.model,
	}

	// Map GenOpts
	if options != nil {
		if options.Temperature != nil {
			params.Temperature = oai.Float(*options.Temperature)
		}
		if options.TopP != nil {
			params.TopP = oai.Float(*options.TopP)
		}
		if options.MaxGenerationTokens != nil {
			params.MaxCompletionTokens = oai.Int(int64(*options.MaxGenerationTokens))
		}
		if len(options.StopSequences) > 0 {
			if len(options.StopSequences) == 1 {
				params.Stop = oai.ChatCompletionNewParamsStopUnion{OfString: oai.String(options.StopSequences[0])}
			} else {
				params.Stop = oai.ChatCompletionNewParamsStopUnion{OfStringArray: options.StopSequences}
			}
		}
		if options.ToolChoice != "" {
			switch options.ToolChoice {
			case ToolChoiceAuto:
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String("auto")}
			case ToolChoiceToolsRequired:
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String("required")}
			case "none":
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String("none")}
			default:
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{
					OfFunctionToolChoice: &oai.ChatCompletionNamedToolChoiceParam{
						Function: oai.ChatCompletionNamedToolChoiceFunctionParam{
							Name: options.ToolChoice,
						},
					},
				}
			}
		}
		if len(options.OutputModalities) > 0 {
			for _, m := range options.OutputModalities {
				if m != Text {
					return Response{}, UnsupportedOutputModalityErr(m.String())
				}
			}
		}
	}

	// Add tools if registered
	if len(g.tools) > 0 {
		var tools []oai.ChatCompletionToolUnionParam
		for _, tool := range g.tools {
			tools = append(tools, tool)
		}
		params.Tools = tools
	}

	// Get Z.AI-specific options
	reqOpts := g.getRequestOptions()
	// Override messages with our custom structure that includes reasoning_content
	reqOpts = append(reqOpts, option.WithJSONSet("messages", messages))

	resp, err := g.client.New(ctx, params, reqOpts...)
	if err != nil {
		return Response{}, g.mapError(err)
	}

	result := Response{UsageMetadata: make(Metadata)}
	if resp.Usage.PromptTokens > 0 {
		result.UsageMetadata[UsageMetricInputTokens] = int(resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens > 0 {
		result.UsageMetadata[UsageMetricGenerationTokens] = int(resp.Usage.CompletionTokens)
	}
	// Z.AI uses OpenAI-compatible API, check for cached tokens
	if resp.Usage.PromptTokensDetails.CachedTokens > 0 {
		result.UsageMetadata[UsageMetricCacheReadTokens] = int(resp.Usage.PromptTokensDetails.CachedTokens)
	}

	var hasToolCalls bool
	for _, choice := range resp.Choices {
		var blocks []Block

		// Extract reasoning_content from extra fields
		if rc := extractReasoningContent(choice.Message); rc != "" {
			blocks = append(blocks, Block{
				BlockType:    Thinking,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str(rc),
				ExtraFields: map[string]interface{}{
					ThinkingExtraFieldGeneratorKey: ThinkingGeneratorZai,
				},
			})
		}

		if choice.Message.Content != "" {
			blocks = append(blocks, Block{
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str(choice.Message.Content),
			})
		}

		if len(choice.Message.ToolCalls) > 0 {
			hasToolCalls = true
			for _, tc := range choice.Message.ToolCalls {
					var params map[string]any
					if tc.Function.Arguments != "" {
						_ = json.Unmarshal([]byte(tc.Function.Arguments), &params)
					}
					tj, _ := json.Marshal(ToolCallInput{
						Name:       tc.Function.Name,
						Parameters: params,
					})
					blocks = append(blocks, Block{
						ID:           tc.ID,
						BlockType:    ToolCall,
						ModalityType: Text,
						MimeType:     "application/json",
						Content:      Str(tj),
					})
			}
		}
		result.Candidates = append(result.Candidates, Message{Role: Assistant, Blocks: blocks})
	}

	if len(resp.Choices) > 0 {
		switch resp.Choices[0].FinishReason {
		case "stop":
			result.FinishReason = EndTurn
		case "length":
			result.FinishReason = MaxGenerationLimit
			return result, MaxGenerationLimitErr
		case "tool_calls":
			result.FinishReason = ToolUse
		case "content_filter":
			result.FinishReason = Unknown
			return result, ContentPolicyErr("content filtered")
		default:
			// Check for Z.AI-specific finish reasons in extra fields
			if resp.Choices[0].JSON.ExtraFields != nil {
				if fr, ok := resp.Choices[0].JSON.ExtraFields["finish_reason"]; ok {
					var reason string
					if json.Unmarshal([]byte(fr.Raw()), &reason) == nil && reason == "sensitive" {
						result.FinishReason = Unknown
						return result, ContentPolicyErr("content flagged as sensitive")
					}
				}
			}
			result.FinishReason = Unknown
		}
	}
	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}
	return result, nil
}

// Stream implements StreamingGenerator
func (g *ZaiGenerator) Stream(ctx context.Context, dialog Dialog, options *GenOpts) iter.Seq2[StreamChunk, error] {
	return func(yield func(StreamChunk, error) bool) {
		if g.client == nil {
			yield(StreamChunk{}, fmt.Errorf("zai: client not initialized"))
			return
		}
		if len(dialog) == 0 {
			yield(StreamChunk{}, EmptyDialogErr)
			return
		}

		messages, err := g.buildMessages(dialog)
		if err != nil {
			yield(StreamChunk{}, err)
			return
		}

		params := oai.ChatCompletionNewParams{
			Model: g.model,
			StreamOptions: oai.ChatCompletionStreamOptionsParam{
				IncludeUsage: oai.Bool(true),
			},
		}

		// Map GenOpts
		if options != nil {
			if options.Temperature != nil {
				params.Temperature = oai.Float(*options.Temperature)
			}
			if options.TopP != nil {
				params.TopP = oai.Float(*options.TopP)
			}
			if options.MaxGenerationTokens != nil {
				params.MaxCompletionTokens = oai.Int(int64(*options.MaxGenerationTokens))
			}
			if len(options.StopSequences) > 0 {
				if len(options.StopSequences) == 1 {
					params.Stop = oai.ChatCompletionNewParamsStopUnion{OfString: oai.String(options.StopSequences[0])}
				} else {
					params.Stop = oai.ChatCompletionNewParamsStopUnion{OfStringArray: options.StopSequences}
				}
			}
			if options.ToolChoice != "" {
				switch options.ToolChoice {
				case ToolChoiceAuto:
					params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String("auto")}
				case ToolChoiceToolsRequired:
					params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String("required")}
				case "none":
					params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String("none")}
				default:
					params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{
						OfFunctionToolChoice: &oai.ChatCompletionNamedToolChoiceParam{
							Function: oai.ChatCompletionNamedToolChoiceFunctionParam{
								Name: options.ToolChoice,
							},
						},
					}
				}
			}
			if len(options.OutputModalities) > 0 {
				for _, m := range options.OutputModalities {
					if m != Text {
						yield(StreamChunk{}, UnsupportedOutputModalityErr(m.String()))
						return
					}
				}
			}
		}

		// Add tools if registered
		if len(g.tools) > 0 {
			var tools []oai.ChatCompletionToolUnionParam
			for _, tool := range g.tools {
				tools = append(tools, tool)
			}
			params.Tools = tools
		}

		// Get Z.AI-specific options
		reqOpts := g.getRequestOptions()
		reqOpts = append(reqOpts, option.WithJSONSet("messages", messages))

		stream := g.client.NewStreaming(ctx, params, reqOpts...)
		defer stream.Close()

		var finalUsage *oai.CompletionUsage

		for stream.Next() {
			chunk := stream.Current()

			// Capture usage if present
			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				finalUsage = &chunk.Usage
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			switch choice.FinishReason {
			case "length":
				yield(StreamChunk{}, MaxGenerationLimitErr)
				return
			case "content_filter", "sensitive":
				yield(StreamChunk{}, ContentPolicyErr("content filtered"))
				return
			}

			if choice.Delta.Refusal != "" {
				yield(StreamChunk{}, ContentPolicyErr("content refused"))
				return
			}

			// Yield reasoning content
			if rc := extractStreamReasoningContent(choice.Delta); rc != "" {
				if !yield(StreamChunk{
					Block: Block{
						BlockType:    Thinking,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str(rc),
						ExtraFields: map[string]interface{}{
							ThinkingExtraFieldGeneratorKey: ThinkingGeneratorZai,
						},
					},
					CandidatesIndex: int(choice.Index),
				}, nil) {
					return
				}
			}

			// Yield content
			if choice.Delta.Content != "" {
				if !yield(StreamChunk{
					Block: Block{
						BlockType:    Content,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str(choice.Delta.Content),
					},
					CandidatesIndex: int(choice.Index),
				}, nil) {
					return
				}
			}

			// Yield tool calls
			for _, tc := range choice.Delta.ToolCalls {
				if tc.Function.Name != "" {
					if !yield(StreamChunk{
						Block: Block{
							ID:           tc.ID,
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(tc.Function.Name),
						},
						CandidatesIndex: int(choice.Index),
					}, nil) {
						return
					}
				}
				if tc.Function.Arguments != "" {
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(tc.Function.Arguments),
						},
						CandidatesIndex: int(choice.Index),
					}, nil) {
						return
					}
				}
			}
		}

		if stream.Err() != nil {
			yield(StreamChunk{}, g.mapError(stream.Err()))
			return
		}

		// Emit metadata block as final block
		if finalUsage != nil {
			metadata := make(Metadata)
			if finalUsage.PromptTokens > 0 {
				metadata[UsageMetricInputTokens] = int(finalUsage.PromptTokens)
			}
			if finalUsage.CompletionTokens > 0 {
				metadata[UsageMetricGenerationTokens] = int(finalUsage.CompletionTokens)
			}
			// Z.AI uses OpenAI-compatible API, check for cached tokens
			if finalUsage.PromptTokensDetails.CachedTokens > 0 {
				metadata[UsageMetricCacheReadTokens] = int(finalUsage.PromptTokensDetails.CachedTokens)
			}
			if len(metadata) > 0 {
				yield(StreamChunk{
					Block:           MetadataBlock(metadata),
					CandidatesIndex: 0,
				}, nil)
			}
		}
	}
}

func (g *ZaiGenerator) mapError(err error) error {
	if err == nil {
		return nil
	}
	errStr := err.Error()
	
	// Map common error patterns
	if strings.Contains(errStr, "401") || strings.Contains(errStr, "authentication") {
		return AuthenticationErr(errStr)
	}
	if strings.Contains(errStr, "429") || strings.Contains(errStr, "rate limit") {
		return RateLimitErr(errStr)
	}
	if strings.Contains(errStr, "1113") {
		return RateLimitErr(errStr)
	}
	
	return err
}

var _ Generator = (*ZaiGenerator)(nil)
var _ ToolRegister = (*ZaiGenerator)(nil)
var _ StreamingGenerator = (*ZaiGenerator)(nil)
