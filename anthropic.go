package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"iter"
	"strconv"
)

// AnthropicGenerator implements the gai.Generator interface using OpenAI's API
type AnthropicGenerator struct {
	client             AnthropicSvc
	model              string
	tools              map[string]a.ToolParam
	systemInstructions string
}

// convertToolToAnthropic converts our tool definition to Anthropic's format
func convertToolToAnthropic(tool Tool) a.ToolParam {
	var toolProperties any
	if tool.InputSchema != nil {
		toolProperties = tool.InputSchema.Properties
	}
	return a.ToolParam{
		Name:        tool.Name,
		Description: a.String(tool.Description),
		InputSchema: a.ToolInputSchemaParam{
			Properties: toolProperties,
		},
	}
}

const generatorPrefix = "anthropic_"
const thinkingSignatureKey = "thinking_signature"

// toAnthropicMessage converts a gai.Message to an Anthropic message.
// It returns an error if the message contains unsupported modalities or block types.
func toAnthropicMessage(msg Message) (a.MessageParam, error) {
	if len(msg.Blocks) == 0 {
		return a.MessageParam{}, fmt.Errorf("message must have at least one block")
	}

	// Check for video modality in any block
	for _, block := range msg.Blocks {
		if block.ModalityType == Video || block.ModalityType == Audio {
			return a.MessageParam{}, UnsupportedInputModalityErr(block.ModalityType.String())
		}
	}

	switch msg.Role {
	case User:
		// User messages should only have Content blocks
		for _, block := range msg.Blocks {
			if block.BlockType != Content {
				return a.MessageParam{}, fmt.Errorf("unsupported block type for user: %v", block.BlockType)
			}
		}

		// Handle multimodal content
		var parts []a.ContentBlockParamUnion
		for _, block := range msg.Blocks {
			switch block.ModalityType {
			case Text:
				parts = append(parts, a.NewTextBlock(block.Content.String()))
			case Image:
				// Convert image media to an image part
				if block.MimeType == "" {
					return a.MessageParam{}, fmt.Errorf("image media missing mimetype")
				}

				// Special handling for PDFs using NewDocumentBlock
				if block.MimeType == "application/pdf" {
					parts = append(parts, a.NewDocumentBlock(
						a.Base64PDFSourceParam{
							Data: block.Content.String(),
						},
					))
				} else {
					parts = append(parts, a.NewImageBlockBase64(block.MimeType, block.Content.String()))
				}
			default:
				return a.MessageParam{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
		}

		return a.MessageParam{
			Content: parts,
			Role:    a.MessageParamRoleUser,
		}, nil

	case Assistant:
		// Handle multiple blocks
		var contentParts []a.ContentBlockParamUnion

		result := a.MessageParam{
			Role: a.MessageParamRoleAssistant,
		}

		for _, block := range msg.Blocks {
			switch block.BlockType {
			case Content:
				if block.ModalityType != Text {
					return a.MessageParam{}, UnsupportedInputModalityErr(block.ModalityType.String())
				}
				contentParts = append(contentParts, a.NewTextBlock(block.Content.String()))
			case ToolCall:
				// Parse the tool call content as ToolCallInput
				var toolUse ToolCallInput
				if err := json.Unmarshal([]byte(block.Content.String()), &toolUse); err != nil {
					return a.MessageParam{}, fmt.Errorf("invalid tool call content: %w", err)
				}

				// Convert parameters to JSON for Anthropic
				inputJSON, err := json.Marshal(toolUse.Parameters)
				if err != nil {
					return a.MessageParam{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
				}

				contentParts = append(contentParts, a.NewToolUseBlock(
					block.ID,
					json.RawMessage(inputJSON),
					toolUse.Name,
				))
			case Thinking:
				if block.ModalityType != Text {
					return a.MessageParam{}, UnsupportedInputModalityErr(block.ModalityType.String())
				}

				var thinkingSig string
				var ok bool
				if block.ExtraFields[generatorPrefix+thinkingSignatureKey] != nil {
					thinkingSig, ok = block.ExtraFields[generatorPrefix+thinkingSignatureKey].(string)
					if !ok {
						return a.MessageParam{}, fmt.Errorf("invalid thinking signature")
					}
				}

				contentParts = append(contentParts, a.NewThinkingBlock(
					thinkingSig,
					block.Content.String(),
				))
			default:
				return a.MessageParam{}, fmt.Errorf("unsupported block type for assistant: %v", block.BlockType)
			}
		}

		result.Content = contentParts
		return result, nil

	case ToolResult:
		// Anthropic handles tool results differently from OpenAI:
		// - Anthropic: Multiple tool results for parallel tool calls must be in a single message
		//   as separate tool_result blocks, each with its own tool_use_id
		// - OpenAI: Each tool result must be in a separate message with a single tool_call_id.
		//   All blocks in the message must have the same tool ID and be text modality.
		//
		// This implementation supports both approaches by:
		// 1. Allowing a single tool result message to contain blocks with different tool use IDs (Anthropic style)
		// 2. Grouping blocks by their tool use ID and creating separate tool result blocks for each group

		// Validate that there's at least one block
		if len(msg.Blocks) == 0 {
			return a.MessageParam{}, fmt.Errorf("tool result message must have at least one block")
		}

		// Group blocks by tool use ID
		blocksByToolID := make(map[string][]Block)
		for _, block := range msg.Blocks {
			if block.ID == "" {
				return a.MessageParam{}, fmt.Errorf("tool result message block must have a tool use ID")
			}
			blocksByToolID[block.ID] = append(blocksByToolID[block.ID], block)
		}

		// Create tool result content for each group of blocks
		var contentParts []a.ContentBlockParamUnion

		for toolUseID, blocks := range blocksByToolID {
			// A tool result for a specific tool use ID
			resultContent := a.ToolResultBlockParam{
				ToolUseID: toolUseID,
				IsError:   a.Bool(msg.ToolResultError),
			}

			// Process all blocks for this tool use ID
			var blockContent []a.ToolResultBlockParamContentUnion
			for _, block := range blocks {
				var b a.ToolResultBlockParamContentUnion
				switch block.ModalityType {
				case Text:
					b.OfText = &a.TextBlockParam{
						Text: block.Content.String(),
					}
				case Image:
					b.OfImage = &a.ImageBlockParam{
						Source: a.ImageBlockParamSourceUnion{
							OfBase64: &a.Base64ImageSourceParam{
								Data:      block.Content.String(),
								MediaType: a.Base64ImageSourceMediaType(block.MimeType),
							},
						},
					}
				default:
					return a.MessageParam{}, UnsupportedInputModalityErr(block.ModalityType.String())
				}
				blockContent = append(blockContent, b)
			}

			resultContent.Content = blockContent
			contentParts = append(contentParts, a.ContentBlockParamUnion{OfToolResult: &resultContent})
		}

		return a.MessageParam{
			// For tool result blocks, we use the user role
			Role:    a.MessageParamRoleUser,
			Content: contentParts,
		}, nil
	default:
		return a.MessageParam{}, fmt.Errorf("unsupported role: %v", msg.Role)
	}
}

// Register implements gai.ToolRegister
func (g *AnthropicGenerator) Register(tool Tool) error {
	// Validate tool name
	if tool.Name == "" {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool name cannot be empty"),
		}
	}

	// Check for special tool choice values
	if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool name cannot be %s", tool.Name),
		}
	}

	// Initialize tools map if needed
	if g.tools == nil {
		g.tools = make(map[string]a.ToolParam)
	}

	// Check for conflicts with existing tools
	if _, exists := g.tools[tool.Name]; exists {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool already registered"),
		}
	}

	// Convert our tool definition to OpenAI's format and store it
	g.tools[tool.Name] = convertToolToAnthropic(tool)

	return nil
}

// Generate implements gai.Generator
func (g *AnthropicGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("anthropic: client not initialized")
	}

	// Check for empty dialog
	if len(dialog) == 0 {
		return Response{}, EmptyDialogErr
	}

	// Convert each message to Anthropic format
	var messages []a.MessageParam
	for _, msg := range dialog {
		anthropicMsg, err := toAnthropicMessage(msg)
		if err != nil {
			return Response{}, fmt.Errorf("failed to convert message: %w", err)
		}
		messages = append(messages, anthropicMsg)
	}

	// Create Anthropic message params
	params := a.MessageNewParams{
		Model:    a.Model(g.model),
		Messages: messages,
	}

	// Add system instructions if present
	if g.systemInstructions != "" {
		params.System = []a.TextBlockParam{
			{
				Text: g.systemInstructions,
			},
		}
	}

	// Map our options to Anthropic params if options are provided
	if options != nil {
		// Set temperature if non-zero
		if options.Temperature != 0 {
			params.Temperature = a.Float(options.Temperature)
		}

		// Set top_p if non-zero
		if options.TopP != 0 {
			params.TopP = a.Float(options.TopP)
		}

		// Set frequency penalty if non-zero
		if options.FrequencyPenalty != 0 {
			return Response{}, fmt.Errorf("frequency penalty is invalid")
		}

		// Set presence penalty if non-zero
		if options.PresencePenalty != 0 {
			return Response{}, fmt.Errorf("presence penalty is invalid")
		}

		// Set max tokens if specified
		if options.MaxGenerationTokens > 0 {
			params.MaxTokens = int64(options.MaxGenerationTokens)
		}

		// Set number of completions if specified
		if options.N > 0 {
			return Response{}, fmt.Errorf("n is invalid")
		}

		// Set stop sequences if specified
		if len(options.StopSequences) > 0 {
			params.StopSequences = options.StopSequences
		}

		// Set tool choice if specified
		if options.ToolChoice != "" {
			switch options.ToolChoice {
			case ToolChoiceAuto:
				params.ToolChoice = a.ToolChoiceUnionParam{
					OfAuto: &a.ToolChoiceAutoParam{},
				}
			case ToolChoiceToolsRequired:
				params.ToolChoice = a.ToolChoiceUnionParam{
					OfAny: &a.ToolChoiceAnyParam{},
				}
			default:
				// Specific tool name
				params.ToolChoice = a.ToolChoiceUnionParam{
					OfTool: &a.ToolChoiceToolParam{
						Name: options.ToolChoice,
					},
				}
			}
		}

		// Handle multimodality options
		if len(options.OutputModalities) > 0 {
			for _, m := range options.OutputModalities {
				switch m {
				case Audio:
					return Response{}, UnsupportedOutputModalityErr("image output not supported by model")
				case Image:
					return Response{}, UnsupportedOutputModalityErr("image output not supported by model")
				case Video:
					return Response{}, UnsupportedOutputModalityErr("video output not supported by model")
				}
			}
		}

		if options.ThinkingBudget != "" {
			budget, err := strconv.ParseUint(options.ThinkingBudget, 10, 64)
			if err != nil {
				return Response{}, &InvalidParameterErr{
					Parameter: "thinking budget",
					Reason:    fmt.Sprintf("value is not a unsigned int: %s", err),
				}
			}

			params.Thinking = a.ThinkingConfigParamUnion{
				OfEnabled: &a.ThinkingConfigEnabledParam{
					BudgetTokens: int64(budget),
				},
			}
		}
	}

	// Add tools if any are registered
	if len(g.tools) > 0 {
		var tools []a.ToolUnionParam
		for _, tool := range g.tools {
			tools = append(tools, a.ToolUnionParam{
				OfTool: &tool,
			})
		}
		params.Tools = tools
	}

	// Use message streaming, as the anthropic sdk *forces* us to use streaming for large models,
	// even if we _want_ to just the standard http request. As such, we will simply use streaming
	// for all models to keep things simple
	resp, err := g.client.New(ctx, params)
	if err != nil {
		// Convert Anthropic SDK error to our error types based on status code
		var apierr *a.Error
		if errors.As(err, &apierr) {
			// Map HTTP status codes to our error types
			switch apierr.StatusCode {
			case 401:
				return Response{}, AuthenticationErr(apierr.Error())
			case 403:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "permission_error",
					Message:    apierr.Error(),
				}
			case 404:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "not_found_error",
					Message:    apierr.Error(),
				}
			case 413:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "request_too_large",
					Message:    apierr.Error(),
				}
			case 429:
				return Response{}, RateLimitErr(apierr.Error())
			case 500:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "api_error",
					Message:    apierr.Error(),
				}
			case 529:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "overloaded_error",
					Message:    apierr.Error(),
				}
			default:
				// Default to invalid_request_error for 400 and other status codes
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "invalid_request_error",
					Message:    apierr.Error(),
				}
			}
		}
		return Response{}, fmt.Errorf("failed to create new message: %w", err)
	}

	// Convert OpenAI response to our Response type
	result := Response{
		UsageMetrics: make(Metrics),
	}

	// Add usage metrics if available
	inputTokens := resp.Usage.InputTokens + resp.Usage.CacheReadInputTokens + resp.Usage.CacheCreationInputTokens
	result.UsageMetrics[UsageMetricInputTokens] = int(inputTokens)
	result.UsageMetrics[UsageMetricGenerationTokens] = int(resp.Usage.OutputTokens)

	// Convert the message content
	var blocks []Block

	// Handle text content
	for _, contentPart := range resp.Content {
		switch contentPart.Type {
		case "text":
			blocks = append(blocks, Block{
				ID:           contentPart.ID,
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str(contentPart.Text),
			})
		case "tool_use":
			// Create a ToolCallInput with standardized format
			var toolParams map[string]interface{}
			if err := json.Unmarshal(contentPart.Input, &toolParams); err != nil {
				return Response{}, fmt.Errorf("failed to unmarshal tool use input: %w", err)
			}

			toolUse := ToolCallInput{
				Name:       contentPart.Name,
				Parameters: toolParams,
			}

			// Marshal to JSON for consistent representation
			toolUseJSON, err := json.Marshal(toolUse)
			if err != nil {
				return Response{}, fmt.Errorf("failed to marshal tool use: %w", err)
			}

			blocks = append(blocks, Block{
				ID:           contentPart.ID,
				BlockType:    ToolCall,
				ModalityType: Text,
				MimeType:     "application/json",
				Content:      Str(toolUseJSON),
			})
		case "thinking":
			blocks = append(blocks, Block{
				ID:           contentPart.ID,
				BlockType:    Thinking,
				ModalityType: Text,
				Content:      Str(contentPart.Thinking),
				ExtraFields: map[string]interface{}{
					generatorPrefix + thinkingSignatureKey: contentPart.Signature,
				},
			})
		case "redacted_thinking":
			blocks = append(blocks, Block{
				ID:           contentPart.ID,
				BlockType:    "redacted_thinking",
				ModalityType: Text,
				Content:      Str(contentPart.Data),
			})
		default:
			return Response{}, fmt.Errorf("unknown content type: %s", contentPart.Type)
		}
	}

	result.Candidates = append(result.Candidates, Message{
		Role:   Assistant,
		Blocks: blocks,
	})

	// Set finish reason
	switch resp.StopReason {
	case a.StopReasonEndTurn:
		result.FinishReason = EndTurn
	case a.StopReasonMaxTokens:
		result.FinishReason = MaxGenerationLimit
		// Return MaxGenerationLimitErr when the model reaches its token limit,
		// regardless of whether MaxGenerationTokens was explicitly set
		return result, MaxGenerationLimitErr
	case a.StopReasonStopSequence:
		result.FinishReason = StopSequence
	case a.StopReasonToolUse:
		result.FinishReason = ToolUse
	default:
		result.FinishReason = Unknown
	}

	return result, nil
}

func (g *AnthropicGenerator) Stream(ctx context.Context, dialog Dialog, options *GenOpts) iter.Seq2[StreamChunk, error] {
	return func(yield func(StreamChunk, error) bool) {
		if g.client == nil {
			yield(StreamChunk{}, fmt.Errorf("openai: client not initialized"))
			return
		}

		// Check for empty dialog
		if len(dialog) == 0 {
			yield(StreamChunk{}, EmptyDialogErr)
			return
		}

		// Convert each message to Anthropic format
		var messages []a.MessageParam
		for _, msg := range dialog {
			anthropicMsg, err := toAnthropicMessage(msg)
			if err != nil {
				yield(StreamChunk{}, fmt.Errorf("failed to convert message: %w", err))
				return
			}
			messages = append(messages, anthropicMsg)
		}

		// Create Anthropic message params
		params := a.MessageNewParams{
			Model:    a.Model(g.model),
			Messages: messages,
		}

		// Add system instructions if present
		if g.systemInstructions != "" {
			params.System = []a.TextBlockParam{
				{
					Text: g.systemInstructions,
				},
			}
		}

		// Map our options to OpenAI params if options are provided
		if options != nil {
			// Set temperature if non-zero
			if options.Temperature != 0 {
				params.Temperature = a.Float(options.Temperature)
			}

			// Set top_p if non-zero
			if options.TopP != 0 {
				params.TopP = a.Float(options.TopP)
			}

			// Set frequency penalty if non-zero
			if options.FrequencyPenalty != 0 {
				yield(StreamChunk{}, fmt.Errorf("frequency penalty is invalid"))
				return
			}

			// Set presence penalty if non-zero
			if options.PresencePenalty != 0 {
				yield(StreamChunk{}, fmt.Errorf("presence penalty is invalid"))
				return
			}

			// Set max tokens if specified
			if options.MaxGenerationTokens > 0 {
				params.MaxTokens = int64(options.MaxGenerationTokens)
			}

			// Set number of completions if specified
			if options.N > 0 {
				yield(StreamChunk{}, fmt.Errorf("n is invalid"))
				return
			}

			// Set stop sequences if specified
			if len(options.StopSequences) > 0 {
				params.StopSequences = options.StopSequences
			}

			// Set tool choice if specified
			if options.ToolChoice != "" {
				switch options.ToolChoice {
				case ToolChoiceAuto:
					params.ToolChoice = a.ToolChoiceUnionParam{
						OfAuto: &a.ToolChoiceAutoParam{},
					}
				case ToolChoiceToolsRequired:
					params.ToolChoice = a.ToolChoiceUnionParam{
						OfAny: &a.ToolChoiceAnyParam{},
					}
				default:
					// Specific tool name
					params.ToolChoice = a.ToolChoiceUnionParam{
						OfTool: &a.ToolChoiceToolParam{
							Name: options.ToolChoice,
						},
					}
				}
			}

			// Handle multimodality options
			if len(options.OutputModalities) > 0 {
				for _, m := range options.OutputModalities {
					switch m {
					case Audio, Image, Video:
						yield(StreamChunk{}, UnsupportedOutputModalityErr("image output not supported by model"))
						return
					}
				}
			}

			if options.ThinkingBudget != "" {
				budget, err := strconv.ParseUint(options.ThinkingBudget, 10, 64)
				if err != nil {
					yield(StreamChunk{}, InvalidParameterErr{
						Parameter: "thinking budget",
						Reason:    fmt.Sprintf("value is not a unsigned int: %s", err),
					})
				}

				params.Thinking = a.ThinkingConfigParamUnion{
					OfEnabled: &a.ThinkingConfigEnabledParam{
						BudgetTokens: int64(budget),
					},
				}
			}
		}

		// Add tools if any are registered
		if len(g.tools) > 0 {
			var tools []a.ToolUnionParam
			for _, tool := range g.tools {
				tools = append(tools, a.ToolUnionParam{
					OfTool: &tool,
				})
			}
			params.Tools = tools
		}

		// Start the stream
		stream := g.client.NewStreaming(ctx, params)
		defer stream.Close()
		for stream.Next() {
			chunk := stream.Current()

			switch event := chunk.AsAny().(type) {
			case a.ContentBlockStartEvent:
				// if a content block start type event and tool call, then extract the tool call details
				// all types of content block deltas are automatically handled in the `case a.ContentBlockDeltaEvent`
				if event.ContentBlock.Type != "tool_use" {
					continue
				}
				if !yield(StreamChunk{
					Block: Block{
						ID:           event.ContentBlock.ID,
						BlockType:    ToolCall,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str(event.ContentBlock.Name),
					},
					CandidatesIndex: 0,
				}, nil) {
					return
				}
			case a.ContentBlockDeltaEvent:
				switch delta := event.Delta.AsAny().(type) {
				case a.TextDelta:
					if delta.Text == "" {
						continue
					}
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    Content,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(delta.Text),
						},
						CandidatesIndex: 0,
					}, nil) {
						return
					}
				case a.InputJSONDelta:
					// ignore empty partial json chunks
					if delta.PartialJSON == "" {
						continue
					}
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(delta.PartialJSON),
						},
						CandidatesIndex: 0,
					}, nil) {
						return
					}
				case a.ThinkingDelta:
					if delta.Thinking == "" {
						continue
					}
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    Thinking,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(delta.Thinking),
						},
						CandidatesIndex: 0,
					}, nil) {
						return
					}
				case a.SignatureDelta:
					if delta.Signature == "" {
						continue
					}
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    Thinking,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(""),
							ExtraFields: map[string]interface{}{
								generatorPrefix + thinkingSignatureKey: delta.Signature,
							},
						},
						CandidatesIndex: 0,
					}, nil) {
						return
					}
				case a.CitationsDelta:
					panic("citation block type not supported")
				default:
					panic("unknown block type")
				}
			}

		}

		if stream.Err() != nil {
			yield(StreamChunk{}, stream.Err())
		}
	}

}

// AnthropicSvc defines the interface for interacting with the Anthropic API.
// It requires the methods needed for both generation and token counting.
//
// This interface is implemented by the Anthropic SDK's MessageService,
// allowing for direct use or wrapping with additional functionality
// (such as caching via AnthropicServiceWrapper).
type AnthropicSvc interface {
	// New generates a new message using the Anthropic API
	New(ctx context.Context, body a.MessageNewParams, opts ...option.RequestOption) (res *a.Message, err error)

	// NewStreaming generates a new streaming message using the Anthropic API
	NewStreaming(ctx context.Context, body a.MessageNewParams, opts ...option.RequestOption) (stream *ssestream.Stream[a.MessageStreamEventUnion])

	// CountTokens counts tokens for a message without generating a response
	CountTokens(ctx context.Context, body a.MessageCountTokensParams, opts ...option.RequestOption) (res *a.MessageTokensCount, err error)
}

// NewAnthropicGenerator creates a new Anthropic generator with the specified model.
// It returns a ToolCapableGenerator that preprocesses dialog for parallel tool use compatibility.
// This generator fully supports the anyOf JSON Schema feature.
//
// Parameters:
//   - client: An Anthropic service client
//   - model: The Anthropic model to use (e.g., "claude-3-5-sonnet-20241022")
//   - systemInstructions: Optional system instructions that set the model's behavior
//
// Supported modalities:
//   - Text: Both input and output
//   - Image: Input only (base64 encoded, including PDFs with MIME type "application/pdf")
//
// PDF documents are handled specially using Anthropic's NewDocumentBlock function,
// which provides optimized PDF processing. Use the PDFBlock helper function to
// create PDF content blocks.
//
// The returned generator also implements the TokenCounter interface for token counting.
func NewAnthropicGenerator(client AnthropicSvc, model, systemInstructions string) interface {
	ToolCapableGenerator
	StreamingGenerator
	TokenCounter
} {
	inner := &AnthropicGenerator{
		client:             client,
		systemInstructions: systemInstructions,
		model:              model,
		tools:              make(map[string]a.ToolParam),
	}
	return &PreprocessingGenerator{Inner: inner}
}

var _ Generator = (*AnthropicGenerator)(nil)
var _ ToolRegister = (*AnthropicGenerator)(nil)
var _ TokenCounter = (*AnthropicGenerator)(nil)

// Count implements the TokenCounter interface for AnthropicGenerator.
// It converts the dialog to Anthropic's format and uses Anthropic's dedicated CountTokens API.
//
// Unlike the OpenAI implementation which uses a local tokenizer, this method makes an API call
// to the Anthropic service. This provides the most accurate token count as it uses exactly
// the same tokenization logic as the actual generation.
//
// The method accounts for:
//   - System instructions (if set during generator initialization)
//   - All messages in the dialog with their respective blocks
//   - Multi-modal content like images
//   - Tool definitions registered with the generator
//
// The context parameter allows for cancellation of the API call.
//
// Returns:
//   - The total token count as uint, representing input tokens only
//   - An error if the API call fails or if dialog conversion fails
//
// Note: Anthropic's CountTokens API returns only input token count. For an estimate of
// output tokens, you would need to perform a separate calculation.
func (g *AnthropicGenerator) Count(ctx context.Context, dialog Dialog) (uint, error) {
	if g.client == nil {
		return 0, fmt.Errorf("anthropic: client not initialized")
	}

	if len(dialog) == 0 {
		return 0, EmptyDialogErr
	}

	var messages []a.MessageParam
	for _, msg := range dialog {
		anthropicMsg, err := toAnthropicMessage(msg)
		if err != nil {
			return 0, fmt.Errorf("failed to convert message for token counting: %w", err)
		}
		messages = append(messages, anthropicMsg)
	}

	params := a.MessageCountTokensParams{
		Messages: messages,
		Model:    a.Model(g.model),
	}
	// Add system instructions if present
	if g.systemInstructions != "" {
		params.System = a.MessageCountTokensParamsSystemUnion{
			OfTextBlockArray: []a.TextBlockParam{
				{
					Text: g.systemInstructions,
				},
			},
		}
	}

	for _, tool := range g.tools {
		params.Tools = append(params.Tools, a.MessageCountTokensToolUnionParam{
			OfTool: &tool,
		})
	}

	resp, err := g.client.CountTokens(ctx, params)
	if err != nil {
		// Convert Anthropic SDK error to our error types based on status code
		var apierr *a.Error
		if errors.As(err, &apierr) {
			// Map HTTP status codes to our error types
			switch apierr.StatusCode {
			case 401:
				return 0, AuthenticationErr(apierr.Error())
			case 403:
				return 0, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "permission_error",
					Message:    apierr.Error(),
				}
			case 404:
				return 0, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "not_found_error",
					Message:    apierr.Error(),
				}
			case 413:
				return 0, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "request_too_large",
					Message:    apierr.Error(),
				}
			case 429:
				return 0, RateLimitErr(apierr.Error())
			case 500:
				return 0, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "api_error",
					Message:    apierr.Error(),
				}
			case 529:
				return 0, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "overloaded_error",
					Message:    apierr.Error(),
				}
			default:
				// Default to invalid_request_error for 400 and other status codes
				return 0, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "invalid_request_error",
					Message:    apierr.Error(),
				}
			}
		}
		return 0, fmt.Errorf("failed to count tokens: %w", err)
	}

	return uint(resp.InputTokens), nil
}
