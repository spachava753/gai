package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"strconv"
)

// AnthropicGenerator implements the gai.Generator interface using OpenAI's API
type AnthropicGenerator struct {
	client             AnthropicCompletionService
	model              string
	tools              map[string]a.ToolParam
	systemInstructions string
}

// convertToolToAnthropic converts our tool definition to Anthropic's format
func convertToolToAnthropic(tool Tool) a.ToolParam {
	if tool.InputSchema.Type != Object {
		panic("invalid tool type")
	}

	// Convert our tool schema to JSON schema format
	parameters := make(map[string]interface{})

	// Convert each property to Anthropic's format
	for name, prop := range tool.InputSchema.Properties {
		parameters[name] = convertPropertyToAnthropicMap(prop)
	}

	// Create the input schema with the properties
	inputSchema := a.ToolInputSchemaParam{
		Type:        a.F(a.ToolInputSchemaTypeObject),
		Properties:  a.F[any](parameters),
		ExtraFields: make(map[string]interface{}),
	}

	// Add required properties if any
	if len(tool.InputSchema.Required) > 0 {
		inputSchema.ExtraFields["required"] = tool.InputSchema.Required
	}

	return a.ToolParam{
		Name:        a.F(tool.Name),
		Description: a.F(tool.Description),
		InputSchema: a.F[interface{}](inputSchema),
	}
}

// convertPropertyToAnthropicMap converts a Property to a map[string]interface{} suitable for Anthropic's format
func convertPropertyToAnthropicMap(prop Property) map[string]interface{} {
	result := map[string]interface{}{
		"type":        prop.Type.String(),
		"description": prop.Description,
	}

	// Add enum if present
	if len(prop.Enum) > 0 {
		result["enum"] = prop.Enum
	}

	// Add properties for object types
	if prop.Type == Object && prop.Properties != nil {
		properties := make(map[string]interface{})
		for name, p := range prop.Properties {
			properties[name] = convertPropertyToAnthropicMap(p)
		}
		result["properties"] = properties

		// Add required fields for object types
		if len(prop.Required) > 0 {
			result["required"] = prop.Required
		}
	}

	// Add items for array types
	if prop.Type == Array && prop.Items != nil {
		result["items"] = convertPropertyToAnthropicMap(*prop.Items)
	}

	return result
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
				parts = append(parts, a.TextBlockParam{
					Type: a.F(a.TextBlockParamTypeText),
					Text: a.F(block.Content.String()),
				})
			case Image:
				// Convert image media to an image part
				if block.MimeType == "" {
					return a.MessageParam{}, fmt.Errorf("image media missing mimetype")
				}

				var imageBlock a.ImageBlockParamSourceUnion = &a.Base64ImageSourceParam{
					Type:      a.F(a.Base64ImageSourceTypeBase64),
					MediaType: a.F(a.Base64ImageSourceMediaType(block.MimeType)),
					Data:      a.F(block.Content.String()),
				}

				parts = append(parts, a.ImageBlockParam{
					Type:   a.F(a.ImageBlockParamTypeImage),
					Source: a.F(imageBlock),
				})
			default:
				return a.MessageParam{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
		}

		return a.MessageParam{
			Content: a.F(parts),
			Role:    a.F(a.MessageParamRoleUser),
		}, nil

	case Assistant:
		// Handle multiple blocks
		var contentParts []a.ContentBlockParamUnion

		result := a.MessageParam{
			Role: a.F(a.MessageParamRoleAssistant),
		}

		for _, block := range msg.Blocks {
			switch block.BlockType {
			case Content:
				if block.ModalityType != Text {
					return a.MessageParam{}, UnsupportedInputModalityErr(block.ModalityType.String())
				}
				contentParts = append(contentParts, &a.TextBlockParam{
					Text: a.F(block.Content.String()),
					Type: a.F(a.TextBlockParamTypeText),
				})
			case ToolCall:
				// Parse the tool call content as ToolUseInput
				var toolUse ToolUseInput
				if err := json.Unmarshal([]byte(block.Content.String()), &toolUse); err != nil {
					return a.MessageParam{}, fmt.Errorf("invalid tool call content: %w", err)
				}

				// Convert parameters to JSON for Anthropic
				inputJSON, err := json.Marshal(toolUse.Parameters)
				if err != nil {
					return a.MessageParam{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
				}

				contentParts = append(contentParts, &a.ToolUseBlockParam{
					ID:    a.F(block.ID),
					Name:  a.F(toolUse.Name),
					Input: a.F[interface{}](json.RawMessage(inputJSON)),
					Type:  a.F(a.ToolUseBlockParamTypeToolUse),
				})
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

				contentParts = append(contentParts, &a.ThinkingBlockParam{
					Signature: a.F(thinkingSig),
					Thinking:  a.F(block.Content.String()),
					Type:      a.F(a.ThinkingBlockParamTypeThinking),
				})
			default:
				return a.MessageParam{}, fmt.Errorf("unsupported block type for assistant: %v", block.BlockType)
			}
		}

		result.Content = a.F(contentParts)
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
				Type:      a.F(a.ToolResultBlockParamTypeToolResult),
				ToolUseID: a.F(toolUseID),
				IsError:   a.F(msg.ToolResultError),
			}

			// Process all blocks for this tool use ID
			var blockContent []a.ToolResultBlockParamContentUnion
			for _, block := range blocks {
				var b a.ToolResultBlockParamContent
				switch block.ModalityType {
				case Text:
					b.Type = a.F(a.ToolResultBlockParamContentTypeText)
					b.Text = a.F(block.Content.String())
				case Image:
					b.Type = a.F(a.ToolResultBlockParamContentTypeImage)
					b.Source = a.F[interface{}](block.Content.String())
				default:
					return a.MessageParam{}, UnsupportedInputModalityErr(block.ModalityType.String())
				}
				blockContent = append(blockContent, &b)
			}

			resultContent.Content = a.F(blockContent)
			contentParts = append(contentParts, resultContent)
		}

		return a.MessageParam{
			// For tool result blocks, we use the user role
			Role:    a.F(a.MessageParamRoleUser),
			Content: a.F(contentParts),
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

	// First, preprocess the dialog to combine consecutive tool result messages for parallel tool use.
	// This is a key difference from OpenAI: Anthropic requires all tool results for parallel tool calls
	// to be in a single message with multiple tool_result blocks, while our internal Dialog representation
	// (and OpenAI) use separate messages where all blocks must have the same tool ID.
	processedDialog := preprocessToolResults(dialog)

	// Convert each message to Anthropic format
	var messages []a.MessageParam
	for _, msg := range processedDialog {
		anthropicMsg, err := toAnthropicMessage(msg)
		if err != nil {
			return Response{}, fmt.Errorf("failed to convert message: %w", err)
		}
		messages = append(messages, anthropicMsg)
	}

	// Create Anthropic message params
	params := a.MessageNewParams{
		Model:    a.F(g.model),
		Messages: a.F(messages),
	}

	// Add system instructions if present
	if g.systemInstructions != "" {
		params.System = a.F([]a.TextBlockParam{
			{
				Text: a.String(g.systemInstructions),
				Type: a.F(a.TextBlockParamTypeText),
			},
		})
	}

	// Map our options to Anthropic params if options are provided
	if options != nil {
		// Set temperature if non-zero
		if options.Temperature != 0 {
			params.Temperature = a.F(options.Temperature)
		}

		// Set top_p if non-zero
		if options.TopP != 0 {
			params.TopP = a.F(options.TopP)
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
			params.MaxTokens = a.F(int64(options.MaxGenerationTokens))
		}

		// Set number of completions if specified
		if options.N > 0 {
			return Response{}, fmt.Errorf("n is invalid")
		}

		// Set stop sequences if specified
		if len(options.StopSequences) > 0 {
			params.StopSequences = a.F(options.StopSequences)
		}

		// Set tool choice if specified
		if options.ToolChoice != "" {
			switch options.ToolChoice {
			case ToolChoiceAuto:
				params.ToolChoice = a.F[a.ToolChoiceUnionParam](a.ToolChoiceAutoParam{
					Type: a.F(a.ToolChoiceAutoTypeAuto),
				})
			case ToolChoiceToolsRequired:
				params.ToolChoice = a.F[a.ToolChoiceUnionParam](a.ToolChoiceAnyParam{
					Type: a.F(a.ToolChoiceAnyTypeAny),
				})
			default:
				// Specific tool name
				params.ToolChoice = a.F[a.ToolChoiceUnionParam](a.ToolChoiceToolParam{
					Name: a.F(options.ToolChoice),
					Type: a.F(a.ToolChoiceToolTypeTool),
				})
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

			params.Thinking = a.F[a.ThinkingConfigParamUnion](a.ThinkingConfigParam{
				Type:         a.F(a.ThinkingConfigParamTypeEnabled),
				BudgetTokens: a.F(int64(budget)),
			})
		}
	}

	// Add tools if any are registered
	if len(g.tools) > 0 {
		var tools []a.ToolUnionUnionParam
		for _, tool := range g.tools {
			tools = append(tools, &tool)
		}
		params.Tools = a.F(tools)
	}

	// Make the API call
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
	if usage := resp.Usage; usage.InputTokens > 0 || usage.OutputTokens > 0 {
		if promptTokens := usage.InputTokens; promptTokens > 0 {
			result.UsageMetrics[UsageMetricInputTokens] = int(promptTokens)
		}
		if completionTokens := usage.OutputTokens; completionTokens > 0 {
			result.UsageMetrics[UsageMetricGenerationTokens] = int(completionTokens)
		}
	}

	// Convert the message content
	var blocks []Block

	// Handle text content
	for _, contentPart := range resp.Content {
		switch contentPart.Type {
		case a.ContentBlockTypeText:
			blocks = append(blocks, Block{
				ID:           contentPart.ID,
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str(contentPart.Text),
			})
		case a.ContentBlockTypeToolUse:
			// Create a ToolUseInput with standardized format
			var toolParams map[string]interface{}
			if err := json.Unmarshal(contentPart.Input, &toolParams); err != nil {
				return Response{}, fmt.Errorf("failed to unmarshal tool use input: %w", err)
			}

			toolUse := ToolUseInput{
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
		case a.ContentBlockTypeThinking:
			blocks = append(blocks, Block{
				ID:           contentPart.ID,
				BlockType:    Thinking,
				ModalityType: Text,
				Content:      Str(contentPart.Thinking),
				ExtraFields: map[string]interface{}{
					generatorPrefix + thinkingSignatureKey: contentPart.Signature,
				},
			})
		case a.ContentBlockTypeRedactedThinking:
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
	case a.MessageStopReasonEndTurn:
		result.FinishReason = EndTurn
	case a.MessageStopReasonMaxTokens:
		result.FinishReason = MaxGenerationLimit
		// Return MaxGenerationLimitErr when the model reaches its token limit,
		// regardless of whether MaxGenerationTokens was explicitly set
		return result, MaxGenerationLimitErr
	case a.MessageStopReasonStopSequence:
		result.FinishReason = StopSequence
	case a.MessageStopReasonToolUse:
		result.FinishReason = ToolUse
	default:
		result.FinishReason = Unknown
	}

	return result, nil
}

// preprocessToolResults consolidates consecutive tool result messages that respond to tool calls
// from the same previous assistant message.
//
// This function is specific to the Anthropic implementation and addresses a key difference in how
// parallel tool use is handled between Anthropic and OpenAI:
//
//   - Anthropic API expects all tool results for parallel tool calls to be bundled in a single message
//     with multiple tool result blocks, each with its own tool_use_id.
//   - OpenAI API expects each tool result to be a separate message in the conversation,
//     where all blocks in a message must have the same tool ID and be text modality.
//
// Our internal Dialog representation follows OpenAI's approach with separate messages for each tool result,
// so we need this preprocessing step to adapt to Anthropic's API expectations.
func preprocessToolResults(dialog Dialog) Dialog {
	if len(dialog) <= 1 {
		return dialog
	}

	result := make(Dialog, 0, len(dialog))
	i := 0

	for i < len(dialog) {
		// Always add non-tool-result messages as they are
		if dialog[i].Role != ToolResult {
			result = append(result, dialog[i])
			i++
			continue
		}

		// We have found a tool result message

		// First, find the previous assistant message that might have tool calls
		assistantMsgIndex := -1
		for j := i - 1; j >= 0; j-- {
			if dialog[j].Role == Assistant {
				// Check if this assistant message has tool calls
				hasToolCalls := false
				for _, block := range dialog[j].Blocks {
					if block.BlockType == ToolCall {
						hasToolCalls = true
						break
					}
				}

				if hasToolCalls {
					assistantMsgIndex = j
					break
				}
			}
		}

		// If we can't find a previous assistant message with tool calls,
		// just add the tool result as is
		if assistantMsgIndex == -1 {
			result = append(result, dialog[i])
			i++
			continue
		}

		// Check if this is a parallel tool use scenario (multiple tool calls in the assistant message)
		var toolCallIDs []string
		for _, block := range dialog[assistantMsgIndex].Blocks {
			if block.BlockType == ToolCall {
				toolCallIDs = append(toolCallIDs, block.ID)
			}
		}

		// If there's only one tool call, don't do any special handling
		if len(toolCallIDs) <= 1 {
			result = append(result, dialog[i])
			i++
			continue
		}

		// This could be the start of a sequence of tool result messages
		// for parallel tool use. Look ahead to collect consecutive tool results.
		startIndex := i
		j := i + 1

		// Look ahead for consecutive tool result messages
		for j < len(dialog) && dialog[j].Role == ToolResult {
			j++
		}

		// If we found multiple consecutive tool result messages
		if j-startIndex > 1 {
			// Check if these tool results correspond to the tool calls in the previous assistant message
			toolResultMessagesByToolID := make(map[string][]Block)

			// Group all blocks from these consecutive tool result messages by tool ID
			for k := startIndex; k < j; k++ {
				for _, block := range dialog[k].Blocks {
					if block.ID != "" {
						toolResultMessagesByToolID[block.ID] = append(toolResultMessagesByToolID[block.ID], block)
					}
				}
			}

			// Create a consolidated tool result message if we found results for the tools
			isThisParallelToolUse := false
			for _, id := range toolCallIDs {
				if _, found := toolResultMessagesByToolID[id]; found {
					isThisParallelToolUse = true
					break
				}
			}

			if isThisParallelToolUse {
				// Create a consolidated message with all blocks from these consecutive tool results
				var consolidatedBlocks []Block
				anyError := false

				for k := startIndex; k < j; k++ {
					consolidatedBlocks = append(consolidatedBlocks, dialog[k].Blocks...)
					if dialog[k].ToolResultError {
						anyError = true
					}
				}

				result = append(result, Message{
					Role:            ToolResult,
					Blocks:          consolidatedBlocks,
					ToolResultError: anyError,
				})

				i = j // Skip past all the consolidated messages
				continue
			}
		}

		// If not parallel tool use or only one tool result message, add it as is
		result = append(result, dialog[i])
		i++
	}

	return result
}

type AnthropicCompletionService interface {
	New(ctx context.Context, params a.MessageNewParams, opts ...option.RequestOption) (res *a.Message, err error)
}

// NewAnthropicGenerator creates a new OpenAI generator with the specified model.
func NewAnthropicGenerator(client AnthropicCompletionService, model, systemInstructions string) AnthropicGenerator {
	return AnthropicGenerator{
		client:             client,
		systemInstructions: systemInstructions,
		model:              model,
		tools:              make(map[string]a.ToolParam),
	}
}

var _ Generator = (*AnthropicGenerator)(nil)
var _ ToolRegister = (*AnthropicGenerator)(nil)
