package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/openai/openai-go/option"
	"slices"
	"strings"

	oai "github.com/openai/openai-go"
)

// OpenAiGenerator implements the gai.Generator interface using OpenAI's API
type OpenAiGenerator struct {
	client             OpenAICompletionService
	model              string
	tools              map[string]oai.ChatCompletionToolParam
	systemInstructions string
}

// convertToolToOpenAI converts our tool definition to OpenAI's format
func convertToolToOpenAI(tool Tool) oai.ChatCompletionToolParam {
	// Convert our tool schema to OpenAI's JSON schema format
	parameters := make(map[string]interface{})
	parameters["type"] = tool.InputSchema.Type.String()

	// Only include properties and required fields if we have an object type
	if tool.InputSchema.Type == Object && tool.InputSchema.Properties != nil {
		properties := make(map[string]interface{})
		for name, prop := range tool.InputSchema.Properties {
			properties[name] = convertPropertyToMap(prop)
		}
		parameters["properties"] = properties
		if tool.InputSchema.Required == nil {
			tool.InputSchema.Required = []string{}
		}
		parameters["required"] = tool.InputSchema.Required
	}

	return oai.ChatCompletionToolParam{
		Function: oai.FunctionDefinitionParam{
			Name:        tool.Name,
			Description: oai.String(tool.Description),
			Parameters:  parameters,
		},
	}
}

// convertPropertyToMap converts a Property to a map[string]interface{} suitable for OpenAI's format
func convertPropertyToMap(prop Property) map[string]interface{} {
	result := map[string]interface{}{
		"type":        prop.Type.String(),
		"description": prop.Description,
	}

	// Handle string enums
	if prop.Type == String && len(prop.Enum) > 0 {
		result["enum"] = prop.Enum
	}

	// Handle array items
	if prop.Type == Array && prop.Items != nil {
		result["items"] = convertPropertyToMap(*prop.Items)
	}

	// Handle object properties and required fields
	if prop.Type == Object && prop.Properties != nil {
		properties := make(map[string]interface{})
		for name, p := range prop.Properties {
			properties[name] = convertPropertyToMap(p)
		}
		result["properties"] = properties
		if len(prop.Required) > 0 {
			result["required"] = prop.Required
		}
	}

	return result
}

// toOpenAIMessage converts a gai.Message to an OpenAI chat message.
// It returns an error if the message contains unsupported modalities or block types.
func toOpenAIMessage(msg Message) (oai.ChatCompletionMessageParamUnion, error) {
	if len(msg.Blocks) == 0 {
		return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("message must have at least one block")
	}

	// Check for video modality in any block
	for _, block := range msg.Blocks {
		if block.ModalityType == Video {
			return oai.ChatCompletionMessageParamUnion{}, UnsupportedInputModalityErr("video")
		}
	}

	// If the message is a ToolResult, it'll be handled in the switch statement below

	switch msg.Role {
	case User:
		// Special case for single text block, to supply the content directly as a string,
		// instead of as a part of an object. This is especially helpful when using third-party
		// providers like open-router or deepseek, which do not support multiple objects supplied as
		// content for a message.
		if len(msg.Blocks) == 1 && msg.Blocks[0].ModalityType == Text {
			return oai.UserMessage(msg.Blocks[0].Content.String()), nil
		}

		// User messages should only have Content blocks
		for _, block := range msg.Blocks {
			if block.BlockType != Content {
				return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unsupported block type for user: %v", block.BlockType)
			}
		}

		// Handle multimodal content
		var parts []oai.ChatCompletionContentPartUnionParam
		for _, block := range msg.Blocks {
			switch block.ModalityType {
			case Text:
				parts = append(parts, oai.TextContentPart(block.Content.String()))
			case Image:
				// Convert image content to an image part
				if block.MimeType == "" {
					return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("image block missing mimetype")
				}
				dataUrl := fmt.Sprintf("data:%s;base64,%s",
					block.MimeType,
					block.Content.String())

				parts = append(parts, oai.ImageContentPart(oai.ChatCompletionContentPartImageImageURLParam{
					URL: dataUrl,
				}))
			case Audio:
				// Convert audio content to an audio input part
				if block.MimeType == "" {
					return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("audio block missing mimetype")
				}
				// Extract format from mimetype (e.g., "audio/wav" -> "wav")
				format, _ := strings.CutPrefix(block.MimeType, "audio/")
				if !slices.Contains([]string{"wav", "mp3"}, format) {
					return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unsupported audio format: %v", block.MimeType)
				}
				parts = append(parts, oai.InputAudioContentPart(oai.ChatCompletionContentPartInputAudioInputAudioParam{
					Data:   block.Content.String(),
					Format: format,
				}))
			default:
				return oai.ChatCompletionMessageParamUnion{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
		}

		return oai.UserMessage(parts), nil

	case Assistant:
		// Handle multiple blocks
		var contentParts oai.ChatCompletionAssistantMessageParamContentUnion
		var toolCalls []oai.ChatCompletionMessageToolCallParam
		var audioID string

		for _, block := range msg.Blocks {
			switch block.BlockType {
			case Content:
				if block.ModalityType == Text {
					contentParts.OfArrayOfContentParts = append(
						contentParts.OfArrayOfContentParts,
						oai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
							OfText: &oai.ChatCompletionContentPartTextParam{
								Text: block.Content.String(),
							},
						},
					)
				} else if block.ModalityType == Audio {
					if block.ID == "" {
						return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("assistant audio block missing ID")
					}
					// Remember audio ID for later use
					audioID = block.ID
				} else {
					return oai.ChatCompletionMessageParamUnion{}, UnsupportedInputModalityErr(block.ModalityType.String())
				}
			case ToolCall:
				// Parse the tool call content as ToolUseInput
				var toolUse ToolUseInput
				if err := json.Unmarshal([]byte(block.Content.String()), &toolUse); err != nil {
					return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("invalid tool call content: %w", err)
				}

				// Convert parameters to JSON string for OpenAI
				argsJSON, err := json.Marshal(toolUse.Parameters)
				if err != nil {
					return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
				}

				toolCalls = append(toolCalls, oai.ChatCompletionMessageToolCallParam{
					ID: block.ID,
					Function: oai.ChatCompletionMessageToolCallFunctionParam{
						Name:      toolUse.Name,
						Arguments: string(argsJSON),
					},
				})
			default:
				return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unsupported block type for assistant: %v", block.BlockType)
			}
		}

		result := oai.ChatCompletionMessageParamUnion{
			OfAssistant: &oai.ChatCompletionAssistantMessageParam{},
		}
		if len(contentParts.OfArrayOfContentParts) > 0 {
			result.OfAssistant.Content = contentParts
		}
		if len(toolCalls) > 0 {
			result.OfAssistant.ToolCalls = toolCalls
		}
		if audioID != "" {
			result.OfAssistant.Audio = oai.ChatCompletionAssistantMessageParamAudio{
				ID: audioID,
			}
		}
		return result, nil

	case ToolResult:
		// OpenAI handles tool results differently from Anthropic:
		// - OpenAI: Each tool result must be in a separate message with a single tool_call_id.
		//   All blocks in the message must have the same tool ID and be text modality.
		// - Anthropic: Multiple tool results for parallel tool calls must be in a single message
		//   with multiple tool_result blocks, each with its own tool_use_id.
		//
		// OpenAI's API has these requirements for tool results:
		// 1. Each tool result message must reference exactly one tool_call_id
		// 2. All blocks in a tool result message must have the same tool ID
		// 3. All blocks must be text modality
		// 4. Multiple tool results (for parallel tool calls) require separate messages

		// For ToolResult messages, we convert them to OpenAI's tool message format
		if len(msg.Blocks) == 0 {
			return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("tool result message must have at least one block")
		}

		// Get the ID from the first block
		toolID := msg.Blocks[0].ID
		if toolID == "" {
			return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("tool result message block must have an ID")
		}

		// Special case for single text block, to supply the content directly as a string,
		// instead of as a part of an object. This is especially helpful when using third-party
		// providers like open-router or deepseek, which do not support multiple objects supplied as
		// content for a message.
		if len(msg.Blocks) == 1 && msg.Blocks[0].ModalityType == Text {
			return oai.ToolMessage(msg.Blocks[0].Content.String(), toolID), nil
		}

		// Create text parts for each block with the same ID
		var textParts []oai.ChatCompletionContentPartTextParam
		for _, block := range msg.Blocks {
			// Verify all blocks have the same tool ID
			if block.ID != toolID {
				return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("all blocks in tool result message must have the same ID")
			}

			// Verify all blocks are text modality
			if block.ModalityType != Text {
				return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("OpenAI only supports text modality in tool result messages")
			}

			textParts = append(textParts, oai.ChatCompletionContentPartTextParam{
				Text: block.Content.String(),
			})
		}

		// Create the tool message with the text parts
		return oai.ToolMessage(textParts, toolID), nil

	default:
		return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unsupported role: %v", msg.Role)
	}
}

// Register implements gai.ToolRegister
func (g *OpenAiGenerator) Register(tool Tool) error {
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
		g.tools = make(map[string]oai.ChatCompletionToolParam)
	}

	// Check for conflicts with existing tools
	if _, exists := g.tools[tool.Name]; exists {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool already registered"),
		}
	}

	// Convert our tool definition to OpenAI's format and store it
	g.tools[tool.Name] = convertToolToOpenAI(tool)

	return nil
}

// Generate implements gai.Generator
func (g *OpenAiGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("openai: client not initialized")
	}

	// Check for empty dialog
	if len(dialog) == 0 {
		return Response{}, EmptyDialogErr
	}

	// Convert each message to OpenAI format
	var messages []oai.ChatCompletionMessageParamUnion
	for _, msg := range dialog {
		oaiMsg, err := toOpenAIMessage(msg)
		if err != nil {
			return Response{}, fmt.Errorf("failed to convert message: %w", err)
		}
		messages = append(messages, oaiMsg)
	}

	// Create OpenAI chat completion params
	params := oai.ChatCompletionNewParams{
		Model:    g.model,
		Messages: messages,
	}

	// Add system instructions if present
	if g.systemInstructions != "" {
		params.Messages = append([]oai.ChatCompletionMessageParamUnion{
			oai.SystemMessage(g.systemInstructions),
		}, messages...)
	}

	// Map our options to OpenAI params if options are provided
	if options != nil {
		// Set temperature if non-zero
		if options.Temperature != 0 {
			params.Temperature = oai.Float(options.Temperature)
		}

		// Set top_p if non-zero
		if options.TopP != 0 {
			params.TopP = oai.Float(options.TopP)
		}

		// Set frequency penalty if non-zero
		if options.FrequencyPenalty != 0 {
			params.FrequencyPenalty = oai.Float(options.FrequencyPenalty)
		}

		// Set presence penalty if non-zero
		if options.PresencePenalty != 0 {
			params.PresencePenalty = oai.Float(options.PresencePenalty)
		}

		// Set max tokens if specified
		if options.MaxGenerationTokens > 0 {
			params.MaxCompletionTokens = oai.Int(int64(options.MaxGenerationTokens))
		}

		// Set number of completions if specified
		if options.N > 0 {
			params.N = oai.Int(int64(options.N))
		}

		// Set stop sequences if specified
		if len(options.StopSequences) > 0 {
			// OpenAI accepts either a single string or array of strings
			if len(options.StopSequences) == 1 {
				params.Stop = oai.ChatCompletionNewParamsStopUnion{OfString: oai.String(options.StopSequences[0])}
			} else {
				params.Stop = oai.ChatCompletionNewParamsStopUnion{OfChatCompletionNewsStopArray: options.StopSequences}
			}
		}

		// Set tool choice if specified
		if options.ToolChoice != "" {
			switch options.ToolChoice {
			case ToolChoiceAuto:
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String(ToolChoiceAuto)}
			case ToolChoiceToolsRequired:
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String(ToolChoiceToolsRequired)}
			default:
				// Specific tool name
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfChatCompletionNamedToolChoice: &oai.ChatCompletionNamedToolChoiceParam{
					Function: oai.ChatCompletionNamedToolChoiceFunctionParam{
						Name: options.ToolChoice,
					},
				}}
			}
		}

		// Handle multimodality options
		if len(options.OutputModalities) > 0 {
			// Set requested modalities
			modalities := make([]string, 0, len(options.OutputModalities))
			hasAudio := false
			for _, m := range options.OutputModalities {
				switch m {
				case Text:
					modalities = append(modalities, "text")
				case Audio:
					modalities = append(modalities, "audio")
					hasAudio = true
				case Image:
					return Response{}, UnsupportedOutputModalityErr("image output not supported by model")
				case Video:
					return Response{}, UnsupportedOutputModalityErr("video output not supported by model")
				}
			}
			params.Modalities = modalities

			// Set audio configuration if audio output is requested
			if hasAudio {
				if options.AudioConfig.VoiceName == "" {
					return Response{}, InvalidParameterErr{
						Parameter: "AudioConfig.VoiceName",
						Reason:    "voice name is required for audio output",
					}
				}
				if options.AudioConfig.Format == "" {
					return Response{}, InvalidParameterErr{
						Parameter: "AudioConfig.Format",
						Reason:    "format is required for audio output",
					}
				}

				params.Audio = oai.ChatCompletionAudioParam{
					Voice:  oai.ChatCompletionAudioParamVoice(options.AudioConfig.VoiceName),
					Format: oai.ChatCompletionAudioParamFormat(options.AudioConfig.Format),
				}
			}
		}

		if options.ThinkingBudget != "" {
			switch options.ThinkingBudget {
			case "low", "medium", "high":
				params.ReasoningEffort = oai.ReasoningEffort(options.ThinkingBudget)
			default:
				return Response{}, &InvalidParameterErr{
					Parameter: "thinking budget",
					Reason: fmt.Sprintf(
						"invalid thinking budget, expected 'low', 'medium', or 'high': %s",
						options.ThinkingBudget,
					),
				}
			}
		}
	}

	// Add tools if any are registered
	if len(g.tools) > 0 {
		var tools []oai.ChatCompletionToolParam
		for _, tool := range g.tools {
			tools = append(tools, tool)
		}
		params.Tools = tools
	}

	// Make the API call
	resp, err := g.client.New(ctx, params)
	if err != nil {
		// Convert OpenAI SDK error to our error types based on status code
		var apierr *oai.Error
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
			case 429:
				return Response{}, RateLimitErr(apierr.Error())
			case 500:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "api_error",
					Message:    apierr.Error(),
				}
			case 503:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "service_unavailable",
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
	if usage := resp.Usage; usage.PromptTokens > 0 || usage.CompletionTokens > 0 {
		if promptTokens := usage.PromptTokens; promptTokens > 0 {
			result.UsageMetrics[UsageMetricInputTokens] = int(promptTokens)
		}
		if completionTokens := usage.CompletionTokens; completionTokens > 0 {
			result.UsageMetrics[UsageMetricGenerationTokens] = int(completionTokens)
		}
	}

	// Convert all choices to our Message type
	for _, choice := range resp.Choices {
		// Convert the message content
		var blocks []Block

		// Handle text content
		if content := choice.Message.Content; content != "" {
			blocks = append(blocks, Block{
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str(content),
			})
		}

		// Handle audio content if present
		if choice.Message.Audio.ID != "" {
			// Add audio block
			blocks = append(blocks, Block{
				ID:           choice.Message.Audio.ID,
				BlockType:    Content,
				ModalityType: Audio,
				MimeType:     "audio/" + options.AudioConfig.Format,
				Content:      Str(choice.Message.Audio.Data),
			})

			// Add transcript as a separate text block if available
			if choice.Message.Audio.Transcript != "" {
				blocks = append(blocks, Block{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(choice.Message.Audio.Transcript),
				})
			}
		}

		// Handle tool calls
		if toolCalls := choice.Message.ToolCalls; len(toolCalls) > 0 {
			for _, toolCall := range toolCalls {
				// Create a ToolUseInput with standardized format
				toolUse := ToolUseInput{
					Name: toolCall.Function.Name,
				}

				// Parse the arguments string into a map
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &toolUse.Parameters); err != nil {
					return Response{}, fmt.Errorf("failed to parse tool arguments: %w", err)
				}

				// Marshal back to JSON for consistent representation
				toolUseJSON, err := json.Marshal(toolUse)
				if err != nil {
					return Response{}, fmt.Errorf("failed to marshal tool use: %w", err)
				}

				blocks = append(blocks, Block{
					ID:           toolCall.ID,
					BlockType:    ToolCall,
					ModalityType: Text,
					MimeType:     "application/json",
					Content:      Str(string(toolUseJSON)),
				})
			}
		}

		result.Candidates = append(result.Candidates, Message{
			Role:   Assistant,
			Blocks: blocks,
		})
	}

	// Set finish reason
	if len(resp.Choices) > 0 {
		switch resp.Choices[0].FinishReason {
		case "stop":
			result.FinishReason = EndTurn
		case "length":
			result.FinishReason = MaxGenerationLimit
			// Return MaxGenerationLimitErr when the model reaches its token limit,
			// regardless of whether MaxGenerationTokens was explicitly set
			return result, MaxGenerationLimitErr
		case "tool_calls":
			result.FinishReason = ToolUse
		case "content_filter":
			result.FinishReason = Unknown
			// If content was filtered, check for refusal message and return ContentPolicyErr
			refusalMessage := resp.Choices[0].Message.Refusal
			if refusalMessage == "" {
				// Default message if no specific reason provided
				refusalMessage = "content policy violation detected"
			}
			return result, ContentPolicyErr(refusalMessage)
		default:
			result.FinishReason = Unknown
		}
	}

	return result, nil
}

type OpenAICompletionService interface {
	New(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) (res *oai.ChatCompletion, err error)
}

// NewOpenAiGenerator creates a new OpenAI generator with the specified model.
func NewOpenAiGenerator(client OpenAICompletionService, model, systemInstructions string) OpenAiGenerator {
	return OpenAiGenerator{
		client:             client,
		systemInstructions: systemInstructions,
		model:              model,
		tools:              make(map[string]oai.ChatCompletionToolParam),
	}
}

var _ Generator = (*OpenAiGenerator)(nil)
var _ ToolRegister = (*OpenAiGenerator)(nil)
var _ OpenAICompletionService = (*oai.ChatCompletionService)(nil)
