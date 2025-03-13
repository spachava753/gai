package gai

import (
	"context"
	"encoding/json"
	"fmt"
	a "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
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

// toAnthropicMessage converts a gai.Message to an OpenAI chat message.
// It returns an error if the message contains unsupported modalities or block types.
func toAnthropicMessage(msg Message) (a.MessageParam, error) {
	if len(msg.Blocks) == 0 {
		return a.MessageParam{}, fmt.Errorf("message must have at least one block")
	}

	// Check for video modality in any block
	for _, block := range msg.Blocks {
		if block.ModalityType == Video || block.ModalityType == Audio {
			return a.MessageParam{}, fmt.Errorf("unsupported modality: %v", block.ModalityType)
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
				return a.MessageParam{}, fmt.Errorf("unsupported modality for user: %v", block.ModalityType)
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
					return a.MessageParam{}, fmt.Errorf(
						"unsupported modality in multi-block assistant message: %v",
						block.ModalityType,
					)
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
			default:
				return a.MessageParam{}, fmt.Errorf("unsupported block type for assistant: %v", block.BlockType)
			}
		}

		result.Content = a.F(contentParts)
		return result, nil

	case ToolResult:
		// Validate that all blocks have the same tool use ID
		if len(msg.Blocks) == 0 {
			return a.MessageParam{}, fmt.Errorf("tool result message must have at least one block")
		}

		// Get the ID from the first block
		toolUseID := msg.Blocks[0].ID
		if toolUseID == "" {
			return a.MessageParam{}, fmt.Errorf("tool result message must have a tool use ID")
		}

		// Check that all blocks have the same tool use ID
		for i, block := range msg.Blocks {
			if block.ID != toolUseID {
				return a.MessageParam{}, fmt.Errorf(
					"all blocks in a tool result message must have the same tool use ID (block %d has ID %q, expected %q)",
					i,
					block.ID,
					toolUseID,
				)
			}
		}

		resultContent := a.ToolResultBlockParam{
			Type:      a.F(a.ToolResultBlockParamTypeToolResult),
			ToolUseID: a.F(toolUseID),
			IsError:   a.F(msg.ToolResultError),
		}

		// A tool result message can have multiple content blocks of text or image
		var content []a.ToolResultBlockParamContentUnion
		for _, block := range msg.Blocks {
			var b a.ToolResultBlockParamContent
			switch block.ModalityType {
			case Text:
				b.Type = a.F(a.ToolResultBlockParamContentTypeText)
				b.Text = a.F(block.Content.String())
			case Image:
				b.Type = a.F(a.ToolResultBlockParamContentTypeImage)
				b.Source = a.F[interface{}](block.Content.String())
			default:
				return a.MessageParam{}, fmt.Errorf("unsupported modality for tool result: %v", block.ModalityType)
			}
			content = append(content, &b)
		}

		resultContent.Content = a.F(content)

		return a.MessageParam{
			// For tool result blocks, we actually want the role to be user
			Role: a.F(a.MessageParamRoleUser),
			Content: a.F([]a.ContentBlockParamUnion{
				resultContent,
			}),
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
		return Response{}, fmt.Errorf("openai: client not initialized")
	}

	// Convert each message to OpenAI format
	var messages []a.MessageParam
	for _, msg := range dialog {
		anthropicMsg, err := toAnthropicMessage(msg)
		if err != nil {
			return Response{}, fmt.Errorf("failed to convert message: %w", err)
		}
		messages = append(messages, anthropicMsg)
	}

	// Create OpenAI chat completion params
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

	// Map our options to OpenAI params if options are provided
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
	case a.MessageStopReasonStopSequence:
		result.FinishReason = StopSequence
	case a.MessageStopReasonToolUse:
		result.FinishReason = ToolUse
	default:
		result.FinishReason = Unknown
	}

	return result, nil
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
