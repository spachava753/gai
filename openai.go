package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"

	oai "github.com/openai/openai-go"
)

// OpenAiGenerator implements the gai.Generator interface using OpenAI's API
type OpenAiGenerator struct {
	client             ChatCompletionService
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
		parameters["required"] = tool.InputSchema.Required
	}

	return oai.ChatCompletionToolParam{
		Type: oai.F(oai.ChatCompletionToolTypeFunction),
		Function: oai.F(oai.FunctionDefinitionParam{
			Name:        oai.F(tool.Name),
			Description: oai.F(tool.Description),
			Parameters:  oai.F(oai.FunctionParameters(parameters)),
		}),
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
		return nil, fmt.Errorf("message must have at least one block")
	}

	// Check for video modality in any block
	for _, block := range msg.Blocks {
		if block.ModalityType == Video {
			return nil, fmt.Errorf("unsupported modality: %v", block.ModalityType)
		}
	}

	// Check if any block is a tool result - it must be the only block in the message
	for _, block := range msg.Blocks {
		if block.BlockType == ToolResult && len(msg.Blocks) > 1 {
			return nil, fmt.Errorf("tool result block must be in its own message")
		}
	}

	switch msg.Role {
	case User:
		// User messages should only have Content blocks
		if len(msg.Blocks) != 1 || msg.Blocks[0].BlockType != Content {
			return nil, fmt.Errorf("unsupported block type for user: %v", msg.Blocks[0].BlockType)
		}
		return oai.UserMessage(msg.Blocks[0].Content), nil

	case Assistant:
		// Handle different assistant message types
		if len(msg.Blocks) == 1 {
			block := msg.Blocks[0]
			switch block.BlockType {
			case Content:
				// TODO: handle multi modality, like images
				return oai.AssistantMessage(block.Content), nil
			case ToolCall:
				// Parse the tool call content
				var call struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"parameters"`
				}
				if err := json.Unmarshal([]byte(block.Content), &call); err != nil {
					return nil, fmt.Errorf("invalid tool call content: %w", err)
				}

				return oai.ChatCompletionAssistantMessageParam{
					Role: oai.F(oai.ChatCompletionAssistantMessageParamRoleAssistant),
					ToolCalls: oai.F([]oai.ChatCompletionMessageToolCallParam{
						{
							ID: oai.F(block.ID),
							Function: oai.F(oai.ChatCompletionMessageToolCallFunctionParam{
								Name:      oai.F(call.Name),
								Arguments: oai.F(string(call.Arguments)),
							}),
							Type: oai.F(oai.ChatCompletionMessageToolCallTypeFunction),
						},
					}),
				}, nil
			case ToolResult:
				return oai.ToolMessage(block.ID, block.Content), nil
			default:
				return nil, fmt.Errorf("unsupported block type for assistant: %v", block.BlockType)
			}
		}

		// Handle multiple blocks - tool results are not allowed here
		var textContent []oai.ChatCompletionAssistantMessageParamContentUnion
		var toolCalls []oai.ChatCompletionMessageToolCallParam

		for _, block := range msg.Blocks {
			switch block.BlockType {
			case Content:
				textContent = append(textContent, oai.TextPart(block.Content))
			case ToolCall:
				// Parse the tool call content
				var call struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				}
				if err := json.Unmarshal([]byte(block.Content), &call); err != nil {
					return nil, fmt.Errorf("invalid tool call content: %w", err)
				}

				toolCalls = append(toolCalls, oai.ChatCompletionMessageToolCallParam{
					ID: oai.F(block.ID),
					Function: oai.F(oai.ChatCompletionMessageToolCallFunctionParam{
						Name:      oai.F(call.Name),
						Arguments: oai.F(string(call.Arguments)),
					}),
				})
			default:
				return nil, fmt.Errorf("unsupported block type for assistant: %v", block.BlockType)
			}
		}

		result := oai.ChatCompletionAssistantMessageParam{}
		if len(textContent) > 0 {
			result.Content = oai.F(textContent)
		}
		if len(toolCalls) > 0 {
			result.ToolCalls = oai.F(toolCalls)
		}
		return result, nil

	default:
		return nil, fmt.Errorf("unsupported role: %v", msg.Role)
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
		Model:    oai.F(g.model),
		Messages: oai.F(messages),
	}

	// Add system instructions if present
	if g.systemInstructions != "" {
		params.Messages = oai.F(append([]oai.ChatCompletionMessageParamUnion{
			oai.SystemMessage(g.systemInstructions),
		}, messages...))
	}

	// Map our options to OpenAI params if options are provided
	if options != nil {
		// Set temperature if non-zero
		if options.Temperature != 0 {
			params.Temperature = oai.F(options.Temperature)
		}

		// Set top_p if non-zero
		if options.TopP != 0 {
			params.TopP = oai.F(options.TopP)
		}

		// Set frequency penalty if non-zero
		if options.FrequencyPenalty != 0 {
			params.FrequencyPenalty = oai.F(options.FrequencyPenalty)
		}

		// Set presence penalty if non-zero
		if options.PresencePenalty != 0 {
			params.PresencePenalty = oai.F(options.PresencePenalty)
		}

		// Set max tokens if specified
		if options.MaxGenerationTokens > 0 {
			params.MaxCompletionTokens = oai.F(int64(options.MaxGenerationTokens))
		}

		// Set number of completions if specified
		if options.N > 0 {
			params.N = oai.F(int64(options.N))
		}

		// Set stop sequences if specified
		if len(options.StopSequences) > 0 {
			// OpenAI accepts either a single string or array of strings
			if len(options.StopSequences) == 1 {
				params.Stop = oai.F(oai.ChatCompletionNewParamsStopUnion(shared.UnionString(options.StopSequences[0])))
			} else {
				params.Stop = oai.F(oai.ChatCompletionNewParamsStopUnion(oai.ChatCompletionNewParamsStopArray(options.StopSequences)))
			}
		}

		// Set tool choice if specified
		if options.ToolChoice != "" {
			switch options.ToolChoice {
			case ToolChoiceAuto:
				params.ToolChoice = oai.F(oai.ChatCompletionToolChoiceOptionUnionParam(oai.ChatCompletionToolChoiceOptionAuto("auto")))
			case ToolChoiceToolsRequired:
				params.ToolChoice = oai.F(oai.ChatCompletionToolChoiceOptionUnionParam(oai.ChatCompletionToolChoiceOptionAuto("required")))
			default:
				// Specific tool name
				params.ToolChoice = oai.F(oai.ChatCompletionToolChoiceOptionUnionParam(oai.ChatCompletionNamedToolChoiceParam{
					Type: oai.F(oai.ChatCompletionNamedToolChoiceTypeFunction),
					Function: oai.F(oai.ChatCompletionNamedToolChoiceFunctionParam{
						Name: oai.F(options.ToolChoice),
					}),
				}))
			}
		}
	}

	// Add tools if any are registered
	if len(g.tools) > 0 {
		var tools []oai.ChatCompletionToolParam
		for _, tool := range g.tools {
			tools = append(tools, tool)
		}
		params.Tools = oai.F(tools)
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
				Content:      content,
			})
		}

		// Handle tool calls
		if toolCalls := choice.Message.ToolCalls; len(toolCalls) > 0 {
			for _, toolCall := range toolCalls {
				// Create tool call block
				toolCallContent := map[string]interface{}{
					"name":       toolCall.Function.Name,
					"parameters": json.RawMessage(toolCall.Function.Arguments),
				}
				toolCallJSON, err := json.Marshal(toolCallContent)
				if err != nil {
					return Response{}, fmt.Errorf("failed to marshal tool call: %w", err)
				}

				blocks = append(blocks, Block{
					ID:           toolCall.ID,
					BlockType:    ToolCall,
					ModalityType: Text,
					Content:      string(toolCallJSON),
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
		case oai.ChatCompletionChoicesFinishReasonStop:
			result.FinishReason = EndTurn
		case oai.ChatCompletionChoicesFinishReasonLength:
			result.FinishReason = MaxGenerationLimit
		case oai.ChatCompletionChoicesFinishReasonToolCalls:
			result.FinishReason = ToolUse
		default:
			result.FinishReason = Unknown
		}
	}

	return result, nil
}

type ChatCompletionService interface {
	New(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) (res *oai.ChatCompletion, err error)
}

// NewOpenAiGenerator creates a new OpenAI generator with the specified model.
func NewOpenAiGenerator(client ChatCompletionService, model, systemInstructions string) OpenAiGenerator {
	return OpenAiGenerator{
		client:             client,
		systemInstructions: systemInstructions,
		model:              model,
		tools:              make(map[string]oai.ChatCompletionToolParam),
	}
}

var _ Generator = (*OpenAiGenerator)(nil)
var _ ToolRegister = (*OpenAiGenerator)(nil)
