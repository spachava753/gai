package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"

	oai "github.com/openai/openai-go"
	"github.com/spachava753/gai"
)

// registeredTool represents a tool that has been registered with the generator
type registeredTool struct {
	callback gai.ToolCallback
	oaiTool  oai.ChatCompletionToolParam // OpenAI's representation of the tool
}

// generator implements the gai.Generator interface using OpenAI's API
type generator struct {
	client             ChatCompletionService
	model              string
	tools              map[string]registeredTool
	systemInstructions string
}

// convertToolToOpenAI converts our tool definition to OpenAI's format
func convertToolToOpenAI(tool gai.Tool) oai.ChatCompletionToolParam {
	// Convert our tool schema to OpenAI's JSON schema format
	parameters := make(map[string]interface{})
	parameters["type"] = tool.InputSchema.Type.String()

	// Only include properties and required fields if we have an object type
	if tool.InputSchema.Type == gai.Object && tool.InputSchema.Properties != nil {
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
func convertPropertyToMap(prop gai.Property) map[string]interface{} {
	result := map[string]interface{}{
		"type":        prop.Type.String(),
		"description": prop.Description,
	}

	// Handle string enums
	if prop.Type == gai.String && len(prop.Enum) > 0 {
		result["enum"] = prop.Enum
	}

	// Handle array items
	if prop.Type == gai.Array && prop.Items != nil {
		result["items"] = convertPropertyToMap(*prop.Items)
	}

	// Handle object properties and required fields
	if prop.Type == gai.Object && prop.Properties != nil {
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
func toOpenAIMessage(msg gai.Message) (oai.ChatCompletionMessageParamUnion, error) {
	if len(msg.Blocks) == 0 {
		return nil, fmt.Errorf("message must have at least one block")
	}

	// Check for video modality in any block
	for _, block := range msg.Blocks {
		if block.ModalityType == gai.Video {
			return nil, fmt.Errorf("unsupported modality: %v", block.ModalityType)
		}
	}

	// Check if any block is a tool result - it must be the only block in the message
	for _, block := range msg.Blocks {
		if block.BlockType == gai.ToolResult && len(msg.Blocks) > 1 {
			return nil, fmt.Errorf("tool result block must be in its own message")
		}
	}

	switch msg.Role {
	case gai.User:
		// User messages should only have Content blocks
		if len(msg.Blocks) != 1 || msg.Blocks[0].BlockType != gai.Content {
			return nil, fmt.Errorf("unsupported block type for user: %v", msg.Blocks[0].BlockType)
		}
		return oai.UserMessage(msg.Blocks[0].Content), nil

	case gai.Assistant:
		// Handle different assistant message types
		if len(msg.Blocks) == 1 {
			block := msg.Blocks[0]
			switch block.BlockType {
			case gai.Content:
				// TODO: handle multi modality, like images
				return oai.AssistantMessage(block.Content), nil
			case gai.ToolCall:
				// Parse the tool call content
				var call struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				}
				if err := json.Unmarshal([]byte(block.Content), &call); err != nil {
					return nil, fmt.Errorf("invalid tool call content: %w", err)
				}

				return oai.ChatCompletionAssistantMessageParam{
					ToolCalls: oai.F([]oai.ChatCompletionMessageToolCallParam{
						{
							ID: oai.F(block.ID),
							Function: oai.F(oai.ChatCompletionMessageToolCallFunctionParam{
								Name:      oai.F(call.Name),
								Arguments: oai.F(string(call.Arguments)),
							}),
						},
					}),
				}, nil
			case gai.ToolResult:
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
			case gai.Content:
				textContent = append(textContent, oai.TextPart(block.Content))
			case gai.ToolCall:
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

// RegisterTool implements gai.ToolGenerator
func (g *generator) RegisterTool(tool gai.Tool, callback gai.ToolCallback) error {
	if g.tools == nil {
		g.tools = make(map[string]registeredTool)
	}

	// Check for conflicts with existing tools
	if existing, exists := g.tools[tool.Name]; exists {
		return gai.ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool already registered with callback: %v", existing.callback != nil),
		}
	}

	// Convert our tool definition to OpenAI's format
	oaiTool := convertToolToOpenAI(tool)

	// Store the tool and callback
	g.tools[tool.Name] = registeredTool{
		callback: callback,
		oaiTool:  oaiTool,
	}

	return nil
}

// Generate implements gai.Generator
func (g *generator) Generate(ctx context.Context, dialog gai.Dialog, options *gai.GenOpts) (gai.Response, error) {
	if g.client == nil {
		return gai.Response{}, fmt.Errorf("openai: client not initialized")
	}

	// Transform dialog to ensure tool results are in their own messages
	var transformedDialog gai.Dialog
	for _, msg := range dialog {
		// Check if we need to split this message
		var toolResults []gai.Block
		var otherBlocks []gai.Block

		for _, block := range msg.Blocks {
			if block.BlockType == gai.ToolResult {
				toolResults = append(toolResults, block)
			} else {
				otherBlocks = append(otherBlocks, block)
			}
		}

		// If we have both tool results and other blocks, split them
		if len(toolResults) > 0 && len(otherBlocks) > 0 {
			// Add other blocks besides tool result blocks first if they exist
			if len(otherBlocks) > 0 {
				transformedDialog = append(transformedDialog, gai.Message{
					Role:   msg.Role,
					Blocks: otherBlocks,
				})
			}

			// Add each tool result in its own message
			for _, block := range toolResults {
				transformedDialog = append(transformedDialog, gai.Message{
					Role:   msg.Role,
					Blocks: []gai.Block{block},
				})
			}
		} else {
			// No transformation needed
			transformedDialog = append(transformedDialog, msg)
		}
	}

	// Convert each message to OpenAI format
	var messages []oai.ChatCompletionMessageParamUnion
	for _, msg := range transformedDialog {
		oaiMsg, err := toOpenAIMessage(msg)
		if err != nil {
			return gai.Response{}, fmt.Errorf("failed to convert message: %w", err)
		}
		messages = append(messages, oaiMsg)
	}

	// Create OpenAI chat completion params
	params := oai.ChatCompletionNewParams{
		Model:    oai.F(oai.ChatModel(g.model)),
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
			case gai.ToolChoiceAuto:
				params.ToolChoice = oai.F(oai.ChatCompletionToolChoiceOptionUnionParam(oai.ChatCompletionToolChoiceOptionAuto("auto")))
			case gai.ToolChoiceToolsRequired:
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
			tools = append(tools, tool.oaiTool)
		}
		params.Tools = oai.F(tools)
	}

	// Keep generating until we get a text response or a tool call without a callback
	for {
		resp, err := g.client.New(ctx, params)
		if err != nil {
			return gai.Response{}, fmt.Errorf("failed to create new message: %w", err)
		}

		// Get the first choice (we enforce N=1 for tool callbacks)
		choice := resp.Choices[0]

		// Check if it's a tool call
		if len(choice.Message.ToolCalls) > 0 {
			// Get the first tool call (parallel tool calls not supported yet)
			toolCall := choice.Message.ToolCalls[0]

			// Look up the tool
			tool, exists := g.tools[toolCall.Function.Name]
			if !exists {
				return gai.Response{}, fmt.Errorf("tool %q not found", toolCall.Function.Name)
			}

			// If no callback, we're done
			if tool.callback == nil {
				break
			}

			// Parse the arguments
			var args map[string]any
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
				return gai.Response{}, fmt.Errorf("failed to parse tool arguments: %w", err)
			}

			// Execute the callback
			result, err := tool.callback.Call(ctx, args)
			if err != nil {
				return gai.Response{}, fmt.Errorf("tool execution failed: %w", err)
			}

			// Convert result to string
			var resultStr string
			if err, ok := result.(error); ok {
				// If result implements error, use the error message
				resultStr = err.Error()
			} else {
				// Otherwise, JSON marshal the result
				resultBytes, err := json.Marshal(result)
				if err != nil {
					return gai.Response{}, fmt.Errorf("failed to marshal tool result: %w", err)
				}
				resultStr = string(resultBytes)
			}

			// Add the tool call and result to messages
			params.Messages = oai.F(append(messages,
				oai.ChatCompletionAssistantMessageParam{
					ToolCalls: oai.F([]oai.ChatCompletionMessageToolCallParam{
						{
							ID: oai.F(toolCall.ID),
							Function: oai.F(oai.ChatCompletionMessageToolCallFunctionParam{
								Name:      oai.F(toolCall.Function.Name),
								Arguments: oai.F(toolCall.Function.Arguments),
							}),
						},
					}),
				},
				oai.ToolMessage(toolCall.ID, resultStr),
			))
			continue
		}

		// If we get here, it's a text response, so we're done
		break
	}

	return gai.Response{}, nil
}

type ChatCompletionService interface {
	New(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) (res *oai.ChatCompletion, err error)
}

// New creates a new OpenAI generator with the specified model.
func New(client ChatCompletionService, model, systemInstructions string) gai.Generator {
	return &generator{
		client:             client,
		systemInstructions: systemInstructions,
		model:              model,
		tools:              make(map[string]registeredTool),
	}
}

//var _ gai.Generator = (*generator)(nil)
//var _ gai.ToolGenerator = (*generator)(nil)
