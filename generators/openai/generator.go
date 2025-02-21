package openai

import (
	"context"
	"encoding/json"
	"fmt"

	oai "github.com/openai/openai-go"
	"github.com/spachava753/gai"
	"github.com/spachava753/gai/generators/internal"
)

// registeredTool represents a tool that has been registered with the generator
type registeredTool struct {
	callback gai.ToolCallback
	oaiTool  oai.ChatCompletionToolParam // OpenAI's representation of the tool
}

// generator implements the gai.Generator interface using OpenAI's API
type generator struct {
	client *oai.Client
	model  string
	tools  map[string]registeredTool
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

	switch msg.Role {
	case gai.User:
		// User messages should only have unstructured blocks
		if len(msg.Blocks) != 1 || msg.Blocks[0].BlockType != gai.Unstructured {
			return nil, fmt.Errorf("unsupported block type for user: %v", msg.Blocks[0].BlockType)
		}
		return oai.UserMessage(msg.Blocks[0].Content), nil

	case gai.Assistant:
		// Handle different assistant message types
		if len(msg.Blocks) == 1 {
			block := msg.Blocks[0]
			switch block.BlockType {
			case gai.Unstructured:
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

		// Handle multiple blocks
		var textContent []oai.ChatCompletionAssistantMessageParamContentUnion
		var toolCalls []oai.ChatCompletionMessageToolCallParam

		for _, block := range msg.Blocks {
			switch block.BlockType {
			case gai.Unstructured:
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

	return gai.Response{}, nil
}

// New creates a new OpenAI generator with the specified model.
func New(client *oai.Client, model string) gai.ToolGenerator {
	g := internal.NewValidation(&generator{
		client: client,
		model:  model,
		tools:  make(map[string]registeredTool),
	})
	return &g
}

var _ gai.Generator = (*generator)(nil)
var _ gai.ToolGenerator = (*generator)(nil)