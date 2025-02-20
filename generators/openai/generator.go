package openai

import (
	"context"
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
	// For tools with no parameters, don't set any parameters
	if tool.InputSchema.Type == gai.Null {
		return oai.ChatCompletionToolParam{
			Type: oai.F(oai.ChatCompletionToolTypeFunction),
			Function: oai.F(oai.FunctionDefinitionParam{
				Name:        oai.F(tool.Name),
				Description: oai.F(tool.Description),
			}),
		}
	}

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

// toOpenAIMessage converts a gai.Message to an OpenAI chat message.
// It returns an error if the message contains unsupported modalities or block types.
func toOpenAIMessage(msg gai.Message) (oai.ChatCompletionMessageParamUnion, error) {
	var content string
	for _, block := range msg.Blocks {
		if block.BlockType != gai.Unstructured {
			return nil, fmt.Errorf("unsupported block type: %v", block.BlockType)
		}

		switch block.ModalityType {
		case gai.Text:
			content += block.Content
		default:
			return nil, fmt.Errorf("unsupported modality: %v", block.ModalityType)
		}
	}

	switch msg.Role {
	case gai.User:
		return oai.UserMessage(content), nil
	case gai.Assistant:
		return oai.AssistantMessage(content), nil
	default:
		return nil, fmt.Errorf("unsupported role: %v", msg.Role)
	}
}

// toOpenAIChatParams converts a gai.Dialog and gai.GenOpts to OpenAI chat completion parameters.
// It returns an error if any message contains unsupported modalities or block types.
func toOpenAIChatParams(dialog gai.Dialog, opts *gai.GenOpts) (oai.ChatCompletionNewParams, error) {
	messages := make([]oai.ChatCompletionMessageParamUnion, len(dialog))
	for i, msg := range dialog {
		oaiMsg, err := toOpenAIMessage(msg)
		if err != nil {
			return oai.ChatCompletionNewParams{}, fmt.Errorf("failed to convert message at index %d: %w", i, err)
		}
		messages[i] = oaiMsg
	}

	params := oai.ChatCompletionNewParams{
		Messages: oai.F(messages),
	}

	// Only set optional parameters if they are explicitly set in GenOpts
	if opts != nil {
		if opts.Temperature != 0 {
			params.Temperature = oai.F(opts.Temperature)
		}
		if opts.TopP != 0 {
			params.TopP = oai.F(opts.TopP)
		}
		if opts.FrequencyPenalty != 0 {
			params.FrequencyPenalty = oai.F(opts.FrequencyPenalty)
		}
		if opts.PresencePenalty != 0 {
			params.PresencePenalty = oai.F(opts.PresencePenalty)
		}
		if opts.MaxGenerationTokens != 0 {
			params.MaxTokens = oai.F(int64(opts.MaxGenerationTokens))
		}
		if len(opts.StopSequences) > 0 {
			// Convert []string to ChatCompletionNewParamsStopArray which implements ChatCompletionNewParamsStopUnion
			stopArray := oai.ChatCompletionNewParamsStopArray(opts.StopSequences)
			params.Stop = oai.F[oai.ChatCompletionNewParamsStopUnion](stopArray)
		}
		if opts.N != 0 {
			params.N = oai.F(int64(opts.N))
		}
	}

	return params, nil
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