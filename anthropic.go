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
	tools              map[string]a.BetaToolParam
	systemInstructions string
}

// convertToolToAnthropic converts our tool definition to OpenAI's format
func convertToolToAnthropic(tool Tool) a.BetaToolParam {
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

	return a.BetaToolParam{
		Name:        a.F(tool.Name),
		Description: a.F(tool.Description),
		InputSchema: a.F(a.BetaToolInputSchemaParam{
			Type:       a.F(a.BetaToolInputSchemaTypeObject),
			Properties: a.F[any](parameters),
		}),
	}
}

// toAnthropicMessage converts a gai.Message to an OpenAI chat message.
// It returns an error if the message contains unsupported modalities or block types.
func toAnthropicMessage(msg Message) (a.BetaMessageParam, error) {
	if len(msg.Blocks) == 0 {
		return a.BetaMessageParam{}, fmt.Errorf("message must have at least one block")
	}

	// Check for video modality in any block
	for _, block := range msg.Blocks {
		if block.ModalityType == Video || block.ModalityType == Audio {
			return a.BetaMessageParam{}, fmt.Errorf("unsupported modality: %v", block.ModalityType)
		}
	}

	// A message can contain only tool result blocks, or no tool result blocks at all,
	// but cannot mix tool result blocks with other blocks
	// TODO: implement validation

	switch msg.Role {
	case User:
		// User messages should only have Content blocks
		for _, block := range msg.Blocks {
			if block.BlockType != Content {
				return a.BetaMessageParam{}, fmt.Errorf("unsupported block type for user: %v", block.BlockType)
			}
		}

		// Handle multimodal content
		var parts []a.BetaContentBlockParamUnion
		for _, block := range msg.Blocks {
			switch block.ModalityType {
			case Text:
				parts = append(parts, a.BetaTextBlockParam{
					Type: a.F(a.BetaTextBlockParamTypeText),
					Text: a.F(block.Content.String()),
				})
			case Image:
				// Convert image media to an image part
				if block.MimeType == "" {
					return a.BetaMessageParam{}, fmt.Errorf("image media missing mimetype")
				}

				var imageBlock a.BetaImageBlockParamSourceUnion = &a.BetaBase64ImageSourceParam{
					Type:      a.F(a.BetaBase64ImageSourceTypeBase64),
					MediaType: a.F(a.BetaBase64ImageSourceMediaType(block.MimeType)),
					Data:      a.F(block.Content.String()),
				}

				parts = append(parts, a.BetaImageBlockParam{
					Type:   a.F(a.BetaImageBlockParamTypeImage),
					Source: a.F(imageBlock),
				})
			default:
				return a.BetaMessageParam{}, fmt.Errorf("unsupported modality for user: %v", block.ModalityType)
			}
		}

		return a.BetaMessageParam{
			Content: a.F(parts),
			Role:    a.F(a.BetaMessageParamRoleUser),
		}, nil

	case Assistant:
		// Handle multiple blocks
		var contentParts []a.BetaContentBlockParamUnion

		result := a.BetaMessageParam{
			Role: a.F(a.BetaMessageParamRoleAssistant),
		}

		for _, block := range msg.Blocks {
			switch block.BlockType {
			case Content:
				if block.ModalityType != Text {
					return a.BetaMessageParam{}, fmt.Errorf(
						"unsupported modality in multi-block assistant message: %v",
						block.ModalityType,
					)
				}
				contentParts = append(contentParts, &a.BetaTextBlockParam{
					Text: a.F(block.Content.String()),
					Type: a.F(a.BetaTextBlockParamTypeText),
				})
			case ToolCall:
				// Parse the tool call content
				var call struct {
					Name       string          `json:"name"`
					Parameters json.RawMessage `json:"parameters"`
				}
				if err := json.Unmarshal([]byte(block.Content.String()), &call); err != nil {
					return a.BetaMessageParam{}, fmt.Errorf("invalid tool call content: %w", err)
				}

				contentParts = append(contentParts, &a.BetaToolUseBlockParam{
					ID:    a.F(block.ID),
					Name:  a.F(call.Name),
					Input: a.F[interface{}](call.Parameters),
					Type:  a.F(a.BetaToolUseBlockParamTypeToolUse),
				})
			default:
				return a.BetaMessageParam{}, fmt.Errorf("unsupported block type for assistant: %v", block.BlockType)
			}
		}

		result.Content = a.F(contentParts)
		return result, nil

	case ToolResult:
		// Validate that all blocks have the same tool use ID
		if len(msg.Blocks) == 0 {
			return a.BetaMessageParam{}, fmt.Errorf("tool result message must have at least one block")
		}

		// Get the ID from the first block
		toolUseID := msg.Blocks[0].ID
		if toolUseID == "" {
			return a.BetaMessageParam{}, fmt.Errorf("tool result message must have a tool use ID")
		}

		// Check that all blocks have the same tool use ID
		for i, block := range msg.Blocks {
			if block.ID != toolUseID {
				return a.BetaMessageParam{}, fmt.Errorf(
					"all blocks in a tool result message must have the same tool use ID (block %d has ID %q, expected %q)",
					i,
					block.ID,
					toolUseID,
				)
			}
		}

		resultContent := a.BetaToolResultBlockParam{
			ToolUseID: a.F(toolUseID),
			Type:      a.F(a.BetaToolResultBlockParamTypeToolResult),
			IsError:   a.F(msg.ToolResultError),
		}

		// A tool result message can have multiple content blocks of text or image
		var content []a.BetaToolResultBlockParamContentUnion
		for _, block := range msg.Blocks {
			var b a.BetaToolResultBlockParamContent
			switch block.ModalityType {
			case Text:
				b.Type = a.F(a.BetaToolResultBlockParamContentTypeText)
				b.Text = a.F(block.Content.String())
			case Image:
				b.Type = a.F(a.BetaToolResultBlockParamContentTypeImage)
				b.Source = a.F[interface{}](block.Content.String())
			default:
				return a.BetaMessageParam{}, fmt.Errorf("unsupported modality for tool result: %v", block.ModalityType)
			}
			content = append(content, &b)
		}

		resultContent.Content = a.F(content)

		return a.BetaMessageParam{
			// For tool result blocks, we actually want the role to be user
			Role: a.F(a.BetaMessageParamRoleUser),
			Content: a.F([]a.BetaContentBlockParamUnion{
				resultContent,
			}),
		}, nil
	default:
		return a.BetaMessageParam{}, fmt.Errorf("unsupported role: %v", msg.Role)
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
		g.tools = make(map[string]a.BetaToolParam)
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
	var messages []a.BetaMessageParam
	for _, msg := range dialog {
		anthropicMsg, err := toAnthropicMessage(msg)
		if err != nil {
			return Response{}, fmt.Errorf("failed to convert message: %w", err)
		}
		messages = append(messages, anthropicMsg)
	}

	// Create OpenAI chat completion params
	params := a.BetaMessageNewParams{
		Model:    a.F(g.model),
		Messages: a.F(messages),
	}

	// Add system instructions if present
	if g.systemInstructions != "" {
		params.System = a.F([]a.BetaTextBlockParam{
			{
				Text: a.String(g.systemInstructions),
				Type: a.F(a.BetaTextBlockParamTypeText),
				CacheControl: a.F(a.BetaCacheControlEphemeralParam{
					Type: a.F(a.BetaCacheControlEphemeralTypeEphemeral),
				}),
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
				params.ToolChoice = a.F[a.BetaToolChoiceUnionParam](a.BetaToolChoiceAutoParam{
					Type: a.F(a.BetaToolChoiceAutoTypeAuto),
				})
			case ToolChoiceToolsRequired:
				params.ToolChoice = a.F[a.BetaToolChoiceUnionParam](a.BetaToolChoiceAnyParam{
					Type: a.F(a.BetaToolChoiceAnyTypeAny),
				})
			default:
				// Specific tool name
				params.ToolChoice = a.F[a.BetaToolChoiceUnionParam](a.BetaToolChoiceToolParam{
					Name: a.F(options.ToolChoice),
					Type: a.F(a.BetaToolChoiceToolTypeTool),
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
		var tools []a.BetaToolUnionUnionParam
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
		case a.BetaContentBlockTypeText:
			blocks = append(blocks, Block{
				ID:           contentPart.ID,
				BlockType:    Content,
				ModalityType: Text,
				Content:      Str(contentPart.Text),
			})
		case a.BetaContentBlockTypeToolUse:
			inputJson, err := json.Marshal(contentPart.Input)
			if err != nil {
				return Response{}, fmt.Errorf("failed to marshal input: %w", err)
			}
			blocks = append(blocks, Block{
				ID:           contentPart.ID,
				BlockType:    ToolCall,
				ModalityType: Text,
				Content:      Str(inputJson),
			})
		case a.BetaContentBlockTypeThinking:
			blocks = append(blocks, Block{
				ID:           contentPart.ID,
				BlockType:    Thinking,
				ModalityType: Text,
				Content:      Str(contentPart.Thinking),
			})
		case a.BetaContentBlockTypeRedactedThinking:
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
	case a.BetaMessageStopReasonEndTurn:
		result.FinishReason = EndTurn
	case a.BetaMessageStopReasonMaxTokens:
		result.FinishReason = MaxGenerationLimit
	case a.BetaMessageStopReasonStopSequence:
		result.FinishReason = StopSequence
	case a.BetaMessageStopReasonToolUse:
		result.FinishReason = ToolUse
	default:
		result.FinishReason = Unknown
	}

	return result, nil
}

type AnthropicCompletionService interface {
	New(ctx context.Context, params a.BetaMessageNewParams, opts ...option.RequestOption) (res *a.BetaMessage, err error)
}

// NewAnthropicGenerator creates a new OpenAI generator with the specified model.
func NewAnthropicGenerator(client AnthropicCompletionService, model, systemInstructions string) AnthropicGenerator {
	return AnthropicGenerator{
		client:             client,
		systemInstructions: systemInstructions,
		model:              model,
		tools:              make(map[string]a.BetaToolParam),
	}
}

var _ Generator = (*AnthropicGenerator)(nil)
var _ ToolRegister = (*AnthropicGenerator)(nil)
