package gai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"  // Register GIF format
	_ "image/jpeg" // Register JPEG format
	_ "image/png"  // Register PNG format
	"iter"
	"math"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	oaissestream "github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/pkoukk/tiktoken-go" // Added for token counting

	oai "github.com/openai/openai-go/v3"
)

const (
	// OpenAIExtraFieldImageWidth is the int pixel-width key in
	// [Block.ExtraFields]. [OpenAiGenerator.Count] uses it when dimensions cannot
	// be decoded from image data.
	OpenAIExtraFieldImageWidth = "width"

	// OpenAIExtraFieldImageHeight is the int pixel-height key in
	// [Block.ExtraFields]. [OpenAiGenerator.Count] uses it with
	// [OpenAIExtraFieldImageWidth].
	OpenAIExtraFieldImageHeight = "height"

	// OpenAIExtraFieldImageDetail is the string [Block.ExtraFields] key for
	// OpenAI image detail. [OpenAiGenerator.Count] recognizes "low" and "high"
	// and defaults an absent or unrecognized value to "high".
	OpenAIExtraFieldImageDetail = "detail"
)

func init() {
	// The tiktoken library does not have the update-to-date mappings of model names to encodings,
	// so we manually add them here.
	tiktoken.MODEL_TO_ENCODING["o3"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o4-mini"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-mini"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-nano"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-2025-04-14"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-mini-2025-04-14"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-nano--2025-04-14"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelO3Mini] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelO3Mini2025_01_31] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelO1] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelO1_2024_12_17] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelO1Preview] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelO1Preview2024_09_12] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelO1Mini] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelO1Mini2024_09_12] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4o] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4o2024_11_20] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4o2024_08_06] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4o2024_05_13] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4oAudioPreview] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4oAudioPreview2024_10_01] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4oAudioPreview2024_12_17] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4oMiniAudioPreview] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4oMiniAudioPreview2024_12_17] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelChatgpt4oLatest] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4oMini] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4oMini2024_07_18] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4Turbo] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4Turbo2024_04_09] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4_0125Preview] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4TurboPreview] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4_1106Preview] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4VisionPreview] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4_0314] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4_0613] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4_32k] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4_32k0314] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT4_32k0613] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT3_5Turbo] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT3_5Turbo16k] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT3_5Turbo0301] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT3_5Turbo0613] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT3_5Turbo1106] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT3_5Turbo0125] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING[oai.ChatModelGPT3_5Turbo16k0613] = tiktoken.MODEL_CL100K_BASE
}

// OpenAiGenerator adapts OpenAI Chat Completions to [Generator],
// [StreamingGenerator], and [TokenCounter]. It accepts text, image, audio, and
// PDF input, produces text or audio, and supports function tools.
//
// It consumes [WithTemperature], [WithTopP], [WithFrequencyPenalty],
// [WithPresencePenalty], [WithCandidateCount], [WithMaxGenerationTokens],
// [WithToolChoice], [WithStopSequences], [WithOutputModalities],
// [WithAudioConfig], and [WithThinkingBudget]. Image token counting can use
// [OpenAIExtraFieldImageWidth], [OpenAIExtraFieldImageHeight], and
// [OpenAIExtraFieldImageDetail].
type OpenAiGenerator struct {
	client OpenAICompletionService
}

// convertToolToOpenAI converts our tool definition to OpenAI's format
func convertToolToOpenAI(tool Tool) (oai.ChatCompletionToolUnionParam, error) {
	// Convert tool schema to OpenAI's JSON schema format
	parameters := make(map[string]interface{})
	if tool.InputSchema != nil {
		// Serialize the schema to JSON then unmarshal into interface{} for OpenAI
		schemaJSON, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return oai.ChatCompletionFunctionTool(oai.FunctionDefinitionParam{}), err
		} else {
			if err := json.Unmarshal(schemaJSON, &parameters); err != nil {
				return oai.ChatCompletionFunctionTool(oai.FunctionDefinitionParam{}), err
			}
		}
	}

	// if the input schema doesn't contain anything but the type, treat as no schema
	// example: {"type": "object"}
	if len(parameters) == 1 && parameters["type"].(string) == "object" {
		parameters = make(map[string]interface{})
	}

	return oai.ChatCompletionFunctionTool(oai.FunctionDefinitionParam{
		Name:        tool.Name,
		Description: oai.String(tool.Description),
		Parameters:  parameters,
	}), nil
}

func convertToolsToOpenAI(tools []Tool) ([]oai.ChatCompletionToolUnionParam, error) {
	converted := make([]oai.ChatCompletionToolUnionParam, 0, len(tools))
	seen := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		if tool.Name == "" {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be empty")}
		}
		if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be %s", tool.Name)}
		}
		if _, exists := seen[tool.Name]; exists {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool already provided")}
		}
		seen[tool.Name] = struct{}{}

		providerTool, err := convertToolToOpenAI(tool)
		if err != nil {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
		}
		converted = append(converted, providerTool)
	}
	return converted, nil
}

type openAIGenerationOptions struct {
	Temperature         *float64
	TopP                *float64
	FrequencyPenalty    *float64
	PresencePenalty     *float64
	CandidateCount      *uint
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	OutputModalities    []Modality
	AudioConfig         AudioConfig
	ThinkingBudget      string
}

// parseOpenAIGenerationOptions validates common option values and records the typed Chat Completions configuration.
func parseOpenAIGenerationOptions(values GenerationOptions) (*openAIGenerationOptions, error) {
	options := &openAIGenerationOptions{}

	temperature, ok, err := generationOption[float64](values, GenerationOptionTemperature)
	if err != nil {
		return nil, err
	}
	if ok {
		options.Temperature = &temperature
	}
	topP, ok, err := generationOption[float64](values, GenerationOptionTopP)
	if err != nil {
		return nil, err
	}
	if ok {
		options.TopP = &topP
	}
	frequencyPenalty, ok, err := generationOption[float64](values, GenerationOptionFrequencyPenalty)
	if err != nil {
		return nil, err
	}
	if ok {
		options.FrequencyPenalty = &frequencyPenalty
	}
	presencePenalty, ok, err := generationOption[float64](values, GenerationOptionPresencePenalty)
	if err != nil {
		return nil, err
	}
	if ok {
		options.PresencePenalty = &presencePenalty
	}
	candidateCount, ok, err := generationOption[uint](values, GenerationOptionCandidateCount)
	if err != nil {
		return nil, err
	}
	if ok {
		options.CandidateCount = &candidateCount
	}
	maxTokens, ok, err := generationOption[int](values, GenerationOptionMaxGenerationTokens)
	if err != nil {
		return nil, err
	}
	if ok {
		options.MaxGenerationTokens = &maxTokens
	}
	if options.ToolChoice, _, err = generationOption[string](values, GenerationOptionToolChoice); err != nil {
		return nil, err
	}
	if options.StopSequences, _, err = generationOption[[]string](values, GenerationOptionStopSequences); err != nil {
		return nil, err
	}
	if options.OutputModalities, _, err = generationOption[[]Modality](values, GenerationOptionOutputModalities); err != nil {
		return nil, err
	}
	if options.AudioConfig, _, err = generationOption[AudioConfig](values, GenerationOptionAudioConfig); err != nil {
		return nil, err
	}
	if options.ThinkingBudget, _, err = generationOption[string](values, GenerationOptionThinkingBudget); err != nil {
		return nil, err
	}
	return options, nil
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

				switch block.MimeType {
				case "application/pdf":
					val, ok := block.ExtraFields[BlockFieldFilenameKey]
					if !ok {
						return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("filename field missing in extra fields")
					}
					filename, ok := val.(string)
					if !ok {
						return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("filename field is not a string")
					}
					parts = append(parts, oai.FileContentPart(oai.ChatCompletionContentPartFileFileParam{
						FileData: oai.String(dataUrl),
						Filename: oai.String(filename),
					}))
				default:
					parts = append(parts, oai.ImageContentPart(oai.ChatCompletionContentPartImageImageURLParam{
						URL: dataUrl,
					}))
				}
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
		var toolCalls []oai.ChatCompletionMessageToolCallUnionParam
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
				// Parse the tool call content as ToolCallInput
				var toolUse ToolCallInput
				if err := json.Unmarshal([]byte(block.Content.String()), &toolUse); err != nil {
					return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("invalid tool call content: %w", err)
				}

				// Convert parameters to JSON string for OpenAI
				argsJSON, err := json.Marshal(toolUse.Parameters)
				if err != nil {
					return oai.ChatCompletionMessageParamUnion{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
				}

				toolCalls = append(toolCalls, oai.ChatCompletionMessageToolCallUnionParam{
					OfFunction: &oai.ChatCompletionMessageFunctionToolCallParam{
						ID: block.ID,
						Function: oai.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name:      toolUse.Name,
							Arguments: string(argsJSON),
						},
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

// Generate sends one Chat Completions request and normalizes candidates, text,
// audio, tool calls, usage, and provider failures into [Response].
func (g *OpenAiGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("openai: client not initialized")
	}

	dialog := request.Dialog
	if len(dialog) == 0 {
		return Response{}, ErrEmptyDialog
	}
	options, err := parseOpenAIGenerationOptions(request.Options)
	if err != nil {
		return Response{}, err
	}
	tools, err := convertToolsToOpenAI(request.Tools)
	if err != nil {
		return Response{}, err
	}
	instructions, err := textInstructions(request.Instructions)
	if err != nil {
		return Response{}, err
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
		Model:    request.Model,
		Messages: messages,
	}

	if len(instructions) > 0 {
		parts := make([]oai.ChatCompletionContentPartTextParam, 0, len(instructions))
		for _, instruction := range instructions {
			parts = append(parts, oai.ChatCompletionContentPartTextParam{Text: instruction})
		}
		params.Messages = append([]oai.ChatCompletionMessageParamUnion{
			oai.SystemMessage(parts),
		}, messages...)
	}

	// Map our options to OpenAI params if options are provided
	if options != nil {
		// Set temperature if non-zero
		if options.Temperature != nil {
			params.Temperature = oai.Float(*options.Temperature)
		}

		// Set top_p if non-zero
		if options.TopP != nil {
			params.TopP = oai.Float(*options.TopP)
		}

		// Set frequency penalty if non-zero
		if options.FrequencyPenalty != nil {
			params.FrequencyPenalty = oai.Float(*options.FrequencyPenalty)
		}

		// Set presence penalty if non-zero
		if options.PresencePenalty != nil {
			params.PresencePenalty = oai.Float(*options.PresencePenalty)
		}

		// Set max tokens if specified
		if options.MaxGenerationTokens != nil {
			params.MaxCompletionTokens = oai.Int(int64(*options.MaxGenerationTokens))
		}

		// Set number of completions if specified
		if options.CandidateCount != nil {
			params.N = oai.Int(int64(*options.CandidateCount))
		}

		// Set stop sequences if specified
		if len(options.StopSequences) > 0 {
			// OpenAI accepts either a single string or array of strings
			if len(options.StopSequences) == 1 {
				params.Stop = oai.ChatCompletionNewParamsStopUnion{OfString: oai.String(options.StopSequences[0])}
			} else {
				params.Stop = oai.ChatCompletionNewParamsStopUnion{OfStringArray: options.StopSequences}
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
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfFunctionToolChoice: &oai.ChatCompletionNamedToolChoiceParam{
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
					Voice: oai.ChatCompletionAudioParamVoiceUnion{
						OfString: param.NewOpt(options.AudioConfig.VoiceName),
					},
					Format: oai.ChatCompletionAudioParamFormat(options.AudioConfig.Format),
				}
			}
		}

		if options.ThinkingBudget != "" {
			params.ReasoningEffort = oai.ReasoningEffort(options.ThinkingBudget)
		}
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	// Make the API call
	resp, err := g.client.New(ctx, params)
	if err != nil {
		if mapped := mapOpenAISDKError(ProviderOpenAI, err); mapped != nil {
			return Response{}, mapped
		}
		return Response{}, fmt.Errorf("failed to create new message: %w", err)
	}

	// Convert OpenAI response to our Response type
	result := Response{
		UsageMetadata: make(Metadata),
	}

	// Add usage metrics if available
	if usage := resp.Usage; usage.PromptTokens > 0 || usage.CompletionTokens > 0 {
		if promptTokens := usage.PromptTokens; promptTokens > 0 {
			result.UsageMetadata[UsageMetricInputTokens] = int(promptTokens)
		}
		if completionTokens := usage.CompletionTokens; completionTokens > 0 {
			result.UsageMetadata[UsageMetricGenerationTokens] = int(completionTokens)
		}
		if usage.PromptTokensDetails.CachedTokens > 0 {
			result.UsageMetadata[UsageMetricCacheReadTokens] = int(usage.PromptTokensDetails.CachedTokens)
		}
	}

	var hasToolCalls bool

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
			hasToolCalls = true
			for _, toolCall := range toolCalls {
				// Create a ToolCallInput with standardized format
				toolUse := ToolCallInput{
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
					Content:      Str(toolUseJSON),
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
		choice := resp.Choices[0]
		// The API documents refusal as an independent assistant-message field.
		// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
		if choice.Message.Refusal != "" {
			result.FinishReason = ContentPolicyViolation
			return result, ContentPolicyErr(choice.Message.Refusal)
		}
		switch choice.FinishReason {
		case "stop":
			result.FinishReason = EndTurn
		case "length":
			result.FinishReason = MaxGenerationLimit
			// Return ErrMaxGenerationLimit when the model reaches its token limit,
			// regardless of whether MaxGenerationTokens was explicitly set
			return result, ErrMaxGenerationLimit
		case "tool_calls":
			result.FinishReason = ToolUse
		case "content_filter":
			// content_filter means output was omitted by the API's content filters.
			// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
			result.FinishReason = ContentPolicyViolation
			return result, ContentPolicyErr("content policy violation detected")
		default:
			result.FinishReason = Unknown
		}
	}

	// some OpenAI providers return an EndTurn stop reason, despite being a tool call
	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}

	return result, nil
}

// Stream starts one Chat Completions stream when iterated and emits ordered
// text, audio, tool-call, usage, and terminal-error [StreamChunk] values.
func (g *OpenAiGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("openai: client not initialized")})
			return
		}

		dialog := request.Dialog
		if len(dialog) == 0 {
			yield(StreamChunk{Err: ErrEmptyDialog})
			return
		}
		options, err := parseOpenAIGenerationOptions(request.Options)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		tools, err := convertToolsToOpenAI(request.Tools)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		instructions, err := textInstructions(request.Instructions)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}

		// Convert each message to OpenAI format
		var messages []oai.ChatCompletionMessageParamUnion
		for _, msg := range dialog {
			oaiMsg, err := toOpenAIMessage(msg)
			if err != nil {
				yield(StreamChunk{Err: fmt.Errorf("failed to convert message: %w", err)})
				return
			}
			messages = append(messages, oaiMsg)
		}

		// Create OpenAI chat completion params
		params := oai.ChatCompletionNewParams{
			Model:    request.Model,
			Messages: messages,
		}

		if len(instructions) > 0 {
			parts := make([]oai.ChatCompletionContentPartTextParam, 0, len(instructions))
			for _, instruction := range instructions {
				parts = append(parts, oai.ChatCompletionContentPartTextParam{Text: instruction})
			}
			params.Messages = append([]oai.ChatCompletionMessageParamUnion{
				oai.SystemMessage(parts),
			}, messages...)
		}

		// Map our options to OpenAI params if options are provided
		if options != nil {
			// Set temperature if non-zero
			if options.Temperature != nil {
				params.Temperature = oai.Float(*options.Temperature)
			}

			// Set top_p if non-zero
			if options.TopP != nil {
				params.TopP = oai.Float(*options.TopP)
			}

			// Set frequency penalty if non-zero
			if options.FrequencyPenalty != nil {
				params.FrequencyPenalty = oai.Float(*options.FrequencyPenalty)
			}

			// Set presence penalty if non-zero
			if options.PresencePenalty != nil {
				params.PresencePenalty = oai.Float(*options.PresencePenalty)
			}

			// Set max tokens if specified
			if options.MaxGenerationTokens != nil {
				params.MaxCompletionTokens = oai.Int(int64(*options.MaxGenerationTokens))
			}

			// Set number of completions if specified
			if options.CandidateCount != nil {
				params.N = oai.Int(int64(*options.CandidateCount))
			}

			// Set stop sequences if specified
			if len(options.StopSequences) > 0 {
				// OpenAI accepts either a single string or array of strings
				if len(options.StopSequences) == 1 {
					params.Stop = oai.ChatCompletionNewParamsStopUnion{OfString: oai.String(options.StopSequences[0])}
				} else {
					params.Stop = oai.ChatCompletionNewParamsStopUnion{OfStringArray: options.StopSequences}
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
					params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfFunctionToolChoice: &oai.ChatCompletionNamedToolChoiceParam{
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
						yield(StreamChunk{Err: UnsupportedOutputModalityErr("image output not supported by model")})
						return
					case Video:
						yield(StreamChunk{Err: UnsupportedOutputModalityErr("video output not supported by model")})
						return
					}
				}
				params.Modalities = modalities

				// Set audio configuration if audio output is requested
				if hasAudio {
					if options.AudioConfig.VoiceName == "" {
						yield(StreamChunk{Err: InvalidParameterErr{
							Parameter: "AudioConfig.VoiceName",
							Reason:    "voice name is required for audio output",
						}})
						return
					}
					if options.AudioConfig.Format == "" {
						yield(StreamChunk{Err: InvalidParameterErr{
							Parameter: "AudioConfig.Format",
							Reason:    "format is required for audio output",
						}})
						return
					}
					params.Audio = oai.ChatCompletionAudioParam{
						Voice: oai.ChatCompletionAudioParamVoiceUnion{
							OfString: param.NewOpt(options.AudioConfig.VoiceName),
						},
						Format: oai.ChatCompletionAudioParamFormat(options.AudioConfig.Format),
					}
				}
			}

			if options.ThinkingBudget != "" {
				params.ReasoningEffort = oai.ReasoningEffort(options.ThinkingBudget)
			}
		}

		if len(tools) > 0 {
			params.Tools = tools
		}

		// Enable usage in streaming response
		params.StreamOptions = oai.ChatCompletionStreamOptionsParam{
			IncludeUsage: oai.Bool(true),
		}

		// Start the stream
		stream := g.client.NewStreaming(ctx, params)
		defer stream.Close()

		var finalUsage *oai.CompletionUsage

		for stream.Next() {
			chunk := stream.Current()

			// Capture usage if present (only in final chunk)
			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				finalUsage = &chunk.Usage
			}

			// Final chunk may have usage but no choices
			if len(chunk.Choices) == 0 {
				continue
			}

			if len(chunk.Choices) > 1 {
				panic("choices > 1 not supported")
			}

			switch chunk.Choices[0].FinishReason {
			case "length":
				yield(StreamChunk{Err: ErrMaxGenerationLimit})
				return
			case "content_filter":
				yield(StreamChunk{Err: ContentPolicyErr("could not produce response")})
				return
			}

			if chunk.Choices[0].Delta.Refusal != "" {
				yield(StreamChunk{Err: ContentPolicyErr(chunk.Choices[0].Delta.Refusal)})
				return
			}

			if chunk.Choices[0].Delta.Content != "" {
				if !yield(StreamChunk{
					Block: Block{
						BlockType:    Content,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str(chunk.Choices[0].Delta.Content),
					},
					CandidatesIndex: int(chunk.Choices[0].Index),
				}) {
					return
				}
			}

			if len(chunk.Choices[0].Delta.ToolCalls) > 1 {
				panic("tool call > 1 not supported")
			}

			if len(chunk.Choices[0].Delta.ToolCalls) == 1 {
				toolCall := chunk.Choices[0].Delta.ToolCalls[0]
				if toolCall.Function.Name != "" {
					if !yield(StreamChunk{
						Block: Block{
							ID:           toolCall.ID,
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(toolCall.Function.Name),
						},
						CandidatesIndex: int(chunk.Choices[0].Index),
					}) {
						return
					}
				}
				if toolCall.Function.Arguments != "" {
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(toolCall.Function.Arguments),
						},
						CandidatesIndex: int(chunk.Choices[0].Index),
					}) {
						return
					}
				}
			}
		}

		// Check for stream errors
		if stream.Err() != nil {
			if mapped := mapOpenAISDKError(ProviderOpenAI, stream.Err()); mapped != nil {
				yield(StreamChunk{Err: mapped})
			} else {
				yield(StreamChunk{Err: stream.Err()})
			}
			return
		}

		// Emit metadata block as final block if we have usage data
		if finalUsage != nil {
			metadata := make(Metadata)

			if finalUsage.PromptTokens > 0 {
				metadata[UsageMetricInputTokens] = int(finalUsage.PromptTokens)
			}
			if finalUsage.CompletionTokens > 0 {
				metadata[UsageMetricGenerationTokens] = int(finalUsage.CompletionTokens)
			}
			if finalUsage.PromptTokensDetails.CachedTokens > 0 {
				metadata[UsageMetricCacheReadTokens] = int(finalUsage.PromptTokensDetails.CachedTokens)
			}

			if len(metadata) > 0 {
				yield(StreamChunk{
					Block:           MetadataBlock(metadata),
					CandidatesIndex: 0,
				})
			}
		}
	}
}

// OpenAICompletionService is the Chat Completions SDK subset required by
// [OpenAiGenerator]. The OpenAI SDK's client.Chat.Completions service satisfies
// this interface.
type OpenAICompletionService interface {
	// New performs one non-streaming Chat Completions request.
	New(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) (res *oai.ChatCompletion, err error)
	// NewStreaming starts one streaming Chat Completions request.
	NewStreaming(ctx context.Context, body oai.ChatCompletionNewParams, opts ...option.RequestOption) (stream *oaissestream.Stream[oai.ChatCompletionChunk])
}

// NewOpenAiGenerator returns a stateless OpenAI Chat Completions adapter backed
// by client. The OpenAI SDK's client.Chat.Completions service satisfies
// [OpenAICompletionService].
func NewOpenAiGenerator(client OpenAICompletionService) *OpenAiGenerator {
	return &OpenAiGenerator{client: client}
}

var _ Generator = (*OpenAiGenerator)(nil)
var _ StreamingGenerator = (*OpenAiGenerator)(nil)
var _ TokenCounter = (*OpenAiGenerator)(nil)
var _ OpenAICompletionService = (*oai.ChatCompletionService)(nil)

// calculateImageTokens calculates the number of tokens used by an image block
// based on OpenAI's token calculation rules for different models.
//
// The function first attempts to extract dimensions directly from the image data by:
//  1. Decoding the base64-encoded image content
//  2. Using Go's image package to determine width and height
//
// If extraction from image data fails, it tries to get dimensions from ExtraFields.
// If dimensions still cannot be determined, an error is returned.
//
// PDFs are not supported for token counting as they are converted to images server-side
// and exact dimensions cannot be determined.
//
// The detail level ("high" or "low") is determined from ExtraFields if specified,
// with "high" being the default.
//
// Token calculation depends on the model:
//   - For minimal models (GPT-4.1-mini, GPT-4.1-nano, o4-mini), it uses the 32px patch method
//   - For standard models (GPT-4o, etc.), it uses the base+tile method with detail level consideration
//
// Returns:
//   - The number of tokens as an integer
//   - An error if dimensions cannot be determined or if calculation fails
func (g *OpenAiGenerator) calculateImageTokens(block Block, model string) (int, error) {
	if block.MimeType == "application/pdf" {
		return 0, fmt.Errorf("PDF token counting is not supported")
	}

	var width, height int
	var detail string = "high" // Default to high detail

	// Try to extract image dimensions directly from the image data
	imgData, err := base64.StdEncoding.DecodeString(block.Content.String())
	if err == nil {
		// Successfully decoded base64, now try to extract dimensions
		imgReader := bytes.NewReader(imgData)
		config, _, err := image.DecodeConfig(imgReader)
		if err == nil {
			// Successfully extracted dimensions
			width = config.Width
			height = config.Height
		}
	}

	// If we couldn't extract dimensions from the image, try the ExtraFields
	if width == 0 || height == 0 {
		if block.ExtraFields != nil {
			if w, ok := block.ExtraFields[OpenAIExtraFieldImageWidth].(int); ok {
				width = w
			}
			if h, ok := block.ExtraFields[OpenAIExtraFieldImageHeight].(int); ok {
				height = h
			}
		}
	}

	// Return an error if we still couldn't determine dimensions
	if width == 0 || height == 0 {
		return 0, fmt.Errorf("could not determine image dimensions for token calculation")
	}

	// Get detail level from ExtraFields if specified
	if block.ExtraFields != nil {
		if d, ok := block.ExtraFields[OpenAIExtraFieldImageDetail].(string); ok && (d == "low" || d == "high") {
			detail = d
		}
	}

	// Determine which calculation method to use based on model
	if isMinimalModel(model) {
		return calculateMinimalModelImageTokens(width, height, model)
	} else {
		return calculateStandardModelImageTokens(width, height, detail, model)
	}
}

// isMinimalModel checks if the model uses the 32px patch calculation method.
// It returns true for models that use patch-based calculation like gpt-4.1-mini,
// gpt-4.1-nano, and o4-mini.
//
// These models use a different token calculation algorithm that divides images
// into 32x32 pixel patches and applies model-specific multipliers.
func isMinimalModel(model string) bool {
	minimalModels := []string{"gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"}
	for _, m := range minimalModels {
		if strings.Contains(model, m) {
			return true
		}
	}
	return false
}

// calculateMinimalModelImageTokens calculates image tokens for minimal models
// (GPT-4.1-mini, GPT-4.1-nano, o4-mini) based on OpenAI's token calculation rules.
//
// The calculation follows these steps:
//  1. Calculate the number of 32x32 pixel patches needed to cover the image
//  2. If the number exceeds 1536, scale the image to fit within that limit
//  3. Apply a model-specific multiplier to the patch count
//
// Multipliers:
//   - GPT-4.1-mini: 1.62
//   - GPT-4.1-nano: 2.46
//   - o4-mini:      1.72
//
// The result is the final token count for the image.
func calculateMinimalModelImageTokens(width, height int, model string) (int, error) {
	// Calculate patches based on 32px x 32px
	patchesWidth := (width + 32 - 1) / 32   // Ceiling division
	patchesHeight := (height + 32 - 1) / 32 // Ceiling division

	totalPatches := patchesWidth * patchesHeight

	// If exceeds 1536, scale down
	if totalPatches > 1536 {
		// We need to scale down the image while preserving aspect ratio
		// Calculate shrink factor
		shrinkFactor := math.Sqrt(float64(1536*32*32) / float64(width*height))

		// Apply shrink factor
		scaledWidth := int(float64(width) * shrinkFactor)
		scaledHeight := int(float64(height) * shrinkFactor)

		if scaledWidth%32 != 0 {
			widthPatches := float64(scaledWidth) / 32.0
			shrinkFactor = math.Floor(widthPatches) / widthPatches

			// Apply shrink factor with dimensions that fit in a whole patch size (for width)
			scaledWidth = int(float64(scaledWidth) * shrinkFactor)
			scaledHeight = int(float64(scaledHeight) * shrinkFactor)
		}

		// Recalculate patches
		patchesWidth = (scaledWidth + 32 - 1) / 32
		patchesHeight = (scaledHeight + 32 - 1) / 32

		totalPatches = patchesWidth * patchesHeight
	}

	// Apply multiplier based on model
	switch {
	case strings.Contains(model, "gpt-4.1-mini"):
		return int(float64(totalPatches) * 1.62), nil
	case strings.Contains(model, "gpt-4.1-nano"):
		return int(float64(totalPatches) * 2.46), nil
	case strings.Contains(model, "o4-mini"):
		return int(float64(totalPatches) * 1.72), nil
	default:
		// Default to no multiplier if model is in minimal category but not recognized
		return totalPatches, nil
	}
}

// calculateStandardModelImageTokens calculates image tokens for standard models
// (GPT-4o, GPT-4.1, etc.) based on OpenAI's token calculation rules.
//
// For "low" detail images, it returns a fixed base token count dependent on the model.
//
// For "high" detail images, the calculation follows these steps:
//  1. Scale the image to fit within a 2048x2048 square if necessary
//  2. Scale so the shortest side is 768px
//  3. Count the number of 512px tiles needed to cover the image
//  4. Calculate tokens as: base_tokens + (tile_tokens * number_of_tiles)
//
// Base and tile token counts vary by model and are retrieved using getTokensForModel().
//
// Returns the calculated token count for the image.
func calculateStandardModelImageTokens(width, height int, detail string, model string) (int, error) {
	// For low detail, return base tokens
	if detail == "low" {
		baseTokens, _ := getTokensForModel(model)
		return baseTokens, nil
	}

	// For high detail, we need to calculate based on tiles
	// First, get base and tile tokens for the model
	baseTokens, tileTokens := getTokensForModel(model)

	// Scale to fit in 2048px x 2048px square if needed
	scaledWidth, scaledHeight := width, height
	if width > 2048 || height > 2048 {
		// Scale down preserving aspect ratio
		ratio := float64(2048) / math.Max(float64(width), float64(height))
		scaledWidth = int(float64(width) * ratio)
		scaledHeight = int(float64(height) * ratio)
	}

	// Scale so shortest side is 768px
	ratio := float64(768) / math.Min(float64(scaledWidth), float64(scaledHeight))
	scaledWidth = int(float64(scaledWidth) * ratio)
	scaledHeight = int(float64(scaledHeight) * ratio)

	// Count 512px tiles
	tilesWidth := (scaledWidth + 512 - 1) / 512   // Ceiling division
	tilesHeight := (scaledHeight + 512 - 1) / 512 // Ceiling division

	totalTiles := tilesWidth * tilesHeight

	// Calculate total tokens
	return baseTokens + (tileTokens * totalTiles), nil
}

// getTokensForModel returns the base tokens and tile tokens for a given model.
// These values are used in token calculations for standard models (non-minimal models)
// according to OpenAI's documentation.
//
// Model-specific token values:
//   - GPT-4o, GPT-4.1, GPT-4.5:        base=85,  tile=170
//   - GPT-4o-mini:                      base=2833, tile=5667
//   - o1, o1-pro, o3:                   base=75,  tile=150
//   - computer-use-preview:             base=65,  tile=129
//   - Default (unrecognized models):    base=85,  tile=170 (Same as GPT-4o)
//
// Returns the base tokens and tile tokens as integers.
func getTokensForModel(model string) (baseTokens, tileTokens int) {
	switch {
	case strings.Contains(model, "4o") ||
		strings.Contains(model, "4.1") ||
		strings.Contains(model, "4.5"):
		return 85, 170
	case strings.Contains(model, "4o-mini"):
		return 2833, 5667
	case strings.Contains(model, "o1") ||
		strings.Contains(model, "o1-pro") ||
		strings.Contains(model, "o3"):
		return 75, 150
	case strings.Contains(model, "computer-use-preview"):
		return 65, 129
	default:
		// Default to GPT-4o values
		return 85, 170
	}
}

// Count uses tiktoken to count the request model, instructions, dialog, and
// tools without a provider call.
//
// Image counts use dimensions decoded from image data or supplied through
// [OpenAIExtraFieldImageWidth] and [OpenAIExtraFieldImageHeight]. [PDFBlock]
// counting is unsupported because server-side page conversion is not observable.
func (g *OpenAiGenerator) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
	}

	dialog := request.Dialog
	if len(dialog) == 0 {
		return 0, ErrEmptyDialog
	}

	tke, err := tiktoken.EncodingForModel(request.Model)
	if err != nil {
		tke, err = tiktoken.GetEncoding(tiktoken.MODEL_O200K_BASE) // Fallback
		if err != nil {
			return 0, fmt.Errorf("failed to get tiktoken encoding: %w", err)
		}
	}

	var totalTokens int

	instructions, err := textInstructions(request.Instructions)
	if err != nil {
		return 0, err
	}
	for _, instruction := range instructions {
		totalTokens += len(tke.Encode(instruction, nil, nil))
	}

	// See https://platform.openai.com/docs/guides/images-vision?api-mode=chat#calculating-costs
	for _, msg := range dialog {
		for _, block := range msg.Blocks {
			switch block.ModalityType {
			case Text:
				contentToTokenize := block.Content.String()
				totalTokens += len(tke.Encode(contentToTokenize, nil, nil))
			case Image:
				// Extract dimensions from the image content
				imageTokens, err := g.calculateImageTokens(block, request.Model)
				if err != nil {
					return 0, fmt.Errorf("failed to calculate image tokens: %w", err)
				}
				totalTokens += imageTokens
			default:
				return 0, UnsupportedOutputModalityErr("unsupported modality")
			}
		}
	}

	tools, err := convertToolsToOpenAI(request.Tools)
	if err != nil {
		return 0, err
	}
	for _, tool := range tools {
		var toolDefStr string
		toolDefStr += tool.OfFunction.Function.Name + "\n"
		toolDefStr += tool.OfFunction.Function.Description.String() + "\n"
		paramsJSON, err := json.Marshal(tool.OfFunction.Function.Parameters)
		if err == nil {
			toolDefStr += string(paramsJSON) + "\n"
		}
		totalTokens += len(tke.Encode(toolDefStr, nil, nil))
	}

	return uint(totalTokens), nil
}
