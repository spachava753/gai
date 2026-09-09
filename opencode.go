package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"strings"

	"github.com/go-faster/jx"

	"github.com/spachava753/gai/internal/opencode"
)

const (
	// OpenCodeGenerationOptionSessionID is the string [GenerationOptions] key set
	// by [WithOpenCodeSessionID].
	OpenCodeGenerationOptionSessionID = "opencode_session_id"

	// OpenCodeDefaultBaseURL is the endpoint used by [NewOpenCodeGenerator] when
	// baseURL is empty.
	OpenCodeDefaultBaseURL = string(opencode.DefaultServer)

	// OpenCodeResponseExtraFieldID is the string completion-ID key in
	// [Response.ExtraFields].
	OpenCodeResponseExtraFieldID = "opencode_id"
	// OpenCodeResponseExtraFieldModel is the string response-model key in
	// [Response.ExtraFields].
	OpenCodeResponseExtraFieldModel = "opencode_model"
	// OpenCodeResponseExtraFieldCreated is the int64 Unix timestamp key in
	// [Response.ExtraFields].
	OpenCodeResponseExtraFieldCreated = "opencode_created"
	// OpenCodeResponseExtraFieldSystemFingerprint is the string backend
	// fingerprint key in [Response.ExtraFields].
	OpenCodeResponseExtraFieldSystemFingerprint = "opencode_system_fingerprint"
	// OpenCodeResponseExtraFieldCost is the string request-cost key in
	// [Response.ExtraFields]. OpenCode reports "0" for subscription usage.
	OpenCodeResponseExtraFieldCost = "opencode_cost"

	// OpenCodeExtraFieldReasoningField is the string [Block.ExtraFields] key that
	// records whether OpenCode returned reasoning_content, reasoning, or
	// reasoning_details. [OpenCodeGenerator] uses it when replaying an assistant
	// message.
	OpenCodeExtraFieldReasoningField = "opencode_reasoning_field"
	// OpenCodeExtraFieldReasoningDetail is the map[string]any [Block.ExtraFields]
	// key containing one structured reasoning_details item for exact replay.
	OpenCodeExtraFieldReasoningDetail = "opencode_reasoning_detail"
)

// WithOpenCodeSessionID keeps every request in one logical conversation on
// OpenCode's sticky upstream provider. Reuse the value when replaying assistant
// reasoning and tool calls across generation requests.
func WithOpenCodeSessionID(value string) GenerationOption {
	return func(options GenerationOptions) { options[OpenCodeGenerationOptionSessionID] = value }
}

// OpenCodeGenerator adapts OpenCode Chat Completions to [Generator] and
// [StreamingGenerator]. It accepts text plus PNG, JPEG, GIF, or WebP
// [ImageBlock] input, produces text, supports function tools, and preserves
// reasoning_content plus structured reasoning_details on assistant messages for
// tool-call replay. Model IDs and vision capability are validated by OpenCode because its catalog can change
// independently of this package.
//
// OpenCode consumes [WithTemperature], [WithTopP], [WithFrequencyPenalty],
// [WithPresencePenalty], [WithMaxGenerationTokens], [WithToolChoice],
// [WithStopSequences], [WithOutputModalities], [WithThinkingBudget], and
// [WithOpenCodeSessionID]. The thinking budget is passed through as the
// model-specific reasoning_effort string. Reuse one OpenCode session ID across
// a dialog so OpenCode keeps its requests on the same upstream provider.
// Invocation details use the OpenCodeResponseExtraField constants in
// [Response.ExtraFields].
//
// OpenCode API keys come from an OpenCode Go subscription. This constructor
// does not read environment variables.
type OpenCodeGenerator struct {
	client *opencode.Client
}

// NewOpenCodeGenerator constructs a stateless OpenCode adapter. A nil
// httpClient uses the generated client's default transport, an empty baseURL
// uses [OpenCodeDefaultBaseURL], and an empty apiKey returns [ErrMissingAPIKey].
func NewOpenCodeGenerator(httpClient *http.Client, baseURL, apiKey string) (*OpenCodeGenerator, error) {
	if baseURL == "" {
		baseURL = OpenCodeDefaultBaseURL
	}
	if apiKey == "" {
		return nil, fmt.Errorf("opencode: %w", ErrMissingAPIKey)
	}
	options := make([]opencode.ClientOption, 0, 1)
	if httpClient != nil {
		options = append(options, opencode.WithClient(httpClient))
	}
	client, err := opencode.NewClient(baseURL, openCodeSecuritySource{apiKey: apiKey}, options...)
	if err != nil {
		return nil, fmt.Errorf("opencode: create client: %w", err)
	}
	return &OpenCodeGenerator{client: client}, nil
}

type openCodeSecuritySource struct {
	apiKey string
}

func (s openCodeSecuritySource) BearerAuth(ctx context.Context, operationName opencode.OperationName) (opencode.BearerAuth, error) {
	return opencode.BearerAuth{Token: s.apiKey}, nil
}

type openCodeGenerationOptions struct {
	Temperature         *float64
	TopP                *float64
	FrequencyPenalty    *float64
	PresencePenalty     *float64
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	OutputModalities    []Modality
	ReasoningEffort     string
	SessionID           string
}

// parseOpenCodeGenerationOptions validates recognized common options and records a typed request snapshot.
func parseOpenCodeGenerationOptions(values GenerationOptions) (*openCodeGenerationOptions, error) {
	options := &openCodeGenerationOptions{}

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
	if options.ReasoningEffort, _, err = generationOption[string](values, GenerationOptionThinkingBudget); err != nil {
		return nil, err
	}
	if options.SessionID, _, err = generationOption[string](values, OpenCodeGenerationOptionSessionID); err != nil {
		return nil, err
	}
	return options, nil
}

func convertToolToOpenCode(tool Tool) (opencode.FunctionTool, error) {
	parameters := opencode.FunctionDefinitionParameters{
		"type":       jx.Raw(`"object"`),
		"properties": jx.Raw(`{}`),
	}
	if tool.InputSchema != nil {
		schemaJSON, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return opencode.FunctionTool{}, err
		}
		var rawParameters map[string]json.RawMessage
		if err := json.Unmarshal(schemaJSON, &rawParameters); err != nil {
			return opencode.FunctionTool{}, err
		}
		parameters = make(opencode.FunctionDefinitionParameters, len(rawParameters))
		for name, raw := range rawParameters {
			parameters[name] = jx.Raw(raw)
		}
	}
	return opencode.FunctionTool{
		Type: opencode.FunctionToolTypeFunction,
		Function: opencode.FunctionDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  parameters,
		},
	}, nil
}

func convertToolsToOpenCode(tools []Tool) ([]opencode.FunctionTool, error) {
	converted := make([]opencode.FunctionTool, 0, len(tools))
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

		providerTool, err := convertToolToOpenCode(tool)
		if err != nil {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
		}
		converted = append(converted, providerTool)
	}
	return converted, nil
}

// buildOpenCodeMessages prepends instructions and converts each dialog role, including multimodal user input and replayable reasoning.
func buildOpenCodeMessages(request GenerationRequest) ([]opencode.Message, error) {
	messages := make([]opencode.Message, 0, len(request.Dialog)+1)
	instructions, err := joinedTextInstructions(request.Instructions)
	if err != nil {
		return nil, err
	}
	if instructions != "" {
		messages = append(messages, opencode.NewSystemMessageMessage(opencode.SystemMessage{
			Role:    opencode.SystemMessageRoleSystem,
			Content: instructions,
		}))
	}

	for i, message := range request.Dialog {
		switch message.Role {
		case User:
			content, err := openCodeUserContent(message.Blocks)
			if err != nil {
				return nil, err
			}
			messages = append(messages, opencode.NewUserMessageMessage(opencode.UserMessage{
				Role:    opencode.UserMessageRoleUser,
				Content: content,
			}))
		case Assistant:
			assistant, err := buildOpenCodeAssistantMessage(message.Blocks)
			if err != nil {
				return nil, err
			}
			messages = append(messages, opencode.NewAssistantMessageMessage(assistant))
		case ToolResult:
			if len(message.Blocks) == 0 {
				return nil, fmt.Errorf("opencode: tool result message must have at least one block")
			}
			for _, block := range message.Blocks {
				if block.ID == "" {
					return nil, fmt.Errorf("opencode: tool result block must have a tool call ID")
				}
				if block.BlockType != Content {
					return nil, fmt.Errorf("opencode: unsupported tool result block type %q", block.BlockType)
				}
				if block.ModalityType != Text {
					return nil, UnsupportedInputModalityErr(block.ModalityType.String())
				}
				messages = append(messages, opencode.NewToolMessageMessage(opencode.ToolMessage{
					Role:       opencode.ToolMessageRoleTool,
					ToolCallID: block.ID,
					Content:    block.Content.String(),
				}))
			}
		default:
			return nil, fmt.Errorf("opencode: unsupported role at index %d: %v", i, message.Role)
		}
	}
	return messages, nil
}

func openCodeUserContent(blocks []Block) (opencode.UserMessageContent, error) {
	var text strings.Builder
	parts := make([]opencode.UserContentPart, 0, len(blocks))
	var hasImage bool

	for _, block := range blocks {
		if block.BlockType != Content {
			return opencode.UserMessageContent{}, fmt.Errorf("opencode: unsupported block type for user: %q", block.BlockType)
		}
		switch block.ModalityType {
		case Text:
			value := block.Content.String()
			text.WriteString(value)
			parts = append(parts, opencode.NewTextContentPartUserContentPart(opencode.TextContentPart{
				Type: opencode.TextContentPartTypeText,
				Text: value,
			}))
		case Image:
			switch block.MimeType {
			case "image/png", "image/jpeg", "image/gif", "image/webp":
			default:
				return opencode.UserMessageContent{}, fmt.Errorf("opencode: unsupported image MIME type %q", block.MimeType)
			}
			hasImage = true
			parts = append(parts, opencode.NewImageContentPartUserContentPart(opencode.ImageContentPart{
				Type: opencode.ImageContentPartTypeImageURL,
				ImageURL: opencode.ImageURL{
					URL: fmt.Sprintf("data:%s;base64,%s", block.MimeType, block.Content.String()),
				},
			}))
		default:
			return opencode.UserMessageContent{}, UnsupportedInputModalityErr(block.ModalityType.String())
		}
	}
	if !hasImage {
		return opencode.NewStringUserMessageContent(text.String()), nil
	}
	return opencode.NewUserContentPartArrayUserMessageContent(parts), nil
}

// buildOpenCodeAssistantMessage collects visible text, reasoning, and validated function calls into one replayable assistant turn.
func buildOpenCodeAssistantMessage(blocks []Block) (opencode.AssistantMessage, error) {
	message := opencode.AssistantMessage{Role: opencode.AssistantMessageRoleAssistant}
	var content strings.Builder
	var reasoningContent strings.Builder
	var reasoning strings.Builder
	var hasContent bool

	for _, block := range blocks {
		switch block.BlockType {
		case Content:
			if block.ModalityType != Text {
				return opencode.AssistantMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
			hasContent = true
			content.WriteString(block.Content.String())
		case Thinking:
			if block.ModalityType != Text {
				return opencode.AssistantMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
			value := block.Content.String()
			field, _ := block.ExtraFields[OpenCodeExtraFieldReasoningField].(string)
			switch field {
			case "reasoning", "reasoning_details":
				reasoning.WriteString(value)
			default:
				reasoningContent.WriteString(value)
			}
			if field == "reasoning_details" {
				rawDetail, ok := block.ExtraFields[OpenCodeExtraFieldReasoningDetail].(map[string]any)
				if !ok {
					return opencode.AssistantMessage{}, fmt.Errorf("opencode: reasoning detail block missing structured replay metadata")
				}
				encoded, err := json.Marshal(rawDetail)
				if err != nil {
					return opencode.AssistantMessage{}, fmt.Errorf("opencode: encode reasoning detail replay metadata: %w", err)
				}
				var detail opencode.ReasoningDetail
				if err := json.Unmarshal(encoded, &detail); err != nil {
					return opencode.AssistantMessage{}, fmt.Errorf("opencode: decode reasoning detail replay metadata: %w", err)
				}
				message.ReasoningDetails = append(message.ReasoningDetails, detail)
			}
		case ToolCall:
			if block.ID == "" {
				return opencode.AssistantMessage{}, fmt.Errorf("opencode: tool call block missing ID")
			}
			var input ToolCallInput
			if err := json.Unmarshal([]byte(block.Content.String()), &input); err != nil {
				return opencode.AssistantMessage{}, fmt.Errorf("opencode: invalid tool call content: %w", err)
			}
			arguments, err := json.Marshal(input.Parameters)
			if err != nil {
				return opencode.AssistantMessage{}, fmt.Errorf("opencode: marshal tool arguments: %w", err)
			}
			message.ToolCalls = append(message.ToolCalls, opencode.ToolCall{
				ID:   block.ID,
				Type: opencode.ToolCallTypeFunction,
				Function: opencode.ToolCallFunction{
					Name:      input.Name,
					Arguments: string(arguments),
				},
			})
		default:
			return opencode.AssistantMessage{}, fmt.Errorf("opencode: unsupported assistant block type %q", block.BlockType)
		}
	}
	if hasContent {
		message.Content = opencode.NewOptNilString(content.String())
	} else {
		message.Content.SetToNull()
	}
	if reasoningContent.Len() > 0 {
		message.ReasoningContent = opencode.NewOptString(reasoningContent.String())
	}
	if reasoning.Len() > 0 {
		message.Reasoning = opencode.NewOptString(reasoning.String())
	}
	return message, nil
}

// buildRequest converts request-scoped messages, tools, options, and headers into the generated OpenCode request.
func (g *OpenCodeGenerator) buildRequest(generationRequest GenerationRequest, stream bool) (*opencode.ChatCompletionRequest, opencode.CreateChatCompletionParams, error) {
	options, err := parseOpenCodeGenerationOptions(generationRequest.Options)
	if err != nil {
		return nil, opencode.CreateChatCompletionParams{}, err
	}
	params := opencode.CreateChatCompletionParams{}
	if options.SessionID != "" {
		params.XOpencodeSession = opencode.NewOptString(options.SessionID)
	}
	for _, modality := range options.OutputModalities {
		if modality != Text {
			return nil, params, UnsupportedOutputModalityErr(modality.String())
		}
	}
	messages, err := buildOpenCodeMessages(generationRequest)
	if err != nil {
		return nil, params, err
	}
	tools, err := convertToolsToOpenCode(generationRequest.Tools)
	if err != nil {
		return nil, params, err
	}

	request := &opencode.ChatCompletionRequest{
		Model:    generationRequest.Model,
		Messages: messages,
		Tools:    tools,
		Stream:   opencode.NewOptBool(stream),
	}
	if stream {
		request.StreamOptions = opencode.NewOptStreamOptions(opencode.StreamOptions{
			IncludeUsage: opencode.NewOptBool(true),
		})
	}
	if options.Temperature != nil {
		request.Temperature = opencode.NewOptFloat64(*options.Temperature)
	}
	if options.TopP != nil {
		request.TopP = opencode.NewOptFloat64(*options.TopP)
	}
	if options.FrequencyPenalty != nil {
		request.FrequencyPenalty = opencode.NewOptFloat64(*options.FrequencyPenalty)
	}
	if options.PresencePenalty != nil {
		request.PresencePenalty = opencode.NewOptFloat64(*options.PresencePenalty)
	}
	if options.MaxGenerationTokens != nil {
		request.MaxTokens = opencode.NewOptInt(*options.MaxGenerationTokens)
	}
	if len(options.StopSequences) == 1 {
		request.Stop = opencode.NewOptStop(opencode.NewStringStop(options.StopSequences[0]))
	} else if len(options.StopSequences) > 1 {
		request.Stop = opencode.NewOptStop(opencode.NewStringArrayStop(options.StopSequences))
	}
	if options.ReasoningEffort != "" {
		request.ReasoningEffort = opencode.NewOptString(options.ReasoningEffort)
	}
	if err := applyOpenCodeToolChoice(request, options.ToolChoice, generationRequest.Tools); err != nil {
		return nil, params, err
	}
	return request, params, nil
}

func applyOpenCodeToolChoice(request *opencode.ChatCompletionRequest, choice string, tools []Tool) error {
	if choice == "" {
		return nil
	}
	if choice == ToolChoiceToolsRequired && len(tools) == 0 {
		return InvalidToolChoiceErr("required needs at least one tool")
	}
	if choice == "none" || choice == ToolChoiceAuto || choice == ToolChoiceToolsRequired {
		request.ToolChoice = opencode.NewOptToolChoice(
			opencode.NewToolChoice0ToolChoice(opencode.ToolChoice0(choice)),
		)
		return nil
	}
	for _, tool := range tools {
		if tool.Name == choice {
			request.ToolChoice = opencode.NewOptToolChoice(
				opencode.NewNamedToolChoiceToolChoice(opencode.NamedToolChoice{
					Type: opencode.NamedToolChoiceTypeFunction,
					Function: opencode.NamedToolChoiceFunction{
						Name: choice,
					},
				}),
			)
			return nil
		}
	}
	return InvalidToolChoiceErr(fmt.Sprintf("tool %q is not in the request", choice))
}

func openCodeResponseExtraFields(id, model opencode.OptString, created opencode.OptInt64, systemFingerprint, cost opencode.OptString) map[string]interface{} {
	extraFields := make(map[string]interface{})
	if value, ok := id.Get(); ok {
		extraFields[OpenCodeResponseExtraFieldID] = value
	}
	if value, ok := model.Get(); ok {
		extraFields[OpenCodeResponseExtraFieldModel] = value
	}
	if value, ok := created.Get(); ok {
		extraFields[OpenCodeResponseExtraFieldCreated] = value
	}
	if value, ok := systemFingerprint.Get(); ok {
		extraFields[OpenCodeResponseExtraFieldSystemFingerprint] = value
	}
	if value, ok := cost.Get(); ok {
		extraFields[OpenCodeResponseExtraFieldCost] = value
	}
	return extraFields
}

// Generate sends one OpenCode Chat Completions request and normalizes text,
// reasoning, tool calls, usage, cost, and provider failures into [Response].
func (g *OpenCodeGenerator) Generate(ctx context.Context, generationRequest GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("opencode: client not initialized")
	}
	if len(generationRequest.Dialog) == 0 {
		return Response{}, ErrEmptyDialog
	}
	request, params, err := g.buildRequest(generationRequest, false)
	if err != nil {
		return Response{}, err
	}
	rawResponse, err := g.client.CreateChatCompletion(ctx, request, params)
	if err != nil {
		return Response{}, mapOpenCodeError(err)
	}
	response, ok := rawResponse.(*opencode.ChatCompletionResponse)
	if !ok {
		if stream, isStream := rawResponse.(*opencode.CreateChatCompletionOKTextEventStream); isStream {
			_ = stream.Close()
		}
		return Response{}, fmt.Errorf("opencode: expected JSON completion response, got %T", rawResponse)
	}

	result := Response{
		UsageMetadata: make(Metadata),
		ExtraFields: openCodeResponseExtraFields(
			response.ID,
			response.Model,
			response.Created,
			response.SystemFingerprint,
			response.Cost,
		),
	}
	if usage, ok := response.Usage.Get(); ok {
		addOpenCodeUsageMetadata(result.UsageMetadata, usage)
	}
	var hasToolCalls bool
	for _, choice := range response.Choices {
		blocks := make([]Block, 0, 2)
		if message, ok := choice.Message.Get(); ok {
			hasReasoningDetails := false
			for _, detail := range message.ReasoningDetails {
				block, ok, detailErr := openCodeReasoningDetailBlock(detail)
				if detailErr != nil {
					return result, detailErr
				}
				if ok {
					hasReasoningDetails = true
					blocks = append(blocks, block)
				}
			}
			if !hasReasoningDetails {
				if reasoning, ok := message.ReasoningContent.Get(); ok && reasoning != "" {
					blocks = append(blocks, openCodeThinkingBlock(reasoning))
				} else if reasoning, ok := message.Reasoning.Get(); ok && reasoning != "" {
					blocks = append(blocks, openCodeReasoningFieldBlock(reasoning, "reasoning"))
				}
			}
			if content, ok := message.Content.Get(); ok && content != "" {
				blocks = append(blocks, TextBlock(content))
			}
			if calls, ok := message.ToolCalls.Get(); ok {
				hasToolCalls = hasToolCalls || len(calls) > 0
				for _, call := range calls {
					block, err := openCodeToolCallBlock(call)
					if err != nil {
						return result, err
					}
					blocks = append(blocks, block)
				}
			}
		}
		result.Candidates = append(result.Candidates, Message{Role: Assistant, Blocks: blocks})
	}
	if len(response.Choices) > 0 {
		result.FinishReason, err = openCodeFinishReason(response.Choices[0].FinishReason.Or(""))
		if err != nil {
			return result, err
		}
	}
	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}
	return result, nil
}

// Stream starts one OpenCode SSE stream when iterated and emits ordered
// reasoning, text, tool-call, metadata, and terminal-error [StreamChunk] values.
func (g *OpenCodeGenerator) Stream(ctx context.Context, generationRequest GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("opencode: client not initialized")})
			return
		}
		if len(generationRequest.Dialog) == 0 {
			yield(StreamChunk{Err: ErrEmptyDialog})
			return
		}
		request, params, err := g.buildRequest(generationRequest, true)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		rawResponse, err := g.client.CreateChatCompletion(ctx, request, params)
		if err != nil {
			yield(StreamChunk{Err: mapOpenCodeError(err)})
			return
		}
		stream, ok := rawResponse.(*opencode.CreateChatCompletionOKTextEventStream)
		if !ok {
			yield(StreamChunk{Err: fmt.Errorf("opencode: expected event stream response, got %T", rawResponse)})
			return
		}
		defer stream.Close()

		var finalUsage opencode.Usage
		var hasFinalUsage bool
		responseExtraFields := make(map[string]interface{})
		for {
			event, err := stream.Next(ctx)
			if err != nil {
				yield(StreamChunk{Err: mapOpenCodeError(err)})
				return
			}
			if event.Data.IsCreateChatCompletionOKTextEventStreamEventData1() {
				break
			}
			chunk, ok := event.Data.GetChatCompletionChunk()
			if !ok {
				yield(StreamChunk{Err: fmt.Errorf("opencode: unexpected event data type %q", event.Data.Type)})
				return
			}
			chunkExtraFields := openCodeResponseExtraFields(
				chunk.ID,
				chunk.Model,
				chunk.Created,
				chunk.SystemFingerprint,
				chunk.Cost,
			)
			maps.Copy(responseExtraFields, chunkExtraFields)
			if usage, ok := chunk.Usage.Get(); ok {
				finalUsage = usage
				hasFinalUsage = true
			}
			for _, choice := range chunk.Choices {
				candidateIndex := choice.Index.Or(0)
				if finishReason, ok := choice.FinishReason.Get(); ok {
					if _, finishErr := openCodeFinishReason(finishReason); finishErr != nil {
						yield(StreamChunk{Err: finishErr})
						return
					}
				}
				delta, ok := choice.Delta.Get()
				if !ok {
					continue
				}
				yieldBlock := func(block Block) bool {
					return yield(StreamChunk{
						Block:               block,
						ResponseExtraFields: chunkExtraFields,
						CandidatesIndex:     candidateIndex,
					})
				}
				hasReasoningDetails := false
				for _, detail := range delta.ReasoningDetails {
					block, ok, detailErr := openCodeReasoningDetailBlock(detail)
					if detailErr != nil {
						yield(StreamChunk{Err: detailErr})
						return
					}
					if ok {
						hasReasoningDetails = true
						if !yieldBlock(block) {
							return
						}
					}
				}
				if !hasReasoningDetails {
					if reasoning, ok := delta.ReasoningContent.Get(); ok && reasoning != "" {
						if !yieldBlock(openCodeThinkingBlock(reasoning)) {
							return
						}
					} else if reasoning, ok := delta.Reasoning.Get(); ok && reasoning != "" {
						if !yieldBlock(openCodeReasoningFieldBlock(reasoning, "reasoning")) {
							return
						}
					}
				}
				if content, ok := delta.Content.Get(); ok && content != "" {
					if !yieldBlock(TextBlock(content)) {
						return
					}
				}
				if calls, ok := delta.ToolCalls.Get(); ok {
					for _, call := range calls {
						function, _ := call.Function.Get()
						if name := function.Name.Or(""); name != "" {
							if !yieldBlock(Block{
								ID:           call.ID.Or(""),
								BlockType:    ToolCall,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(name),
							}) {
								return
							}
						}
						if arguments := function.Arguments.Or(""); arguments != "" {
							if !yieldBlock(Block{
								BlockType:    ToolCall,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(arguments),
							}) {
								return
							}
						}
					}
				}
			}
		}
		metadata := make(Metadata)
		if hasFinalUsage {
			addOpenCodeUsageMetadata(metadata, finalUsage)
		}
		terminalBlock := SeparatorBlock()
		if len(metadata) > 0 {
			terminalBlock = MetadataBlock(metadata)
		}
		if len(metadata) > 0 || len(responseExtraFields) > 0 {
			yield(StreamChunk{
				Block:               terminalBlock,
				ResponseExtraFields: responseExtraFields,
				CandidatesIndex:     0,
			})
		}
	}
}

func openCodeThinkingBlock(content string) Block {
	return openCodeReasoningFieldBlock(content, "reasoning_content")
}

func openCodeReasoningFieldBlock(content, field string) Block {
	return Block{
		BlockType:    Thinking,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str(content),
		ExtraFields: map[string]interface{}{
			ThinkingExtraFieldGeneratorKey:   ThinkingGeneratorOpenCode,
			OpenCodeExtraFieldReasoningField: field,
		},
	}
}

func openCodeReasoningDetailBlock(detail opencode.ReasoningDetail) (Block, bool, error) {
	var content string
	switch detail.Type {
	case "reasoning.summary":
		content = detail.Summary.Or("")
	case "reasoning.text":
		content = detail.Text.Or("")
	case "reasoning.encrypted":
		content = detail.Data.Or("")
	default:
		return Block{}, false, nil
	}
	if content == "" {
		return Block{}, false, nil
	}
	encoded, err := detail.MarshalJSON()
	if err != nil {
		return Block{}, false, fmt.Errorf("opencode: encode reasoning detail: %w", err)
	}
	var replayDetail map[string]any
	if err := json.Unmarshal(encoded, &replayDetail); err != nil {
		return Block{}, false, fmt.Errorf("opencode: preserve reasoning detail: %w", err)
	}
	return Block{
		ID:           detail.ID.Or(""),
		BlockType:    Thinking,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str(content),
		ExtraFields: map[string]interface{}{
			ThinkingExtraFieldGeneratorKey:    ThinkingGeneratorOpenCode,
			OpenCodeExtraFieldReasoningField:  "reasoning_details",
			OpenCodeExtraFieldReasoningDetail: replayDetail,
		},
	}, true, nil
}

func openCodeToolCallBlock(call opencode.ToolCall) (Block, error) {
	parameters := make(map[string]any)
	if strings.TrimSpace(call.Function.Arguments) != "" {
		if err := json.Unmarshal([]byte(call.Function.Arguments), &parameters); err != nil {
			return Block{}, fmt.Errorf("opencode: malformed tool arguments for %q: %w", call.Function.Name, err)
		}
	}
	return ToolCallBlock(call.ID, call.Function.Name, parameters)
}

func openCodeFinishReason(reason string) (FinishReason, error) {
	switch reason {
	case "stop", "end_turn":
		return EndTurn, nil
	case "tool_calls", "tool_use":
		return ToolUse, nil
	case "length", "max_tokens", "model_length", "model_context_window_exceeded":
		return MaxGenerationLimit, ErrMaxGenerationLimit
	case "content_filter", "sensitive", "refusal":
		return ContentPolicyViolation, ContentPolicyErr("content filtered")
	default:
		return Unknown, nil
	}
}

func addOpenCodeUsageMetadata(metadata Metadata, usage opencode.Usage) {
	if promptTokens := usage.PromptTokens.Or(0); promptTokens > 0 {
		metadata[UsageMetricInputTokens] = promptTokens
	}
	if completionTokens := usage.CompletionTokens.Or(0); completionTokens > 0 {
		metadata[UsageMetricGenerationTokens] = completionTokens
	}
	cachedTokens := usage.CachedTokens.Or(0)
	if details, ok := usage.PromptTokensDetails.Get(); ok {
		if value := details.CachedTokens.Or(0); value > 0 {
			cachedTokens = value
		}
		if value := details.CacheCreationInputTokens.Or(0); value > 0 {
			metadata[UsageMetricCacheWriteTokens] = value
		}
	}
	if cachedTokens == 0 {
		cachedTokens = usage.PromptCacheHitTokens.Or(0)
	}
	if cachedTokens > 0 {
		metadata[UsageMetricCacheReadTokens] = cachedTokens
	}
	if details, ok := usage.CompletionTokensDetails.Get(); ok {
		if reasoningTokens := details.ReasoningTokens.Or(0); reasoningTokens > 0 {
			metadata[UsageMetricReasoningTokens] = reasoningTokens
		}
	}
}

func mapOpenCodeError(err error) error {
	var statusErr *opencode.ErrorResponseStatusCode
	if !errors.As(err, &statusErr) {
		return err
	}
	rawBody, _ := statusErr.Response.MarshalJSON()
	message := statusErr.Response.Message.Or("")
	if detail, ok := statusErr.Response.Error.Get(); ok && detail.Message.Or("") != "" {
		message = detail.Message.Or("")
	}
	return &ApiErr{
		Provider:   ProviderOpenCode,
		Kind:       classifyHTTPStatus(statusErr.StatusCode),
		StatusCode: statusErr.StatusCode,
		Message:    message,
		RawBody:    string(rawBody),
		Cause:      err,
	}
}

var _ Generator = (*OpenCodeGenerator)(nil)
var _ StreamingGenerator = (*OpenCodeGenerator)(nil)
