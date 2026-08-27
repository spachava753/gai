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

	"github.com/spachava753/gai/internal/deepseek"
)

const (
	// DeepSeekDefaultBaseURL is the DeepSeek API server declared by the generated OpenAPI client.
	DeepSeekDefaultBaseURL = string(deepseek.DefaultServer)

	// DeepSeekGenerationOptionThinkingEnabled controls thinking mode for one request.
	DeepSeekGenerationOptionThinkingEnabled = "deepseek_thinking_enabled"
)

const (
	// DeepSeekResponseExtraFieldID stores the completion identifier returned by DeepSeek.
	DeepSeekResponseExtraFieldID = "deepseek_id"
	// DeepSeekResponseExtraFieldModel stores the model reported by DeepSeek.
	DeepSeekResponseExtraFieldModel = "deepseek_model"
	// DeepSeekResponseExtraFieldCreated stores the completion's Unix creation timestamp.
	DeepSeekResponseExtraFieldCreated = "deepseek_created"
	// DeepSeekResponseExtraFieldSystemFingerprint stores the backend configuration fingerprint.
	DeepSeekResponseExtraFieldSystemFingerprint = "deepseek_system_fingerprint"
)

// WithDeepSeekThinking controls thinking mode for one DeepSeek generation request.
func WithDeepSeekThinking(enabled bool) GenerationOption {
	return func(options GenerationOptions) {
		options[DeepSeekGenerationOptionThinkingEnabled] = enabled
	}
}

// DeepSeekGenerator implements Generator and StreamingGenerator using the generated DeepSeek client.
type DeepSeekGenerator struct {
	client *deepseek.Client
}

// NewDeepSeekGenerator creates a stateless DeepSeek generator.
// A nil httpClient uses the default HTTP client. An empty baseURL uses
// DeepSeekDefaultBaseURL. apiKey is required.
func NewDeepSeekGenerator(httpClient *http.Client, baseURL, apiKey string) (*DeepSeekGenerator, error) {
	if baseURL == "" {
		baseURL = DeepSeekDefaultBaseURL
	}
	if apiKey == "" {
		return nil, fmt.Errorf("deepseek: %w", ErrMissingAPIKey)
	}
	options := make([]deepseek.ClientOption, 0, 1)
	if httpClient != nil {
		options = append(options, deepseek.WithClient(httpClient))
	}
	client, err := deepseek.NewClient(baseURL, deepSeekSecuritySource{apiKey: apiKey}, options...)
	if err != nil {
		return nil, fmt.Errorf("deepseek: create client: %w", err)
	}
	return &DeepSeekGenerator{client: client}, nil
}

type deepSeekSecuritySource struct {
	apiKey string
}

func (s deepSeekSecuritySource) BearerAuth(ctx context.Context, operationName deepseek.OperationName) (deepseek.BearerAuth, error) {
	return deepseek.BearerAuth{Token: s.apiKey}, nil
}

type deepSeekGenerationOptions struct {
	Temperature         *float64
	TopP                *float64
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	OutputModalities    []Modality
	ThinkingEnabled     *bool
	ReasoningEffort     string
}

func parseDeepSeekGenerationOptions(values GenerationOptions) (*deepSeekGenerationOptions, error) {
	options := &deepSeekGenerationOptions{}

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
	thinkingEnabled, ok, err := generationOption[bool](values, DeepSeekGenerationOptionThinkingEnabled)
	if err != nil {
		return nil, err
	}
	if ok {
		options.ThinkingEnabled = &thinkingEnabled
	}
	if options.ReasoningEffort, _, err = generationOption[string](values, GenerationOptionThinkingBudget); err != nil {
		return nil, err
	}
	return options, nil
}

func convertToolToDeepSeek(tool Tool) (deepseek.FunctionTool, error) {
	parameters := deepseek.FunctionObjectParameters{
		"type":       jx.Raw(`"object"`),
		"properties": jx.Raw(`{}`),
	}
	if tool.InputSchema != nil {
		schemaJSON, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return deepseek.FunctionTool{}, err
		}
		var rawParameters map[string]json.RawMessage
		if err := json.Unmarshal(schemaJSON, &rawParameters); err != nil {
			return deepseek.FunctionTool{}, err
		}
		parameters = make(deepseek.FunctionObjectParameters, len(rawParameters))
		for name, raw := range rawParameters {
			parameters[name] = jx.Raw(raw)
		}
	}

	converted := deepseek.FunctionTool{
		Type: deepseek.FunctionToolTypeFunction,
		Function: deepseek.FunctionObject{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  parameters,
		},
	}
	if err := converted.Validate(); err != nil {
		return deepseek.FunctionTool{}, err
	}
	return converted, nil
}

func convertToolsToDeepSeek(tools []Tool) ([]deepseek.FunctionTool, error) {
	converted := make([]deepseek.FunctionTool, 0, len(tools))
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

		providerTool, err := convertToolToDeepSeek(tool)
		if err != nil {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
		}
		converted = append(converted, providerTool)
	}
	return converted, nil
}

func buildDeepSeekMessages(dialog Dialog, instructions string) ([]deepseek.Message, error) {
	messages := make([]deepseek.Message, 0, len(dialog)+1)
	if instructions != "" {
		messages = append(messages, deepseek.NewSystemMessageMessage(deepseek.SystemMessage{
			Role:    deepseek.SystemMessageRoleSystem,
			Content: instructions,
		}))
	}

	for i, message := range dialog {
		switch message.Role {
		case User:
			content, err := deepSeekTextContent(message.Blocks, "user")
			if err != nil {
				return nil, err
			}
			messages = append(messages, deepseek.NewUserMessageMessage(deepseek.UserMessage{
				Role:    deepseek.UserMessageRoleUser,
				Content: content,
			}))
		case Assistant:
			assistant, err := buildDeepSeekAssistantMessage(message.Blocks)
			if err != nil {
				return nil, err
			}
			messages = append(messages, deepseek.NewAssistantMessageMessage(assistant))
		case ToolResult:
			if len(message.Blocks) == 0 {
				return nil, fmt.Errorf("deepseek: tool result message must have at least one block")
			}
			for _, block := range message.Blocks {
				if block.ID == "" {
					return nil, fmt.Errorf("deepseek: tool result block must have a tool call ID")
				}
				if block.BlockType != Content {
					return nil, fmt.Errorf("deepseek: unsupported tool result block type %q", block.BlockType)
				}
				if block.ModalityType != Text {
					return nil, UnsupportedInputModalityErr(block.ModalityType.String())
				}
				messages = append(messages, deepseek.NewToolMessageMessage(deepseek.ToolMessage{
					Role:       deepseek.ToolMessageRoleTool,
					Content:    block.Content.String(),
					ToolCallID: block.ID,
				}))
			}
		default:
			return nil, fmt.Errorf("deepseek: unsupported role at index %d: %v", i, message.Role)
		}
	}
	return messages, nil
}

func deepSeekTextContent(blocks []Block, role string) (string, error) {
	var content strings.Builder
	for _, block := range blocks {
		if block.BlockType != Content {
			return "", fmt.Errorf("deepseek: unsupported block type for %s: %q", role, block.BlockType)
		}
		if block.ModalityType != Text {
			return "", UnsupportedInputModalityErr(block.ModalityType.String())
		}
		content.WriteString(block.Content.String())
	}
	return content.String(), nil
}

func buildDeepSeekAssistantMessage(blocks []Block) (deepseek.AssistantMessage, error) {
	message := deepseek.AssistantMessage{Role: deepseek.AssistantMessageRoleAssistant}
	var content strings.Builder
	var reasoning strings.Builder

	for _, block := range blocks {
		switch block.BlockType {
		case Content:
			if block.ModalityType != Text {
				return deepseek.AssistantMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
			content.WriteString(block.Content.String())
		case Thinking:
			if block.ModalityType != Text {
				return deepseek.AssistantMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
			reasoning.WriteString(block.Content.String())
		case ToolCall:
			if block.ID == "" {
				return deepseek.AssistantMessage{}, fmt.Errorf("deepseek: tool call block missing ID")
			}
			var input ToolCallInput
			if err := json.Unmarshal([]byte(block.Content.String()), &input); err != nil {
				return deepseek.AssistantMessage{}, fmt.Errorf("deepseek: invalid tool call content: %w", err)
			}
			arguments, err := json.Marshal(input.Parameters)
			if err != nil {
				return deepseek.AssistantMessage{}, fmt.Errorf("deepseek: marshal tool arguments: %w", err)
			}
			message.ToolCalls = append(message.ToolCalls, deepseek.ToolCall{
				ID:   block.ID,
				Type: deepseek.ToolCallTypeFunction,
				Function: deepseek.ToolCallFunction{
					Name:      input.Name,
					Arguments: string(arguments),
				},
			})
		default:
			return deepseek.AssistantMessage{}, fmt.Errorf("deepseek: unsupported assistant block type %q", block.BlockType)
		}
	}
	if content.Len() > 0 {
		message.Content = deepseek.NewOptNilString(content.String())
	}
	if reasoning.Len() > 0 {
		message.ReasoningContent = deepseek.NewOptString(reasoning.String())
	}
	return message, nil
}

func (g *DeepSeekGenerator) buildRequest(generationRequest GenerationRequest, stream bool) (*deepseek.ChatCompletionRequest, error) {
	options, err := parseDeepSeekGenerationOptions(generationRequest.Options)
	if err != nil {
		return nil, err
	}
	for _, modality := range options.OutputModalities {
		if modality != Text {
			return nil, UnsupportedOutputModalityErr(modality.String())
		}
	}
	instructions, err := joinedTextInstructions(generationRequest.Instructions)
	if err != nil {
		return nil, err
	}
	messages, err := buildDeepSeekMessages(generationRequest.Dialog, instructions)
	if err != nil {
		return nil, err
	}
	tools, err := convertToolsToDeepSeek(generationRequest.Tools)
	if err != nil {
		return nil, err
	}

	request := &deepseek.ChatCompletionRequest{
		Model:    deepseek.ChatCompletionRequestModel(generationRequest.Model),
		Messages: messages,
		Tools:    tools,
		Stream:   deepseek.NewOptBool(stream),
	}
	if stream {
		request.StreamOptions = deepseek.NewOptChatCompletionRequestStreamOptions(deepseek.ChatCompletionRequestStreamOptions{
			IncludeUsage: deepseek.NewOptBool(true),
		})
	}
	if options.Temperature != nil {
		request.Temperature = deepseek.NewOptFloat64(*options.Temperature)
	}
	if options.TopP != nil {
		request.TopP = deepseek.NewOptFloat64(*options.TopP)
	}
	if options.MaxGenerationTokens != nil {
		request.MaxTokens = deepseek.NewOptInt(*options.MaxGenerationTokens)
	}
	if len(options.StopSequences) > 0 {
		request.Stop = deepseek.NewOptChatCompletionRequestStop(
			deepseek.NewStringArrayChatCompletionRequestStop(options.StopSequences),
		)
	}
	if options.ThinkingEnabled != nil {
		thinkingType := deepseek.ChatCompletionRequestThinkingTypeDisabled
		if *options.ThinkingEnabled {
			thinkingType = deepseek.ChatCompletionRequestThinkingTypeEnabled
		}
		request.Thinking = deepseek.NewOptChatCompletionRequestThinking(deepseek.ChatCompletionRequestThinking{
			Type: deepseek.NewOptChatCompletionRequestThinkingType(thinkingType),
		})
	}
	if options.ReasoningEffort != "" {
		effort := deepseek.ChatCompletionRequestReasoningEffort(options.ReasoningEffort)
		if err := effort.Validate(); err != nil {
			return nil, &InvalidParameterErr{Parameter: GenerationOptionThinkingBudget, Reason: err.Error()}
		}
		request.ReasoningEffort = deepseek.NewOptChatCompletionRequestReasoningEffort(effort)
	}
	if err := applyDeepSeekToolChoice(request, options.ToolChoice, generationRequest.Tools); err != nil {
		return nil, err
	}
	return request, nil
}

func applyDeepSeekToolChoice(request *deepseek.ChatCompletionRequest, choice string, tools []Tool) error {
	if choice == "" {
		return nil
	}
	if choice == ToolChoiceToolsRequired && len(tools) == 0 {
		return InvalidToolChoiceErr("required needs at least one tool")
	}
	if choice == "none" || choice == ToolChoiceAuto || choice == ToolChoiceToolsRequired {
		providerChoice := deepseek.ChatCompletionRequestToolChoice0(choice)
		request.ToolChoice = deepseek.NewOptChatCompletionRequestToolChoice(
			deepseek.NewChatCompletionRequestToolChoice0ChatCompletionRequestToolChoice(providerChoice),
		)
		return nil
	}
	for _, tool := range tools {
		if tool.Name == choice {
			request.ToolChoice = deepseek.NewOptChatCompletionRequestToolChoice(
				deepseek.NewToolChoiceObjectChatCompletionRequestToolChoice(deepseek.ToolChoiceObject{
					Type: deepseek.ToolChoiceObjectTypeFunction,
					Function: deepseek.ToolChoiceObjectFunction{
						Name: choice,
					},
				}),
			)
			return nil
		}
	}
	return InvalidToolChoiceErr(fmt.Sprintf("tool %q is not in the request", choice))
}

func deepSeekResponseExtraFields(id, model string, created int, systemFingerprint string) map[string]interface{} {
	extraFields := map[string]interface{}{
		DeepSeekResponseExtraFieldID:      id,
		DeepSeekResponseExtraFieldModel:   model,
		DeepSeekResponseExtraFieldCreated: created,
	}
	if systemFingerprint != "" {
		extraFields[DeepSeekResponseExtraFieldSystemFingerprint] = systemFingerprint
	}
	return extraFields
}

// Generate implements Generator.
func (g *DeepSeekGenerator) Generate(ctx context.Context, generationRequest GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("deepseek: client not initialized")
	}
	if len(generationRequest.Dialog) == 0 {
		return Response{}, ErrEmptyDialog
	}
	request, err := g.buildRequest(generationRequest, false)
	if err != nil {
		return Response{}, err
	}
	rawResponse, err := g.client.ChatCompletionsPost(ctx, request)
	if err != nil {
		return Response{}, mapDeepSeekError(err)
	}
	response, ok := rawResponse.(*deepseek.ChatCompletionResponse)
	if !ok {
		if stream, isStream := rawResponse.(*deepseek.ChatCompletionsPostOKTextEventStream); isStream {
			_ = stream.Close()
		}
		return Response{}, fmt.Errorf("deepseek: expected JSON completion response, got %T", rawResponse)
	}

	result := Response{
		UsageMetadata: make(Metadata),
		ExtraFields: deepSeekResponseExtraFields(
			response.ID,
			response.Model,
			response.Created,
			response.SystemFingerprint.Or(""),
		),
	}
	if usage, ok := response.Usage.Get(); ok {
		addDeepSeekUsageMetadata(result.UsageMetadata, usage)
	}
	var hasToolCalls bool
	for _, choice := range response.Choices {
		blocks := make([]Block, 0, 2)
		if reasoning, ok := choice.Message.ReasoningContent.Get(); ok && reasoning != "" {
			blocks = append(blocks, deepSeekThinkingBlock(reasoning))
		}
		if content, ok := choice.Message.Content.Get(); ok && content != "" {
			blocks = append(blocks, TextBlock(content))
		}
		if calls, ok := choice.Message.ToolCalls.Get(); ok {
			hasToolCalls = hasToolCalls || len(calls) > 0
			for _, call := range calls {
				block, err := deepSeekToolCallBlock(call)
				if err != nil {
					return result, err
				}
				blocks = append(blocks, block)
			}
		}
		result.Candidates = append(result.Candidates, Message{Role: Assistant, Blocks: blocks})
	}
	if len(response.Choices) > 0 {
		result.FinishReason, err = deepSeekFinishReason(string(response.Choices[0].FinishReason))
		if err != nil {
			return result, err
		}
	}
	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}
	return result, nil
}

// Stream implements StreamingGenerator.
func (g *DeepSeekGenerator) Stream(ctx context.Context, generationRequest GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("deepseek: client not initialized")})
			return
		}
		if len(generationRequest.Dialog) == 0 {
			yield(StreamChunk{Err: ErrEmptyDialog})
			return
		}
		request, err := g.buildRequest(generationRequest, true)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		rawResponse, err := g.client.ChatCompletionsPost(ctx, request)
		if err != nil {
			yield(StreamChunk{Err: mapDeepSeekError(err)})
			return
		}
		stream, ok := rawResponse.(*deepseek.ChatCompletionsPostOKTextEventStream)
		if !ok {
			yield(StreamChunk{Err: fmt.Errorf("deepseek: expected event stream response, got %T", rawResponse)})
			return
		}
		defer stream.Close()

		var finalUsage deepseek.Usage
		var hasFinalUsage bool
		responseExtraFields := make(map[string]interface{})
		for {
			event, err := stream.Next(ctx)
			if err != nil {
				yield(StreamChunk{Err: mapDeepSeekError(err)})
				return
			}
			if event.Data.IsChatCompletionsPostOKTextEventStreamEventData1() {
				break
			}
			chunk, ok := event.Data.GetChatCompletionChunk()
			if !ok {
				yield(StreamChunk{Err: fmt.Errorf("deepseek: unexpected event data type %q", event.Data.Type)})
				return
			}
			if usage, ok := chunk.Usage.Get(); ok {
				finalUsage = usage
				hasFinalUsage = true
			}
			chunkExtraFields := deepSeekResponseExtraFields(
				chunk.ID,
				chunk.Model,
				chunk.Created,
				chunk.SystemFingerprint.Or(""),
			)
			maps.Copy(responseExtraFields, chunkExtraFields)
			for _, choice := range chunk.Choices {
				if finishReason, ok := choice.FinishReason.Get(); ok {
					_, finishErr := deepSeekFinishReason(string(finishReason))
					if finishErr != nil {
						yield(StreamChunk{Err: finishErr})
						return
					}
				}
				if reasoning, ok := choice.Delta.ReasoningContent.Get(); ok && reasoning != "" {
					if !yield(StreamChunk{
						Block:               deepSeekThinkingBlock(reasoning),
						ResponseExtraFields: chunkExtraFields,
						CandidatesIndex:     choice.Index,
					}) {
						return
					}
				}
				if content, ok := choice.Delta.Content.Get(); ok && content != "" {
					if !yield(StreamChunk{
						Block:               TextBlock(content),
						ResponseExtraFields: chunkExtraFields,
						CandidatesIndex:     choice.Index,
					}) {
						return
					}
				}
				for _, call := range choice.Delta.ToolCalls {
					if name := call.Function.Name.Or(""); name != "" {
						if !yield(StreamChunk{
							Block: Block{
								ID:           call.ID.Or(""),
								BlockType:    ToolCall,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(name),
							},
							ResponseExtraFields: chunkExtraFields,
							CandidatesIndex:     choice.Index,
						}) {
							return
						}
					}
					if arguments := call.Function.Arguments.Or(""); arguments != "" {
						if !yield(StreamChunk{
							Block: Block{
								BlockType:    ToolCall,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(arguments),
							},
							ResponseExtraFields: chunkExtraFields,
							CandidatesIndex:     choice.Index,
						}) {
							return
						}
					}
				}
			}
		}
		metadata := make(Metadata)
		if hasFinalUsage {
			addDeepSeekUsageMetadata(metadata, finalUsage)
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

func deepSeekThinkingBlock(content string) Block {
	return Block{
		BlockType:    Thinking,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str(content),
		ExtraFields: map[string]interface{}{
			ThinkingExtraFieldGeneratorKey: ThinkingGeneratorDeepSeek,
		},
	}
}

func deepSeekToolCallBlock(call deepseek.ToolCall) (Block, error) {
	parameters := make(map[string]any)
	if strings.TrimSpace(call.Function.Arguments) != "" {
		if err := json.Unmarshal([]byte(call.Function.Arguments), &parameters); err != nil {
			return Block{}, fmt.Errorf("deepseek: malformed tool arguments for %q: %w", call.Function.Name, err)
		}
	}
	return ToolCallBlock(call.ID, call.Function.Name, parameters)
}

func deepSeekFinishReason(reason string) (FinishReason, error) {
	switch reason {
	case "stop":
		return EndTurn, nil
	case "tool_calls":
		return ToolUse, nil
	case "length":
		return MaxGenerationLimit, ErrMaxGenerationLimit
	case "content_filter":
		return ContentPolicyViolation, ContentPolicyErr("content filtered")
	case "insufficient_system_resource":
		return Unknown, &ApiErr{
			Provider: ProviderDeepSeek,
			Kind:     APIErrorKindServiceUnavailable,
			Message:  "generation stopped because DeepSeek had insufficient system resources",
		}
	case "":
		return Unknown, nil
	default:
		return Unknown, nil
	}
}

func addDeepSeekUsageMetadata(metadata Metadata, usage deepseek.Usage) {
	if usage.PromptTokens > 0 {
		metadata[UsageMetricInputTokens] = usage.PromptTokens
	}
	if usage.CompletionTokens > 0 {
		metadata[UsageMetricGenerationTokens] = usage.CompletionTokens
	}
	if usage.PromptCacheHitTokens > 0 {
		metadata[UsageMetricCacheReadTokens] = usage.PromptCacheHitTokens
	}
	if details, ok := usage.CompletionTokensDetails.Get(); ok && details.ReasoningTokens.Or(0) > 0 {
		metadata[UsageMetricReasoningTokens] = details.ReasoningTokens.Or(0)
	}
}

func mapDeepSeekError(err error) error {
	var statusErr *deepseek.ErrorResponseStatusCode
	if !errors.As(err, &statusErr) {
		return err
	}
	rawBody, _ := json.Marshal(statusErr.Response)
	message := ""
	if detail, ok := statusErr.Response.Error.Get(); ok {
		message = detail.Message
	}
	return &ApiErr{
		Provider:   ProviderDeepSeek,
		Kind:       classifyHTTPStatus(statusErr.StatusCode),
		StatusCode: statusErr.StatusCode,
		Message:    message,
		RawBody:    string(rawBody),
		Cause:      err,
	}
}
