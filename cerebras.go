package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"strings"

	"github.com/go-faster/jx"

	"github.com/spachava753/gai/internal/cerebras"
)

// CerebrasDefaultBaseURL is the Cerebras API server declared by the generated OpenAPI client.
const CerebrasDefaultBaseURL = string(cerebras.DefaultServer)

// CerebrasGenerator implements Generator using the generated Cerebras client.
type CerebrasGenerator struct {
	client *cerebras.Client
}

// NewCerebrasGenerator creates a stateless Cerebras generator.
// A nil httpClient uses the default HTTP client. An empty baseURL uses
// CerebrasDefaultBaseURL. apiKey is required.
func NewCerebrasGenerator(httpClient *http.Client, baseURL, apiKey string) (*CerebrasGenerator, error) {
	if baseURL == "" {
		baseURL = CerebrasDefaultBaseURL
	}
	if apiKey == "" {
		return nil, fmt.Errorf("cerebras: %w", ErrMissingAPIKey)
	}
	options := make([]cerebras.ClientOption, 0, 1)
	if httpClient != nil {
		options = append(options, cerebras.WithClient(httpClient))
	}
	client, err := cerebras.NewClient(baseURL, cerebrasSecuritySource{apiKey: apiKey}, options...)
	if err != nil {
		return nil, fmt.Errorf("cerebras: create client: %w", err)
	}
	return &CerebrasGenerator{client: client}, nil
}

type cerebrasSecuritySource struct {
	apiKey string
}

func (s cerebrasSecuritySource) BearerAuth(ctx context.Context, operationName cerebras.OperationName) (cerebras.BearerAuth, error) {
	return cerebras.BearerAuth{Token: s.apiKey}, nil
}

func convertToolToCerebras(tool Tool) (cerebras.FunctionTool, error) {
	function := cerebras.FunctionObject{Name: tool.Name}
	if tool.Description != "" {
		function.Description = cerebras.NewOptString(tool.Description)
	}
	if tool.InputSchema != nil {
		schemaJSON, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return cerebras.FunctionTool{}, err
		}
		var rawParameters map[string]json.RawMessage
		if err := json.Unmarshal(schemaJSON, &rawParameters); err != nil {
			return cerebras.FunctionTool{}, err
		}
		parameters := make(cerebras.FunctionObjectParameters, len(rawParameters))
		for name, raw := range rawParameters {
			parameters[name] = jx.Raw(raw)
		}
		function.Parameters = cerebras.NewOptFunctionObjectParameters(parameters)
	}

	converted := cerebras.FunctionTool{
		Type:     cerebras.FunctionToolTypeFunction,
		Function: function,
	}
	if err := converted.Validate(); err != nil {
		return cerebras.FunctionTool{}, err
	}
	return converted, nil
}

func convertToolsToCerebras(tools []Tool) ([]cerebras.FunctionTool, error) {
	converted := make([]cerebras.FunctionTool, 0, len(tools))
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

		providerTool, err := convertToolToCerebras(tool)
		if err != nil {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
		}
		converted = append(converted, providerTool)
	}
	return converted, nil
}

type cerebrasGenerationOptions struct {
	Temperature         *float64
	TopP                *float64
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	OutputModalities    []Modality
	ThinkingBudget      string
}

func parseCerebrasGenerationOptions(values GenerationOptions) (*cerebrasGenerationOptions, error) {
	options := &cerebrasGenerationOptions{}

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
	if options.ThinkingBudget, _, err = generationOption[string](values, GenerationOptionThinkingBudget); err != nil {
		return nil, err
	}
	return options, nil
}

func buildCerebrasMessages(request GenerationRequest) ([]cerebras.Message, error) {
	messages := make([]cerebras.Message, 0, len(request.Dialog)+1)
	instructions, err := joinedTextInstructions(request.Instructions)
	if err != nil {
		return nil, err
	}
	if instructions != "" {
		messages = append(messages, cerebras.NewSystemMessageMessage(cerebras.SystemMessage{
			Role:    cerebras.SystemMessageRoleSystem,
			Content: instructions,
		}))
	}

	for i, message := range request.Dialog {
		switch message.Role {
		case User:
			content, err := cerebrasTextContent(message.Blocks, "user")
			if err != nil {
				return nil, err
			}
			messages = append(messages, cerebras.NewUserMessageMessage(cerebras.UserMessage{
				Role:    cerebras.UserMessageRoleUser,
				Content: content,
			}))
		case Assistant:
			assistant, err := buildCerebrasAssistantMessage(message.Blocks)
			if err != nil {
				return nil, err
			}
			messages = append(messages, cerebras.NewAssistantMessageMessage(assistant))
		case ToolResult:
			if len(message.Blocks) == 0 {
				return nil, fmt.Errorf("cerebras: tool result message must have at least one block")
			}
			for _, block := range message.Blocks {
				if block.ID == "" {
					return nil, fmt.Errorf("cerebras: tool result block must have a tool call ID")
				}
				if block.BlockType != Content {
					return nil, fmt.Errorf("cerebras: unsupported tool result block type %q", block.BlockType)
				}
				if block.ModalityType != Text {
					return nil, UnsupportedInputModalityErr(block.ModalityType.String())
				}
				messages = append(messages, cerebras.NewToolMessageMessage(cerebras.ToolMessage{
					Role:       cerebras.ToolMessageRoleTool,
					Content:    block.Content.String(),
					ToolCallID: block.ID,
				}))
			}
		default:
			return nil, fmt.Errorf("cerebras: unsupported role at index %d: %v", i, message.Role)
		}
	}
	return messages, nil
}

func cerebrasTextContent(blocks []Block, role string) (string, error) {
	var content strings.Builder
	for _, block := range blocks {
		if block.BlockType != Content {
			return "", fmt.Errorf("cerebras: unsupported block type for %s: %q", role, block.BlockType)
		}
		if block.ModalityType != Text {
			return "", UnsupportedInputModalityErr(block.ModalityType.String())
		}
		content.WriteString(block.Content.String())
	}
	return content.String(), nil
}

func buildCerebrasAssistantMessage(blocks []Block) (cerebras.AssistantMessage, error) {
	message := cerebras.AssistantMessage{Role: cerebras.AssistantMessageRoleAssistant}
	var content strings.Builder
	var reasoning strings.Builder

	for _, block := range blocks {
		switch block.BlockType {
		case Content:
			if block.ModalityType != Text {
				return cerebras.AssistantMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
			content.WriteString(block.Content.String())
		case Thinking:
			if block.ModalityType != Text {
				return cerebras.AssistantMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
			reasoning.WriteString(block.Content.String())
		case ToolCall:
			if block.ID == "" {
				return cerebras.AssistantMessage{}, fmt.Errorf("cerebras: tool call block missing ID")
			}
			var input ToolCallInput
			if err := json.Unmarshal([]byte(block.Content.String()), &input); err != nil {
				return cerebras.AssistantMessage{}, fmt.Errorf("cerebras: invalid tool call content: %w", err)
			}
			arguments, err := json.Marshal(input.Parameters)
			if err != nil {
				return cerebras.AssistantMessage{}, fmt.Errorf("cerebras: marshal tool arguments: %w", err)
			}
			message.ToolCalls = append(message.ToolCalls, cerebras.ToolCall{
				ID:   block.ID,
				Type: cerebras.ToolCallTypeFunction,
				Function: cerebras.ToolCallFunction{
					Name:      input.Name,
					Arguments: string(arguments),
				},
			})
		default:
			return cerebras.AssistantMessage{}, fmt.Errorf("cerebras: unsupported assistant block type %q", block.BlockType)
		}
	}
	if content.Len() > 0 {
		message.Content = cerebras.NewOptNilString(content.String())
	}
	if reasoning.Len() > 0 {
		message.Reasoning = cerebras.NewOptNilString(reasoning.String())
	}
	return message, nil
}

func (g *CerebrasGenerator) buildRequest(request GenerationRequest) (*cerebras.ChatCompletionRequest, error) {
	options, err := parseCerebrasGenerationOptions(request.Options)
	if err != nil {
		return nil, err
	}
	for _, modality := range options.OutputModalities {
		if modality != Text {
			return nil, UnsupportedOutputModalityErr(modality.String())
		}
	}
	messages, err := buildCerebrasMessages(request)
	if err != nil {
		return nil, err
	}
	tools, err := convertToolsToCerebras(request.Tools)
	if err != nil {
		return nil, err
	}

	providerRequest := &cerebras.ChatCompletionRequest{
		Model:    request.Model,
		Messages: messages,
		Stream:   cerebras.NewOptBool(false),
	}
	if len(tools) > 0 {
		providerRequest.Tools = tools
	}
	if options.Temperature != nil {
		providerRequest.Temperature = cerebras.NewOptFloat64(*options.Temperature)
	}
	if options.TopP != nil {
		providerRequest.TopP = cerebras.NewOptFloat64(*options.TopP)
	}
	if options.MaxGenerationTokens != nil {
		providerRequest.MaxCompletionTokens = cerebras.NewOptInt(*options.MaxGenerationTokens)
	}
	if len(options.StopSequences) == 1 {
		providerRequest.Stop = cerebras.NewOptStop(cerebras.NewStringStop(options.StopSequences[0]))
	} else if len(options.StopSequences) > 1 {
		providerRequest.Stop = cerebras.NewOptStop(cerebras.NewStringArrayStop(options.StopSequences))
	}
	if options.ThinkingBudget != "" {
		effort := cerebras.ReasoningEffort(options.ThinkingBudget)
		if err := effort.Validate(); err != nil {
			return nil, &InvalidParameterErr{Parameter: GenerationOptionThinkingBudget, Reason: err.Error()}
		}
		providerRequest.ReasoningEffort = cerebras.NewOptReasoningEffort(effort)
	}
	if err := applyCerebrasToolChoice(providerRequest, options.ToolChoice, request.Tools); err != nil {
		return nil, err
	}
	return providerRequest, nil
}

func applyCerebrasToolChoice(request *cerebras.ChatCompletionRequest, choice string, tools []Tool) error {
	if choice == "" {
		return nil
	}
	if choice == ToolChoiceToolsRequired && len(tools) == 0 {
		return InvalidToolChoiceErr("required needs at least one tool")
	}
	if choice == "none" || choice == ToolChoiceAuto || choice == ToolChoiceToolsRequired {
		request.ToolChoice = cerebras.NewOptToolChoice(
			cerebras.NewToolChoiceModeToolChoice(cerebras.ToolChoiceMode(choice)),
		)
		return nil
	}
	for _, tool := range tools {
		if tool.Name == choice {
			request.ToolChoice = cerebras.NewOptToolChoice(
				cerebras.NewToolChoiceObjectToolChoice(cerebras.ToolChoiceObject{
					Type: cerebras.ToolChoiceObjectTypeFunction,
					Function: cerebras.ToolChoiceObjectFunction{
						Name: choice,
					},
				}),
			)
			return nil
		}
	}
	return InvalidToolChoiceErr(fmt.Sprintf("tool %q is not in the request", choice))
}

// Generate implements Generator.
func (g *CerebrasGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("cerebras: client not initialized")
	}
	if len(request.Dialog) == 0 {
		return Response{}, ErrEmptyDialog
	}
	providerRequest, err := g.buildRequest(request)
	if err != nil {
		return Response{}, err
	}
	rawCompletion, err := g.client.CreateChatCompletion(ctx, providerRequest)
	if err != nil {
		return Response{}, mapCerebrasError(err)
	}
	completion, ok := rawCompletion.(*cerebras.ChatCompletionResponse)
	if !ok {
		if stream, isStream := rawCompletion.(*cerebras.CreateChatCompletionOKTextEventStream); isStream {
			_ = stream.Close()
		}
		return Response{}, fmt.Errorf("cerebras: expected JSON completion response, got %T", rawCompletion)
	}

	result := Response{UsageMetadata: make(Metadata)}
	if usage, ok := completion.Usage.Get(); ok {
		addCerebrasUsageMetadata(result.UsageMetadata, usage)
	}
	var hasToolCalls bool
	for _, choice := range completion.Choices {
		if refusal, ok := choice.Message.Refusal.Get(); ok && refusal != "" {
			result.FinishReason = ContentPolicyViolation
			return result, ContentPolicyErr(refusal)
		}
		blocks := make([]Block, 0, 2)
		if reasoning, ok := choice.Message.Reasoning.Get(); ok && reasoning != "" {
			blocks = append(blocks, cerebrasThinkingBlock(reasoning))
		}
		if content, ok := choice.Message.Content.Get(); ok && content != "" {
			blocks = append(blocks, TextBlock(content))
		}
		if calls, ok := choice.Message.ToolCalls.Get(); ok {
			hasToolCalls = hasToolCalls || len(calls) > 0
			for _, call := range calls {
				block, err := cerebrasToolCallBlock(call)
				if err != nil {
					return result, err
				}
				blocks = append(blocks, block)
			}
		}
		result.Candidates = append(result.Candidates, Message{Role: Assistant, Blocks: blocks})
	}
	if len(completion.Choices) > 0 {
		result.FinishReason, err = cerebrasFinishReason(string(completion.Choices[0].FinishReason))
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
func (g *CerebrasGenerator) Stream(ctx context.Context, generationRequest GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("cerebras: client not initialized")})
			return
		}
		if len(generationRequest.Dialog) == 0 {
			yield(StreamChunk{Err: ErrEmptyDialog})
			return
		}
		request, err := g.buildRequest(generationRequest)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		request.Stream = cerebras.NewOptBool(true)

		rawResponse, err := g.client.CreateChatCompletion(ctx, request)
		if err != nil {
			yield(StreamChunk{Err: mapCerebrasError(err)})
			return
		}
		stream, ok := rawResponse.(*cerebras.CreateChatCompletionOKTextEventStream)
		if !ok {
			yield(StreamChunk{Err: fmt.Errorf("cerebras: expected event stream response, got %T", rawResponse)})
			return
		}
		defer stream.Close()

		var finalUsage cerebras.Usage
		var hasFinalUsage bool
		for {
			event, err := stream.Next(ctx)
			if err != nil {
				yield(StreamChunk{Err: mapCerebrasError(err)})
				return
			}
			if event.Data.IsCreateChatCompletionOKTextEventStreamEventData1() {
				break
			}
			chunk, ok := event.Data.GetChatCompletionChunk()
			if !ok {
				yield(StreamChunk{Err: fmt.Errorf("cerebras: unexpected event data type %q", event.Data.Type)})
				return
			}
			if usage, ok := chunk.Usage.Get(); ok {
				finalUsage = usage
				hasFinalUsage = true
			}
			for _, choice := range chunk.Choices {
				if finishReason, ok := choice.FinishReason.Get(); ok {
					if _, finishErr := cerebrasFinishReason(string(finishReason)); finishErr != nil {
						yield(StreamChunk{Err: finishErr})
						return
					}
				}
				if reasoning, ok := choice.Delta.Reasoning.Get(); ok && reasoning != "" {
					if !yield(StreamChunk{Block: cerebrasThinkingBlock(reasoning), CandidatesIndex: choice.Index}) {
						return
					}
				}
				if content, ok := choice.Delta.Content.Get(); ok && content != "" {
					if !yield(StreamChunk{Block: TextBlock(content), CandidatesIndex: choice.Index}) {
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
							CandidatesIndex: choice.Index,
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
							CandidatesIndex: choice.Index,
						}) {
							return
						}
					}
				}
			}
		}
		if hasFinalUsage {
			metadata := make(Metadata)
			addCerebrasUsageMetadata(metadata, finalUsage)
			if len(metadata) > 0 {
				yield(StreamChunk{Block: MetadataBlock(metadata), CandidatesIndex: 0})
			}
		}
	}
}

func cerebrasThinkingBlock(content string) Block {
	return Block{
		BlockType:    Thinking,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str(content),
		ExtraFields: map[string]interface{}{
			ThinkingExtraFieldGeneratorKey: ThinkingGeneratorCerebras,
		},
	}
}

func cerebrasToolCallBlock(call cerebras.ToolCall) (Block, error) {
	parameters := make(map[string]any)
	if strings.TrimSpace(call.Function.Arguments) != "" {
		if err := json.Unmarshal([]byte(call.Function.Arguments), &parameters); err != nil {
			return Block{}, fmt.Errorf("cerebras: malformed tool arguments for %q: %w", call.Function.Name, err)
		}
	}
	return ToolCallBlock(call.ID, call.Function.Name, parameters)
}

func cerebrasFinishReason(reason string) (FinishReason, error) {
	switch reason {
	case "stop":
		return EndTurn, nil
	case "tool_calls":
		return ToolUse, nil
	case "length":
		return MaxGenerationLimit, ErrMaxGenerationLimit
	case "content_filter":
		return ContentPolicyViolation, ContentPolicyErr("content policy violation detected")
	default:
		return Unknown, nil
	}
}

func addCerebrasUsageMetadata(metadata Metadata, usage cerebras.Usage) {
	if promptTokens := usage.PromptTokens.Or(0); promptTokens > 0 {
		metadata[UsageMetricInputTokens] = promptTokens
	}
	if completionTokens := usage.CompletionTokens.Or(0); completionTokens > 0 {
		metadata[UsageMetricGenerationTokens] = completionTokens
	}
	if details, ok := usage.PromptTokensDetails.Get(); ok && details.CachedTokens.Or(0) > 0 {
		metadata[UsageMetricCacheReadTokens] = details.CachedTokens.Or(0)
	}
	if details, ok := usage.CompletionTokensDetails.Get(); ok && details.ReasoningTokens.Or(0) > 0 {
		metadata[UsageMetricReasoningTokens] = details.ReasoningTokens.Or(0)
	}
}

func mapCerebrasError(err error) error {
	var statusErr *cerebras.ErrorResponseStatusCode
	if !errors.As(err, &statusErr) {
		return err
	}
	rawBody, _ := json.Marshal(statusErr.Response)
	message := ""
	if detail, ok := statusErr.Response.Error.Get(); ok {
		message = detail.Message
	}
	return &ApiErr{
		Provider:   ProviderCerebras,
		Kind:       classifyHTTPStatus(statusErr.StatusCode),
		StatusCode: statusErr.StatusCode,
		Message:    message,
		RawBody:    string(rawBody),
		Cause:      err,
	}
}

var _ Generator = (*CerebrasGenerator)(nil)
var _ StreamingGenerator = (*CerebrasGenerator)(nil)
