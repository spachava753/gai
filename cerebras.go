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

	"github.com/spachava753/gai/internal/cerebras"
)

const (
	// CerebrasUsageMetricImageTokens is the number of tokens used for image inputs.
	CerebrasUsageMetricImageTokens = "image_tokens"
	// CerebrasUsageMetricQueueTimeSeconds is the time a request spent queued.
	CerebrasUsageMetricQueueTimeSeconds = "queue_time_seconds"
	// CerebrasUsageMetricPromptTimeSeconds is the time spent processing input tokens.
	CerebrasUsageMetricPromptTimeSeconds = "prompt_time_seconds"
	// CerebrasUsageMetricCompletionTimeSeconds is the time spent generating output tokens.
	CerebrasUsageMetricCompletionTimeSeconds = "completion_time_seconds"
	// CerebrasUsageMetricTotalTimeSeconds is the request's total processing time.
	CerebrasUsageMetricTotalTimeSeconds = "total_time_seconds"
	// CerebrasUsageMetricTimeInfoCreated is the Unix timestamp in Cerebras time_info.
	CerebrasUsageMetricTimeInfoCreated = "time_info_created"
)

const (
	// CerebrasGenerationOptionLogitBias stores a map of token IDs to logit adjustments.
	CerebrasGenerationOptionLogitBias = "cerebras_logit_bias"
	// CerebrasGenerationOptionLogprobs controls token log-probability output.
	CerebrasGenerationOptionLogprobs = "cerebras_logprobs"
	// CerebrasGenerationOptionParallelToolCalls controls parallel function calling.
	CerebrasGenerationOptionParallelToolCalls = "cerebras_parallel_tool_calls"
	// CerebrasGenerationOptionPrediction stores known predicted output text.
	CerebrasGenerationOptionPrediction = "cerebras_prediction"
	// CerebrasGenerationOptionPromptCacheKey stores a prompt cache routing key.
	CerebrasGenerationOptionPromptCacheKey = "cerebras_prompt_cache_key"
	// CerebrasGenerationOptionResponseFormat stores a Cerebras response_format object.
	CerebrasGenerationOptionResponseFormat = "cerebras_response_format"
	// CerebrasGenerationOptionSeed stores the sampling seed.
	CerebrasGenerationOptionSeed = "cerebras_seed"
	// CerebrasGenerationOptionServiceTier stores the requested processing tier.
	CerebrasGenerationOptionServiceTier = "cerebras_service_tier"
	// CerebrasGenerationOptionTopLogprobs stores the number of alternative tokens per position.
	CerebrasGenerationOptionTopLogprobs = "cerebras_top_logprobs"
	// CerebrasGenerationOptionUser stores a provider-side end-user identifier.
	CerebrasGenerationOptionUser = "cerebras_user"
)

const (
	// CerebrasResponseExtraFieldID stores the completion identifier.
	CerebrasResponseExtraFieldID = "cerebras_id"
	// CerebrasResponseExtraFieldModel stores the model reported in the response.
	CerebrasResponseExtraFieldModel = "cerebras_model"
	// CerebrasResponseExtraFieldCreated stores the completion's Unix creation timestamp.
	CerebrasResponseExtraFieldCreated = "cerebras_created"
	// CerebrasResponseExtraFieldSystemFingerprint stores the backend configuration fingerprint.
	CerebrasResponseExtraFieldSystemFingerprint = "cerebras_system_fingerprint"
	// CerebrasResponseExtraFieldServiceTier stores the response's service tier.
	CerebrasResponseExtraFieldServiceTier = "cerebras_service_tier"
	// CerebrasResponseExtraFieldServiceTierUsed stores the processing tier Cerebras actually used.
	CerebrasResponseExtraFieldServiceTierUsed = "cerebras_service_tier_used"
	// CerebrasMessageExtraFieldLogprobs stores candidate token log probabilities.
	CerebrasMessageExtraFieldLogprobs = "cerebras_logprobs"
)

// CerebrasServiceTier controls request prioritization.
type CerebrasServiceTier string

const (
	// CerebrasServiceTierPriority requests priority processing.
	CerebrasServiceTierPriority CerebrasServiceTier = "priority"
	// CerebrasServiceTierDefault requests standard processing.
	CerebrasServiceTierDefault CerebrasServiceTier = "default"
	// CerebrasServiceTierAuto lets Cerebras choose the processing tier.
	CerebrasServiceTierAuto CerebrasServiceTier = "auto"
	// CerebrasServiceTierFlex requests flex processing.
	CerebrasServiceTierFlex CerebrasServiceTier = "flex"
)

// WithCerebrasLogitBias sets token logit adjustments for one Cerebras request.
func WithCerebrasLogitBias(value map[string]float64) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionLogitBias] = maps.Clone(value)
	}
}

// WithCerebrasLogprobs controls whether Cerebras returns token log probabilities.
func WithCerebrasLogprobs(enabled bool) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionLogprobs] = enabled
	}
}

// WithCerebrasParallelToolCalls controls parallel function calling.
func WithCerebrasParallelToolCalls(enabled bool) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionParallelToolCalls] = enabled
	}
}

// WithCerebrasPrediction supplies known text that Cerebras can match as predicted output.
func WithCerebrasPrediction(content string) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionPrediction] = content
	}
}

// WithCerebrasPromptCacheKey sets the Cerebras prompt cache routing key.
func WithCerebrasPromptCacheKey(value string) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionPromptCacheKey] = value
	}
}

// WithCerebrasResponseFormat sets a Cerebras response_format object.
func WithCerebrasResponseFormat(value map[string]any) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionResponseFormat] = maps.Clone(value)
	}
}

// WithCerebrasSeed sets the best-effort deterministic sampling seed.
func WithCerebrasSeed(value int) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionSeed] = value
	}
}

// WithCerebrasServiceTier sets the Cerebras processing tier.
func WithCerebrasServiceTier(value CerebrasServiceTier) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionServiceTier] = string(value)
	}
}

// WithCerebrasTopLogprobs sets the number of alternative tokens returned per position.
func WithCerebrasTopLogprobs(value int) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionTopLogprobs] = value
	}
}

// WithCerebrasUser sets the provider-side end-user identifier.
func WithCerebrasUser(value string) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionUser] = value
	}
}

// CerebrasDefaultBaseURL is the Cerebras API server declared by the generated OpenAPI client.
const CerebrasDefaultBaseURL = string(cerebras.DefaultServer)

// CerebrasGenerator implements Generator and StreamingGenerator using the generated Cerebras client.
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
	FrequencyPenalty    *float64
	PresencePenalty     *float64
	LogitBias           map[string]float64
	Logprobs            *bool
	ParallelToolCalls   *bool
	Prediction          *string
	PromptCacheKey      string
	ResponseFormat      map[string]any
	Seed                *int
	ServiceTier         string
	TopLogprobs         *int
	User                string
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	OutputModalities    []Modality
	ThinkingBudget      string
}

// parseCerebrasGenerationOptions validates common and native option types before request construction.
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
	if options.LogitBias, _, err = generationOption[map[string]float64](values, CerebrasGenerationOptionLogitBias); err != nil {
		return nil, err
	}
	logprobs, ok, err := generationOption[bool](values, CerebrasGenerationOptionLogprobs)
	if err != nil {
		return nil, err
	}
	if ok {
		options.Logprobs = &logprobs
	}
	parallelToolCalls, ok, err := generationOption[bool](values, CerebrasGenerationOptionParallelToolCalls)
	if err != nil {
		return nil, err
	}
	if ok {
		options.ParallelToolCalls = &parallelToolCalls
	}
	prediction, ok, err := generationOption[string](values, CerebrasGenerationOptionPrediction)
	if err != nil {
		return nil, err
	}
	if ok {
		options.Prediction = &prediction
	}
	if options.PromptCacheKey, _, err = generationOption[string](values, CerebrasGenerationOptionPromptCacheKey); err != nil {
		return nil, err
	}
	if options.ResponseFormat, _, err = generationOption[map[string]any](values, CerebrasGenerationOptionResponseFormat); err != nil {
		return nil, err
	}
	seed, ok, err := generationOption[int](values, CerebrasGenerationOptionSeed)
	if err != nil {
		return nil, err
	}
	if ok {
		options.Seed = &seed
	}
	if options.ServiceTier, _, err = generationOption[string](values, CerebrasGenerationOptionServiceTier); err != nil {
		return nil, err
	}
	topLogprobs, ok, err := generationOption[int](values, CerebrasGenerationOptionTopLogprobs)
	if err != nil {
		return nil, err
	}
	if ok {
		options.TopLogprobs = &topLogprobs
	}
	if options.User, _, err = generationOption[string](values, CerebrasGenerationOptionUser); err != nil {
		return nil, err
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

// buildCerebrasMessages prepends instructions and converts each dialog role into its matching provider message.
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
			content, err := cerebrasUserContent(message.Blocks)
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

func cerebrasUserContent(blocks []Block) (cerebras.UserMessageContent, error) {
	var text strings.Builder
	parts := make([]cerebras.UserContentPart, 0, len(blocks))
	hasImage := false

	for _, block := range blocks {
		if block.BlockType != Content {
			return cerebras.UserMessageContent{}, fmt.Errorf("cerebras: unsupported block type for user: %q", block.BlockType)
		}
		switch block.ModalityType {
		case Text:
			value := block.Content.String()
			text.WriteString(value)
			parts = append(parts, cerebras.NewTextContentPartUserContentPart(cerebras.TextContentPart{
				Type: cerebras.TextContentPartTypeText,
				Text: value,
			}))
		case Image:
			if block.MimeType != "image/png" && block.MimeType != "image/jpeg" {
				return cerebras.UserMessageContent{}, fmt.Errorf("cerebras: unsupported image MIME type %q", block.MimeType)
			}
			hasImage = true
			parts = append(parts, cerebras.NewImageContentPartUserContentPart(cerebras.ImageContentPart{
				Type: cerebras.ImageContentPartTypeImageURL,
				ImageURL: cerebras.ImageURL{
					URL: fmt.Sprintf("data:%s;base64,%s", block.MimeType, block.Content.String()),
				},
			}))
		default:
			return cerebras.UserMessageContent{}, UnsupportedInputModalityErr(block.ModalityType.String())
		}
	}
	if !hasImage {
		return cerebras.NewStringUserMessageContent(text.String()), nil
	}
	return cerebras.NewUserContentPartArrayUserMessageContent(parts), nil
}

// buildCerebrasAssistantMessage collects visible text, reasoning, and validated tool calls into one assistant turn.
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

// buildRequest converts request-scoped messages, tools, and validated options into the generated Cerebras type.
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
	if options.FrequencyPenalty != nil {
		providerRequest.FrequencyPenalty = cerebras.NewOptFloat64(*options.FrequencyPenalty)
	}
	if options.PresencePenalty != nil {
		providerRequest.PresencePenalty = cerebras.NewOptFloat64(*options.PresencePenalty)
	}
	if options.LogitBias != nil {
		for token, bias := range options.LogitBias {
			if bias < -100 || bias > 100 {
				return nil, &InvalidParameterErr{
					Parameter: CerebrasGenerationOptionLogitBias,
					Reason:    fmt.Sprintf("bias for token %q must be between -100 and 100", token),
				}
			}
		}
		providerRequest.LogitBias = cerebras.NewOptChatCompletionRequestLogitBias(
			cerebras.ChatCompletionRequestLogitBias(options.LogitBias),
		)
	}
	if options.Logprobs != nil {
		providerRequest.Logprobs = cerebras.NewOptBool(*options.Logprobs)
	}
	if options.ParallelToolCalls != nil {
		providerRequest.ParallelToolCalls = cerebras.NewOptBool(*options.ParallelToolCalls)
	}
	if options.Prediction != nil {
		data, marshalErr := json.Marshal(map[string]any{"type": "content", "content": *options.Prediction})
		if marshalErr != nil {
			return nil, &InvalidParameterErr{Parameter: CerebrasGenerationOptionPrediction, Reason: marshalErr.Error()}
		}
		var prediction cerebras.Prediction
		if decodeErr := json.Unmarshal(data, &prediction); decodeErr != nil {
			return nil, &InvalidParameterErr{Parameter: CerebrasGenerationOptionPrediction, Reason: decodeErr.Error()}
		}
		providerRequest.Prediction = cerebras.NewOptPrediction(prediction)
	}
	if options.PromptCacheKey != "" {
		if len(options.PromptCacheKey) > 1024 {
			return nil, &InvalidParameterErr{Parameter: CerebrasGenerationOptionPromptCacheKey, Reason: "must be at most 1024 bytes"}
		}
		providerRequest.PromptCacheKey = cerebras.NewOptString(options.PromptCacheKey)
	}
	if options.ResponseFormat != nil {
		data, marshalErr := json.Marshal(options.ResponseFormat)
		if marshalErr != nil {
			return nil, &InvalidParameterErr{Parameter: CerebrasGenerationOptionResponseFormat, Reason: marshalErr.Error()}
		}
		var responseFormat cerebras.ResponseFormat
		if decodeErr := json.Unmarshal(data, &responseFormat); decodeErr != nil {
			return nil, &InvalidParameterErr{Parameter: CerebrasGenerationOptionResponseFormat, Reason: decodeErr.Error()}
		}
		if validateErr := responseFormat.Validate(); validateErr != nil {
			return nil, &InvalidParameterErr{Parameter: CerebrasGenerationOptionResponseFormat, Reason: validateErr.Error()}
		}
		providerRequest.ResponseFormat = cerebras.NewOptResponseFormat(responseFormat)
	}
	if options.Seed != nil {
		providerRequest.Seed = cerebras.NewOptInt(*options.Seed)
	}
	if options.ServiceTier != "" {
		tier := cerebras.ServiceTier(options.ServiceTier)
		if validateErr := tier.Validate(); validateErr != nil {
			return nil, &InvalidParameterErr{Parameter: CerebrasGenerationOptionServiceTier, Reason: validateErr.Error()}
		}
		providerRequest.ServiceTier = cerebras.NewOptServiceTier(tier)
	}
	if options.TopLogprobs != nil {
		if *options.TopLogprobs < 0 || *options.TopLogprobs > 20 {
			return nil, &InvalidParameterErr{Parameter: CerebrasGenerationOptionTopLogprobs, Reason: "must be between 0 and 20"}
		}
		if options.Logprobs == nil || !*options.Logprobs {
			return nil, &InvalidParameterErr{Parameter: CerebrasGenerationOptionTopLogprobs, Reason: "requires logprobs to be enabled"}
		}
		providerRequest.TopLogprobs = cerebras.NewOptInt(*options.TopLogprobs)
	}
	if options.User != "" {
		providerRequest.User = cerebras.NewOptString(options.User)
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

func cerebrasResponseExtraFields(id, model string, created int64, systemFingerprint, serviceTier, serviceTierUsed string) map[string]interface{} {
	extraFields := map[string]interface{}{
		CerebrasResponseExtraFieldID:      id,
		CerebrasResponseExtraFieldModel:   model,
		CerebrasResponseExtraFieldCreated: created,
	}
	if systemFingerprint != "" {
		extraFields[CerebrasResponseExtraFieldSystemFingerprint] = systemFingerprint
	}
	if serviceTier != "" {
		extraFields[CerebrasResponseExtraFieldServiceTier] = serviceTier
	}
	if serviceTierUsed != "" {
		extraFields[CerebrasResponseExtraFieldServiceTierUsed] = serviceTierUsed
	}
	return extraFields
}

func cerebrasLogprobsValue(logprobs cerebras.LogProbs) (map[string]any, error) {
	encoded, err := json.Marshal(logprobs)
	if err != nil {
		return nil, fmt.Errorf("cerebras: encode logprobs: %w", err)
	}
	var value map[string]any
	if err := json.Unmarshal(encoded, &value); err != nil {
		return nil, fmt.Errorf("cerebras: decode logprobs: %w", err)
	}
	return value, nil
}

func mergeCerebrasLogprobs(destination, source map[string]any) {
	for key, value := range source {
		if sourceItems, ok := value.([]any); ok {
			if destinationItems, exists := destination[key].([]any); exists {
				destination[key] = append(destinationItems, sourceItems...)
				continue
			}
		}
		destination[key] = value
	}
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

	result := Response{
		UsageMetadata: make(Metadata),
		ExtraFields: cerebrasResponseExtraFields(
			completion.ID,
			completion.Model,
			completion.Created,
			completion.SystemFingerprint.Or(""),
			completion.ServiceTier.Or(""),
			string(completion.ServiceTierUsed.Or("")),
		),
	}
	if usage, ok := completion.Usage.Get(); ok {
		addCerebrasUsageMetadata(result.UsageMetadata, usage)
	}
	if timeInfo, ok := completion.TimeInfo.Get(); ok {
		addCerebrasTimeMetadata(result.UsageMetadata, timeInfo)
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
		message := Message{Role: Assistant, Blocks: blocks}
		if logprobs, ok := choice.Logprobs.Get(); ok {
			value, valueErr := cerebrasLogprobsValue(logprobs)
			if valueErr != nil {
				return result, valueErr
			}
			message.ExtraFields = map[string]interface{}{CerebrasMessageExtraFieldLogprobs: value}
		}
		result.Candidates = append(result.Candidates, message)
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
		var finalTimeInfo cerebras.TimeInfo
		var hasFinalTimeInfo bool
		responseExtraFields := make(map[string]interface{})
		streamLogprobs := make(map[string]any)
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
			chunkExtraFields := cerebrasResponseExtraFields(
				chunk.ID,
				chunk.Model,
				chunk.Created,
				chunk.SystemFingerprint.Or(""),
				chunk.ServiceTier.Or(""),
				string(chunk.ServiceTierUsed.Or("")),
			)
			maps.Copy(responseExtraFields, chunkExtraFields)
			yieldBlock := func(block Block, candidateIndex int) bool {
				return yield(StreamChunk{
					Block:               block,
					ResponseExtraFields: chunkExtraFields,
					CandidatesIndex:     candidateIndex,
				})
			}
			if usage, ok := chunk.Usage.Get(); ok {
				finalUsage = usage
				hasFinalUsage = true
			}
			if timeInfo, ok := chunk.TimeInfo.Get(); ok {
				finalTimeInfo = timeInfo
				hasFinalTimeInfo = true
			}
			for _, choice := range chunk.Choices {
				if logprobs, ok := choice.Logprobs.Get(); ok && choice.Index == 0 {
					value, valueErr := cerebrasLogprobsValue(logprobs)
					if valueErr != nil {
						yield(StreamChunk{Err: valueErr})
						return
					}
					mergeCerebrasLogprobs(streamLogprobs, value)
				}
				if finishReason, ok := choice.FinishReason.Get(); ok {
					if _, finishErr := cerebrasFinishReason(string(finishReason)); finishErr != nil {
						yield(StreamChunk{Err: finishErr})
						return
					}
				}
				if reasoning, ok := choice.Delta.Reasoning.Get(); ok && reasoning != "" {
					if !yieldBlock(cerebrasThinkingBlock(reasoning), choice.Index) {
						return
					}
				}
				if content, ok := choice.Delta.Content.Get(); ok && content != "" {
					if !yieldBlock(TextBlock(content), choice.Index) {
						return
					}
				}
				for _, call := range choice.Delta.ToolCalls {
					if name := call.Function.Name.Or(""); name != "" {
						if !yieldBlock(Block{
							ID:           call.ID.Or(""),
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(name),
						}, choice.Index) {
							return
						}
					}
					if arguments := call.Function.Arguments.Or(""); arguments != "" {
						if !yieldBlock(Block{
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(arguments),
						}, choice.Index) {
							return
						}
					}
				}
			}
		}
		metadata := make(Metadata)
		if hasFinalUsage {
			addCerebrasUsageMetadata(metadata, finalUsage)
		}
		if hasFinalTimeInfo {
			addCerebrasTimeMetadata(metadata, finalTimeInfo)
		}
		var messageExtraFields map[string]interface{}
		if len(streamLogprobs) > 0 {
			messageExtraFields = map[string]interface{}{CerebrasMessageExtraFieldLogprobs: streamLogprobs}
		}
		terminalBlock := SeparatorBlock()
		if len(metadata) > 0 {
			terminalBlock = MetadataBlock(metadata)
		}
		if len(metadata) > 0 || len(messageExtraFields) > 0 || len(responseExtraFields) > 0 {
			yield(StreamChunk{
				Block:               terminalBlock,
				MessageExtraFields:  messageExtraFields,
				ResponseExtraFields: responseExtraFields,
				CandidatesIndex:     0,
			})
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
	if imageTokens := usage.ImageTokens.Or(0); imageTokens > 0 {
		metadata[CerebrasUsageMetricImageTokens] = imageTokens
	}
	if details, ok := usage.PromptTokensDetails.Get(); ok && details.CachedTokens.Or(0) > 0 {
		metadata[UsageMetricCacheReadTokens] = details.CachedTokens.Or(0)
	}
	if details, ok := usage.CompletionTokensDetails.Get(); ok && details.ReasoningTokens.Or(0) > 0 {
		metadata[UsageMetricReasoningTokens] = details.ReasoningTokens.Or(0)
	}
}

func addCerebrasTimeMetadata(metadata Metadata, timeInfo cerebras.TimeInfo) {
	if value, ok := timeInfo.QueueTime.Get(); ok {
		metadata[CerebrasUsageMetricQueueTimeSeconds] = value
	}
	if value, ok := timeInfo.PromptTime.Get(); ok {
		metadata[CerebrasUsageMetricPromptTimeSeconds] = value
	}
	if value, ok := timeInfo.CompletionTime.Get(); ok {
		metadata[CerebrasUsageMetricCompletionTimeSeconds] = value
	}
	if value, ok := timeInfo.TotalTime.Get(); ok {
		metadata[CerebrasUsageMetricTotalTimeSeconds] = value
	}
	if value, ok := timeInfo.Created.Get(); ok {
		metadata[CerebrasUsageMetricTimeInfoCreated] = value
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
