package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-faster/jx"

	"github.com/spachava753/gai/internal/openrouter"
)

const (
	// OpenRouterExtraFieldReasoningType stores the reasoning detail type (e.g., "reasoning.summary", "reasoning.text", "reasoning.encrypted").
	// Present in Block.ExtraFields for Thinking blocks from OpenRouter responses.
	OpenRouterExtraFieldReasoningType = "reasoning_type"

	// OpenRouterExtraFieldReasoningFormat stores the reasoning detail format (e.g., "anthropic-claude-v1").
	// Present in Block.ExtraFields for Thinking blocks from OpenRouter responses.
	OpenRouterExtraFieldReasoningFormat = "reasoning_format"

	// OpenRouterExtraFieldReasoningIndex stores the zero-based index of the reasoning detail in the response.
	// Present in Block.ExtraFields for Thinking blocks from OpenRouter responses.
	OpenRouterExtraFieldReasoningIndex = "reasoning_index"

	// OpenRouterExtraFieldReasoningSignature stores the signature for encrypted reasoning details.
	// Present in Block.ExtraFields for Thinking blocks with type "reasoning.text" when a signature is provided.
	OpenRouterExtraFieldReasoningSignature = "reasoning_signature"

	// OpenRouterUsageMetricReasoningDetailsAvailable indicates whether reasoning_details were present in the response.
	// Stored in Response.UsageMetadata as a boolean value.
	OpenRouterUsageMetricReasoningDetailsAvailable = "reasoning_details_available"

	// OpenRouterUsageMetricCost is the completion cost reported by OpenRouter.
	OpenRouterUsageMetricCost = "cost"

	// OpenRouterUsageMetricIsBYOK reports whether OpenRouter used a caller-provided provider key.
	OpenRouterUsageMetricIsBYOK = "is_byok"

	// OpenRouterUsageMetricCostDetails stores OpenRouter's native cost breakdown.
	OpenRouterUsageMetricCostDetails = "cost_details"

	// OpenRouterUsageMetricServerToolUseDetails stores server-side tool usage counts.
	OpenRouterUsageMetricServerToolUseDetails = "server_tool_use_details"

	// OpenRouterUsageMetricPromptTokenDetails stores OpenRouter's full prompt token breakdown.
	OpenRouterUsageMetricPromptTokenDetails = "prompt_token_details"

	// OpenRouterUsageMetricCompletionTokenDetails stores OpenRouter's full completion token breakdown.
	OpenRouterUsageMetricCompletionTokenDetails = "completion_token_details"
)

const (
	// OpenRouterGenerationOptionLogitBias stores a map of token IDs to logit adjustments.
	OpenRouterGenerationOptionLogitBias = "openrouter_logit_bias"
	// OpenRouterGenerationOptionLogprobs controls token log-probability output.
	OpenRouterGenerationOptionLogprobs = "openrouter_logprobs"
	// OpenRouterGenerationOptionMinP stores the minimum probability sampling threshold.
	OpenRouterGenerationOptionMinP = "openrouter_min_p"
	// OpenRouterGenerationOptionModels stores the ordered fallback model list.
	OpenRouterGenerationOptionModels = "openrouter_models"
	// OpenRouterGenerationOptionParallelToolCalls controls parallel function calling.
	OpenRouterGenerationOptionParallelToolCalls = "openrouter_parallel_tool_calls"
	// OpenRouterGenerationOptionPrediction stores known predicted output text.
	OpenRouterGenerationOptionPrediction = "openrouter_prediction"
	// OpenRouterGenerationOptionPromptCacheKey stores a prompt cache routing key.
	OpenRouterGenerationOptionPromptCacheKey = "openrouter_prompt_cache_key"
	// OpenRouterGenerationOptionProvider stores OpenRouter provider routing preferences.
	OpenRouterGenerationOptionProvider = "openrouter_provider"
	// OpenRouterGenerationOptionRepetitionPenalty stores the repetition penalty.
	OpenRouterGenerationOptionRepetitionPenalty = "openrouter_repetition_penalty"
	// OpenRouterGenerationOptionResponseFormat stores an OpenRouter response_format object.
	OpenRouterGenerationOptionResponseFormat = "openrouter_response_format"
	// OpenRouterGenerationOptionSeed stores the sampling seed.
	OpenRouterGenerationOptionSeed = "openrouter_seed"
	// OpenRouterGenerationOptionServiceTier stores the requested processing tier.
	OpenRouterGenerationOptionServiceTier = "openrouter_service_tier"
	// OpenRouterGenerationOptionSessionID stores the session identifier.
	OpenRouterGenerationOptionSessionID = "openrouter_session_id"
	// OpenRouterGenerationOptionTopA stores the top-a sampling threshold.
	OpenRouterGenerationOptionTopA = "openrouter_top_a"
	// OpenRouterGenerationOptionTopLogprobs stores the number of alternative tokens per position.
	OpenRouterGenerationOptionTopLogprobs = "openrouter_top_logprobs"
	// OpenRouterGenerationOptionUser stores a provider-side end-user identifier.
	OpenRouterGenerationOptionUser = "openrouter_user"
)

const (
	// OpenRouterResponseExtraFieldID stores the completion identifier.
	OpenRouterResponseExtraFieldID = "openrouter_id"
	// OpenRouterResponseExtraFieldModel stores the model reported in the response.
	OpenRouterResponseExtraFieldModel = "openrouter_model"
	// OpenRouterResponseExtraFieldCreated stores the completion's Unix creation timestamp.
	OpenRouterResponseExtraFieldCreated = "openrouter_created"
	// OpenRouterResponseExtraFieldSystemFingerprint stores the backend configuration fingerprint.
	OpenRouterResponseExtraFieldSystemFingerprint = "openrouter_system_fingerprint"
	// OpenRouterResponseExtraFieldServiceTier stores the processing tier OpenRouter used.
	OpenRouterResponseExtraFieldServiceTier = "openrouter_service_tier"
	// OpenRouterResponseExtraFieldMetadata stores OpenRouter's native response metadata object.
	OpenRouterResponseExtraFieldMetadata = "openrouter_metadata"
	// OpenRouterMessageExtraFieldLogprobs stores candidate token log probabilities.
	OpenRouterMessageExtraFieldLogprobs = "openrouter_logprobs"
)

// OpenRouterServiceTier selects the processing tier requested from OpenRouter.
type OpenRouterServiceTier string

const (
	// OpenRouterServiceTierAuto lets OpenRouter choose the processing tier.
	OpenRouterServiceTierAuto OpenRouterServiceTier = "auto"
	// OpenRouterServiceTierDefault requests standard processing.
	OpenRouterServiceTierDefault OpenRouterServiceTier = "default"
	// OpenRouterServiceTierFast requests fast processing.
	OpenRouterServiceTierFast OpenRouterServiceTier = "fast"
	// OpenRouterServiceTierFlex requests flex processing.
	OpenRouterServiceTierFlex OpenRouterServiceTier = "flex"
	// OpenRouterServiceTierPriority requests priority processing.
	OpenRouterServiceTierPriority OpenRouterServiceTier = "priority"
	// OpenRouterServiceTierScale requests scale processing.
	OpenRouterServiceTierScale OpenRouterServiceTier = "scale"
)

// WithOpenRouterLogitBias sets token logit adjustments for one OpenRouter request.
func WithOpenRouterLogitBias(value map[string]float64) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionLogitBias] = maps.Clone(value) }
}

// WithOpenRouterLogprobs controls whether OpenRouter returns token log probabilities.
func WithOpenRouterLogprobs(enabled bool) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionLogprobs] = enabled }
}

// WithOpenRouterMinP sets OpenRouter's minimum probability sampling threshold.
func WithOpenRouterMinP(value float64) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionMinP] = value }
}

// WithOpenRouterFallbackModels sets the ordered model fallback list.
func WithOpenRouterFallbackModels(values ...string) GenerationOption {
	return func(options GenerationOptions) {
		options[OpenRouterGenerationOptionModels] = append([]string(nil), values...)
	}
}

// WithOpenRouterParallelToolCalls controls parallel function calling.
func WithOpenRouterParallelToolCalls(enabled bool) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionParallelToolCalls] = enabled }
}

// WithOpenRouterPrediction supplies known text that OpenRouter can match as predicted output.
func WithOpenRouterPrediction(content string) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionPrediction] = content }
}

// WithOpenRouterPromptCacheKey sets the OpenRouter prompt cache routing key.
func WithOpenRouterPromptCacheKey(value string) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionPromptCacheKey] = value }
}

// WithOpenRouterProviderPreferences sets OpenRouter's provider routing object.
func WithOpenRouterProviderPreferences(value map[string]any) GenerationOption {
	return func(options GenerationOptions) {
		options[OpenRouterGenerationOptionProvider] = maps.Clone(value)
	}
}

// WithOpenRouterRepetitionPenalty sets OpenRouter's repetition penalty.
func WithOpenRouterRepetitionPenalty(value float64) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionRepetitionPenalty] = value }
}

// WithOpenRouterResponseFormat sets an OpenRouter response_format object.
func WithOpenRouterResponseFormat(value map[string]any) GenerationOption {
	return func(options GenerationOptions) {
		options[OpenRouterGenerationOptionResponseFormat] = maps.Clone(value)
	}
}

// WithOpenRouterSeed sets the best-effort deterministic sampling seed.
func WithOpenRouterSeed(value int) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionSeed] = value }
}

// WithOpenRouterServiceTier sets the requested OpenRouter processing tier.
func WithOpenRouterServiceTier(value OpenRouterServiceTier) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionServiceTier] = string(value) }
}

// WithOpenRouterSessionID sets the OpenRouter session identifier.
func WithOpenRouterSessionID(value string) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionSessionID] = value }
}

// WithOpenRouterTopA sets OpenRouter's top-a sampling threshold.
func WithOpenRouterTopA(value float64) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionTopA] = value }
}

// WithOpenRouterTopLogprobs sets the number of alternative tokens returned per position.
func WithOpenRouterTopLogprobs(value int) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionTopLogprobs] = value }
}

// WithOpenRouterUser sets the provider-side end-user identifier.
func WithOpenRouterUser(value string) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionUser] = value }
}

// OpenRouterDefaultBaseURL is the OpenRouter API server declared by the generated OpenAPI client.
const OpenRouterDefaultBaseURL = string(openrouter.DefaultServer)

// OpenRouterGenerator implements Generator and StreamingGenerator using the generated OpenRouter client.
// It normalizes replayable reasoning details, function tools, multimodal input,
// provider errors, and usage data from OpenRouter Chat Completions.
type OpenRouterGenerator struct {
	client *openrouter.Client
}

func classifyOpenRouterError(statusCode int, errorType string) APIErrorKind {
	switch errorType {
	case "authentication":
		return APIErrorKindAuthentication
	case "permission_denied":
		return APIErrorKindPermission
	case "rate_limit_exceeded":
		return APIErrorKindRateLimit
	case "provider_overloaded":
		return APIErrorKindOverloaded
	case "provider_unavailable":
		return APIErrorKindServiceUnavailable
	case "timeout":
		return APIErrorKindTimeout
	case "server", "unmapped":
		return APIErrorKindServer
	case "not_found", "image_not_found":
		return APIErrorKindNotFound
	case "payload_too_large":
		return APIErrorKindRequestTooLarge
	case "content_policy_violation", "refusal":
		return APIErrorKindContentPolicy
	case "payment_required", "invalid_request", "invalid_prompt", "precondition_failed", "unprocessable",
		"context_length_exceeded", "max_tokens_exceeded", "token_limit_exceeded", "string_too_long",
		"invalid_image", "image_too_small", "unsupported_image_format", "image_download_failed":
		return APIErrorKindInvalidRequest
	case "image_too_large":
		return APIErrorKindRequestTooLarge
	default:
		return classifyHTTPStatus(statusCode)
	}
}

func openRouterErrorType(detail openrouter.ErrorDetail) string {
	if metadata, ok := detail.Metadata.Get(); ok {
		if errorType := metadata.ErrorType.Or(""); errorType != "" {
			return errorType
		}
	}
	return detail.ErrorType.Or("")
}

func mapOpenRouterErrorDetail(detail openrouter.ErrorDetail, statusCode int, rawBody string) *ApiErr {
	if statusCode <= 0 {
		statusCode = detail.Code
	}
	if statusCode <= 0 {
		statusCode = 500
	}
	return &ApiErr{
		Provider:   ProviderOpenRouter,
		Kind:       classifyOpenRouterError(statusCode, openRouterErrorType(detail)),
		StatusCode: statusCode,
		Message:    detail.Message,
		RawBody:    rawBody,
	}
}

func mapOpenRouterTransportError(err error) error {
	var statusErr *openrouter.ErrorEnvelopeStatusCodeWithHeaders
	if !errors.As(err, &statusErr) {
		return err
	}
	rawBody, _ := json.Marshal(statusErr.Response)
	mapped := mapOpenRouterErrorDetail(statusErr.Response.Error, statusErr.StatusCode, string(rawBody))
	if retryAfter := statusErr.RetryAfter.Or(""); retryAfter != "" {
		if delay, ok := parseRetryAfter(retryAfter, time.Now()); ok {
			mapped.RetryAfterDuration = &delay
		}
	}
	mapped.Cause = err
	return mapped
}

type openRouterGenerationOptions struct {
	Temperature         *float64
	TopP                *float64
	TopK                *uint
	LogitBias           map[string]float64
	Logprobs            *bool
	MinP                *float64
	FrequencyPenalty    *float64
	PresencePenalty     *float64
	CandidateCount      *uint
	Models              []string
	ParallelToolCalls   *bool
	Prediction          *string
	PromptCacheKey      string
	Provider            map[string]any
	RepetitionPenalty   *float64
	ResponseFormat      map[string]any
	Seed                *int
	ServiceTier         string
	SessionID           string
	TopA                *float64
	TopLogprobs         *int
	User                string
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	ThinkingBudget      string
}

func parseOpenRouterGenerationOptions(values GenerationOptions) (*openRouterGenerationOptions, error) {
	options := &openRouterGenerationOptions{}

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
	topK, ok, err := generationOption[uint](values, GenerationOptionTopK)
	if err != nil {
		return nil, err
	}
	if ok {
		options.TopK = &topK
	}
	if options.LogitBias, _, err = generationOption[map[string]float64](values, OpenRouterGenerationOptionLogitBias); err != nil {
		return nil, err
	}
	logprobs, ok, err := generationOption[bool](values, OpenRouterGenerationOptionLogprobs)
	if err != nil {
		return nil, err
	}
	if ok {
		options.Logprobs = &logprobs
	}
	minP, ok, err := generationOption[float64](values, OpenRouterGenerationOptionMinP)
	if err != nil {
		return nil, err
	}
	if ok {
		options.MinP = &minP
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
	if options.Models, _, err = generationOption[[]string](values, OpenRouterGenerationOptionModels); err != nil {
		return nil, err
	}
	parallelToolCalls, ok, err := generationOption[bool](values, OpenRouterGenerationOptionParallelToolCalls)
	if err != nil {
		return nil, err
	}
	if ok {
		options.ParallelToolCalls = &parallelToolCalls
	}
	prediction, ok, err := generationOption[string](values, OpenRouterGenerationOptionPrediction)
	if err != nil {
		return nil, err
	}
	if ok {
		options.Prediction = &prediction
	}
	if options.PromptCacheKey, _, err = generationOption[string](values, OpenRouterGenerationOptionPromptCacheKey); err != nil {
		return nil, err
	}
	if options.Provider, _, err = generationOption[map[string]any](values, OpenRouterGenerationOptionProvider); err != nil {
		return nil, err
	}
	repetitionPenalty, ok, err := generationOption[float64](values, OpenRouterGenerationOptionRepetitionPenalty)
	if err != nil {
		return nil, err
	}
	if ok {
		options.RepetitionPenalty = &repetitionPenalty
	}
	if options.ResponseFormat, _, err = generationOption[map[string]any](values, OpenRouterGenerationOptionResponseFormat); err != nil {
		return nil, err
	}
	seed, ok, err := generationOption[int](values, OpenRouterGenerationOptionSeed)
	if err != nil {
		return nil, err
	}
	if ok {
		options.Seed = &seed
	}
	if options.ServiceTier, _, err = generationOption[string](values, OpenRouterGenerationOptionServiceTier); err != nil {
		return nil, err
	}
	if options.SessionID, _, err = generationOption[string](values, OpenRouterGenerationOptionSessionID); err != nil {
		return nil, err
	}
	topA, ok, err := generationOption[float64](values, OpenRouterGenerationOptionTopA)
	if err != nil {
		return nil, err
	}
	if ok {
		options.TopA = &topA
	}
	topLogprobs, ok, err := generationOption[int](values, OpenRouterGenerationOptionTopLogprobs)
	if err != nil {
		return nil, err
	}
	if ok {
		options.TopLogprobs = &topLogprobs
	}
	if options.User, _, err = generationOption[string](values, OpenRouterGenerationOptionUser); err != nil {
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
	if options.ThinkingBudget, _, err = generationOption[string](values, GenerationOptionThinkingBudget); err != nil {
		return nil, err
	}
	return options, nil
}

// NewOpenRouterGenerator creates a stateless OpenRouter generator.
// A nil httpClient uses the default HTTP client. An empty baseURL uses
// OpenRouterDefaultBaseURL. apiKey is required.
func NewOpenRouterGenerator(httpClient *http.Client, baseURL, apiKey string) (*OpenRouterGenerator, error) {
	if baseURL == "" {
		baseURL = OpenRouterDefaultBaseURL
	}
	if apiKey == "" {
		return nil, fmt.Errorf("openrouter: %w", ErrMissingAPIKey)
	}
	options := make([]openrouter.ClientOption, 0, 1)
	if httpClient != nil {
		options = append(options, openrouter.WithClient(httpClient))
	}
	client, err := openrouter.NewClient(baseURL, openRouterSecuritySource{apiKey: apiKey}, options...)
	if err != nil {
		return nil, fmt.Errorf("openrouter: create client: %w", err)
	}
	return &OpenRouterGenerator{client: client}, nil
}

type openRouterSecuritySource struct {
	apiKey string
}

func (s openRouterSecuritySource) BearerAuth(ctx context.Context, operationName openrouter.OperationName) (openrouter.BearerAuth, error) {
	return openrouter.BearerAuth{Token: s.apiKey}, nil
}

func buildOpenRouterReasoningDetails(blocks []Block) []openrouter.ReasoningDetail {
	details := make([]openrouter.ReasoningDetail, 0, len(blocks))
	for i, block := range blocks {
		if block.BlockType != Thinking {
			continue
		}
		detail := openrouter.ReasoningDetail{
			Type:   "reasoning.text",
			Format: openrouter.NewOptNilString("anthropic-claude-v1"),
			Index:  openrouter.NewOptInt(i),
		}
		if block.ID != "" {
			detail.ID = openrouter.NewOptNilString(block.ID)
		}
		if reasoningType, ok := block.ExtraFields[OpenRouterExtraFieldReasoningType].(string); ok {
			detail.Type = reasoningType
		}
		if format, ok := block.ExtraFields[OpenRouterExtraFieldReasoningFormat].(string); ok {
			detail.Format = openrouter.NewOptNilString(format)
		}
		if index, ok := block.ExtraFields[OpenRouterExtraFieldReasoningIndex].(int); ok {
			detail.Index = openrouter.NewOptInt(index)
		}

		switch detail.Type {
		case "reasoning.summary":
			detail.Summary = openrouter.NewOptString(block.Content.String())
		case "reasoning.encrypted":
			detail.Data = openrouter.NewOptString(block.Content.String())
		default:
			detail.Text = openrouter.NewOptNilString(block.Content.String())
			if signature, ok := block.ExtraFields[OpenRouterExtraFieldReasoningSignature].(string); ok {
				detail.Signature = openrouter.NewOptNilString(signature)
			}
		}
		details = append(details, detail)
	}
	return details
}

func convertToolToOpenRouter(tool Tool) (openrouter.FunctionTool, error) {
	function := openrouter.FunctionObject{Name: tool.Name}
	if tool.Description != "" {
		function.Description = openrouter.NewOptString(tool.Description)
	}
	if tool.InputSchema != nil {
		schemaJSON, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return openrouter.FunctionTool{}, err
		}
		var rawParameters map[string]json.RawMessage
		if err := json.Unmarshal(schemaJSON, &rawParameters); err != nil {
			return openrouter.FunctionTool{}, err
		}
		parameters := make(openrouter.FunctionObjectParameters, len(rawParameters))
		for name, raw := range rawParameters {
			parameters[name] = jx.Raw(raw)
		}
		function.Parameters = openrouter.NewOptFunctionObjectParameters(parameters)
	}
	converted := openrouter.FunctionTool{Type: openrouter.FunctionToolTypeFunction, Function: function}
	if err := converted.Validate(); err != nil {
		return openrouter.FunctionTool{}, err
	}
	return converted, nil
}

func convertToolsToOpenRouter(tools []Tool) ([]openrouter.FunctionTool, error) {
	converted := make([]openrouter.FunctionTool, 0, len(tools))
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
		providerTool, err := convertToolToOpenRouter(tool)
		if err != nil {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
		}
		converted = append(converted, providerTool)
	}
	return converted, nil
}

func buildOpenRouterMessages(request GenerationRequest) ([]openrouter.Message, error) {
	messages := make([]openrouter.Message, 0, len(request.Dialog)+1)
	instructions, err := joinedTextInstructions(request.Instructions)
	if err != nil {
		return nil, err
	}
	if instructions != "" {
		messages = append(messages, openrouter.NewSystemMessageMessage(openrouter.SystemMessage{
			Role: openrouter.SystemMessageRoleSystem, Content: instructions,
		}))
	}
	for i, message := range request.Dialog {
		switch message.Role {
		case User:
			user, err := buildOpenRouterUserMessage(message.Blocks)
			if err != nil {
				return nil, err
			}
			messages = append(messages, openrouter.NewUserMessageMessage(user))
		case Assistant:
			assistant, err := buildOpenRouterAssistantMessage(message.Blocks)
			if err != nil {
				return nil, err
			}
			messages = append(messages, openrouter.NewAssistantMessageMessage(assistant))
		case ToolResult:
			toolResult, err := buildOpenRouterToolMessage(message.Blocks)
			if err != nil {
				return nil, err
			}
			messages = append(messages, openrouter.NewToolMessageMessage(toolResult))
		default:
			return nil, fmt.Errorf("openrouter: unsupported role at index %d: %v", i, message.Role)
		}
	}
	return messages, nil
}

func buildOpenRouterUserMessage(blocks []Block) (openrouter.UserMessage, error) {
	if len(blocks) == 0 {
		return openrouter.UserMessage{}, fmt.Errorf("openrouter: user message must have at least one block")
	}
	if len(blocks) == 1 && blocks[0].BlockType == Content && blocks[0].ModalityType == Text {
		return openrouter.UserMessage{
			Role:    openrouter.UserMessageRoleUser,
			Content: openrouter.NewStringUserMessageContent(blocks[0].Content.String()),
		}, nil
	}
	parts := make([]openrouter.UserContentPart, 0, len(blocks))
	for _, block := range blocks {
		if block.BlockType != Content {
			return openrouter.UserMessage{}, fmt.Errorf("openrouter: unsupported user block type %q", block.BlockType)
		}
		switch block.ModalityType {
		case Text:
			parts = append(parts, openrouter.NewTextContentPartUserContentPart(openrouter.TextContentPart{
				Type: openrouter.TextContentPartTypeText, Text: block.Content.String(),
			}))
		case Image:
			if block.MimeType == "" {
				return openrouter.UserMessage{}, fmt.Errorf("openrouter: image block missing MIME type")
			}
			dataURL := fmt.Sprintf("data:%s;base64,%s", block.MimeType, block.Content.String())
			if block.MimeType == "application/pdf" {
				filenameValue, ok := block.ExtraFields[BlockFieldFilenameKey]
				if !ok {
					return openrouter.UserMessage{}, fmt.Errorf("openrouter: PDF block missing filename")
				}
				filename, ok := filenameValue.(string)
				if !ok {
					return openrouter.UserMessage{}, fmt.Errorf("openrouter: PDF filename is not a string")
				}
				parts = append(parts, openrouter.NewFileContentPartUserContentPart(openrouter.FileContentPart{
					Type: openrouter.FileContentPartTypeFile,
					File: openrouter.FileContentPartFile{FileData: dataURL, Filename: filename},
				}))
				continue
			}
			parts = append(parts, openrouter.NewImageContentPartUserContentPart(openrouter.ImageContentPart{
				Type:     openrouter.ImageContentPartTypeImageURL,
				ImageURL: openrouter.ImageContentPartImageURL{URL: dataURL},
			}))
		case Audio:
			format, ok := strings.CutPrefix(block.MimeType, "audio/")
			if !ok || (format != "wav" && format != "mp3") {
				return openrouter.UserMessage{}, fmt.Errorf("openrouter: unsupported audio format %q", block.MimeType)
			}
			parts = append(parts, openrouter.NewAudioContentPartUserContentPart(openrouter.AudioContentPart{
				Type:       openrouter.AudioContentPartTypeInputAudio,
				InputAudio: openrouter.AudioContentPartInputAudio{Data: block.Content.String(), Format: format},
			}))
		default:
			return openrouter.UserMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
		}
	}
	return openrouter.UserMessage{
		Role:    openrouter.UserMessageRoleUser,
		Content: openrouter.NewUserContentPartArrayUserMessageContent(parts),
	}, nil
}

func buildOpenRouterAssistantMessage(blocks []Block) (openrouter.AssistantMessage, error) {
	message := openrouter.AssistantMessage{Role: openrouter.AssistantMessageRoleAssistant}
	var content strings.Builder
	var thinking []Block
	for _, block := range blocks {
		switch block.BlockType {
		case Content:
			switch block.ModalityType {
			case Text:
				content.WriteString(block.Content.String())
			case Audio:
				if block.ID == "" {
					return openrouter.AssistantMessage{}, fmt.Errorf("openrouter: assistant audio block missing ID")
				}
				message.Audio = openrouter.NewOptAssistantAudio(openrouter.AssistantAudio{ID: openrouter.NewOptString(block.ID)})
			default:
				return openrouter.AssistantMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
		case Thinking:
			if block.ModalityType != Text {
				return openrouter.AssistantMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
			}
			thinking = append(thinking, block)
		case ToolCall:
			if block.ID == "" {
				return openrouter.AssistantMessage{}, fmt.Errorf("openrouter: tool call block missing ID")
			}
			var input ToolCallInput
			if err := json.Unmarshal([]byte(block.Content.String()), &input); err != nil {
				return openrouter.AssistantMessage{}, fmt.Errorf("openrouter: invalid tool call content: %w", err)
			}
			arguments, err := json.Marshal(input.Parameters)
			if err != nil {
				return openrouter.AssistantMessage{}, fmt.Errorf("openrouter: marshal tool arguments: %w", err)
			}
			message.ToolCalls = append(message.ToolCalls, openrouter.ToolCall{
				ID: block.ID, Type: openrouter.ToolCallTypeFunction,
				Function: openrouter.ToolCallFunction{Name: input.Name, Arguments: string(arguments)},
			})
		default:
			return openrouter.AssistantMessage{}, fmt.Errorf("openrouter: unsupported assistant block type %q", block.BlockType)
		}
	}
	if content.Len() > 0 {
		message.Content = openrouter.NewOptNilString(content.String())
	}
	message.ReasoningDetails = buildOpenRouterReasoningDetails(thinking)
	return message, nil
}

func buildOpenRouterToolMessage(blocks []Block) (openrouter.ToolMessage, error) {
	if len(blocks) == 0 {
		return openrouter.ToolMessage{}, fmt.Errorf("openrouter: tool result message must have at least one block")
	}
	toolCallID := blocks[0].ID
	if toolCallID == "" {
		return openrouter.ToolMessage{}, fmt.Errorf("openrouter: tool result block must have a tool call ID")
	}
	var content strings.Builder
	for _, block := range blocks {
		if block.ID != toolCallID {
			return openrouter.ToolMessage{}, fmt.Errorf("openrouter: all tool result blocks must have the same ID")
		}
		if block.BlockType != Content {
			return openrouter.ToolMessage{}, fmt.Errorf("openrouter: unsupported tool result block type %q", block.BlockType)
		}
		if block.ModalityType != Text {
			return openrouter.ToolMessage{}, UnsupportedInputModalityErr(block.ModalityType.String())
		}
		content.WriteString(block.Content.String())
	}
	return openrouter.ToolMessage{Role: openrouter.ToolMessageRoleTool, Content: content.String(), ToolCallID: toolCallID}, nil
}

func (g *OpenRouterGenerator) buildRequest(request GenerationRequest) (*openrouter.ChatCompletionRequest, error) {
	options, err := parseOpenRouterGenerationOptions(request.Options)
	if err != nil {
		return nil, err
	}
	messages, err := buildOpenRouterMessages(request)
	if err != nil {
		return nil, err
	}
	tools, err := convertToolsToOpenRouter(request.Tools)
	if err != nil {
		return nil, err
	}
	providerRequest := &openrouter.ChatCompletionRequest{Model: request.Model, Messages: messages, Stream: openrouter.NewOptBool(false)}
	if len(tools) > 0 {
		providerRequest.Tools = tools
	}
	if options.Temperature != nil {
		providerRequest.Temperature = openrouter.NewOptFloat64(*options.Temperature)
	}
	if options.TopP != nil {
		providerRequest.TopP = openrouter.NewOptFloat64(*options.TopP)
	}
	if options.TopK != nil {
		providerRequest.TopK = openrouter.NewOptInt(int(*options.TopK))
	}
	if options.LogitBias != nil {
		providerRequest.LogitBias = openrouter.NewOptChatCompletionRequestLogitBias(
			openrouter.ChatCompletionRequestLogitBias(options.LogitBias),
		)
	}
	if options.Logprobs != nil {
		providerRequest.Logprobs = openrouter.NewOptBool(*options.Logprobs)
	}
	if options.MinP != nil {
		providerRequest.MinP = openrouter.NewOptFloat64(*options.MinP)
	}
	if options.FrequencyPenalty != nil {
		providerRequest.FrequencyPenalty = openrouter.NewOptFloat64(*options.FrequencyPenalty)
	}
	if options.PresencePenalty != nil {
		providerRequest.PresencePenalty = openrouter.NewOptFloat64(*options.PresencePenalty)
	}
	if options.CandidateCount != nil {
		providerRequest.N = openrouter.NewOptInt(int(*options.CandidateCount))
	}
	if len(options.Models) > 0 {
		providerRequest.Models = options.Models
	}
	if options.ParallelToolCalls != nil {
		providerRequest.ParallelToolCalls = openrouter.NewOptBool(*options.ParallelToolCalls)
	}
	if options.Prediction != nil {
		providerRequest.Prediction = openrouter.NewOptPrediction(openrouter.Prediction{
			Type:    openrouter.PredictionTypeContent,
			Content: *options.Prediction,
		})
	}
	if options.PromptCacheKey != "" {
		providerRequest.PromptCacheKey = openrouter.NewOptString(options.PromptCacheKey)
	}
	if options.Provider != nil {
		encoded, marshalErr := json.Marshal(options.Provider)
		if marshalErr != nil {
			return nil, &InvalidParameterErr{Parameter: OpenRouterGenerationOptionProvider, Reason: marshalErr.Error()}
		}
		var provider openrouter.ChatCompletionRequestProvider
		if decodeErr := json.Unmarshal(encoded, &provider); decodeErr != nil {
			return nil, &InvalidParameterErr{Parameter: OpenRouterGenerationOptionProvider, Reason: decodeErr.Error()}
		}
		providerRequest.Provider = openrouter.NewOptChatCompletionRequestProvider(provider)
	}
	if options.RepetitionPenalty != nil {
		providerRequest.RepetitionPenalty = openrouter.NewOptFloat64(*options.RepetitionPenalty)
	}
	if options.ResponseFormat != nil {
		if typ, ok := options.ResponseFormat["type"].(string); !ok || typ == "" {
			return nil, &InvalidParameterErr{Parameter: OpenRouterGenerationOptionResponseFormat, Reason: "type must be a non-empty string"}
		}
		encoded, marshalErr := json.Marshal(options.ResponseFormat)
		if marshalErr != nil {
			return nil, &InvalidParameterErr{Parameter: OpenRouterGenerationOptionResponseFormat, Reason: marshalErr.Error()}
		}
		var responseFormat openrouter.ChatCompletionRequestResponseFormat
		if decodeErr := json.Unmarshal(encoded, &responseFormat); decodeErr != nil {
			return nil, &InvalidParameterErr{Parameter: OpenRouterGenerationOptionResponseFormat, Reason: decodeErr.Error()}
		}
		providerRequest.ResponseFormat = openrouter.NewOptChatCompletionRequestResponseFormat(responseFormat)
	}
	if options.Seed != nil {
		providerRequest.Seed = openrouter.NewOptInt(*options.Seed)
	}
	if options.ServiceTier != "" {
		tier := openrouter.ServiceTier(options.ServiceTier)
		if validateErr := tier.Validate(); validateErr != nil {
			return nil, &InvalidParameterErr{Parameter: OpenRouterGenerationOptionServiceTier, Reason: validateErr.Error()}
		}
		providerRequest.ServiceTier = openrouter.NewOptServiceTier(tier)
	}
	if options.SessionID != "" {
		if len(options.SessionID) > 256 {
			return nil, &InvalidParameterErr{Parameter: OpenRouterGenerationOptionSessionID, Reason: "must be at most 256 bytes"}
		}
		providerRequest.SessionID = openrouter.NewOptString(options.SessionID)
	}
	if options.TopA != nil {
		providerRequest.TopA = openrouter.NewOptFloat64(*options.TopA)
	}
	if options.TopLogprobs != nil {
		if *options.TopLogprobs < 0 || *options.TopLogprobs > 20 {
			return nil, &InvalidParameterErr{Parameter: OpenRouterGenerationOptionTopLogprobs, Reason: "must be between 0 and 20"}
		}
		if options.Logprobs == nil || !*options.Logprobs {
			return nil, &InvalidParameterErr{Parameter: OpenRouterGenerationOptionTopLogprobs, Reason: "requires logprobs to be enabled"}
		}
		providerRequest.TopLogprobs = openrouter.NewOptInt(*options.TopLogprobs)
	}
	if options.User != "" {
		providerRequest.User = openrouter.NewOptString(options.User)
	}
	if options.MaxGenerationTokens != nil {
		providerRequest.MaxCompletionTokens = openrouter.NewOptInt(*options.MaxGenerationTokens)
	}
	if len(options.StopSequences) == 1 {
		providerRequest.Stop = openrouter.NewOptStop(openrouter.NewStringStop(options.StopSequences[0]))
	} else if len(options.StopSequences) > 1 {
		providerRequest.Stop = openrouter.NewOptStop(openrouter.NewStringArrayStop(options.StopSequences))
	}
	if options.ThinkingBudget != "" {
		reasoning, err := openRouterReasoningConfig(options.ThinkingBudget)
		if err != nil {
			return nil, err
		}
		providerRequest.Reasoning = openrouter.NewOptReasoningConfig(reasoning)
	}
	if err := applyOpenRouterToolChoice(providerRequest, options.ToolChoice, request.Tools); err != nil {
		return nil, err
	}
	return providerRequest, nil
}

func openRouterReasoningConfig(value string) (openrouter.ReasoningConfig, error) {
	effort := openrouter.ReasoningConfigEffort(value)
	if err := effort.Validate(); err == nil {
		return openrouter.ReasoningConfig{Effort: openrouter.NewOptReasoningConfigEffort(effort)}, nil
	}
	maxTokens, err := strconv.Atoi(value)
	if err != nil || maxTokens < 1 {
		return openrouter.ReasoningConfig{}, &InvalidParameterErr{
			Parameter: GenerationOptionThinkingBudget,
			Reason:    "must be a supported effort or a positive integer token budget",
		}
	}
	return openrouter.ReasoningConfig{MaxTokens: openrouter.NewOptInt(maxTokens)}, nil
}

func applyOpenRouterToolChoice(request *openrouter.ChatCompletionRequest, choice string, tools []Tool) error {
	if choice == "" {
		return nil
	}
	if choice == ToolChoiceToolsRequired && len(tools) == 0 {
		return InvalidToolChoiceErr("required needs at least one tool")
	}
	if choice == "none" || choice == ToolChoiceAuto || choice == ToolChoiceToolsRequired {
		request.ToolChoice = openrouter.NewOptToolChoice(openrouter.NewToolChoiceModeToolChoice(openrouter.ToolChoiceMode(choice)))
		return nil
	}
	for _, tool := range tools {
		if tool.Name == choice {
			request.ToolChoice = openrouter.NewOptToolChoice(openrouter.NewNamedToolChoiceToolChoice(openrouter.NamedToolChoice{
				Type:     openrouter.NamedToolChoiceTypeFunction,
				Function: openrouter.NamedToolChoiceFunction{Name: choice},
			}))
			return nil
		}
	}
	return InvalidToolChoiceErr(fmt.Sprintf("tool %q is not in the request", choice))
}

func openRouterJSONMap(value any) (map[string]any, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("openrouter: encode provider metadata: %w", err)
	}
	var result map[string]any
	if err := json.Unmarshal(encoded, &result); err != nil {
		return nil, fmt.Errorf("openrouter: decode provider metadata: %w", err)
	}
	return result, nil
}

func mergeOpenRouterJSONMap(destination, source map[string]any) {
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

func openRouterResponseExtraFields(id, model string, created int64, systemFingerprint, serviceTier string, metadata map[string]any) map[string]interface{} {
	extraFields := map[string]interface{}{
		OpenRouterResponseExtraFieldID:      id,
		OpenRouterResponseExtraFieldModel:   model,
		OpenRouterResponseExtraFieldCreated: created,
	}
	if systemFingerprint != "" {
		extraFields[OpenRouterResponseExtraFieldSystemFingerprint] = systemFingerprint
	}
	if serviceTier != "" {
		extraFields[OpenRouterResponseExtraFieldServiceTier] = serviceTier
	}
	if len(metadata) > 0 {
		extraFields[OpenRouterResponseExtraFieldMetadata] = metadata
	}
	return extraFields
}

// Generate implements Generator
func (g *OpenRouterGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("openrouter: client not initialized")
	}

	if len(request.Dialog) == 0 {
		return Response{}, ErrEmptyDialog
	}
	providerRequest, err := g.buildRequest(request)
	if err != nil {
		return Response{}, err
	}
	rawResponse, err := g.client.CreateChatCompletion(ctx, providerRequest)
	if err != nil {
		return Response{}, mapOpenRouterTransportError(err)
	}
	providerResponse, ok := rawResponse.(*openrouter.OpenRouterResponse)
	if !ok {
		if stream, isStream := rawResponse.(*openrouter.CreateChatCompletionOKTextEventStream); isStream {
			_ = stream.Close()
		}
		return Response{}, fmt.Errorf("openrouter: expected JSON completion response, got %T", rawResponse)
	}

	rawBody, _ := json.Marshal(providerResponse)
	if errorEnvelope, ok := providerResponse.GetErrorEnvelope(); ok {
		return Response{}, mapOpenRouterErrorDetail(errorEnvelope.Error, errorEnvelope.Error.Code, string(rawBody))
	}
	completion, ok := providerResponse.GetChatCompletionResponse()
	if !ok {
		return Response{}, fmt.Errorf("openrouter: unexpected response type %q", providerResponse.Type)
	}

	var nativeMetadata map[string]any
	if metadata, ok := completion.OpenrouterMetadata.Get(); ok {
		nativeMetadata, err = openRouterJSONMap(metadata)
		if err != nil {
			return Response{}, err
		}
	}
	result := Response{
		UsageMetadata: make(Metadata),
		ExtraFields: openRouterResponseExtraFields(
			completion.ID,
			completion.Model,
			completion.Created,
			completion.SystemFingerprint.Or(""),
			completion.ServiceTier.Or(""),
			nativeMetadata,
		),
	}
	if usage, ok := completion.Usage.Get(); ok {
		addOpenRouterUsageMetadata(result.UsageMetadata, usage)
	}

	var hasToolCalls bool

	for _, choice := range completion.Choices {
		blocks := make([]Block, 0, 2)
		if len(choice.Message.ReasoningDetails) > 0 {
			result.UsageMetadata[OpenRouterUsageMetricReasoningDetailsAvailable] = true
		}
		for _, detail := range choice.Message.ReasoningDetails {
			block, ok := openRouterReasoningBlock(detail)
			if ok {
				blocks = append(blocks, block)
			}
		}
		if content, ok := choice.Message.Content.Get(); ok && content != "" {
			blocks = append(blocks, TextBlock(content))
		}
		hasToolCalls = hasToolCalls || len(choice.Message.ToolCalls) > 0
		for _, call := range choice.Message.ToolCalls {
			block, err := openRouterToolCallBlock(call)
			if err != nil {
				return result, err
			}
			blocks = append(blocks, block)
		}
		message := Message{Role: Assistant, Blocks: blocks}
		if logprobs, ok := choice.Logprobs.Get(); ok {
			value, valueErr := openRouterJSONMap(logprobs)
			if valueErr != nil {
				return result, valueErr
			}
			message.ExtraFields = map[string]interface{}{OpenRouterMessageExtraFieldLogprobs: value}
		}
		result.Candidates = append(result.Candidates, message)
		if detail, ok := choice.Error.Get(); ok {
			return result, mapOpenRouterErrorDetail(detail, detail.Code, string(rawBody))
		}
	}

	if len(completion.Choices) > 0 {
		first := completion.Choices[0]
		if refusal, ok := first.Message.Refusal.Get(); ok && refusal != "" {
			result.FinishReason = ContentPolicyViolation
			return result, ContentPolicyErr(refusal)
		}
		result.FinishReason, err = openRouterFinishReason(first.FinishReason.Or(""))
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
func (g *OpenRouterGenerator) Stream(ctx context.Context, generationRequest GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("openrouter: client not initialized")})
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
		request.Stream = openrouter.NewOptBool(true)

		rawResponse, err := g.client.CreateChatCompletion(ctx, request)
		if err != nil {
			yield(StreamChunk{Err: mapOpenRouterTransportError(err)})
			return
		}
		stream, ok := rawResponse.(*openrouter.CreateChatCompletionOKTextEventStream)
		if !ok {
			if providerResponse, isJSON := rawResponse.(*openrouter.OpenRouterResponse); isJSON {
				rawBody, _ := json.Marshal(providerResponse)
				if errorEnvelope, hasError := providerResponse.GetErrorEnvelope(); hasError {
					yield(StreamChunk{Err: mapOpenRouterErrorDetail(errorEnvelope.Error, errorEnvelope.Error.Code, string(rawBody))})
					return
				}
			}
			yield(StreamChunk{Err: fmt.Errorf("openrouter: expected event stream response, got %T", rawResponse)})
			return
		}
		defer stream.Close()

		var finalUsage openrouter.Usage
		var hasFinalUsage bool
		hasReasoningDetails := false
		lastReasoningKey := make(map[int]string)
		responseExtraFields := make(map[string]interface{})
		streamLogprobs := make(map[string]any)
		for {
			event, err := stream.Next(ctx)
			if err != nil {
				yield(StreamChunk{Err: mapOpenRouterTransportError(err)})
				return
			}
			if event.Data.IsCreateChatCompletionOKTextEventStreamEventData1() {
				break
			}
			chunk, ok := event.Data.GetChatCompletionChunk()
			if !ok {
				yield(StreamChunk{Err: fmt.Errorf("openrouter: unexpected event data type %q", event.Data.Type)})
				return
			}
			rawChunk, _ := json.Marshal(chunk)
			if detail, ok := chunk.Error.Get(); ok {
				yield(StreamChunk{Err: mapOpenRouterErrorDetail(detail, detail.Code, string(rawChunk))})
				return
			}
			var nativeMetadata map[string]any
			if metadata, ok := chunk.OpenrouterMetadata.Get(); ok {
				nativeMetadata, err = openRouterJSONMap(metadata)
				if err != nil {
					yield(StreamChunk{Err: err})
					return
				}
			}
			chunkExtraFields := openRouterResponseExtraFields(
				chunk.ID,
				chunk.Model,
				chunk.Created,
				chunk.SystemFingerprint.Or(""),
				chunk.ServiceTier.Or(""),
				nativeMetadata,
			)
			maps.Copy(responseExtraFields, chunkExtraFields)
			yieldBlock := func(block Block, candidateIndex int, messageExtraFields map[string]interface{}) bool {
				return yield(StreamChunk{
					Block:               block,
					MessageExtraFields:  messageExtraFields,
					ResponseExtraFields: chunkExtraFields,
					CandidatesIndex:     candidateIndex,
				})
			}
			if usage, ok := chunk.Usage.Get(); ok {
				finalUsage = usage
				hasFinalUsage = true
			}
			for _, choice := range chunk.Choices {
				if detail, ok := choice.Error.Get(); ok {
					yield(StreamChunk{Err: mapOpenRouterErrorDetail(detail, detail.Code, string(rawChunk))})
					return
				}
				if logprobs, ok := choice.Logprobs.Get(); ok && choice.Index == 0 {
					value, valueErr := openRouterJSONMap(logprobs)
					if valueErr != nil {
						yield(StreamChunk{Err: valueErr})
						return
					}
					mergeOpenRouterJSONMap(streamLogprobs, value)
				}
				if finishReason, ok := choice.FinishReason.Get(); ok {
					if _, finishErr := openRouterFinishReason(finishReason); finishErr != nil {
						yield(StreamChunk{Err: finishErr})
						return
					}
				}
				if refusal, ok := choice.Delta.Refusal.Get(); ok && refusal != "" {
					yield(StreamChunk{Err: ContentPolicyErr(refusal)})
					return
				}
				for _, detail := range choice.Delta.ReasoningDetails {
					hasReasoningDetails = true
					key := openRouterReasoningDetailKey(detail)
					if previous, exists := lastReasoningKey[choice.Index]; exists && previous != key {
						if !yieldBlock(SeparatorBlock(), choice.Index, nil) {
							return
						}
					}
					lastReasoningKey[choice.Index] = key
					if block, ok := openRouterReasoningBlock(detail); ok {
						if !yieldBlock(block, choice.Index, nil) {
							return
						}
					}
				}
				if content, ok := choice.Delta.Content.Get(); ok && content != "" {
					if !yieldBlock(TextBlock(content), choice.Index, nil) {
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
						}, choice.Index, nil) {
							return
						}
					}
					if arguments := call.Function.Arguments.Or(""); arguments != "" {
						if !yieldBlock(Block{
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(arguments),
						}, choice.Index, nil) {
							return
						}
					}
				}
			}
		}

		metadata := make(Metadata)
		if hasFinalUsage {
			addOpenRouterUsageMetadata(metadata, finalUsage)
		}
		if hasReasoningDetails {
			metadata[OpenRouterUsageMetricReasoningDetailsAvailable] = true
		}
		var messageExtraFields map[string]interface{}
		if len(streamLogprobs) > 0 {
			messageExtraFields = map[string]interface{}{OpenRouterMessageExtraFieldLogprobs: streamLogprobs}
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

func openRouterReasoningDetailKey(detail openrouter.ReasoningDetail) string {
	if index, ok := detail.Index.Get(); ok {
		return fmt.Sprintf("%s:%d", detail.Type, index)
	}
	return detail.Type + ":" + detail.ID.Or("")
}

func openRouterReasoningBlock(detail openrouter.ReasoningDetail) (Block, bool) {
	var content string
	switch detail.Type {
	case "reasoning.summary":
		content = detail.Summary.Or("")
	case "reasoning.text":
		content = detail.Text.Or("")
	case "reasoning.encrypted":
		content = detail.Data.Or("")
	default:
		return Block{}, false
	}
	if content == "" {
		return Block{}, false
	}
	extraFields := map[string]interface{}{
		ThinkingExtraFieldGeneratorKey:      ThinkingGeneratorOpenRouter,
		OpenRouterExtraFieldReasoningType:   detail.Type,
		OpenRouterExtraFieldReasoningFormat: detail.Format.Or(""),
		OpenRouterExtraFieldReasoningIndex:  detail.Index.Or(0),
	}
	if signature := detail.Signature.Or(""); signature != "" {
		extraFields[OpenRouterExtraFieldReasoningSignature] = signature
	}
	return Block{
		ID:           detail.ID.Or(""),
		BlockType:    Thinking,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str(content),
		ExtraFields:  extraFields,
	}, true
}

func openRouterToolCallBlock(call openrouter.ToolCall) (Block, error) {
	parameters := make(map[string]any)
	if strings.TrimSpace(call.Function.Arguments) != "" {
		if err := json.Unmarshal([]byte(call.Function.Arguments), &parameters); err != nil {
			return Block{}, fmt.Errorf("openrouter: malformed tool arguments for %q: %w", call.Function.Name, err)
		}
	}
	return ToolCallBlock(call.ID, call.Function.Name, parameters)
}

func openRouterFinishReason(reason string) (FinishReason, error) {
	switch reason {
	case "stop":
		return EndTurn, nil
	case "tool_calls":
		return ToolUse, nil
	case "length":
		return MaxGenerationLimit, ErrMaxGenerationLimit
	case "content_filter":
		return ContentPolicyViolation, ContentPolicyErr("content policy violation detected")
	case "error":
		return Unknown, &ApiErr{
			Provider: ProviderOpenRouter,
			Kind:     APIErrorKindServer,
			Message:  "generation stopped because OpenRouter reported an error",
		}
	default:
		return Unknown, nil
	}
}

func addOpenRouterUsageMetadata(metadata Metadata, usage openrouter.Usage) {
	if promptTokens := usage.PromptTokens.Or(0); promptTokens > 0 {
		metadata[UsageMetricInputTokens] = promptTokens
	}
	if completionTokens := usage.CompletionTokens.Or(0); completionTokens > 0 {
		metadata[UsageMetricGenerationTokens] = completionTokens
	}
	if cost, ok := usage.Cost.Get(); ok {
		metadata[OpenRouterUsageMetricCost] = cost
	}
	if isBYOK, ok := usage.IsByok.Get(); ok {
		metadata[OpenRouterUsageMetricIsBYOK] = isBYOK
	}
	if details, ok := usage.CostDetails.Get(); ok {
		if value, err := openRouterJSONMap(details); err == nil {
			metadata[OpenRouterUsageMetricCostDetails] = value
		}
	}
	if details, ok := usage.ServerToolUseDetails.Get(); ok {
		if value, err := openRouterJSONMap(details); err == nil {
			metadata[OpenRouterUsageMetricServerToolUseDetails] = value
		}
	}
	if details, ok := usage.PromptTokensDetails.Get(); ok {
		if value, err := openRouterJSONMap(details); err == nil {
			metadata[OpenRouterUsageMetricPromptTokenDetails] = value
		}
		if cachedTokens := details.CachedTokens.Or(0); cachedTokens > 0 {
			metadata[UsageMetricCacheReadTokens] = cachedTokens
		}
		if cacheWriteTokens := details.CacheWriteTokens.Or(0); cacheWriteTokens > 0 {
			metadata[UsageMetricCacheWriteTokens] = cacheWriteTokens
		}
	}
	if details, ok := usage.CompletionTokensDetails.Get(); ok {
		if value, err := openRouterJSONMap(details); err == nil {
			metadata[OpenRouterUsageMetricCompletionTokenDetails] = value
		}
		if reasoningTokens := details.ReasoningTokens.Or(0); reasoningTokens > 0 {
			metadata[UsageMetricReasoningTokens] = reasoningTokens
		}
	}
}

var _ Generator = (*OpenRouterGenerator)(nil)
var _ StreamingGenerator = (*OpenRouterGenerator)(nil)
