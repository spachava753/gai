package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"net/http"
)

const (
	// OpenRouterGenerationOptionMinP is the float64 relative probability threshold set by [WithOpenRouterMinP].
	OpenRouterGenerationOptionMinP = "openrouter_min_p"
	// OpenRouterGenerationOptionTopA is the float64 adaptive probability threshold set by [WithOpenRouterTopA].
	OpenRouterGenerationOptionTopA = "openrouter_top_a"
	// OpenRouterGenerationOptionRepetitionPenalty is the float64 repetition control set by [WithOpenRouterRepetitionPenalty].
	OpenRouterGenerationOptionRepetitionPenalty = "openrouter_repetition_penalty"
	// OpenRouterGenerationOptionModels is the []string fallback-model key set by
	// [WithOpenRouterFallbackModels].
	OpenRouterGenerationOptionModels = "openrouter_models"
	// OpenRouterGenerationOptionPrediction is the string key set by
	// [WithOpenRouterPrediction].
	OpenRouterGenerationOptionPrediction = "openrouter_prediction"
	// OpenRouterGenerationOptionProvider is the map[string]any routing key set by
	// [WithOpenRouterProviderPreferences].
	OpenRouterGenerationOptionProvider = "openrouter_provider"
	// OpenRouterGenerationOptionResponseFormat is the map[string]any key set by
	// [WithOpenRouterResponseFormat].
	OpenRouterGenerationOptionResponseFormat = "openrouter_response_format"
	// OpenRouterGenerationOptionServiceTier is the string key set by
	// [WithOpenRouterServiceTier].
	OpenRouterGenerationOptionServiceTier = "openrouter_service_tier"
	// OpenRouterGenerationOptionSessionID is the string key set by
	// [WithOpenRouterSessionID].
	OpenRouterGenerationOptionSessionID = "openrouter_session_id"
)

// WithOpenRouterMinP sets a minimum token probability relative to the most likely
// token through [OpenRouterGenerationOptionMinP]. OpenRouter receives min_p and
// validates the range and model support.
func WithOpenRouterMinP(value float64) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionMinP] = value }
}

// WithOpenRouterTopA sets an adaptive probability cutoff based on the most likely
// token through [OpenRouterGenerationOptionTopA]. OpenRouter receives top_a and
// validates the range and model support.
func WithOpenRouterTopA(value float64) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionTopA] = value }
}

// WithOpenRouterRepetitionPenalty adjusts the likelihood of tokens already used
// through [OpenRouterGenerationOptionRepetitionPenalty]. OpenRouter receives
// repetition_penalty and defines its accepted range and neutral value.
func WithOpenRouterRepetitionPenalty(value float64) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionRepetitionPenalty] = value }
}

// OpenRouterServiceTier selects processing priority for
// [WithOpenRouterServiceTier].
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

// WithOpenRouterFallbackModels stores a copy of values under
// [OpenRouterGenerationOptionModels] in fallback order.
func WithOpenRouterFallbackModels(values ...string) GenerationOption {
	return func(options GenerationOptions) {
		options[OpenRouterGenerationOptionModels] = append([]string(nil), values...)
	}
}

// WithOpenRouterPrediction stores content under
// [OpenRouterGenerationOptionPrediction] as known predicted output.
func WithOpenRouterPrediction(content string) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionPrediction] = content }
}

// WithOpenRouterProviderPreferences stores a shallow copy of value under
// [OpenRouterGenerationOptionProvider].
func WithOpenRouterProviderPreferences(value map[string]any) GenerationOption {
	return func(options GenerationOptions) {
		options[OpenRouterGenerationOptionProvider] = maps.Clone(value)
	}
}

// WithOpenRouterResponseFormat stores a shallow copy of value under
// [OpenRouterGenerationOptionResponseFormat].
func WithOpenRouterResponseFormat(value map[string]any) GenerationOption {
	return func(options GenerationOptions) {
		options[OpenRouterGenerationOptionResponseFormat] = maps.Clone(value)
	}
}

// WithOpenRouterServiceTier stores value under
// [OpenRouterGenerationOptionServiceTier].
func WithOpenRouterServiceTier(value OpenRouterServiceTier) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionServiceTier] = string(value) }
}

// WithOpenRouterSessionID stores value under
// [OpenRouterGenerationOptionSessionID].
func WithOpenRouterSessionID(value string) GenerationOption {
	return func(options GenerationOptions) { options[OpenRouterGenerationOptionSessionID] = value }
}

// classifyOpenRouterError maps typed router and upstream failures before falling back to the HTTP status.
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

// OpenRouterDefaultBaseURL is used when the constructor receives an empty URL.
const OpenRouterDefaultBaseURL = "https://openrouter.ai/api/v1"

// OpenRouterGenerator delegates Chat Completions and streaming to [OpenAiGenerator].
// Keep returned candidate ExtraFields when adding answers to later requests:
// [OpenAIExtraFieldWireFields] holds native reasoning details, including signatures
// and encrypted data. Thinking blocks show readable reasoning text and identify
// its source with [OpenAIExtraFieldReasoningSource]. Candidate log probabilities
// use [OpenAIExtraFieldChoice]. Response IDs and other native response fields use
// [OpenAIResponseExtraFieldWireFields]; complete usage, including costs and
// server-tool details, uses [OpenAIResponseExtraFieldUsage]. These native objects
// are map[string]json.RawMessage values. It does not implement [TokenCounter].
//
// Output limits default to max_tokens. Native options are translated without
// capability or range validation; explicit OpenAI extra-body fields override them.
// Effort becomes reasoning.effort and numeric budgets become reasoning.max_tokens;
// combining the two is an error. [GenerationRequest.SafetyIdentifier] sets user
// for end-user abuse isolation. [GenerationRequest.PromptCacheKey] sets
// prompt_cache_key for cache routing; an explicit session ID takes routing
// precedence. Neither field is derived from the other.
type OpenRouterGenerator struct{ client *OpenAiGenerator }

// NewOpenRouterGenerator uses http.DefaultClient for a nil client and
// [OpenRouterDefaultBaseURL] for an empty URL. Credentials are explicit; no retries
// or environment configuration are added. The URL includes the API prefix.
func NewOpenRouterGenerator(httpClient *http.Client, baseURL, apiKey string) (*OpenRouterGenerator, error) {
	if baseURL == "" {
		baseURL = OpenRouterDefaultBaseURL
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	client, err := NewOpenAiGenerator(httpClient, baseURL, apiKey)
	if err != nil {
		return nil, err
	}
	return &OpenRouterGenerator{client: client}, nil
}

// Generate sends one request without retrying. Embedded errors, including choice
// errors, return an ApiErr alongside any partial response, headers, and usage.
// A numeric error code supplies the status for errors carried by HTTP 200.
func (g *OpenRouterGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("openrouter: uninitialized generator")
	}
	request, err := g.prepareRequest(request)
	if err != nil {
		return Response{}, err
	}
	response, err := g.client.Generate(ctx, request)
	return response, openRouterError(err)
}

// Stream yields ordered chunks without reconnecting or retrying.
func (g *OpenRouterGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("openrouter: uninitialized generator")})
			return
		}
		request, err := g.prepareRequest(request)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		for chunk := range g.client.Stream(ctx, request) {
			chunk.Err = openRouterError(chunk.Err)
			if !yield(chunk) {
				return
			}
		}
	}
}

var _ Generator = (*OpenRouterGenerator)(nil)
var _ StreamingGenerator = (*OpenRouterGenerator)(nil)

// prepareRequest translates native options without changing caller-owned maps.
func (g *OpenRouterGenerator) prepareRequest(request GenerationRequest) (GenerationRequest, error) {
	options := maps.Clone(request.Options)
	if options == nil {
		options = GenerationOptions{}
	}
	explicit, _, err := generationOption[map[string]json.RawMessage](options, OpenAIGenerationOptionExtraBody)
	if err != nil {
		return request, err
	}
	fields := map[string]json.RawMessage{}
	if request.SafetyIdentifier != "" {
		fields["user"], _ = json.Marshal(request.SafetyIdentifier)
	}
	request.SafetyIdentifier = ""
	if _, ok := options[OpenAIGenerationOptionTokenLimitField]; !ok {
		options[OpenAIGenerationOptionTokenLimitField] = "max_tokens"
	}
	effort, _, err := generationOption[string](options, GenerationOptionReasoningEffort)
	if err != nil {
		return request, err
	}
	budget, hasBudget, err := generationOption[int](options, GenerationOptionThinkingBudget)
	if err != nil {
		return request, err
	}
	if effort != "" && hasBudget {
		return request, &InvalidParameterErr{Parameter: GenerationOptionThinkingBudget, Reason: "cannot combine thinking budget with reasoning effort"}
	}
	if effort != "" {
		fields["reasoning"], _ = json.Marshal(map[string]any{"effort": effort})
	}
	if hasBudget {
		fields["reasoning"], _ = json.Marshal(map[string]any{"max_tokens": budget})
	}
	delete(options, GenerationOptionReasoningEffort)
	delete(options, GenerationOptionThinkingBudget)
	for _, option := range []struct{ key, field string }{
		{OpenRouterGenerationOptionPrediction, "prediction"},
		{OpenRouterGenerationOptionResponseFormat, "response_format"},
		{OpenRouterGenerationOptionServiceTier, "service_tier"},
		{GenerationOptionTopK, "top_k"},
		{OpenRouterGenerationOptionMinP, "min_p"},
		{OpenRouterGenerationOptionModels, "models"},
		{OpenRouterGenerationOptionProvider, "provider"},
		{OpenRouterGenerationOptionRepetitionPenalty, "repetition_penalty"},
		{OpenRouterGenerationOptionSessionID, "session_id"},
		{OpenRouterGenerationOptionTopA, "top_a"},
	} {
		var value any
		var present bool
		switch option.key {
		case GenerationOptionTopK:
			value, present, err = generationOption[uint](options, option.key)
		case OpenRouterGenerationOptionMinP, OpenRouterGenerationOptionRepetitionPenalty, OpenRouterGenerationOptionTopA:
			value, present, err = generationOption[float64](options, option.key)
		case OpenRouterGenerationOptionModels:
			value, present, err = generationOption[[]string](options, option.key)
		case OpenRouterGenerationOptionResponseFormat, OpenRouterGenerationOptionProvider:
			value, present, err = generationOption[map[string]any](options, option.key)
		default:
			value, present, err = generationOption[string](options, option.key)
		}
		if err != nil {
			return request, err
		}
		if !present {
			continue
		}
		if option.key == OpenRouterGenerationOptionPrediction {
			value = map[string]any{"type": "content", "content": value}
		}
		fields[option.field], err = json.Marshal(value)
		if err != nil {
			return request, &InvalidParameterErr{Parameter: option.key, Reason: err.Error()}
		}
	}
	maps.Copy(fields, explicit)
	options[OpenAIGenerationOptionExtraBody] = fields
	request.Options = options
	return request, nil
}

// openRouterError classifies router-specific error types while retaining shared
// status, retry headers, raw response, and cause information.
func openRouterError(err error) error {
	err = compatibleError(ProviderOpenRouter, err)
	var apiErr *ApiErr
	if !errors.As(err, &apiErr) {
		return err
	}
	type detail struct {
		ErrorType string `json:"error_type"`
		Metadata  struct {
			ErrorType string `json:"error_type"`
		} `json:"metadata"`
	}
	var envelope struct {
		Error   *detail `json:"error"`
		Choices []struct {
			Error *detail `json:"error"`
		} `json:"choices"`
	}
	if json.Unmarshal([]byte(apiErr.RawBody), &envelope) != nil {
		return err
	}
	selected := envelope.Error
	if selected == nil {
		for _, choice := range envelope.Choices {
			if choice.Error != nil {
				selected = choice.Error
				break
			}
		}
	}
	if selected == nil {
		return err
	}
	errorType := selected.Metadata.ErrorType
	if errorType == "" {
		errorType = selected.ErrorType
	}
	if errorType == "" {
		return err
	}
	mapped := *apiErr
	mapped.Kind = classifyOpenRouterError(mapped.StatusCode, errorType)
	return &mapped
}
