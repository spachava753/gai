package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"net/http"
)

// CerebrasDefaultBaseURL is used when the constructor receives an empty URL.
const CerebrasDefaultBaseURL = "https://api.cerebras.ai/v1"

// CerebrasGenerator delegates Chat Completions and streaming to [OpenAiGenerator].
// It uses the shared OpenAI wire, choice, and usage metadata keys, including
// reasoning replay and lossless native response fields. Provider timing fields
// remain in [OpenAIResponseExtraFieldWireFields]; native usage details remain in
// [OpenAIResponseExtraFieldUsage]. It does not implement [TokenCounter].
//
// Output limits default to max_completion_tokens and streaming requests usage.
// Native Cerebras options are translated without model-name heuristics; the
// server validates supported values and capabilities. Explicit OpenAI extra-body
// fields override translated native options. SafetyIdentifier maps to user;
// PromptCacheKey maps to prompt_cache_key.
type CerebrasGenerator struct{ client *OpenAiGenerator }

const (
	// CerebrasGenerationOptionLogitBias is the map[string]float64 key set by
	// [WithCerebrasLogitBias].
	CerebrasGenerationOptionLogitBias = "cerebras_logit_bias"
	// CerebrasGenerationOptionLogprobs is the bool key set by
	// [WithCerebrasLogprobs].
	CerebrasGenerationOptionLogprobs = "cerebras_logprobs"
	// CerebrasGenerationOptionParallelToolCalls is the bool key set by
	// [WithCerebrasParallelToolCalls].
	CerebrasGenerationOptionParallelToolCalls = "cerebras_parallel_tool_calls"
	// CerebrasGenerationOptionPrediction is the string key set by
	// [WithCerebrasPrediction].
	CerebrasGenerationOptionPrediction = "cerebras_prediction"
	// CerebrasGenerationOptionResponseFormat is the map[string]any key set by
	// [WithCerebrasResponseFormat].
	CerebrasGenerationOptionResponseFormat = "cerebras_response_format"
	// CerebrasGenerationOptionSeed is the int key set by [WithCerebrasSeed].
	CerebrasGenerationOptionSeed = "cerebras_seed"
	// CerebrasGenerationOptionServiceTier is the string key set by
	// [WithCerebrasServiceTier].
	CerebrasGenerationOptionServiceTier = "cerebras_service_tier"
	// CerebrasGenerationOptionTopLogprobs is the int key set by
	// [WithCerebrasTopLogprobs].
	CerebrasGenerationOptionTopLogprobs = "cerebras_top_logprobs"
)

// CerebrasServiceTier selects request prioritization for
// [WithCerebrasServiceTier].
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

// WithCerebrasLogitBias stores a copy of value under
// [CerebrasGenerationOptionLogitBias].
func WithCerebrasLogitBias(value map[string]float64) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionLogitBias] = maps.Clone(value)
	}
}

// WithCerebrasLogprobs stores enabled under
// [CerebrasGenerationOptionLogprobs]. Returned log probabilities use
// [OpenAIExtraFieldChoice].
func WithCerebrasLogprobs(enabled bool) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionLogprobs] = enabled
	}
}

// WithCerebrasParallelToolCalls stores enabled under
// [CerebrasGenerationOptionParallelToolCalls].
func WithCerebrasParallelToolCalls(enabled bool) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionParallelToolCalls] = enabled
	}
}

// WithCerebrasPrediction stores content under
// [CerebrasGenerationOptionPrediction] as known predicted output.
func WithCerebrasPrediction(content string) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionPrediction] = content
	}
}

// WithCerebrasResponseFormat stores a shallow copy of value under
// [CerebrasGenerationOptionResponseFormat].
func WithCerebrasResponseFormat(value map[string]any) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionResponseFormat] = maps.Clone(value)
	}
}

// WithCerebrasSeed stores value under [CerebrasGenerationOptionSeed] for
// best-effort deterministic sampling.
func WithCerebrasSeed(value int) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionSeed] = value
	}
}

// WithCerebrasServiceTier stores value under
// [CerebrasGenerationOptionServiceTier].
func WithCerebrasServiceTier(value CerebrasServiceTier) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionServiceTier] = string(value)
	}
}

// WithCerebrasTopLogprobs stores value under
// [CerebrasGenerationOptionTopLogprobs].
func WithCerebrasTopLogprobs(value int) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionTopLogprobs] = value
	}
}

// NewCerebrasGenerator uses http.DefaultClient for a nil client and
// [CerebrasDefaultBaseURL] for an empty URL. Credentials are explicit; no retries
// or environment configuration are added. The URL includes the API prefix.
func NewCerebrasGenerator(httpClient *http.Client, baseURL, apiKey string) (*CerebrasGenerator, error) {
	if baseURL == "" {
		baseURL = CerebrasDefaultBaseURL
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	client, err := NewOpenAiGenerator(httpClient, baseURL, apiKey)
	if err != nil {
		return nil, err
	}
	return &CerebrasGenerator{client: client}, nil
}

// Generate sends one request without retrying.
func (g *CerebrasGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("cerebras: uninitialized generator")
	}
	request, err := g.prepareRequest(request)
	if err != nil {
		return Response{}, err
	}
	response, err := g.client.Generate(ctx, request)
	return response, compatibleError(ProviderCerebras, err)
}

// Stream yields ordered chunks without reconnecting or retrying.
func (g *CerebrasGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("cerebras: uninitialized generator")})
			return
		}
		request, err := g.prepareRequest(request)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		for chunk := range g.client.Stream(ctx, request) {
			chunk.Err = compatibleError(ProviderCerebras, chunk.Err)
			if !yield(chunk) {
				return
			}
		}
	}
}

var _ Generator = (*CerebrasGenerator)(nil)
var _ StreamingGenerator = (*CerebrasGenerator)(nil)

// prepareRequest translates native options without changing caller-owned maps.
func (g *CerebrasGenerator) prepareRequest(request GenerationRequest) (GenerationRequest, error) {
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
	for _, option := range []struct{ key, field string }{
		{CerebrasGenerationOptionLogitBias, "logit_bias"},
		{CerebrasGenerationOptionLogprobs, "logprobs"},
		{CerebrasGenerationOptionParallelToolCalls, "parallel_tool_calls"},
		{CerebrasGenerationOptionPrediction, "prediction"},
		{CerebrasGenerationOptionResponseFormat, "response_format"},
		{CerebrasGenerationOptionSeed, "seed"},
		{CerebrasGenerationOptionServiceTier, "service_tier"},
		{CerebrasGenerationOptionTopLogprobs, "top_logprobs"},
	} {
		var value any
		var present bool
		switch option.key {
		case CerebrasGenerationOptionLogitBias:
			value, present, err = generationOption[map[string]float64](options, option.key)
		case CerebrasGenerationOptionLogprobs, CerebrasGenerationOptionParallelToolCalls:
			value, present, err = generationOption[bool](options, option.key)
		case CerebrasGenerationOptionResponseFormat:
			value, present, err = generationOption[map[string]any](options, option.key)
		case CerebrasGenerationOptionSeed, CerebrasGenerationOptionTopLogprobs:
			value, present, err = generationOption[int](options, option.key)
		default:
			value, present, err = generationOption[string](options, option.key)
		}
		if err != nil {
			return request, err
		}
		if !present {
			continue
		}
		if option.key == CerebrasGenerationOptionPrediction {
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
