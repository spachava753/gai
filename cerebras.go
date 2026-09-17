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
	// CerebrasGenerationOptionPrediction is the string key set by
	// [WithCerebrasPrediction].
	CerebrasGenerationOptionPrediction = "cerebras_prediction"
	// CerebrasGenerationOptionResponseFormat is the map[string]any key set by
	// [WithCerebrasResponseFormat].
	CerebrasGenerationOptionResponseFormat = "cerebras_response_format"
	// CerebrasGenerationOptionServiceTier is the string key set by
	// [WithCerebrasServiceTier].
	CerebrasGenerationOptionServiceTier = "cerebras_service_tier"
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

// WithCerebrasServiceTier stores value under
// [CerebrasGenerationOptionServiceTier].
func WithCerebrasServiceTier(value CerebrasServiceTier) GenerationOption {
	return func(options GenerationOptions) {
		options[CerebrasGenerationOptionServiceTier] = string(value)
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
		{CerebrasGenerationOptionPrediction, "prediction"},
		{CerebrasGenerationOptionResponseFormat, "response_format"},
		{CerebrasGenerationOptionServiceTier, "service_tier"},
	} {
		var value any
		var present bool
		switch option.key {
		case CerebrasGenerationOptionResponseFormat:
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
