package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"strings"
)

// MoonshotDefaultBaseURL is used when the constructor receives an empty base URL.
const MoonshotDefaultBaseURL = "https://api.moonshot.ai/v1"

// MoonshotGenerator delegates Kimi Chat Completions and streaming to
// [OpenAiGenerator]. It preserves reasoning and opaque metadata using the
// OpenAIExtraField and OpenAIResponseExtraField constants. Output limits use
// max_completion_tokens; streamed usage is requested by default. SafetyIdentifier
// and PromptCacheKey map directly to their snake_case wire fields.
//
// Thinking and sampling defaults are left to the server. K2.6 supports
// [WithMoonshotThinking] and [WithMoonshotKeepThinking]; K2.7-code always thinks
// and keeps reasoning. K3 requests should omit these thinking controls and use
// [WithReasoningEffort] instead. These are caller responsibilities: this adapter
// does not rewrite settings or discard historical reasoning based on model names.
// Current models may fix sampling parameters and restrict required/named tools.
// JSON-object/schema output uses [WithOpenAIResponseFormat].
//
// [MoonshotGenerator.Count] calls the native estimator for text and supported
// images. Requests containing tools are rejected because the counting endpoint
// does not document tool accounting; there is no OpenAI-tokenizer fallback.
type MoonshotGenerator struct {
	client     *OpenAiGenerator
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

// NewMoonshotGenerator uses http.DefaultClient for a nil client and
// [MoonshotDefaultBaseURL] for an empty URL. Credentials are explicit; no retries
// or environment configuration are added. The URL includes the API prefix.
func NewMoonshotGenerator(httpClient *http.Client, baseURL, apiKey string) (*MoonshotGenerator, error) {
	if baseURL == "" {
		baseURL = MoonshotDefaultBaseURL
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	client, err := NewOpenAiGenerator(httpClient, baseURL, apiKey)
	if err != nil {
		return nil, err
	}
	return &MoonshotGenerator{client: client, httpClient: httpClient, baseURL: baseURL, apiKey: apiKey}, nil
}

// Generate sends one request without retrying.
func (g *MoonshotGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("moonshot: uninitialized generator")
	}
	request, err := g.prepareRequest(request)
	if err != nil {
		return Response{}, err
	}
	response, err := g.client.Generate(ctx, request)
	return response, compatibleError(ProviderMoonshot, err)
}

// Stream yields ordered chunks without reconnecting or retrying.
func (g *MoonshotGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("moonshot: uninitialized generator")})
			return
		}
		request, err := g.prepareRequest(request)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		for chunk := range g.client.Stream(ctx, request) {
			chunk.Err = compatibleError(ProviderMoonshot, chunk.Err)
			if !yield(chunk) {
				return
			}
		}
	}
}

var _ Generator = (*MoonshotGenerator)(nil)
var _ StreamingGenerator = (*MoonshotGenerator)(nil)

// Count calls the native tokenizer endpoint. Moonshot rejects requests with tools
// because its endpoint does not document tool counting. No local estimate is used.
func (g *MoonshotGenerator) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	if len(request.Tools) > 0 {
		return 0, &InvalidParameterErr{Parameter: "tools", Reason: "Moonshot's token-count endpoint does not document tool counting"}
	}
	params, _, err := openAIRequest(request, false)
	if err != nil {
		return 0, err
	}
	body := map[string]any{"model": params.Model, "messages": params.Messages}
	var result struct {
		Data struct {
			TotalTokens *uint `json:"total_tokens"`
		} `json:"data"`
	}
	err = compatiblePostJSON(ctx, g.httpClient, strings.TrimRight(g.baseURL, "/")+"/tokenizers/estimate-token-count", g.apiKey, ProviderMoonshot, request.Headers, body, &result)
	if err != nil {
		return 0, err
	}
	if result.Data.TotalTokens == nil {
		return 0, fmt.Errorf("moonshot: missing total_tokens")
	}
	return *result.Data.TotalTokens, nil
}

var _ TokenCounter = (*MoonshotGenerator)(nil)

// MoonshotGenerationOptionThinkingEnabled is the bool key set by [WithMoonshotThinking].
const MoonshotGenerationOptionThinkingEnabled = "moonshot_thinking_enabled"

// WithMoonshotThinking enables or disables reasoning; the server may reject disabling always-thinking models.
func WithMoonshotThinking(value bool) GenerationOption {
	return func(options GenerationOptions) { options[MoonshotGenerationOptionThinkingEnabled] = value }
}

// MoonshotGenerationOptionKeepThinking is the bool key set by [WithMoonshotKeepThinking].
const MoonshotGenerationOptionKeepThinking = "moonshot_keep_thinking"

// WithMoonshotKeepThinking sends thinking.keep as all when true or JSON null when false; omit for server defaults.
func WithMoonshotKeepThinking(value bool) GenerationOption {
	return func(options GenerationOptions) { options[MoonshotGenerationOptionKeepThinking] = value }
}

// prepareRequest applies Moonshot wire conventions without mutating caller data.
// Explicit extra-body fields take precedence over translated options.
func (g *MoonshotGenerator) prepareRequest(request GenerationRequest) (GenerationRequest, error) {
	options := maps.Clone(request.Options)
	if options == nil {
		options = GenerationOptions{}
	}
	explicit, _, err := generationOption[map[string]json.RawMessage](options, OpenAIGenerationOptionExtraBody)
	if err != nil {
		return request, err
	}
	fields := map[string]json.RawMessage{}
	thinking := map[string]any{}
	if enabled, present, err := generationOption[bool](options, MoonshotGenerationOptionThinkingEnabled); err != nil {
		return request, err
	} else if present {
		thinking["type"] = "disabled"
		if enabled {
			thinking["type"] = "enabled"
		}
	}
	if keep, present, err := generationOption[bool](options, MoonshotGenerationOptionKeepThinking); err != nil {
		return request, err
	} else if present {
		thinking["keep"] = nil
		if keep {
			thinking["keep"] = "all"
		}
	}

	if len(thinking) > 0 {
		fields["thinking"], _ = json.Marshal(thinking)
	}
	maps.Copy(fields, explicit)
	options[OpenAIGenerationOptionExtraBody] = fields
	request.Options = options
	return request, nil
}
