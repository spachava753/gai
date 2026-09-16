package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"net/http"
)

// DeepSeekDefaultBaseURL is used when the constructor receives an empty base URL.
const DeepSeekDefaultBaseURL = "https://api.deepseek.com"

// DeepSeekGenerator delegates Chat Completions and streaming to [OpenAiGenerator].
// It preserves reasoning and native fields using the OpenAIExtraField and
// OpenAIResponseExtraField constants. Output limits default to max_tokens;
// streamed usage is requested by default. It does not implement [TokenCounter]:
// DeepSeek documents an offline tokenizer rather than a counting endpoint.
//
// [WithDeepSeekThinking] controls thinking.type; [WithReasoningEffort] passes
// effort labels unchanged. DeepSeek may ignore sampling settings while thinking
// and restrict required/named tool choice to non-thinking requests. The adapter
// does not infer capabilities from model names or rewrite caller settings.
// Historical reasoning is always replayed, including when tools are present.
//
// SafetyIdentifier maps to user_id, which DeepSeek also uses for KV-cache and
// scheduling isolation. PromptCacheKey is ignored. Function tools, JSON-object
// output, and logprobs use the shared OpenAI helpers; the server validates model
// support. No strict-beta tool defaults are injected.
type DeepSeekGenerator struct {
	client *OpenAiGenerator
}

// NewDeepSeekGenerator uses http.DefaultClient for a nil client and
// [DeepSeekDefaultBaseURL] for an empty URL. Credentials are explicit; no retries
// or environment configuration are added. The URL includes the API prefix.
func NewDeepSeekGenerator(httpClient *http.Client, baseURL, apiKey string) (*DeepSeekGenerator, error) {
	if baseURL == "" {
		baseURL = DeepSeekDefaultBaseURL
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	client, err := NewOpenAiGenerator(httpClient, baseURL, apiKey)
	if err != nil {
		return nil, err
	}
	return &DeepSeekGenerator{client: client}, nil
}

// Generate sends one request without retrying.
func (g *DeepSeekGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("deepseek: uninitialized generator")
	}
	request, err := g.prepareRequest(request)
	if err != nil {
		return Response{}, err
	}
	response, err := g.client.Generate(ctx, request)
	return response, compatibleError(ProviderDeepSeek, err)
}

// Stream yields ordered chunks without reconnecting or retrying.
func (g *DeepSeekGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("deepseek: uninitialized generator")})
			return
		}
		request, err := g.prepareRequest(request)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		for chunk := range g.client.Stream(ctx, request) {
			chunk.Err = compatibleError(ProviderDeepSeek, chunk.Err)
			if !yield(chunk) {
				return
			}
		}
	}
}

var _ Generator = (*DeepSeekGenerator)(nil)
var _ StreamingGenerator = (*DeepSeekGenerator)(nil)

// DeepSeekGenerationOptionThinkingEnabled is the bool key set by [WithDeepSeekThinking].
const DeepSeekGenerationOptionThinkingEnabled = "deepseek_thinking_enabled"

// WithDeepSeekThinking enables or disables reasoning; the server may reject disabling always-thinking models.
func WithDeepSeekThinking(value bool) GenerationOption {
	return func(options GenerationOptions) { options[DeepSeekGenerationOptionThinkingEnabled] = value }
}

// prepareRequest applies DeepSeek wire conventions without mutating caller data.
// Explicit extra-body fields take precedence over translated options.
func (g *DeepSeekGenerator) prepareRequest(request GenerationRequest) (GenerationRequest, error) {
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
	if _, ok := options[OpenAIGenerationOptionTokenLimitField]; !ok {
		options[OpenAIGenerationOptionTokenLimitField] = "max_tokens"
	}
	if request.SafetyIdentifier != "" {
		fields["user_id"], _ = json.Marshal(request.SafetyIdentifier)
	}
	request.SafetyIdentifier = ""
	request.PromptCacheKey = ""
	if enabled, present, err := generationOption[bool](options, DeepSeekGenerationOptionThinkingEnabled); err != nil {
		return request, err
	} else if present {
		thinking["type"] = "disabled"
		if enabled {
			thinking["type"] = "enabled"
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
