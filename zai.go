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

// ZaiDefaultBaseURL is used when the constructor receives an empty base URL.
const ZaiDefaultBaseURL = "https://api.z.ai/api/paas/v4"

// ZaiGenerator delegates Chat Completions and streaming to [OpenAiGenerator].
// It supports the shared text, data-URL image, inline PDF, audio, and tool wire
// representations where accepted by the chosen model. Video and the former
// ZaiExtraFieldURL representation are not converted. It preserves reasoning
// and opaque metadata using the OpenAIExtraField and OpenAIResponseExtraField
// constants, including hosted-search response fields.
//
// Output limits default to max_tokens; streamed usage is not requested unless
// explicitly enabled with [WithOpenAIStreamUsage]. [WithZaiToolStream] controls
// tool streaming separately. Thinking and sampling defaults are left to the
// server. [WithZaiClearThinking] never removes reasoning from caller history;
// Coding Plan and standard endpoints may have different reuse defaults.
// GLM always-thinking models may reject disabling thinking. Effort labels and
// tool-choice support depend on the model; caller settings are never rewritten.
//
// SafetyIdentifier maps to user_id (the provider documents 6–128 characters).
// PromptCacheKey is ignored. [ZaiGenerator.Count] uses the native tokenizer, including tools
// and model-supported image input. PDF acceptance is backend-dependent; this
// adapter does not upload, extract, or preprocess documents.
type ZaiGenerator struct {
	client     *OpenAiGenerator
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

// NewZaiGenerator uses http.DefaultClient for a nil client and
// [ZaiDefaultBaseURL] for an empty URL. Credentials are explicit; no retries
// or environment configuration are added. The URL includes the API prefix.
func NewZaiGenerator(httpClient *http.Client, baseURL, apiKey string) (*ZaiGenerator, error) {
	if baseURL == "" {
		baseURL = ZaiDefaultBaseURL
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	client, err := NewOpenAiGenerator(httpClient, baseURL, apiKey)
	if err != nil {
		return nil, err
	}
	return &ZaiGenerator{client: client, httpClient: httpClient, baseURL: baseURL, apiKey: apiKey}, nil
}

// Generate sends one request without retrying.
func (g *ZaiGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("zai: uninitialized generator")
	}
	request, err := g.prepareRequest(request)
	if err != nil {
		return Response{}, err
	}
	response, err := g.client.Generate(ctx, request)
	return response, compatibleError(ProviderZAI, err)
}

// Stream yields ordered chunks without reconnecting or retrying.
func (g *ZaiGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("zai: uninitialized generator")})
			return
		}
		request, err := g.prepareRequest(request)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		for chunk := range g.client.Stream(ctx, request) {
			chunk.Err = compatibleError(ProviderZAI, chunk.Err)
			if !yield(chunk) {
				return
			}
		}
	}
}

var _ Generator = (*ZaiGenerator)(nil)
var _ StreamingGenerator = (*ZaiGenerator)(nil)

// Count calls Z.AI's native tokenizer endpoint with model, messages, and tools.
// It uses the same content conversion as generation, honors cancellation, and
// rejects missing, negative, fractional, or overflowing total_tokens values.
func (g *ZaiGenerator) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	params, _, err := openAIRequest(request, false)
	if err != nil {
		return 0, err
	}
	body := map[string]any{"model": params.Model, "messages": params.Messages}
	if len(request.Tools) > 0 {
		body["tools"] = params.Tools
	}
	var result struct {
		Usage struct {
			TotalTokens *uint `json:"total_tokens"`
		} `json:"usage"`
	}
	err = compatiblePostJSON(ctx, g.httpClient, strings.TrimRight(g.baseURL, "/")+"/tokenizer", g.apiKey, ProviderZAI, body, &result)
	if err != nil {
		return 0, err
	}
	if result.Usage.TotalTokens == nil {
		return 0, fmt.Errorf("zai: missing total_tokens")
	}
	return *result.Usage.TotalTokens, nil
}

var _ TokenCounter = (*ZaiGenerator)(nil)

// ZaiGenerationOptionThinkingEnabled is the bool key set by [WithZaiThinking].
const ZaiGenerationOptionThinkingEnabled = "zai_thinking_enabled"

// WithZaiThinking enables or disables reasoning; the server may reject disabling always-thinking models.
func WithZaiThinking(value bool) GenerationOption {
	return func(options GenerationOptions) { options[ZaiGenerationOptionThinkingEnabled] = value }
}

// ZaiGenerationOptionClearThinking is the bool key set by [WithZaiClearThinking].
const ZaiGenerationOptionClearThinking = "zai_clear_thinking"

// WithZaiClearThinking controls server-side historical reasoning reuse without deleting caller history.
func WithZaiClearThinking(value bool) GenerationOption {
	return func(options GenerationOptions) { options[ZaiGenerationOptionClearThinking] = value }
}

// ZaiGenerationOptionDoSample is the bool key set by [WithZaiDoSample].
const ZaiGenerationOptionDoSample = "zai_do_sample"

// WithZaiDoSample controls sampling; false makes the server bypass temperature and top_p.
func WithZaiDoSample(value bool) GenerationOption {
	return func(options GenerationOptions) { options[ZaiGenerationOptionDoSample] = value }
}

// ZaiGenerationOptionToolStream is the bool key set by [WithZaiToolStream].
const ZaiGenerationOptionToolStream = "zai_tool_stream"

// WithZaiToolStream enables streamed tool-call output.
func WithZaiToolStream(value bool) GenerationOption {
	return func(options GenerationOptions) { options[ZaiGenerationOptionToolStream] = value }
}

// prepareRequest applies Zai wire conventions without mutating caller data.
// Explicit extra-body fields take precedence over translated options.
func (g *ZaiGenerator) prepareRequest(request GenerationRequest) (GenerationRequest, error) {
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
	if _, ok := options[OpenAIGenerationOptionStreamUsage]; !ok {
		options[OpenAIGenerationOptionStreamUsage] = false
	}
	if enabled, present, err := generationOption[bool](options, ZaiGenerationOptionThinkingEnabled); err != nil {
		return request, err
	} else if present {
		thinking["type"] = "disabled"
		if enabled {
			thinking["type"] = "enabled"
		}
	}
	if clear, present, err := generationOption[bool](options, ZaiGenerationOptionClearThinking); err != nil {
		return request, err
	} else if present {
		thinking["clear_thinking"] = clear
	}
	for _, option := range []struct{ key, field string }{{ZaiGenerationOptionDoSample, "do_sample"}, {ZaiGenerationOptionToolStream, "tool_stream"}} {
		if value, present, err := generationOption[bool](options, option.key); err != nil {
			return request, err
		} else if present {
			fields[option.field], _ = json.Marshal(value)
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
