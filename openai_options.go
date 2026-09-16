package gai

import (
	"encoding/json"
	"maps"
)

// WithOpenAIResponseFormat sends a native response_format object, such as
// {"type":"json_object"} or {"type":"json_schema","json_schema":{...}}.
// Model support and schema restrictions are provider-defined. The raw JSON is
// copied when applying the option. Invalid JSON is rejected before HTTP.
func WithOpenAIResponseFormat(format json.RawMessage) GenerationOption {
	return openAIFieldOption("response_format", format)
}

// WithOpenAILogprobs requests token log probabilities. They are retained in
// choice metadata under [OpenAIExtraFieldChoice], including when streaming.
func WithOpenAILogprobs(enabled bool) GenerationOption {
	value, _ := json.Marshal(enabled)
	return openAIFieldOption("logprobs", value)
}

// WithOpenAITopLogprobs requests the provider-defined number of alternatives per
// output token. Use with [WithOpenAILogprobs]; the server validates its range.
func WithOpenAITopLogprobs(count int) GenerationOption {
	value, _ := json.Marshal(count)
	return openAIFieldOption("top_logprobs", value)
}

// WithOpenAIPromptCacheRetention sets OpenAI's prompt_cache_retention policy
// (for example "in_memory" or "24h"). It is independent of PromptCacheKey;
// support and permitted values depend on the model.
func WithOpenAIPromptCacheRetention(retention string) GenerationOption {
	value, _ := json.Marshal(retention)
	return openAIFieldOption("prompt_cache_retention", value)
}

// WithOpenAIPromptCacheOptions sends OpenAI's native prompt_cache_options object.
// This is a cache policy, not a cache grouping key. The provider validates
// supported mode/ttl combinations and model support; no defaults are inferred.
func WithOpenAIPromptCacheOptions(options json.RawMessage) GenerationOption {
	return openAIFieldOption("prompt_cache_options", options)
}

func openAIFieldOption(name string, value json.RawMessage) GenerationOption {
	return func(options GenerationOptions) {
		fields, ok := options[OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)
		if _, present := options[OpenAIGenerationOptionExtraBody]; present && !ok {
			return
		} // Preserve invalid values for request validation.
		fields = maps.Clone(fields)
		if fields == nil {
			fields = map[string]json.RawMessage{}
		}
		fields[name] = append(json.RawMessage(nil), value...)
		options[OpenAIGenerationOptionExtraBody] = fields
	}
}
