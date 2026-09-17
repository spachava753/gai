package gai

import (
	"encoding/json"
	"maps"
)

const (
	// OpenAIGenerationOptionOutputModalities is the []Modality output selection set by [WithOpenAIOutputModalities].
	OpenAIGenerationOptionOutputModalities = "openai_output_modalities"
	// OpenAIGenerationOptionAudioConfig is the [OpenAIAudioConfig] output configuration set by [WithOpenAIAudioConfig].
	OpenAIGenerationOptionAudioConfig = "openai_audio_config"
)

// OpenAIAudioConfig selects the voice and encoding for Chat Completions audio
// output. Use [WithOpenAIAudioConfig] and request [Audio] through
// [WithOpenAIOutputModalities]. The provider validates supported voices and formats.
// Format also determines the MIME type assigned to returned audio blocks.
type OpenAIAudioConfig struct {
	// VoiceName is the provider-defined built-in voice name.
	VoiceName string `json:"voice_name,omitempty" yaml:"voice_name,omitempty"`
	// Format is the audio encoding, such as "wav" or "mp3".
	Format string `json:"format,omitempty" yaml:"format,omitempty"`
}

// WithOpenAIOutputModalities selects Chat Completions output modalities using
// [OpenAIGenerationOptionOutputModalities]. Text and Audio are supported by the
// adapter; the server validates model support. Requesting Audio also requires
// [WithOpenAIAudioConfig]. The values are copied when the option is applied.
func WithOpenAIOutputModalities(values ...Modality) GenerationOption {
	return func(options GenerationOptions) {
		options[OpenAIGenerationOptionOutputModalities] = append([]Modality(nil), values...)
	}
}

// WithOpenAIAudioConfig configures Chat Completions audio output under
// [OpenAIGenerationOptionAudioConfig]. Request Audio through
// [WithOpenAIOutputModalities] to send the voice and format to the provider.
// The format is also used to decode audio responses, including streamed output.
func WithOpenAIAudioConfig(value OpenAIAudioConfig) GenerationOption {
	return func(options GenerationOptions) { options[OpenAIGenerationOptionAudioConfig] = value }
}

// WithOpenAIResponseFormat sends a native response_format object, such as
// {"type":"json_object"} or {"type":"json_schema","json_schema":{...}}.
// Model support and schema restrictions are provider-defined. The raw JSON is
// copied when applying the option. Invalid JSON is rejected before HTTP.
func WithOpenAIResponseFormat(format json.RawMessage) GenerationOption {
	return openAIFieldOption("response_format", format)
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
