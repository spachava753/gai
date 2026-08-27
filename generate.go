package gai

import (
	"context"
	"fmt"
	"strings"
)

const (
	// ToolChoiceAuto lets the provider decide whether to call a tool. Pass it to
	// [WithToolChoice] or store it under [GenerationOptionToolChoice].
	ToolChoiceAuto = "auto"
	// ToolChoiceToolsRequired requires at least one tool call. Providers that do
	// not support required tool choice return [InvalidToolChoiceErr].
	ToolChoiceToolsRequired = "required"
)

// Common keys in [GenerationOptions]. Provider adapters ignore unsupported
// keys and validate recognized values when building a request.
const (
	// GenerationOptionTemperature is the float64 sampling-temperature key set by
	// [WithTemperature]. Valid ranges are provider-defined.
	GenerationOptionTemperature = "temperature"
	// GenerationOptionTopP is the float64 nucleus-sampling key set by [WithTopP].
	GenerationOptionTopP = "top_p"
	// GenerationOptionTopK is the uint top-k sampling key set by [WithTopK].
	GenerationOptionTopK = "top_k"
	// GenerationOptionFrequencyPenalty is the float64 frequency-penalty key set
	// by [WithFrequencyPenalty].
	GenerationOptionFrequencyPenalty = "frequency_penalty"
	// GenerationOptionPresencePenalty is the float64 presence-penalty key set by
	// [WithPresencePenalty].
	GenerationOptionPresencePenalty = "presence_penalty"
	// GenerationOptionCandidateCount is the uint candidate-count key set by
	// [WithCandidateCount]. It is supported by OpenAI Chat Completions, Gemini,
	// and OpenRouter generation; [StreamingAdapter] supports one candidate.
	GenerationOptionCandidateCount = "candidate_count"
	// GenerationOptionMaxGenerationTokens is the int output-token limit key set
	// by [WithMaxGenerationTokens].
	GenerationOptionMaxGenerationTokens = "max_generation_tokens"
	// GenerationOptionToolChoice is the string tool-selection key set by
	// [WithToolChoice]. Values can be "none", [ToolChoiceAuto],
	// [ToolChoiceToolsRequired], or a tool name when the provider supports it.
	GenerationOptionToolChoice = "tool_choice"
	// GenerationOptionStopSequences is the []string stop-sequence key set by
	// [WithStopSequences].
	GenerationOptionStopSequences = "stop_sequences"
	// GenerationOptionOutputModalities is the []Modality output key set by
	// [WithOutputModalities]. Providers reject unsupported requested modalities.
	GenerationOptionOutputModalities = "output_modalities"
	// GenerationOptionAudioConfig is the [AudioConfig] key set by
	// [WithAudioConfig]. OpenAI Chat Completions consumes this option when audio
	// output is requested.
	GenerationOptionAudioConfig = "audio_config"
	// GenerationOptionThinkingBudget is the string reasoning-effort or token-budget
	// key set by [WithThinkingBudget]. Accepted values differ by provider.
	GenerationOptionThinkingBudget = "thinking_budget"
)

// AudioConfig selects the voice and encoding for generated audio. Use it with
// [WithAudioConfig] and request [Audio] through [WithOutputModalities]. The
// provider validates supported voices and formats.
type AudioConfig struct {
	// VoiceName is the provider-defined built-in voice name.
	VoiceName string `json:"voice_name,omitempty" yaml:"voice_name,omitempty"`

	// Format is the provider-defined audio encoding, such as "wav" or "mp3".
	Format string `json:"format,omitempty" yaml:"format,omitempty"`
}

// GenerationOptions stores common and provider-specific request controls.
// Prefer [NewGenerationOptions] and typed [GenerationOption] helpers. Direct
// assignment is supported for exported keys and experimental provider values;
// callers must use the concrete value type documented by the key.
type GenerationOptions map[string]any

// GenerationOption mutates [GenerationOptions]. Options are applied in order by
// [NewGenerationOptions], so a later option can replace an earlier value.
type GenerationOption func(GenerationOptions)

// NewGenerationOptions allocates an option map and applies options in order.
// It returns a non-nil empty map when called without options.
func NewGenerationOptions(options ...GenerationOption) GenerationOptions {
	values := make(GenerationOptions, len(options))
	for _, option := range options {
		option(values)
	}
	return values
}

// WithTemperature returns a [GenerationOption] that stores value under
// [GenerationOptionTemperature]. Providers define the accepted range.
func WithTemperature(value float64) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionTemperature] = value }
}

// WithTopP returns a [GenerationOption] that stores value under
// [GenerationOptionTopP].
func WithTopP(value float64) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionTopP] = value }
}

// WithTopK returns a [GenerationOption] that stores value under
// [GenerationOptionTopK]. Providers without top-k sampling ignore it.
func WithTopK(value uint) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionTopK] = value }
}

// WithFrequencyPenalty returns a [GenerationOption] that stores value under
// [GenerationOptionFrequencyPenalty].
func WithFrequencyPenalty(value float64) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionFrequencyPenalty] = value }
}

// WithPresencePenalty returns a [GenerationOption] that stores value under
// [GenerationOptionPresencePenalty].
func WithPresencePenalty(value float64) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionPresencePenalty] = value }
}

// WithCandidateCount requests value independently generated candidates through
// [GenerationOptionCandidateCount]. A zero value is provider-defined.
func WithCandidateCount(value uint) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionCandidateCount] = value }
}

// WithMaxGenerationTokens limits generated output through
// [GenerationOptionMaxGenerationTokens]. Providers reject invalid limits and
// may return [ErrMaxGenerationLimit] when generation reaches the limit.
func WithMaxGenerationTokens(value int) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionMaxGenerationTokens] = value }
}

// WithToolChoice controls tool selection through [GenerationOptionToolChoice].
// Pass "none", [ToolChoiceAuto], [ToolChoiceToolsRequired], or a name from
// [GenerationRequest.Tools]. Unsupported choices return [InvalidToolChoiceErr].
func WithToolChoice(value string) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionToolChoice] = value }
}

// WithStopSequences stores a copy of values under
// [GenerationOptionStopSequences].
func WithStopSequences(values ...string) GenerationOption {
	return func(options GenerationOptions) {
		options[GenerationOptionStopSequences] = append([]string(nil), values...)
	}
}

// WithOutputModalities stores a copy of values under
// [GenerationOptionOutputModalities]. Unsupported requests return
// [UnsupportedOutputModalityErr].
func WithOutputModalities(values ...Modality) GenerationOption {
	return func(options GenerationOptions) {
		options[GenerationOptionOutputModalities] = append([]Modality(nil), values...)
	}
}

// WithAudioConfig stores value under [GenerationOptionAudioConfig]. Use it with
// [WithOutputModalities] when requesting [Audio] output.
func WithAudioConfig(value AudioConfig) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionAudioConfig] = value }
}

// WithThinkingBudget stores a provider-specific effort label or decimal token
// budget under [GenerationOptionThinkingBudget]. Provider adapters document and
// validate their accepted values.
func WithThinkingBudget(value string) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionThinkingBudget] = value }
}

// GenerationRequest contains every semantic input to one [Generator.Generate]
// invocation. It is safe to reuse a generator concurrently with different
// requests when the underlying provider client is concurrency-safe.
type GenerationRequest struct {
	// Model is the provider model identifier for this invocation.
	Model string `json:"model" yaml:"model"`
	// Instructions contains optional [System]-role content outside [Dialog]. Use
	// [SystemMessage] to construct non-empty instructions.
	Instructions Message `json:"instructions,omitempty" yaml:"instructions,omitempty"`
	// Dialog is the complete conversation. An empty dialog returns [ErrEmptyDialog].
	Dialog Dialog `json:"dialog" yaml:"dialog"`
	// Tools is the complete caller-defined function set available to this
	// invocation. Tool names must be unique.
	Tools []Tool `json:"tools,omitempty" yaml:"tools,omitempty"`
	// Options contains common and provider-specific controls. Providers ignore
	// keys they do not recognize.
	Options GenerationOptions `json:"options,omitempty" yaml:"options,omitempty"`
}

func generationOption[T any](options GenerationOptions, key string) (T, bool, error) {
	var zero T
	value, exists := options[key]
	if !exists {
		return zero, false, nil
	}
	typed, ok := value.(T)
	if !ok {
		return zero, false, &InvalidParameterErr{
			Parameter: key,
			Reason:    fmt.Sprintf("must have type %T, got %T", zero, value),
		}
	}
	return typed, true, nil
}

// textInstructions validates text-only system instructions and returns each
// instruction block as a separate string.
func textInstructions(instructions Message) ([]string, error) {
	if len(instructions.Blocks) == 0 {
		return nil, nil
	}
	if instructions.Role != System {
		return nil, &InvalidParameterErr{
			Parameter: "instructions",
			Reason:    "non-empty instructions must use the system role",
		}
	}

	parts := make([]string, 0, len(instructions.Blocks))
	for _, block := range instructions.Blocks {
		if block.BlockType != Content {
			return nil, &InvalidParameterErr{
				Parameter: "instructions",
				Reason:    fmt.Sprintf("block type %q is not supported", block.BlockType),
			}
		}
		if block.ModalityType != Text {
			return nil, UnsupportedInputModalityErr(block.ModalityType.String())
		}
		if block.Content == nil {
			return nil, &InvalidParameterErr{
				Parameter: "instructions",
				Reason:    "block content cannot be nil",
			}
		}
		parts = append(parts, block.Content.String())
	}
	return parts, nil
}

func joinedTextInstructions(instructions Message) (string, error) {
	parts, err := textInstructions(instructions)
	if err != nil {
		return "", err
	}
	return strings.Join(parts, "\n\n"), nil
}

// FinishReason classifies why generation stopped. It is normalized across
// providers and stored in [Response.FinishReason].
type FinishReason uint8

const (
	// Unknown is the zero value and means no finish reason was reported.
	Unknown FinishReason = iota

	// EndTurn means the model completed its response without another required
	// caller action.
	EndTurn

	// StopSequence means the model emitted a sequence requested through
	// [WithStopSequences].
	StopSequence

	// MaxGenerationLimit means generation reached a provider output limit,
	// commonly one requested through [WithMaxGenerationTokens].
	MaxGenerationLimit

	// ToolUse means the model emitted one or more [ToolCall] blocks and is waiting
	// for caller-supplied [ToolResultMessage] values.
	ToolUse

	// ContentPolicyViolation means the provider stopped because input or output
	// violated its content policy.
	ContentPolicyViolation
)

// Response contains normalized output from one [Generator.Generate] call.
// Provider-specific invocation details use [Response.ExtraFields]; candidate or
// replay data stays on the corresponding [Message] or [Block].
type Response struct {
	// Candidates contains generated assistant messages. Providers commonly return
	// one candidate; supported generators honor [WithCandidateCount].
	Candidates []Message `json:"candidates" yaml:"candidates"`

	// FinishReason reports why generation stopped.
	FinishReason FinishReason `json:"finish_reason" yaml:"finish_reason"`

	// UsageMetadata contains common and provider-specific measurements such as
	// token counts, cost, cache usage, and timing. Use [InputTokens],
	// [OutputTokens], and [GetMetric] to retrieve values.
	UsageMetadata Metadata `json:"usage_metadata,omitempty" yaml:"usage_metadata,omitempty"`

	// ExtraFields contains provider-specific invocation data such as completion
	// IDs, model names, timestamps, fingerprints, and service tiers. Replay data
	// belongs on [Message.ExtraFields] or [Block.ExtraFields].
	ExtraFields map[string]interface{} `json:"extra_fields,omitempty" yaml:"extra_fields,omitempty"`
}

// Generator performs non-streaming model generation. Implementations consume
// only the supplied [GenerationRequest]; callers may reuse one generator with
// independent request state.
//
// Generate honors context cancellation and can return validation errors before
// contacting a provider. Provider failures are returned as [ApiErr]. A terminal
// condition such as [ErrMaxGenerationLimit] can return both a partial [Response]
// and a non-nil error.
type Generator interface {
	// Generate sends request to the provider and returns its normalized response.
	Generate(ctx context.Context, request GenerationRequest) (Response, error)
}

// TokenCounter is an optional generator capability for estimating or querying
// the input tokens consumed by a [GenerationRequest]. Counting includes the
// model, instructions, dialog, and tools, but provider behavior can differ for
// multimodal content.
//
// [OpenAiGenerator] counts locally with tiktoken. [AnthropicGenerator] and
// [GeminiGenerator] call provider token-counting endpoints.
type TokenCounter interface {
	// Count returns the provider's input-token count for request. The context can
	// cancel implementations that perform a remote call.
	Count(ctx context.Context, request GenerationRequest) (uint, error)
}
