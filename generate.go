package gai

import (
	"context"
	"fmt"
	"strings"
)

const (
	ToolChoiceAuto          = "auto"
	ToolChoiceToolsRequired = "required"
)

// Generation option keys shared by provider generators.
const (
	GenerationOptionTemperature         = "temperature"
	GenerationOptionTopP                = "top_p"
	GenerationOptionTopK                = "top_k"
	GenerationOptionFrequencyPenalty    = "frequency_penalty"
	GenerationOptionPresencePenalty     = "presence_penalty"
	GenerationOptionCandidateCount      = "candidate_count"
	GenerationOptionMaxGenerationTokens = "max_generation_tokens"
	GenerationOptionToolChoice          = "tool_choice"
	GenerationOptionStopSequences       = "stop_sequences"
	GenerationOptionOutputModalities    = "output_modalities"
	GenerationOptionAudioConfig         = "audio_config"
	GenerationOptionThinkingBudget      = "thinking_budget"
)

type AudioConfig struct {
	// VoiceName represents what voice to use when generating an audio output as
	// A Generator usually offers an option to generate speech using a specific built-in voice
	VoiceName string `json:"voice_name,omitempty" yaml:"voice_name,omitempty"`

	// Format specifies the output audio format. Must be one a valid audio file format, such as wav or mp3.
	// A Generator's supported file formats will be specified in its docs
	Format string `json:"format,omitempty" yaml:"format,omitempty"`
}

// GenerationOptions contains common and provider-specific generation parameters.
// Recognized common keys use the GenerationOption* constants. Providers ignore
// keys they do not recognize.
type GenerationOptions map[string]any

// GenerationOption applies one typed common option to GenerationOptions.
type GenerationOption func(GenerationOptions)

// NewGenerationOptions creates generation options and applies options in order.
func NewGenerationOptions(options ...GenerationOption) GenerationOptions {
	values := make(GenerationOptions, len(options))
	for _, option := range options {
		option(values)
	}
	return values
}

// WithTemperature sets GenerationOptionTemperature.
func WithTemperature(value float64) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionTemperature] = value }
}

// WithTopP sets GenerationOptionTopP.
func WithTopP(value float64) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionTopP] = value }
}

// WithTopK sets GenerationOptionTopK.
func WithTopK(value uint) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionTopK] = value }
}

// WithFrequencyPenalty sets GenerationOptionFrequencyPenalty.
func WithFrequencyPenalty(value float64) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionFrequencyPenalty] = value }
}

// WithPresencePenalty sets GenerationOptionPresencePenalty.
func WithPresencePenalty(value float64) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionPresencePenalty] = value }
}

// WithCandidateCount sets GenerationOptionCandidateCount.
func WithCandidateCount(value uint) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionCandidateCount] = value }
}

// WithMaxGenerationTokens sets GenerationOptionMaxGenerationTokens.
func WithMaxGenerationTokens(value int) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionMaxGenerationTokens] = value }
}

// WithToolChoice sets GenerationOptionToolChoice.
func WithToolChoice(value string) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionToolChoice] = value }
}

// WithStopSequences sets GenerationOptionStopSequences and copies values.
func WithStopSequences(values ...string) GenerationOption {
	return func(options GenerationOptions) {
		options[GenerationOptionStopSequences] = append([]string(nil), values...)
	}
}

// WithOutputModalities sets GenerationOptionOutputModalities and copies values.
func WithOutputModalities(values ...Modality) GenerationOption {
	return func(options GenerationOptions) {
		options[GenerationOptionOutputModalities] = append([]Modality(nil), values...)
	}
}

// WithAudioConfig sets GenerationOptionAudioConfig.
func WithAudioConfig(value AudioConfig) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionAudioConfig] = value }
}

// WithThinkingBudget sets GenerationOptionThinkingBudget.
func WithThinkingBudget(value string) GenerationOption {
	return func(options GenerationOptions) { options[GenerationOptionThinkingBudget] = value }
}

// GenerationRequest contains all semantic inputs to one model invocation.
type GenerationRequest struct {
	// Model identifies the provider model for this invocation.
	Model string `json:"model" yaml:"model"`
	// Instructions contains system-role content outside the conversation dialog.
	Instructions Message `json:"instructions,omitempty" yaml:"instructions,omitempty"`
	// Dialog is the complete conversation supplied to this invocation.
	Dialog Dialog `json:"dialog" yaml:"dialog"`
	// Tools is the complete tool set available to this invocation.
	Tools []Tool `json:"tools,omitempty" yaml:"tools,omitempty"`
	// Options contains common and provider-specific generation parameters.
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

// FinishReason represents the reason why a Generator stopped generating and returned a Response
type FinishReason uint8

const (
	// Unknown represents an invalid FinishReason, likely only seen with a zero value Response
	Unknown FinishReason = iota

	// EndTurn represents the end of the Generator generating an output.
	// Note that this is different to the ToolUse reason,
	// which the Generator waits for a tool call result
	EndTurn

	// StopSequence represents the Generator generating one of the stop sequences
	// requested with GenerationOptionStopSequences and stopping generation
	StopSequence

	// MaxGenerationLimit represents the Generator reaching the maximum number of
	// tokens requested with GenerationOptionMaxGenerationTokens.
	MaxGenerationLimit

	// ToolUse represents the Generator pausing generated output after calling a
	// tool to wait for a tool call result.
	ToolUse

	// ContentPolicyViolation represents generation stopping because the input or
	// generated output violated the provider's content policy.
	ContentPolicyViolation
)

// Response is what is returned by a Generator
type Response struct {
	// Candidates contains the generated messages. Its length is controlled by
	// GenerationOptionCandidateCount, with a common default of one candidate.
	Candidates []Message `json:"candidates" yaml:"candidates"`

	// FinishReason represents the reason why a Generator stopped generating
	FinishReason FinishReason `json:"finish_reason" yaml:"finish_reason"`

	// UsageMetadata contains common and provider-specific measurements such as
	// token counts, cost, cache usage, and timing information.
	UsageMetadata Metadata `json:"usage_metadata,omitempty" yaml:"usage_metadata,omitempty"`

	// ExtraFields contains provider-specific information about the invocation.
	// Replay-critical candidate and content metadata belongs on Message.ExtraFields
	// or Block.ExtraFields instead.
	ExtraFields map[string]interface{} `json:"extra_fields,omitempty" yaml:"extra_fields,omitempty"`
}

// A Generator accepts a self-contained GenerationRequest and returns a Response or an error.
// The context provides cancellation, deadlines, and request-scoped values.
//
// A Generator implementation may return several types of errors:
//   - [ErrMaxGenerationLimit] when the maximum token generation limit is exceeded
//   - [UnsupportedInputModalityErr] when encountering an unsupported input modality
//   - [UnsupportedOutputModalityErr] when requested to generate an unsupported output modality
//   - [InvalidToolChoiceErr] when an invalid tool choice is specified
//   - [InvalidParameterErr] when generation parameters are invalid or out of range
//   - [ErrContextLengthExceeded] when input dialog is too long
//   - [ContentPolicyErr] when content violates usage policies
//   - [ErrEmptyDialog] when no messages are provided in the dialog
//   - [ApiErr] when a provider returns a server/API error
type Generator interface {
	Generate(ctx context.Context, request GenerationRequest) (Response, error)
}

// TokenCounter is an interface for a generator that can count the number of tokens in a Dialog.
// This is useful for:
//  1. Estimating costs before sending a request to the API
//  2. Checking if a dialog exceeds the context window limits of a model
//  3. Optimizing prompt design by analyzing token usage
//  4. Managing rate limits that are based on token counts
//
// The exact method of token counting varies by provider:
//   - OpenAI uses tiktoken to count tokens without making an API call
//   - Anthropic calls a dedicated counting API endpoint
//   - Gemini calls a dedicated counting API endpoint
//
// In all cases, Count receives the same request used for generation so model,
// instructions, dialog, and tools are counted consistently.
type TokenCounter interface {
	Count(ctx context.Context, request GenerationRequest) (uint, error)
}
