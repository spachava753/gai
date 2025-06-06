package gai

import (
	"context"
)

const (
	ToolChoiceAuto          = "auto"
	ToolChoiceToolsRequired = "required"
)

type AudioConfig struct {
	// VoiceName represents what voice to use when generating an audio output as
	// A Generator usually offers an option to generate speech using a specific built-in voice
	VoiceName string

	// Format specifies the output audio format. Must be one a valid audio file format, such as wav or mp3.
	// A Generator's supported file formats will be specified in its docs
	Format string
}

// GenOpts represents the parameters that customize how a response is generated by a Generator
type GenOpts struct {
	// Temperature is a parameter that controls the randomness of a Generator when calling [Generator.Generate].
	// Higher temperatures lead to more creative and diverse outputs, while lower temperatures result in more
	// conservative and deterministic outputs
	//
	// Must be between 0.0 and 1.0. Default value is 0.0
	Temperature float64

	// TopP is a parameter that uses nucleus sampling. The API computes the cumulative distribution over all
	// options for each subsequent token in decreasing probability order and cuts it off once it reaches the
	// specified probability. You should either alter Temperature or TopP, but not both
	//
	// Must be between 0.0 and 1.0. Default value is 0.0
	TopP float64

	// TopK is used to only sample from the top k options for each subsequent token, and generally
	// used to remove "long tail" low probability responses.
	//
	// Requires a value greater than 0, so if the value is set to 0, it is treated as if the value is not set
	//
	// Recommended for advanced use cases only - you usually only need to use temperature
	TopK uint

	// FrequencyPenalty is a number between -2.0 and 2.0. Positive values penalize new tokens based on their
	// existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
	//
	// Default value is 0
	//
	// Note that this parameter is not supported by every Generator, in which case this parameter will be ignored
	FrequencyPenalty float64

	// PresencePenalty is a number between -2.0 and 2.0. Positive values penalize new tokens based on whether
	// they appear in the text so far, increasing the model's likelihood to talk about new topics
	//
	// Default value is 0
	//
	// Note that this parameter is not supported by every Generator, in which case this parameter will be ignored
	PresencePenalty float64

	// N represents how many [Response.Candidates] to generate.
	//
	// If N is not set (equal to 0), then the default value of 1 is used
	//
	// IMPORTANT: Note that you will be charged based on the number of generated tokens across all the choices.
	// Keep N as 1 to minimize costs
	N uint

	// MaxGenerationTokens is the maximum number of tokens to generate before stopping for each [Response.Candidates].
	//
	// Note that a Generator may stop before reaching this maximum.
	// This parameter only specifies the absolute maximum number of tokens to generate
	MaxGenerationTokens int

	// ToolChoice represents how the Generator should use the provided tools.
	// The Generator can use a specific tool, any available tool, or decide by itself
	//
	// Setting ToolChoice to specific value enables different behavior:
	// - If set to ToolChoiceAuto, the Generator decides for itself whether it should call tools
	// - If set to ToolChoiceToolsRequired, the Generator is required to generate a response with tool calls
	// - If set to some non-empty value, it is interpreted as a tool name,
	//   and requires that Generator call the specific tool provided by name
	ToolChoice string

	// StopSequences are custom text sequences that will cause the model to stop generating
	StopSequences []string

	// OutputModalities is an optional parameter that represents what type of outputs a Generator can generate.
	// If OutputModalities is nil or empty, then a default of only Text Modality is used.
	// OutputModalities only needs to be specified when generating modalities other than Text.
	OutputModalities []Modality

	// AudioConfig are parameters for audio output.
	// Required when audio output is requested with Modality Audio in OutputModalities
	AudioConfig AudioConfig

	// ThinkingBudget is an optional parameter used for a Generator that can perform reasoning.
	//
	// Note that if a Generator does not support this parameter, it will simply be ignored, even if set
	ThinkingBudget string

	// ExtraArgs is an optional parameter used to pass a Generator-specific generation parameters not
	// already supported by any of the above fields
	ExtraArgs map[string]any
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

	// StopSequence represents the Generator generating one of the [GenOpts.StopSequences]
	// and stopping generation
	StopSequence

	// MaxGenerationLimit represents the Generator generating too many tokens and
	// reaching the specified [GenOpts.MaxGenerationTokens]
	MaxGenerationLimit

	// ToolUse represents the Generator generating pausing generating output after
	// calling a tool to wait for a tool call result.
	ToolUse
)

// Response is what is returned by a Generator
type Response struct {
	// Candidates represents the list of possible generations that a Generator generates,
	// equal to the number specified in [GenOpts.N]. Since the default value of [GenOpts.N] is 1,
	// you can expect at least one Message to be present
	Candidates []Message

	// FinishReason represents the reason why a Generator stopped generating
	FinishReason FinishReason

	// UsageMetrics represents some arbitrary metrics that a Generator can return.
	// The metric UsageMetricInputTokens and UsageMetricGenerationTokens is most commonly returned by an
	// implementation of a Generator, representing the total input tokens and output tokens consumed, however
	// it is not guaranteed to have those metrics be present. In addition, a Generator may return additional metrics
	// specific to the implementation. An example might be cached input tokens used, or perhaps the cost of a request
	UsageMetrics Metrics
}

// A Generator takes a Dialog and optional GenOpts and generates a Response or an error.
// A [context.Context] is provided to the Generator as to provide not only cancellation and request specific values,
// but also to pass Generator implementation specific parameters if needed.
//
// An example would be an implementation of Generator offering a beta feature not yet offered as common functionality,
// or utilizing a special feature specific to an implementation of Generator.
//
// A Generator implementation may return several types of errors:
//   - [MaxGenerationLimitErr] when the maximum token generation limit is exceeded
//   - [UnsupportedInputModalityErr] when encountering an unsupported input modality
//   - [UnsupportedOutputModalityErr] when requested to generate an unsupported output modality
//   - [InvalidToolChoiceErr] when an invalid tool choice is specified
//   - [InvalidParameterErr] when generation parameters are invalid or out of range
//   - [ContextLengthExceededErr] when input dialog is too long
//   - [ContentPolicyErr] when content violates usage policies
//   - [EmptyDialogErr] when no messages are provided in the dialog
//   - [AuthenticationErr] when there are authentication or authorization issues
type Generator interface {
	Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error)
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
// In all cases, the Count method takes a context for cancellation and a Dialog to analyze.
// The number of tokens is returned as a uint.
//
// Note that some providers count system instructions separately, but this interface
// will include them in the returned count if the generator was initialized with them.
type TokenCounter interface {
	Count(ctx context.Context, dialog Dialog) (uint, error)
}
