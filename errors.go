package gai

import (
	"errors"
	"fmt"
)

// MaxGenerationLimitErr is returned when a Generator has generated the maximum number of tokens
// specified by GenOpts.MaxGenerationTokens. This error indicates that the generation was terminated
// due to reaching the token limit rather than natural completion.
var MaxGenerationLimitErr = errors.New("maximum generation limit reached")

// UnsupportedInputModalityErr is returned when a Generator encounters an input Message
// with a Block that contains an unsupported Modality. The string value of this error
// contains the name of the unsupported modality.
//
// For example, if a Generator only supports text input but receives an audio input,
// it will return this error with details about the unsupported audio modality.
type UnsupportedInputModalityErr string

func (u UnsupportedInputModalityErr) Error() string {
	return fmt.Sprintf("unsupported input modality: %s", string(u))
}

// UnsupportedOutputModalityErr is returned when a Generator is requested to generate
// a response in a Modality that it does not support via GenOpts.OutputModalities.
// The string value of this error contains the name of the unsupported modality.
//
// For example, if a Generator only supports text output but is asked to generate
// audio content, it will return this error with details about the unsupported
// audio modality.
type UnsupportedOutputModalityErr string

func (u UnsupportedOutputModalityErr) Error() string {
	return fmt.Sprintf("unsupported output modality: %s", string(u))
}

// InvalidToolChoiceErr is returned when an invalid tool choice is specified in
// GenOpts.ToolChoice. This can occur in several scenarios:
//   - When a specific tool is requested but doesn't exist
//   - When tools are required (ToolChoiceToolsRequired) but no tools are provided
//
// The string value of this error contains details about why the tool choice was invalid.
type InvalidToolChoiceErr string

func (i InvalidToolChoiceErr) Error() string {
	return fmt.Sprintf("invalid tool choice: %s", string(i))
}

// InvalidParameterErr is returned when a generation parameter in GenOpts is invalid.
// This can occur in several scenarios:
//   - [GenOpts.Temperature], [GenOpts.TopP], or [GenOpts.TopK] values are out of valid range
//   - [GenOpts.FrequencyPenalty] or [GenOpts.PresencePenalty] are out of valid range
//   - [GenOpts.MaxGenerationTokens] is negative or zero
//   - Invalid combination of parameters (e.g., both [GenOpts.Temperature] and [GenOpts.TopP] set)
type InvalidParameterErr struct {
	// Parameter is the name of the invalid parameter
	Parameter string `json:"parameter" yaml:"parameter"`
	// Reason describes why the parameter is invalid
	Reason string `json:"reason" yaml:"reason"`
}

func (i InvalidParameterErr) Error() string {
	return fmt.Sprintf("invalid parameter %s: %s", i.Parameter, i.Reason)
}

// ContextLengthExceededErr is returned when the total number of tokens in the input Dialog
// exceeds the maximum context length supported by the Generator. Different Generator
// implementations may have different context length limits.
var ContextLengthExceededErr = errors.New("context length exceeded")

// ContentPolicyErr is returned when the input or generated content violates the Generator's
// content policy. This can include:
//   - Unsafe or inappropriate content
//   - Prohibited topics or language
//   - Content that violates usage terms
//
// The string value contains details about the specific policy violation.
type ContentPolicyErr string

func (c ContentPolicyErr) Error() string {
	return fmt.Sprintf("content policy violation: %s", string(c))
}

// ToolRegistrationErr is returned when registering a tool fails. This can occur in several scenarios:
//   - Empty tool name
//   - Tool name conflicts with an existing or built-in tool
//   - Tool name matches special values (ToolChoiceAuto, ToolChoiceToolsRequired)
//   - Invalid tool schema (e.g., Array type without Items field)
//
// The Cause field contains the underlying error that caused the registration to fail.
type ToolRegistrationErr struct {
	// Tool is the name of the tool that failed to register
	Tool string `json:"tool" yaml:"tool"`
	// Cause is the underlying error that caused the registration to fail
	Cause error `json:"cause,omitempty" yaml:"cause,omitempty"`
}

func (t ToolRegistrationErr) Error() string {
	return fmt.Sprintf("failed to register tool %q: %v", t.Tool, t.Cause)
}

// Unwrap returns the underlying cause of the tool registration failure
func (t ToolRegistrationErr) Unwrap() error {
	return t.Cause
}

// EmptyDialogErr is returned when an empty Dialog is provided to Generate.
// At least one Message must be present in the Dialog.
var EmptyDialogErr = errors.New("empty dialog: at least one message required")

// Provider identifies the upstream service that returned an API/server error.
type Provider string

const (
	ProviderAnthropic  Provider = "anthropic"
	ProviderCerebras   Provider = "cerebras"
	ProviderGemini     Provider = "gemini"
	ProviderOpenAI     Provider = "openai"
	ProviderOpenRouter Provider = "openrouter"
	ProviderResponses  Provider = "responses"
	ProviderZAI        Provider = "zai"
)

// APIErrorKind classifies server-originated errors in a provider-agnostic way.
type APIErrorKind string

const (
	APIErrorKindUnknown            APIErrorKind = "unknown"
	APIErrorKindInvalidRequest     APIErrorKind = "invalid_request"
	APIErrorKindAuthentication     APIErrorKind = "authentication"
	APIErrorKindPermission         APIErrorKind = "permission"
	APIErrorKindNotFound           APIErrorKind = "not_found"
	APIErrorKindRateLimit          APIErrorKind = "rate_limit"
	APIErrorKindRequestTooLarge    APIErrorKind = "request_too_large"
	APIErrorKindTimeout            APIErrorKind = "timeout"
	APIErrorKindServer             APIErrorKind = "server"
	APIErrorKindServiceUnavailable APIErrorKind = "service_unavailable"
	APIErrorKindOverloaded         APIErrorKind = "overloaded"
	APIErrorKindContentPolicy      APIErrorKind = "content_policy"
)

// ApiErr is returned when a provider responds with a server/API error. It stores
// a normalized classification, the HTTP status code when available, the raw body
// when available, and wraps the original provider error in Cause.
type ApiErr struct {
	Provider Provider     `json:"provider" yaml:"provider"`
	Kind     APIErrorKind `json:"kind" yaml:"kind"`

	// StatusCode is the HTTP status code returned by the API when available.
	StatusCode int `json:"status_code,omitempty" yaml:"status_code,omitempty"`
	// Message is the best-effort human-readable message extracted from the server response.
	Message string `json:"message,omitempty" yaml:"message,omitempty"`
	// RawBody is the unmodified server response body when the provider exposes it.
	RawBody string `json:"raw_body,omitempty" yaml:"raw_body,omitempty"`
	// Cause is the original provider error or a synthetic internal error representing
	// the raw provider response when no SDK error object exists.
	Cause error `json:"cause,omitempty" yaml:"cause,omitempty"`
}

func (a *ApiErr) Error() string {
	if a == nil {
		return "<nil>"
	}
	if a.StatusCode > 0 && a.Message != "" {
		return fmt.Sprintf("%s %s (%d): %s", a.Provider, a.Kind, a.StatusCode, a.Message)
	}
	if a.Message != "" {
		return fmt.Sprintf("%s %s: %s", a.Provider, a.Kind, a.Message)
	}
	if a.StatusCode > 0 {
		return fmt.Sprintf("%s %s (%d)", a.Provider, a.Kind, a.StatusCode)
	}
	return fmt.Sprintf("%s %s", a.Provider, a.Kind)
}

// Unwrap returns the original provider error when one exists.
func (a *ApiErr) Unwrap() error {
	if a == nil {
		return nil
	}
	return a.Cause
}

// Retryable reports whether the error represents a retryable upstream failure.
func (a *ApiErr) Retryable() bool {
	if a == nil {
		return false
	}
	if a.StatusCode >= 500 && a.StatusCode < 600 {
		return true
	}
	switch a.Kind {
	case APIErrorKindRateLimit, APIErrorKindTimeout, APIErrorKindServer, APIErrorKindServiceUnavailable, APIErrorKindOverloaded:
		return true
	default:
		return false
	}
}
