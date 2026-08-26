package gai

import (
	"errors"
	"fmt"
	"time"
)

// ErrMaxGenerationLimit is returned when a Generator reaches the token limit set by
// GenerationOptionMaxGenerationTokens. Generation stopped because of the limit rather
// than a natural completion.
var ErrMaxGenerationLimit = errors.New("maximum generation limit reached")

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

// UnsupportedOutputModalityErr is returned when GenerationOptionOutputModalities
// requests a modality that a Generator does not support.
//
// For example, if a Generator only supports text output but is asked to generate
// audio content, it will return this error with details about the unsupported
// audio modality.
type UnsupportedOutputModalityErr string

func (u UnsupportedOutputModalityErr) Error() string {
	return fmt.Sprintf("unsupported output modality: %s", string(u))
}

// InvalidToolChoiceErr is returned when GenerationOptionToolChoice is invalid.
// This can occur when a named tool is absent from GenerationRequest.Tools or tools
// are required but the request provides none.
//
// The string value of this error contains details about why the tool choice was invalid.
type InvalidToolChoiceErr string

func (i InvalidToolChoiceErr) Error() string {
	return fmt.Sprintf("invalid tool choice: %s", string(i))
}

// InvalidParameterErr is returned when a recognized GenerationOptions value has
// the wrong type, is outside the provider's valid range, or conflicts with another
// option.
type InvalidParameterErr struct {
	// Parameter is the name of the invalid parameter
	Parameter string `json:"parameter" yaml:"parameter"`
	// Reason describes why the parameter is invalid
	Reason string `json:"reason" yaml:"reason"`
}

func (i InvalidParameterErr) Error() string {
	return fmt.Sprintf("invalid parameter %s: %s", i.Parameter, i.Reason)
}

// ErrContextLengthExceeded is returned when the total number of tokens in the input Dialog
// exceeds the maximum context length supported by the Generator. Different Generator
// implementations may have different context length limits.
var ErrContextLengthExceeded = errors.New("context length exceeded")

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

// InvalidToolErr is returned when a tool in GenerationRequest.Tools is invalid.
// Empty and reserved names, duplicate names, and unsupported schemas are rejected.
// Cause contains the provider conversion or validation error.
type InvalidToolErr struct {
	// Tool is the invalid tool's name.
	Tool string `json:"tool" yaml:"tool"`
	// Cause is the underlying validation or conversion error.
	Cause error `json:"cause,omitempty" yaml:"cause,omitempty"`
}

func (t InvalidToolErr) Error() string {
	return fmt.Sprintf("invalid tool %q: %v", t.Tool, t.Cause)
}

// Unwrap returns the underlying validation or conversion error.
func (t InvalidToolErr) Unwrap() error {
	return t.Cause
}

// ErrEmptyDialog is returned when an empty Dialog is provided to Generate.
// At least one Message must be present in the Dialog.
var ErrEmptyDialog = errors.New("empty dialog: at least one message required")

// Provider identifies the upstream service that returned an API/server error.
type Provider string

const (
	ProviderAnthropic  Provider = "anthropic"
	ProviderCerebras   Provider = "cerebras"
	ProviderDeepSeek   Provider = "deepseek"
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

// ApiErr represents an error returned by an upstream provider. Kind provides a
// provider-independent classification for retry and fallback decisions, while
// the remaining fields retain provider details when available.
//
// ApiErr unwraps to Cause so callers can inspect the underlying error with
// [errors.Is] or [errors.As].
type ApiErr struct {
	// Provider identifies the upstream service that returned the error.
	Provider Provider `json:"provider" yaml:"provider"`
	// Kind is the provider-independent classification of the error.
	Kind APIErrorKind `json:"kind" yaml:"kind"`

	// StatusCode is the HTTP status code returned by the API, or zero when unavailable.
	StatusCode int `json:"status_code,omitempty" yaml:"status_code,omitempty"`
	// Message is the best-effort human-readable message extracted from the provider response.
	Message string `json:"message,omitempty" yaml:"message,omitempty"`
	// RawBody is the provider response body when available.
	RawBody string `json:"raw_body,omitempty" yaml:"raw_body,omitempty"`
	// RetryAfterDuration is the provider-requested delay before another attempt,
	// or nil when the provider supplied no retry timing information.
	RetryAfterDuration *time.Duration `json:"retry_after,omitempty" yaml:"retry_after,omitempty"`
	// Cause is the underlying SDK or transport error when one is available.
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

// Unwrap returns the underlying SDK or transport error when one is available.
func (a *ApiErr) Unwrap() error {
	if a == nil {
		return nil
	}
	return a.Cause
}

// RetryAfter reports the provider-requested delay before another attempt.
// The boolean is false when the provider supplied no retry timing information.
func (a *ApiErr) RetryAfter() (time.Duration, bool) {
	if a == nil || a.RetryAfterDuration == nil {
		return 0, false
	}
	return *a.RetryAfterDuration, true
}

// Retryable reports whether the error represents a retryable upstream failure.
func (a *ApiErr) Retryable() bool {
	if a == nil {
		return false
	}
	// 501 and 505 report unsupported server capabilities, not transient failures.
	// https://www.rfc-editor.org/rfc/rfc9110.html#section-15.6
	switch a.StatusCode {
	case 501, 505:
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
