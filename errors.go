package gai

import (
	"errors"
	"fmt"
	"time"
)

// ErrMaxGenerationLimit reports that generation stopped at an output limit.
// The provider can return a partial [Response] whose finish reason is
// [MaxGenerationLimit]. A caller-supplied limit normally comes from
// [WithMaxGenerationTokens].
var ErrMaxGenerationLimit = errors.New("maximum generation limit reached")

// UnsupportedInputModalityErr reports that a [GenerationRequest] contains a
// [Block.ModalityType] unsupported by the selected provider or model. Its string
// value names the rejected modality.
type UnsupportedInputModalityErr string

// Error formats the rejected input modality.
func (u UnsupportedInputModalityErr) Error() string {
	return fmt.Sprintf("unsupported input modality: %s", string(u))
}

// UnsupportedOutputModalityErr reports that [WithOutputModalities] requested a
// modality unsupported by the selected provider or model. Its string value
// names the rejected modality.
type UnsupportedOutputModalityErr string

// Error formats the rejected output modality.
func (u UnsupportedOutputModalityErr) Error() string {
	return fmt.Sprintf("unsupported output modality: %s", string(u))
}

// InvalidToolChoiceErr reports a value rejected from [WithToolChoice]. Common
// causes are a missing named [Tool], required tool use with no tools, or a
// selection mode unsupported by the provider.
type InvalidToolChoiceErr string

// Error formats the invalid tool-choice reason.
func (i InvalidToolChoiceErr) Error() string {
	return fmt.Sprintf("invalid tool choice: %s", string(i))
}

// InvalidParameterErr reports a recognized [GenerationOptions] value with an
// invalid concrete type, provider range, enum value, or option combination.
type InvalidParameterErr struct {
	// Parameter is the exported option key or request field that failed validation.
	Parameter string `json:"parameter" yaml:"parameter"`
	// Reason describes the violated contract.
	Reason string `json:"reason" yaml:"reason"`
}

// Error formats the invalid parameter and reason.
func (i InvalidParameterErr) Error() string {
	return fmt.Sprintf("invalid parameter %s: %s", i.Parameter, i.Reason)
}

// ErrMissingAPIKey is returned by provider constructors that accept an explicit
// API key when that key is empty.
var ErrMissingAPIKey = errors.New("API key is required")

// ErrContextLengthExceeded is available to generators that can identify a
// context-window overflow independently of other invalid requests. A provider
// that does not expose a distinct reason can instead return [ApiErr] with
// [APIErrorKindInvalidRequest].
var ErrContextLengthExceeded = errors.New("context length exceeded")

// ContentPolicyErr reports a provider content-policy stop that includes a
// human-readable reason. [Response.FinishReason] can also be
// [ContentPolicyViolation] when the provider supplies a response.
type ContentPolicyErr string

// Error formats the content-policy reason.
func (c ContentPolicyErr) Error() string {
	return fmt.Sprintf("content policy violation: %s", string(c))
}

// InvalidToolErr reports an invalid declaration in [GenerationRequest.Tools].
// Empty, reserved, and duplicate names are rejected before a provider call;
// Cause can also contain a provider schema-conversion error.
type InvalidToolErr struct {
	// Tool is the invalid tool's name.
	Tool string `json:"tool" yaml:"tool"`
	// Cause is the underlying validation or conversion error.
	Cause error `json:"cause,omitempty" yaml:"cause,omitempty"`
}

// Error formats the invalid tool name and cause.
func (t InvalidToolErr) Error() string {
	return fmt.Sprintf("invalid tool %q: %v", t.Tool, t.Cause)
}

// Unwrap returns the underlying validation or conversion error.
func (t InvalidToolErr) Unwrap() error {
	return t.Cause
}

// ErrEmptyDialog is returned by built-in generators and token counters when
// [GenerationRequest.Dialog] contains no messages.
var ErrEmptyDialog = errors.New("empty dialog: at least one message required")

// Provider identifies the adapter that produced an [ApiErr].
type Provider string

const (
	// ProviderAnthropic identifies Anthropic Messages API failures.
	ProviderAnthropic Provider = "anthropic"
	// ProviderCerebras identifies Cerebras Chat Completions failures.
	ProviderCerebras Provider = "cerebras"
	// ProviderDeepSeek identifies DeepSeek Chat Completions failures.
	ProviderDeepSeek Provider = "deepseek"
	// ProviderGemini identifies Google Gemini failures.
	ProviderGemini Provider = "gemini"
	// ProviderOpenAI identifies OpenAI Chat Completions failures.
	ProviderOpenAI Provider = "openai"
	// ProviderOpenRouter identifies OpenRouter failures.
	ProviderOpenRouter Provider = "openrouter"
	// ProviderResponses identifies OpenAI Responses API failures.
	ProviderResponses Provider = "responses"
	// ProviderZAI identifies Z.AI failures.
	ProviderZAI Provider = "zai"
)

// APIErrorKind is a provider-neutral [ApiErr] classification used by
// [ApiErr.Retryable], [RetryGenerator], and fallback policies.
type APIErrorKind string

const (
	// APIErrorKindUnknown means the provider failure could not be classified.
	APIErrorKindUnknown APIErrorKind = "unknown"
	// APIErrorKindInvalidRequest means the request was malformed or unsupported.
	APIErrorKindInvalidRequest APIErrorKind = "invalid_request"
	// APIErrorKindAuthentication means credentials were absent or invalid.
	APIErrorKindAuthentication APIErrorKind = "authentication"
	// APIErrorKindPermission means valid credentials lack access to the operation.
	APIErrorKindPermission APIErrorKind = "permission"
	// APIErrorKindNotFound means a requested model or resource does not exist.
	APIErrorKindNotFound APIErrorKind = "not_found"
	// APIErrorKindRateLimit means the provider rejected work because of a quota or
	// rate limit. [ApiErr.RetryAfter] can provide a requested delay.
	APIErrorKindRateLimit APIErrorKind = "rate_limit"
	// APIErrorKindRequestTooLarge means the request body exceeded a provider limit.
	APIErrorKindRequestTooLarge APIErrorKind = "request_too_large"
	// APIErrorKindTimeout means the provider timed out while processing the request.
	APIErrorKindTimeout APIErrorKind = "timeout"
	// APIErrorKindServer means an otherwise unclassified provider server failure.
	APIErrorKindServer APIErrorKind = "server"
	// APIErrorKindServiceUnavailable means the provider cannot currently accept work.
	APIErrorKindServiceUnavailable APIErrorKind = "service_unavailable"
	// APIErrorKindOverloaded means provider capacity is temporarily exhausted.
	APIErrorKindOverloaded APIErrorKind = "overloaded"
	// APIErrorKindContentPolicy means the provider rejected content under its
	// safety or usage policy.
	APIErrorKindContentPolicy APIErrorKind = "content_policy"
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

// Error formats the provider, classification, status, and message available on a.
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
