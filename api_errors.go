package gai

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	a "github.com/anthropics/anthropic-sdk-go"
	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

type syntheticAPIErrorCause struct {
	provider Provider
	message  string
	rawBody  string
}

func (s syntheticAPIErrorCause) Error() string {
	switch {
	case s.message != "":
		return s.message
	case s.rawBody != "":
		return s.rawBody
	default:
		return fmt.Sprintf("%s api error", s.provider)
	}
}

type genericAPIErrorPayload struct {
	Code    any                     `json:"code"`
	Type    string                  `json:"type"`
	Status  string                  `json:"status"`
	Message string                  `json:"message"`
	Error   *genericAPIErrorPayload `json:"error"`
}

func newAPIError(provider Provider, kind APIErrorKind, statusCode int, message, rawBody string, cause error) *ApiErr {
	message = strings.TrimSpace(message)
	rawBody = strings.TrimSpace(rawBody)
	if message == "" {
		message = extractGenericErrorMessage(rawBody)
	}
	if cause == nil && (message != "" || rawBody != "") {
		cause = syntheticAPIErrorCause{provider: provider, message: message, rawBody: rawBody}
	}
	return &ApiErr{
		Provider:   provider,
		Kind:       kind,
		StatusCode: statusCode,
		Message:    message,
		RawBody:    rawBody,
		Cause:      cause,
	}
}

func classifyByStatus(statusCode int) APIErrorKind {
	switch statusCode {
	case 0:
		return APIErrorKindUnknown
	case 400:
		return APIErrorKindInvalidRequest
	case 401:
		return APIErrorKindAuthentication
	case 403:
		return APIErrorKindPermission
	case 404:
		return APIErrorKindNotFound
	case 413:
		return APIErrorKindRequestTooLarge
	case 429:
		return APIErrorKindRateLimit
	case 500, 502:
		return APIErrorKindServer
	case 503:
		return APIErrorKindServiceUnavailable
	case 504:
		return APIErrorKindTimeout
	case 529:
		return APIErrorKindOverloaded
	default:
		if statusCode >= 500 {
			return APIErrorKindServer
		}
		if statusCode >= 400 {
			return APIErrorKindInvalidRequest
		}
		return APIErrorKindUnknown
	}
}

func containsAnyFold(s string, values ...string) bool {
	s = strings.ToLower(s)
	for _, value := range values {
		if strings.Contains(s, strings.ToLower(value)) {
			return true
		}
	}
	return false
}

func normalizeOpenAICompatibleKind(statusCode int, typ, code, message string) APIErrorKind {
	joined := strings.ToLower(strings.Join([]string{typ, code, message}, " "))
	switch {
	case containsAnyFold(joined, "authentication_error", "invalid_api_key", "incorrect_api_key", "api key", "unauthenticated", "unauthorized"):
		return APIErrorKindAuthentication
	case containsAnyFold(joined, "permission_error", "permission denied", "forbidden", "insufficient_permissions"):
		return APIErrorKindPermission
	case containsAnyFold(joined, "rate_limit", "too many requests", "quota exceeded", "rate limit"):
		return APIErrorKindRateLimit
	case containsAnyFold(joined, "not_found"):
		return APIErrorKindNotFound
	case containsAnyFold(joined, "request_too_large", "payload too large"):
		return APIErrorKindRequestTooLarge
	case containsAnyFold(joined, "service_unavailable", "unavailable"):
		return APIErrorKindServiceUnavailable
	case containsAnyFold(joined, "timeout", "deadline exceeded"):
		return APIErrorKindTimeout
	case containsAnyFold(joined, "overloaded"):
		return APIErrorKindOverloaded
	case containsAnyFold(joined, "content policy"):
		return APIErrorKindContentPolicy
	case containsAnyFold(joined, "server_error", "api_error", "internal_error", "internal server"):
		return APIErrorKindServer
	default:
		return classifyByStatus(statusCode)
	}
}

func normalizeAnthropicKind(statusCode int, typ, message string) APIErrorKind {
	switch strings.ToLower(strings.TrimSpace(typ)) {
	case "authentication_error":
		return APIErrorKindAuthentication
	case "permission_error":
		return APIErrorKindPermission
	case "not_found_error":
		return APIErrorKindNotFound
	case "rate_limit_error":
		return APIErrorKindRateLimit
	case "timeout_error":
		return APIErrorKindTimeout
	case "api_error":
		return APIErrorKindServer
	case "overloaded_error":
		return APIErrorKindOverloaded
	case "invalid_request_error", "billing_error":
		return APIErrorKindInvalidRequest
	default:
		return normalizeOpenAICompatibleKind(statusCode, typ, "", message)
	}
}

func normalizeGeminiKind(statusCode int, status, message string, details []map[string]any) APIErrorKind {
	status = strings.ToUpper(strings.TrimSpace(status))
	detailsJSON, _ := json.Marshal(details)
	detailsText := strings.ToLower(string(detailsJSON))
	messageLower := strings.ToLower(message)

	if strings.Contains(detailsText, "api_key_invalid") || strings.Contains(messageLower, "api key") {
		return APIErrorKindAuthentication
	}

	switch status {
	case "UNAUTHENTICATED":
		return APIErrorKindAuthentication
	case "PERMISSION_DENIED":
		return APIErrorKindPermission
	case "NOT_FOUND":
		return APIErrorKindNotFound
	case "RESOURCE_EXHAUSTED":
		return APIErrorKindRateLimit
	case "INVALID_ARGUMENT":
		return APIErrorKindInvalidRequest
	case "DEADLINE_EXCEEDED":
		return APIErrorKindTimeout
	case "UNAVAILABLE":
		return APIErrorKindServiceUnavailable
	case "INTERNAL":
		return APIErrorKindServer
	default:
		return classifyByStatus(statusCode)
	}
}

func normalizeResponsesEventKind(code, message string) APIErrorKind {
	joined := strings.ToLower(strings.Join([]string{code, message}, " "))
	switch {
	case containsAnyFold(joined, "rate_limit_exceeded", "rate limit"):
		return APIErrorKindRateLimit
	case containsAnyFold(joined, "server_error"):
		return APIErrorKindServer
	case containsAnyFold(joined, "timeout"):
		return APIErrorKindTimeout
	case containsAnyFold(joined, "content_policy"):
		return APIErrorKindContentPolicy
	case containsAnyFold(joined, "invalid_prompt", "invalid_image", "invalid_base64_image", "invalid_image_url", "image_too_large", "image_too_small", "image_parse_error", "invalid_image_mode", "image_file_too_large", "unsupported_image_media_type", "empty_image_file", "failed_to_download_image", "image_file_not_found"):
		return APIErrorKindInvalidRequest
	default:
		return APIErrorKindUnknown
	}
}

func extractGenericErrorMessage(rawBody string) string {
	if rawBody == "" {
		return ""
	}
	var payload genericAPIErrorPayload
	if err := json.Unmarshal([]byte(rawBody), &payload); err != nil {
		return strings.TrimSpace(rawBody)
	}
	for payload.Error != nil {
		payload = *payload.Error
	}
	if strings.TrimSpace(payload.Message) != "" {
		return strings.TrimSpace(payload.Message)
	}
	return strings.TrimSpace(rawBody)
}

func extractGenericErrorFields(rawBody string) (message, typ, code, status string) {
	if rawBody == "" {
		return "", "", "", ""
	}
	var payload genericAPIErrorPayload
	if err := json.Unmarshal([]byte(rawBody), &payload); err != nil {
		return strings.TrimSpace(rawBody), "", "", ""
	}
	for payload.Error != nil {
		payload = *payload.Error
	}
	return strings.TrimSpace(payload.Message), strings.TrimSpace(payload.Type), stringifyCode(payload.Code), strings.TrimSpace(payload.Status)
}

func stringifyCode(code any) string {
	switch v := code.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(v)
	case float64:
		return fmt.Sprintf("%.0f", v)
	case json.Number:
		return v.String()
	default:
		return strings.TrimSpace(fmt.Sprint(v))
	}
}

func mapOpenAICompatibleError(provider Provider, err error) error {
	var apierr *oai.Error
	if !errors.As(err, &apierr) {
		return nil
	}
	kind := normalizeOpenAICompatibleKind(apierr.StatusCode, apierr.Type, apierr.Code, apierr.Message)
	return newAPIError(provider, kind, apierr.StatusCode, apierr.Message, apierr.RawJSON(), err)
}

func mapResponsesRequestError(err error) error {
	var apierr *responses.Error
	if !errors.As(err, &apierr) {
		return nil
	}
	kind := normalizeOpenAICompatibleKind(apierr.StatusCode, apierr.Type, apierr.Code, apierr.Message)
	return newAPIError(ProviderResponses, kind, apierr.StatusCode, apierr.Message, apierr.RawJSON(), err)
}

func mapAnthropicError(err error) error {
	var apierr *a.Error
	if !errors.As(err, &apierr) {
		return nil
	}
	message, typ, _, _ := extractGenericErrorFields(apierr.RawJSON())
	kind := normalizeAnthropicKind(apierr.StatusCode, typ, message)
	return newAPIError(ProviderAnthropic, kind, apierr.StatusCode, message, apierr.RawJSON(), err)
}

func mapGeminiError(err error) error {
	var apierr *genai.APIError
	if errors.As(err, &apierr) && apierr != nil {
		kind := normalizeGeminiKind(apierr.Code, apierr.Status, apierr.Message, apierr.Details)
		return newAPIError(ProviderGemini, kind, apierr.Code, apierr.Message, "", err)
	}

	var apierrValue genai.APIError
	if errors.As(err, &apierrValue) {
		kind := normalizeGeminiKind(apierrValue.Code, apierrValue.Status, apierrValue.Message, apierrValue.Details)
		return newAPIError(ProviderGemini, kind, apierrValue.Code, apierrValue.Message, "", err)
	}

	return nil
}

func newHTTPAPIError(provider Provider, statusCode int, rawBody string) error {
	message, typ, code, status := extractGenericErrorFields(rawBody)
	kind := normalizeOpenAICompatibleKind(statusCode, typ, code, status+" "+message)
	return newAPIError(provider, kind, statusCode, message, rawBody, nil)
}

func newResponsesStreamAPIError(code, message, rawBody string) error {
	kind := normalizeResponsesEventKind(code, message)
	return newAPIError(ProviderResponses, kind, 0, message, rawBody, nil)
}

func newZAIHeuristicAPIError(err error) error {
	if err == nil {
		return nil
	}
	errText := err.Error()
	switch {
	case containsAnyFold(errText, "401", "authentication", "unauthorized", "api key"):
		return newAPIError(ProviderZAI, APIErrorKindAuthentication, 401, errText, "", err)
	case containsAnyFold(errText, "403", "permission", "forbidden"):
		return newAPIError(ProviderZAI, APIErrorKindPermission, 403, errText, "", err)
	case containsAnyFold(errText, "429", "rate limit", "quota exceeded", "1113"):
		return newAPIError(ProviderZAI, APIErrorKindRateLimit, 429, errText, "", err)
	case containsAnyFold(errText, "503", "service unavailable", "unavailable"):
		return newAPIError(ProviderZAI, APIErrorKindServiceUnavailable, 503, errText, "", err)
	case containsAnyFold(errText, "500", "502", "504", "internal server", "server error"):
		return newAPIError(ProviderZAI, APIErrorKindServer, 500, errText, "", err)
	default:
		return nil
	}
}
