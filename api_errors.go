package gai

import (
	"encoding/json"
	"errors"
	"strings"
)

type apiErrorPayload struct {
	Code    json.RawMessage `json:"code"`
	Type    string          `json:"type"`
	Status  string          `json:"status"`
	Message string          `json:"message"`
}

func newAPIError(provider Provider, statusCode int, message, rawBody string, cause error, clues ...string) *ApiErr {
	message = strings.TrimSpace(message)
	rawBody = strings.TrimSpace(rawBody)

	if message == "" {
		message = rawBody
	}
	clues = append(clues, message)

	if cause == nil {
		switch {
		case message != "":
			cause = errors.New(message)
		case rawBody != "":
			cause = errors.New(rawBody)
		}
	}

	return &ApiErr{
		Provider:   provider,
		Kind:       classifyAPIError(statusCode, clues...),
		StatusCode: statusCode,
		Message:    message,
		RawBody:    rawBody,
		Cause:      cause,
	}
}

func classifyAPIError(statusCode int, clues ...string) APIErrorKind {
	text := strings.ToLower(strings.Join(clues, " "))
	switch {
	case containsAny(text, "authentication_error", "invalid_api_key", "incorrect_api_key", "api_key_invalid", "api key", "unauthenticated", "unauthorized"):
		return APIErrorKindAuthentication
	case containsAny(text, "permission_error", "permission denied", "permission_denied", "forbidden", "insufficient_permissions"):
		return APIErrorKindPermission
	case containsAny(text, "rate_limit", "rate limit", "too many requests", "quota exceeded", "resource_exhausted"):
		return APIErrorKindRateLimit
	case containsAny(text, "not_found", "not found"):
		return APIErrorKindNotFound
	case containsAny(text, "request_too_large", "request too large", "payload too large"):
		return APIErrorKindRequestTooLarge
	case containsAny(text, "service_unavailable", "service unavailable", "unavailable"):
		return APIErrorKindServiceUnavailable
	case containsAny(text, "timeout", "deadline exceeded", "deadline_exceeded"):
		return APIErrorKindTimeout
	case containsAny(text, "overloaded"):
		return APIErrorKindOverloaded
	case containsAny(text, "content policy", "content_policy"):
		return APIErrorKindContentPolicy
	case containsAny(text, "server_error", "api_error", "internal_error", "internal server", "internal"):
		return APIErrorKindServer
	case containsAny(text,
		"invalid_request", "invalid_argument", "billing_error", "invalid_prompt", "invalid_image",
		"image_too_large", "image_too_small", "image_parse_error", "image_file_too_large",
		"unsupported_image_media_type", "empty_image_file", "failed_to_download_image", "image_file_not_found"):
		return APIErrorKindInvalidRequest
	}

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

func containsAny(text string, values ...string) bool {
	for _, value := range values {
		if strings.Contains(text, value) {
			return true
		}
	}
	return false
}

func parseAPIErrorResponse(rawBody string) apiErrorPayload {
	rawBody = strings.TrimSpace(rawBody)
	if rawBody == "" {
		return apiErrorPayload{}
	}

	var response struct {
		Error apiErrorPayload `json:"error"`
	}
	if err := json.Unmarshal([]byte(rawBody), &response); err != nil {
		return apiErrorPayload{Message: rawBody}
	}

	payload := response.Error
	payload.Type = strings.TrimSpace(payload.Type)
	payload.Status = strings.TrimSpace(payload.Status)
	payload.Message = strings.TrimSpace(payload.Message)
	payload.Code = json.RawMessage(strings.TrimSpace(string(payload.Code)))
	if payload.Message == "" {
		payload.Message = rawBody
	}
	return payload
}

func newHTTPAPIError(provider Provider, statusCode int, rawBody string) *ApiErr {
	payload := parseAPIErrorResponse(rawBody)
	return newAPIError(provider, statusCode, payload.Message, rawBody, nil, payload.Type, string(payload.Code), payload.Status)
}
