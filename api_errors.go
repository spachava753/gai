package gai

import (
	"encoding/json"
	"strings"
)

type apiErrorMessage struct {
	Message string           `json:"message"`
	Error   *apiErrorMessage `json:"error"`
}

func classifyHTTPStatus(statusCode int) APIErrorKind {
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
	case 408, 504:
		return APIErrorKindTimeout
	case 413:
		return APIErrorKindRequestTooLarge
	case 429:
		return APIErrorKindRateLimit
	case 500, 502:
		return APIErrorKindServer
	case 503:
		return APIErrorKindServiceUnavailable
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

func parseAPIErrorMessage(rawBody string) string {
	rawBody = strings.TrimSpace(rawBody)
	if rawBody == "" {
		return ""
	}

	var payload apiErrorMessage
	if err := json.Unmarshal([]byte(rawBody), &payload); err != nil {
		return rawBody
	}
	for payload.Error != nil {
		payload = *payload.Error
	}
	if message := strings.TrimSpace(payload.Message); message != "" {
		return message
	}
	return rawBody
}

func mapHTTPAPIError(provider Provider, statusCode int, rawBody string) *ApiErr {
	return &ApiErr{
		Provider:   provider,
		Kind:       classifyHTTPStatus(statusCode),
		StatusCode: statusCode,
		Message:    parseAPIErrorMessage(rawBody),
		RawBody:    rawBody,
	}
}
