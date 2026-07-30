package gai

import (
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
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

func parseRetryAfter(value string, now time.Time) (time.Duration, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}

	if seconds, err := strconv.ParseUint(value, 10, 64); err == nil {
		const maxDurationSeconds = uint64((1<<63 - 1) / int64(time.Second))
		if seconds > maxDurationSeconds {
			return 0, false
		}
		return time.Duration(seconds) * time.Second, true
	}

	retryAt, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	return max(time.Duration(0), retryAt.Sub(now)), true
}

func retryAfterFromResponse(response *http.Response) *time.Duration {
	if response == nil {
		return nil
	}

	now := time.Now()
	if serverDate, err := http.ParseTime(response.Header.Get("Date")); err == nil {
		now = serverDate
	}
	delay, ok := parseRetryAfter(response.Header.Get("Retry-After"), now)
	if !ok {
		return nil
	}
	return &delay
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

// mapHTTPAPIError consumes and closes response.Body.
func mapHTTPAPIError(provider Provider, response *http.Response) *ApiErr {
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	rawBody := string(body)

	return &ApiErr{
		Provider:           provider,
		Kind:               classifyHTTPStatus(response.StatusCode),
		StatusCode:         response.StatusCode,
		Message:            parseAPIErrorMessage(rawBody),
		RawBody:            rawBody,
		RetryAfterDuration: retryAfterFromResponse(response),
	}
}
