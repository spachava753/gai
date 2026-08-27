package gai

import (
	"encoding/json"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type apiErrorMessage struct {
	Message string           `json:"message"`
	Error   *apiErrorMessage `json:"error"`
}

// classifyHTTPStatus groups standard HTTP failures into retry and fallback categories shared by provider adapters.
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

func parseRetryAfterNumber(value string, unit time.Duration) (time.Duration, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}

	amount, err := strconv.ParseFloat(value, 64)
	if err != nil || math.IsNaN(amount) || math.IsInf(amount, 0) {
		return 0, false
	}
	if amount < 0 {
		return 0, false
	}
	if amount == 0 {
		return 0, true
	}

	nanoseconds := amount * float64(unit)
	if math.IsInf(nanoseconds, 0) || nanoseconds >= float64(math.MaxInt64) {
		return 0, false
	}
	return time.Duration(nanoseconds), true
}

func parseRetryAfter(value string, now time.Time) (time.Duration, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}

	if delay, ok := parseRetryAfterNumber(value, time.Second); ok {
		return delay, true
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

	if delay, ok := parseRetryAfterNumber(response.Header.Get("Retry-After-Ms"), time.Millisecond); ok {
		return &delay
	}

	now := time.Now()
	if serverDate, err := http.ParseTime(response.Header.Get("Date")); err == nil {
		now = serverDate
	}
	if delay, ok := parseRetryAfter(response.Header.Get("Retry-After"), now); ok {
		return &delay
	}
	return nil
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
