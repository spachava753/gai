package gai

import (
	"errors"

	oai "github.com/openai/openai-go/v3"
)

// mapOpenAISDKError maps the HTTP errors shared by OpenAI SDK services.
func mapOpenAISDKError(provider Provider, err error) *ApiErr {
	var apiErr *oai.Error
	if !errors.As(err, &apiErr) {
		return nil
	}
	return &ApiErr{
		Provider:   provider,
		Kind:       classifyHTTPStatus(apiErr.StatusCode),
		StatusCode: apiErr.StatusCode,
		Message:    apiErr.Message,
		RawBody:    apiErr.RawJSON(),
		Cause:      err,
	}
}
