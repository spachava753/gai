package gai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

func compatibleError(provider Provider, err error) error {
	if apiErr, ok := errors.AsType[*ApiErr](err); ok {
		copy := *apiErr
		copy.Provider = provider
		return &copy
	}
	return err
}

// compatiblePostJSON sends a single authenticated JSON request and decodes the
// response into result. Endpoint selection, payloads, and validation belong to
// the calling provider. It does not retry or interpret provider response fields.
func compatiblePostJSON(ctx context.Context, client *http.Client, endpoint, apiKey string, provider Provider, body, result any) error {
	if client == nil {
		return fmt.Errorf("%s: uninitialized generator", provider)
	}
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	response, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("%s: request: %w", provider, err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return mapHTTPAPIError(provider, response)
	}
	if err := json.NewDecoder(response.Body).Decode(result); err != nil {
		return fmt.Errorf("%s: decode response: %w", provider, err)
	}
	return nil
}
