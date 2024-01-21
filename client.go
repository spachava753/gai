package gai

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/sashabaranov/go-openai"
)

type TogetherRateLimiter struct {
	wait time.Duration
}

func (t TogetherRateLimiter) SetDurationFromHeaders(header http.Header) error {
	remaining, remainingErr := strconv.ParseInt(header.Get("x-ratelimit-remaining"), 10, 64)
	if remainingErr != nil {
		return remainingErr
	}

	if remaining > 0 {
		t.wait = 0
		return nil
	}

	waitDuration, waitDurationErr := strconv.ParseInt(header.Get("x-ratelimit-reset"), 10, 64)
	if waitDurationErr != nil {
		return waitDurationErr
	}
	t.wait = time.Duration(waitDuration) * time.Second
	return nil
}

func (t TogetherRateLimiter) NextBackOff() time.Duration {
	return t.wait
}

func (t TogetherRateLimiter) Reset() {
	t.wait = 0
}

type RateLimitBackOff interface {
	SetDurationFromHeaders(header http.Header) error
}

type Client struct {
	OaiClient *openai.Client
}

func NewClient(endpoint, apiKey string) Client {
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = endpoint
	client := openai.NewClientWithConfig(config)
	return Client{
		OaiClient: client,
	}
}

func (c Client) ChatCompletion(
	ctx context.Context, request openai.ChatCompletionRequest, retry backoff.BackOff,
) (openai.ChatCompletionResponse, error) {

	var resp openai.ChatCompletionResponse
	operation := func() error {
		var respErr error
		resp, respErr = c.OaiClient.CreateChatCompletion(ctx, request)

		if respErr != nil {
			var statusCode int
			if reqErr := new(openai.RequestError); errors.As(respErr, &reqErr) {
				if reqErr.HTTPStatusCode == http.StatusUnauthorized || reqErr.HTTPStatusCode == http.StatusBadRequest {
					return backoff.Permanent(reqErr)
				}
				statusCode = reqErr.HTTPStatusCode
			}
			if apiErr := new(openai.APIError); errors.As(respErr, &apiErr) {
				if apiErr.HTTPStatusCode == http.StatusUnauthorized || apiErr.HTTPStatusCode == http.StatusBadRequest {
					return backoff.Permanent(apiErr)
				}
				statusCode = apiErr.HTTPStatusCode
			}
			rateLimitBackoff, ok := retry.(RateLimitBackOff)
			if statusCode == http.StatusTooManyRequests && ok {
				if err := rateLimitBackoff.SetDurationFromHeaders(resp.Header()); err != nil {
					return backoff.Permanent(err)
				}
			}
		}

		return respErr
	}

	err := backoff.Retry(operation, retry)
	if err != nil {
		return resp, err
	}

	return resp, nil
}
