package gai

import (
	"context"
	"net/http"

	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	openaioption "github.com/openai/openai-go/v3/option"

	"github.com/spachava753/gai/internal/httpheaders"
)

func validateRequestHeaders(headers http.Header) error {
	if err := httpheaders.Validate(headers); err != nil {
		return &InvalidParameterErr{Parameter: "headers", Reason: err.Error()}
	}
	return nil
}

// setRequestHeaders replaces named headers without borrowing caller slices.
// Ambiguous case-insensitive duplicates are rejected before changing the request.
func setRequestHeaders(request *http.Request, headers http.Header) error {
	if err := validateRequestHeaders(headers); err != nil {
		return err
	}
	for name, values := range headers {
		request.Header.Del(name)
		for _, value := range values {
			request.Header.Add(name, value)
		}
	}
	return nil
}

func requestHeaderEditor(headers http.Header) func(context.Context, *http.Request) error {
	headers = headers.Clone()
	return func(_ context.Context, request *http.Request) error {
		return setRequestHeaders(request, headers)
	}
}

type requestHeadersContextKey struct{}

func editContextRequestHeaders(ctx context.Context, request *http.Request) error {
	headers, _ := ctx.Value(requestHeadersContextKey{}).(http.Header)
	return setRequestHeaders(request, headers)
}

func anthropicRequestHeaders(headers http.Header) ([]anthropicoption.RequestOption, error) {
	if err := validateRequestHeaders(headers); err != nil {
		return nil, err
	}
	var options []anthropicoption.RequestOption
	for name, values := range headers {
		options = append(options, anthropicoption.WithHeaderDel(name))
		for _, value := range values {
			options = append(options, anthropicoption.WithHeaderAdd(name, value))
		}
	}
	return options, nil
}

func openAIRequestHeaders(headers http.Header) ([]openaioption.RequestOption, error) {
	if err := validateRequestHeaders(headers); err != nil {
		return nil, err
	}
	var options []openaioption.RequestOption
	for name, values := range headers {
		options = append(options, openaioption.WithHeaderDel(name))
		for _, value := range values {
			options = append(options, openaioption.WithHeaderAdd(name, value))
		}
	}
	return options, nil
}
