package gai

import (
	"context"
	"net/http"

	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	openaioption "github.com/openai/openai-go/v3/option"
)

// setRequestHeaders replaces named headers without borrowing caller slices.
func setRequestHeaders(request *http.Request, headers http.Header) {
	for name, values := range headers {
		request.Header.Del(name)
		for _, value := range values {
			request.Header.Add(name, value)
		}
	}
}

func requestHeaderEditor(headers http.Header) func(context.Context, *http.Request) error {
	headers = headers.Clone()
	return func(_ context.Context, request *http.Request) error {
		setRequestHeaders(request, headers)
		return nil
	}
}

type requestHeadersContextKey struct{}

func editContextRequestHeaders(ctx context.Context, request *http.Request) error {
	headers, _ := ctx.Value(requestHeadersContextKey{}).(http.Header)
	setRequestHeaders(request, headers)
	return nil
}

func anthropicRequestHeaders(headers http.Header) []anthropicoption.RequestOption {
	var options []anthropicoption.RequestOption
	for name, values := range headers {
		options = append(options, anthropicoption.WithHeaderDel(name))
		for _, value := range values {
			options = append(options, anthropicoption.WithHeaderAdd(name, value))
		}
	}
	return options
}

func openAIRequestHeaders(headers http.Header) []openaioption.RequestOption {
	var options []openaioption.RequestOption
	for name, values := range headers {
		options = append(options, openaioption.WithHeaderDel(name))
		for _, value := range values {
			options = append(options, openaioption.WithHeaderAdd(name, value))
		}
	}
	return options
}
