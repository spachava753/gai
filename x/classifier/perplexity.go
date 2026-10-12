package classifier

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	wire "github.com/spachava753/gai/internal/perplexity"
	systemone "github.com/spachava753/gai/internal/typesafe"
)

// Perplexity is a client for the Perplexity Decisions API.
// Use [NewPerplexity] to create one; the zero value cannot make requests.
type Perplexity struct {
	client *wire.Client
}

// NewPerplexity returns a client for the Perplexity Decisions API.
// If client is nil, it uses [http.DefaultClient]. It does not modify client.
//
// If baseURL is empty, NewPerplexity uses https://api.perplexity.ai/v1. A custom
// URL must include the API prefix, such as /v1, but not /decisions. It must be an
// absolute HTTP or HTTPS URL without user information, a query, or a fragment.
// An invalid URL returns gai.InvalidParameterErr.
//
// apiKey is the API key to send with requests. An empty or whitespace-only key
// returns gai.ErrMissingAPIKey. NewPerplexity does not read environment variables
// or make a network request. Set [Request.Model] when calling Classify.
func NewPerplexity(client *http.Client, baseURL, apiKey string) (*Perplexity, error) {
	config, err := newClientConfig(client, baseURL, "https://api.perplexity.ai/v1", apiKey)
	if err != nil {
		return nil, err
	}
	generated, err := wire.NewClient(config.baseURL, wire.WithHTTPClient(config.client), wire.WithRequestEditorFn(config.authorize))
	if err != nil {
		return nil, err
	}
	return &Perplexity{client: generated}, nil
}

// Classify sends request to POST /decisions and returns the answers in question
// order. See [Classifier.Classify] for how cancellation, errors, and request
// data are handled.
//
// A request may contain up to 128 questions. Choices require string values and
// allow at most 255 options; scores allow at most 10 levels. Instructions may be
// nil if the options, levels, or predicate criteria define the question.
// Instructions and descriptions are sent as JSON values.
//
// Media may contain PNG, JPEG, or WebP images. Images are sent as data URLs in
// an array with Data. Other formats are rejected. The server checks image sizes
// and model support; this client does not resize images.
//
// SafetyIdentifier is ignored. Confidence is copied from the
// response when present. Reported output-token counts are kept even if the
// provider does not charge for output.
func (c *Perplexity) Classify(ctx context.Context, request Request) (Response, error) {
	if c == nil || c.client == nil {
		return Response{}, fmt.Errorf("perplexity: uninitialized classifier")
	}
	prepared, err := prepareRequest(request)
	if err != nil {
		return Response{}, err
	}
	if len(prepared.questions) > 128 {
		return Response{}, invalid("Questions", "Perplexity accepts at most 128 questions")
	}
	if err := systemOneLimits(prepared.questions, false); err != nil {
		return Response{}, err
	}
	questions, err := systemOneQuestions(prepared.questions)
	if err != nil {
		return Response{}, err
	}
	state := emptyState(prepared.data)
	if len(request.Input.Media) > 0 {
		parts := []any{}
		if prepared.data != nil {
			parts = append(parts, prepared.data)
		}
		for _, block := range request.Input.Media {
			url, err := imageURL(block, "image/png", "image/jpeg", "image/webp")
			if err != nil {
				return Response{}, err
			}
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]string{"url": url}})
		}
		state, err = json.Marshal(parts)
		if err != nil {
			return Response{}, err
		}
	}
	body := wire.ClassificationRequest{Model: request.Model, State: state, Questions: questions}
	httpResponse, err := c.client.Classify(ctx, body, requestHeaders(request.Headers))
	data, response, err := readHTTPResponse(ProviderPerplexity, httpResponse, err)
	if err != nil {
		return response, err
	}
	return decodePerplexity(data, response, prepared)
}

func decodePerplexity(data json.RawMessage, response Response, request preparedRequest) (Response, error) {
	if err := checkJSON(data); err != nil {
		return response, responseError("", err.Error())
	}
	var metadata wire.ResponseMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return response, responseError("", err.Error())
	}
	setResponseMetadata(&response, metadata.Model, metadata.AdditionalProperties)
	if usage, err := metadata.Usage.Get(); err == nil {
		if err := readUsage(&response, usage, usage.InputTokens, usage.OutputTokens); err != nil {
			return response, responseError("", err.Error())
		}
	}
	var native wire.ClassificationResponse
	if err := json.Unmarshal(data, &native); err != nil {
		return response, responseError("", err.Error())
	}
	answers := make(map[string]systemone.Answer, len(native.Answers))
	for name, answer := range native.Answers {
		answers[name] = systemone.Answer(answer)
	}
	return convertSystemOne(answers, response, request)
}

var _ Classifier = (*Perplexity)(nil)
