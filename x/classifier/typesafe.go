package classifier

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	wire "github.com/spachava753/gai/internal/typesafe"
)

// TypeSafe is a client for TypeSafe's System One API, used by Jev.
// Use [NewTypeSafe] to create one; the zero value cannot make requests.
type TypeSafe struct {
	client *wire.Client
}

// NewTypeSafe returns a client for TypeSafe's System One API.
// If client is nil, it uses [http.DefaultClient]. It does not modify client.
//
// If baseURL is empty, NewTypeSafe uses https://api.typesafe.ai/v1. A custom URL
// must include the API prefix, such as /v1, but not /systemone. It must be an
// absolute HTTP or HTTPS URL without user information, a query, or a fragment.
// An invalid URL returns [gai.InvalidParameterErr].
//
// apiKey is the API key to send with requests. An empty or whitespace-only key
// returns [gai.ErrMissingAPIKey]. NewTypeSafe does not read environment variables
// or make a network request. Set [Request.Model] when calling Classify.
func NewTypeSafe(client *http.Client, baseURL, apiKey string) (*TypeSafe, error) {
	config, err := newClientConfig(client, baseURL, "https://api.typesafe.ai/v1", apiKey)
	if err != nil {
		return nil, err
	}
	generated, err := wire.NewClient(config.baseURL, wire.WithHTTPClient(config.client), wire.WithRequestEditorFn(config.authorize))
	if err != nil {
		return nil, err
	}
	return &TypeSafe{client: generated}, nil
}

// Classify sends request to POST /systemone and returns the answers in question
// order. See [Classifier.Classify] for how cancellation, errors, and request
// data are handled.
//
// Data, Instructions, and descriptions are sent as JSON values. Instructions
// may be omitted when criteria define the question. Box requires instructions.
// Choices require string values and allow at most 255 options; scores allow
// at most 10 levels.
// PNG and JPEG images and Box questions are experimental. Image blocks with IDs
// use those IDs as references in Data, instructions, or criteria; each must be
// referenced. Unnamed images are placed before Data. The adapter sends base64
// JSON assets and TypeSafe-Danger-Zone-Images: true when images or boxes are used.
// Request.Headers can override this header. The server validates model support
// and image dimensions/size; the adapter never resizes or uploads files.
//
// SafetyIdentifier is ignored. Confidence is copied from the
// response when present. Token counts that are not reported remain absent.
func (c *TypeSafe) Classify(ctx context.Context, request Request) (Response, error) {
	if c == nil || c.client == nil {
		return Response{}, fmt.Errorf("typesafe: uninitialized classifier")
	}
	prepared, err := prepareRequest(request)
	if err != nil {
		return Response{}, err
	}
	if err := requireQuestionKinds(prepared.questions, "predicate", "choice", "score", "box"); err != nil {
		return Response{}, err
	}
	if err := systemOneLimits(prepared.questions, false); err != nil {
		return Response{}, err
	}
	questions, err := systemOneQuestions(prepared.questions)
	if err != nil {
		return Response{}, err
	}
	body := wire.ClassificationRequest{Model: request.Model, State: emptyState(prepared.data), Questions: questions}
	if err := typeSafeImages(&body, request.Input.Media); err != nil {
		return Response{}, err
	}
	images := len(request.Input.Media) > 0
	for _, q := range prepared.questions {
		if q.kind == "box" && !images {
			return Response{}, invalid(q.name, "box requires an image")
		}
	}
	httpResponse, err := c.client.Classify(ctx, body, func(_ context.Context, r *http.Request) error {
		if images {
			r.Header.Set("TypeSafe-Danger-Zone-Images", "true")
		}
		return nil
	}, requestHeaders(request.Headers))
	data, response, err := readHTTPResponse(ProviderTypeSafe, httpResponse, err)
	if err != nil {
		return response, err
	}
	return decodeTypeSafe(data, response, prepared)
}

func decodeTypeSafe(data json.RawMessage, response Response, request preparedRequest) (Response, error) {
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
	return convertSystemOne(native.Answers, response, request)
}

var _ Classifier = (*TypeSafe)(nil)
