package classifier

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/spachava753/gai"
	wire "github.com/spachava753/gai/internal/cloudflare"
	systemone "github.com/spachava753/gai/internal/typesafe"
)

// Cloudflare is a client for Clef models hosted on Cloudflare Workers AI.
// It does not support self-hosted Clef servers. Use [NewCloudflare] to create one;
// the zero value cannot make requests.
type Cloudflare struct {
	client    *wire.Client
	accountID string
}

// NewCloudflare returns a client for Clef on Cloudflare Workers AI.
// If client is nil, it uses [http.DefaultClient]. It does not modify client.
//
// If baseURL is empty, NewCloudflare uses https://api.cloudflare.com/client/v4.
// A custom URL includes the API prefix, but not the /accounts path. It must be
// an absolute HTTP or HTTPS URL without user information, a query, or a fragment.
// accountID is the Cloudflare account ID, not an account URL or path. An invalid
// URL or account ID returns [gai.InvalidParameterErr].
//
// apiKey is the Cloudflare API token to send with requests. An empty or
// whitespace-only token returns [gai.ErrMissingAPIKey]. NewCloudflare does not
// read environment variables or make a network request. Set [Request.Model]
// to a full model name, such as "@cf/cloudflare/clef", when calling Classify.
func NewCloudflare(client *http.Client, baseURL, accountID, apiKey string) (*Cloudflare, error) {
	if !cloudflarePathComponent(accountID) {
		return nil, invalid("accountID", "requires a nonempty account path component")
	}
	config, err := newClientConfig(client, baseURL, "https://api.cloudflare.com/client/v4", apiKey)
	if err != nil {
		return nil, err
	}
	generated, err := wire.NewClient(config.baseURL, wire.WithHTTPClient(config.client), wire.WithRequestEditorFn(config.authorize))
	if err != nil {
		return nil, err
	}
	return &Cloudflare{client: generated, accountID: accountID}, nil
}

var cloudflareQuestionName = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,100}$`)

func cloudflarePathComponent(value string) bool {
	return value != "" && value != "." && value != ".." && !strings.ContainsAny(value, "/\\?#% \t\r\n")
}

// Classify sends request to the Workers AI model named by request.Model and
// returns the answers in question order. See [Classifier.Classify] for how
// cancellation, errors, and request data are handled.
//
// Model must be a full Workers AI name, such as "@cf/cloudflare/clef" or
// "@cf/cloudflare/clef-omni". The last component, such as "clef", is also sent
// in the request body's model field.
//
// A request may contain up to 64 questions. Names must be 1 to 100 ASCII letters,
// digits, dots, underscores, or hyphens. Every question requires Instructions;
// string instructions must not be blank. Choices require nonempty string values
// and allow at most 255 options; scores allow at most 10 levels. Data,
// Instructions, and descriptions are sent as JSON values.
//
// Media may contain up to four PNG, JPEG, or WebP images, four audio attachments,
// and two video attachments. Audio and video use Clef-Omni's request fields and
// require audio/* and video/* MIME types. The server checks format support,
// size, and duration limits for the selected model.
//
// The hosted API places images before Data and may truncate long text to fit
// the model's context limit. This client does not truncate the input itself.
//
// SafetyIdentifier is ignored. Confidence is copied from the
// response when present. Cloudflare's response wrapper fields, such as success
// and errors, are available under [ExtraFieldEnvelope].
func (c *Cloudflare) Classify(ctx context.Context, request Request) (Response, error) {
	if c == nil || c.client == nil {
		return Response{}, fmt.Errorf("cloudflare: uninitialized classifier")
	}
	prepared, err := prepareRequest(request)
	if err != nil {
		return Response{}, err
	}
	selector, err := cloudflareModel(request.Model)
	if err != nil {
		return Response{}, err
	}
	if err := cloudflareQuestions(prepared.questions); err != nil {
		return Response{}, err
	}
	questions, err := systemOneQuestions(prepared.questions)
	if err != nil {
		return Response{}, err
	}
	body := wire.ClassificationRequest{Model: selector, State: emptyState(prepared.data), Questions: questions}
	if err := cloudflareMedia(&body, request.Input.Media); err != nil {
		return Response{}, err
	}
	httpResponse, err := c.client.Classify(ctx, c.accountID, request.Model, body, requestHeaders(request.Headers))
	data, response, err := readHTTPResponse(ProviderCloudflare, httpResponse, err)
	if err != nil {
		return response, err
	}
	return decodeCloudflare(data, response, prepared)
}

func cloudflareModel(model string) (string, error) {
	parts := strings.Split(model, "/")
	if len(parts) < 2 || !strings.HasPrefix(parts[0], "@") {
		return "", invalid("Model", "requires a full Workers AI model route, such as @cf/cloudflare/clef")
	}
	for _, part := range parts {
		if !cloudflarePathComponent(part) {
			return "", invalid("Model", "invalid model path component")
		}
	}
	return parts[len(parts)-1], nil
}

func cloudflareQuestions(questions []preparedQuestion) error {
	if len(questions) > 64 {
		return invalid("Questions", "Cloudflare accepts at most 64 questions")
	}
	if err := systemOneLimits(questions, true); err != nil {
		return err
	}
	for _, q := range questions {
		if !cloudflareQuestionName.MatchString(q.name) {
			return invalid("Questions.Name", "Cloudflare requires 1–100 ASCII letters, digits, dots, underscores, or hyphens")
		}
		if strings.TrimSpace(dataText(q.instructions)) == "" {
			return invalid(q.name+".Instructions", "Cloudflare requires nonempty instructions")
		}
		for _, option := range q.options {
			if string(option.value) == `""` {
				return invalid(q.name+".Choice", "Cloudflare requires nonempty choice IDs")
			}
		}
	}
	return nil
}

// cloudflareMedia groups attachments into the native modality fields, preserving
// order within each field without claiming a cross-modality interleaving contract.
func cloudflareMedia(body *wire.ClassificationRequest, media []gai.Block) error {
	groups := map[string][]wire.Attachment{}
	for _, block := range media {
		field := ""
		switch {
		case block.ModalityType == gai.Image:
			if _, err := imageURL(block, "image/png", "image/jpeg", "image/webp"); err != nil {
				return err
			}
			field = "images"
		case block.ModalityType == gai.Audio && strings.HasPrefix(block.MimeType, "audio/"):
			field = "audio"
		case block.ModalityType == gai.Video && strings.HasPrefix(block.MimeType, "video/"):
			field = "videos"
		default:
			return gai.UnsupportedInputModalityErr(block.MimeType)
		}
		encoded, err := json.Marshal(map[string]string{"content_type": block.MimeType, "base64": block.Content.String()})
		if err != nil {
			return err
		}
		groups[field] = append(groups[field], encoded)
	}
	for field, parts := range groups {
		limit := 4
		if field == "videos" {
			limit = 2
		}
		if len(parts) > limit {
			return invalid("Input.Media", fmt.Sprintf("Cloudflare accepts at most %d %s attachments", limit, field))
		}
		switch field {
		case "images":
			body.Images = &parts
		case "audio":
			body.Audio = &parts
		case "videos":
			body.Videos = &parts
		}
	}
	return nil
}

// decodeCloudflare checks the generated envelope before converting its typed
// result. The metadata projection preserves usage when answer decoding fails.
func decodeCloudflare(data json.RawMessage, response Response, request preparedRequest) (Response, error) {
	if err := checkJSON(data); err != nil {
		return response, responseError("", err.Error())
	}
	var metadata wire.EnvelopeMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return response, responseError("", err.Error())
	}
	fields, err := metadataObject(metadata)
	if err != nil {
		return response, responseError("", err.Error())
	}
	delete(fields, "result")
	response.ExtraFields[ExtraFieldEnvelope] = fields
	if metadata.Success.IsNull() {
		return response, responseError("", "invalid Cloudflare success flag")
	}
	success, successErr := metadata.Success.Get()
	if (successErr == nil && !success) || len(metadata.Errors.GetOrEmpty()) != 0 {
		return response, &gai.ApiErr{Provider: ProviderCloudflare, Kind: gai.APIErrorKindUnknown, StatusCode: http.StatusOK, Message: apiMessage(ProviderCloudflare, data), RawBody: string(data)}
	}
	resultMetadata, err := metadata.Result.Get()
	if err != nil {
		return response, responseError("", "missing or null Cloudflare result")
	}
	setResponseMetadata(&response, resultMetadata.Model, resultMetadata.AdditionalProperties)
	if usage, err := resultMetadata.Usage.Get(); err == nil {
		if err := readUsage(&response, usage, usage.InputTokens, usage.OutputTokens); err != nil {
			return response, responseError("", err.Error())
		}
	}
	var envelope wire.ClassificationEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		return response, responseError("", err.Error())
	}
	native, err := envelope.Result.Get()
	if err != nil {
		return response, responseError("", "missing or null Cloudflare result")
	}
	answers := make(map[string]systemone.Answer, len(native.Answers))
	for name, answer := range native.Answers {
		answers[name] = systemone.Answer(answer)
	}
	return convertSystemOne(answers, response, request)
}

var _ Classifier = (*Cloudflare)(nil)
