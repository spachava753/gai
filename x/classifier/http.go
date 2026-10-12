package classifier

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/oapi-codegen/nullable"

	"github.com/spachava753/gai"
	openaiwire "github.com/spachava753/gai/internal/openai"
	systemone "github.com/spachava753/gai/internal/typesafe"
	"github.com/spachava753/gai/internal/wirejson"
)

const (
	// ProviderOpenAI identifies OpenAI Decisions in [gai.ApiErr] and [ExtraFieldProvider].
	ProviderOpenAI gai.Provider = "openai_decisions"
	// ProviderTypeSafe identifies TypeSafe in [gai.ApiErr] and [ExtraFieldProvider].
	ProviderTypeSafe gai.Provider = "typesafe"
	// ProviderCloudflare identifies Cloudflare Workers AI in [gai.ApiErr] and [ExtraFieldProvider].
	ProviderCloudflare gai.Provider = "cloudflare"
	// ProviderPerplexity identifies Perplexity Decisions in [gai.ApiErr] and [ExtraFieldProvider].
	ProviderPerplexity gai.Provider = "perplexity_decisions"

	// ExtraFieldHeaders is the [Response.ExtraFields] key for HTTP response
	// headers. Its value is an [http.Header] copied from the HTTP response.
	ExtraFieldHeaders = "classifier.headers"
	// ExtraFieldProvider is the [Response.ExtraFields] key for the provider
	// that handled the request. Its value is a [gai.Provider]: [ProviderOpenAI],
	// [ProviderTypeSafe], [ProviderCloudflare], or [ProviderPerplexity].
	ExtraFieldProvider = "classifier.provider"
	// ExtraFieldUsage is the [Response.ExtraFields] key for the provider's
	// original usage object. Its value is a map[string]json.RawMessage.
	// For common token counts, use [Response.UsageMetadata] instead.
	ExtraFieldUsage = "classifier.usage"
	// ExtraFieldWireFields is the key for additional fields from the provider's
	// JSON response. It appears in [Response.ExtraFields] for response fields
	// and [Answer.ExtraFields] for answer fields, such as a score's legend.
	// Its value is a map[string]json.RawMessage. When decoding into any, use
	// [json.Decoder.UseNumber] to avoid rounding numbers to float64.
	ExtraFieldWireFields = "classifier.wire_fields"
	// ExtraFieldEnvelope is the [Response.ExtraFields] key for Cloudflare's
	// response wrapper fields, such as success and errors. Its value is a
	// map[string]json.RawMessage. The result field is excluded because it is
	// decoded into the Response itself.
	ExtraFieldEnvelope = "classifier.envelope"
)

type clientConfig struct {
	client  *http.Client
	baseURL string
	apiKey  string
}

// newClientConfig validates constructor arguments for generated clients without
// reading global configuration or modifying the supplied HTTP client.
func newClientConfig(client *http.Client, baseURL, defaultURL, apiKey string) (*clientConfig, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, gai.ErrMissingAPIKey
	}
	if baseURL == "" {
		baseURL = defaultURL
	}
	u, err := url.Parse(baseURL)
	if err != nil || u == nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.ForceQuery || strings.Contains(baseURL, "#") {
		return nil, invalid("baseURL", "requires an absolute HTTP(S) URL without user info, query, or fragment")
	}
	if client == nil {
		client = http.DefaultClient
	}
	return &clientConfig{client: client, baseURL: strings.TrimRight(baseURL, "/"), apiKey: apiKey}, nil
}

func (c *clientConfig) authorize(_ context.Context, request *http.Request) error {
	request.Header.Set("Authorization", "Bearer "+c.apiKey)
	return nil
}

func requestHeaders(headers http.Header) func(context.Context, *http.Request) error {
	return func(_ context.Context, request *http.Request) error {
		for name, values := range headers {
			request.Header.Del(name)
			for _, value := range values {
				request.Header.Add(name, value)
			}
		}
		return nil
	}
}

// readHTTPResponse handles a generated client's raw response. Request creation,
// path encoding, serialization, and execution belong to the generated client.
// This helper only bounds and closes the response body and translates API errors.
func readHTTPResponse(provider gai.Provider, httpResponse *http.Response, err error) (json.RawMessage, Response, error) {
	response := Response{ExtraFields: map[string]any{ExtraFieldProvider: provider}}
	if err != nil {
		return nil, response, fmt.Errorf("%s: request: %w", provider, err)
	}
	defer httpResponse.Body.Close()
	response.ExtraFields[ExtraFieldHeaders] = httpResponse.Header.Clone()
	const maximumBody = 32 << 20
	data, err := io.ReadAll(io.LimitReader(httpResponse.Body, maximumBody+1))
	if err != nil {
		return nil, response, fmt.Errorf("%s: read response: %w", provider, err)
	}
	if len(data) > maximumBody {
		return nil, response, responseError("", "response body exceeds 32 MiB")
	}
	if httpResponse.StatusCode < 200 || httpResponse.StatusCode >= 300 {
		return nil, response, &gai.ApiErr{
			Provider: provider, Kind: httpErrorKind(httpResponse.StatusCode),
			StatusCode: httpResponse.StatusCode, RawBody: string(data),
			Message: apiMessage(provider, data), RetryAfterDuration: retryAfter(httpResponse.Header),
		}
	}
	return data, response, nil
}

// httpErrorKind classifies transport status independently of provider payloads.
func httpErrorKind(code int) gai.APIErrorKind {
	switch code {
	case 401:
		return gai.APIErrorKindAuthentication
	case 403:
		return gai.APIErrorKindPermission
	case 404:
		return gai.APIErrorKindNotFound
	case 408, 504:
		return gai.APIErrorKindTimeout
	case 413:
		return gai.APIErrorKindRequestTooLarge
	case 429:
		return gai.APIErrorKindRateLimit
	case 503:
		return gai.APIErrorKindServiceUnavailable
	case 529:
		return gai.APIErrorKindOverloaded
	default:
		if code >= 500 {
			return gai.APIErrorKindServer
		}
		return gai.APIErrorKindInvalidRequest
	}
}

// apiMessage reads generated error models. The System One services share the
// message/error/errors layout. Non-JSON and unrecognized bodies remain intact.
func apiMessage(provider gai.Provider, data []byte) string {
	if provider == ProviderOpenAI {
		var body openaiwire.ErrorResponse
		if json.Unmarshal(data, &body) == nil {
			if value, err := body.Error.Get(); err == nil {
				if detail, err := value.AsErrorDetail(); err == nil && detail.Message.GetOrEmpty() != "" {
					return detail.Message.GetOrEmpty()
				}
			}
			if body.Message.GetOrEmpty() != "" {
				return body.Message.GetOrEmpty()
			}
		}
	} else {
		var body systemone.ErrorResponse
		if json.Unmarshal(data, &body) == nil {
			if body.Error != nil && body.Error.Message != nil && *body.Error.Message != "" {
				return *body.Error.Message
			}
			if body.Errors != nil && len(*body.Errors) > 0 && (*body.Errors)[0].Message != nil && *(*body.Errors)[0].Message != "" {
				return *(*body.Errors)[0].Message
			}
			if body.Message != nil && *body.Message != "" {
				return *body.Message
			}
		}
	}
	return strings.TrimSpace(string(data))
}

func retryAfter(headers http.Header) *time.Duration {
	for _, field := range []struct {
		name string
		unit string
	}{{"Retry-After-Ms", "ms"}, {"Retry-After", "s"}} {
		value := strings.TrimSpace(headers.Get(field.name))
		if value == "" || strings.HasPrefix(value, "-") {
			continue
		}
		if delay, err := time.ParseDuration(value + field.unit); err == nil && delay >= 0 {
			return &delay
		}
	}
	if target, err := http.ParseTime(headers.Get("Retry-After")); err == nil {
		now := time.Now()
		if date, err := http.ParseTime(headers.Get("Date")); err == nil {
			now = date
		}
		delay := max(time.Duration(0), target.Sub(now))
		return &delay
	}
	return nil
}

func object(data json.RawMessage) (map[string]json.RawMessage, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || data[0] != '{' {
		return nil, fmt.Errorf("expected JSON object")
	}
	var result map[string]json.RawMessage
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// checkJSON rejects duplicate names at every nesting level before maps can lose
// them, rejects trailing values, and bounds nesting to encoding/json's usual
// 10000-level limit. Numeric tokens remain json.Number.
func checkJSON(data json.RawMessage) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := checkJSONValue(decoder, 0); err != nil {
		return err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return fmt.Errorf("unexpected data after JSON value")
	}
	return nil
}

func checkJSONValue(decoder *json.Decoder, depth int) error {
	if depth > 10000 {
		return fmt.Errorf("JSON nesting exceeds 10000 levels")
	}
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	seen := map[string]bool{}
	for decoder.More() {
		if delimiter == '{' {
			key, err := decoder.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return fmt.Errorf("invalid or duplicate JSON object key")
			}
			seen[name] = true
		}
		if err := checkJSONValue(decoder, depth+1); err != nil {
			return err
		}
	}
	_, err = decoder.Token()
	return err
}

func requiredNumber(value nullable.Nullable[wirejson.Number]) (json.Number, error) {
	number, err := value.Get()
	if err != nil {
		return "", fmt.Errorf("missing or null number")
	}
	return json.Number(number), nil
}

func optionalConfidence(value nullable.Nullable[wirejson.Number]) *json.Number {
	number, err := value.Get()
	if err != nil {
		return nil
	}
	result := json.Number(number)
	return &result
}

func nativeFields(fields map[string]json.RawMessage) map[string]any {
	if len(fields) == 0 {
		return nil
	}
	return map[string]any{ExtraFieldWireFields: fields}
}

// metadataObject preserves generated models as the documented raw metadata map.
// Known response fields are read from generated types, never from this map.
func metadataObject(value any) (map[string]json.RawMessage, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return object(data)
}

// setResponseMetadata retains the generated metadata projection before answer
// decoding. The omitted answers field belongs to the full response model, not
// to the invocation's additional fields.
func setResponseMetadata(response *Response, model string, fields map[string]json.RawMessage) {
	response.Model = model
	delete(fields, "answers")
	if len(fields) != 0 {
		response.ExtraFields[ExtraFieldWireFields] = fields
	}
}

func readUsage(response *Response, usage any, input, output nullable.Nullable[int]) error {
	fields, err := metadataObject(usage)
	if err != nil {
		return err
	}
	response.ExtraFields[ExtraFieldUsage] = fields
	response.UsageMetadata = gai.Metadata{}
	if err := tokenMetric(response.UsageMetadata, input, gai.UsageMetricInputTokens); err != nil {
		return err
	}
	return tokenMetric(response.UsageMetadata, output, gai.UsageMetricGenerationTokens)
}

func tokenMetric(metrics gai.Metadata, count nullable.Nullable[int], key string) error {
	value, err := count.Get()
	if err != nil {
		return nil
	}
	if value < 0 {
		return fmt.Errorf("invalid nonnegative integer usage field %s", key)
	}
	metrics[key] = value
	return nil
}

func dataText(data json.RawMessage) string {
	if len(data) == 0 {
		return ""
	}
	var text string
	if data[0] == '"' && json.Unmarshal(data, &text) == nil {
		return text
	}
	return string(data)
}

func imageURL(block gai.Block, allowed ...string) (string, error) {
	for _, mime := range allowed {
		if block.ModalityType == gai.Image && block.MimeType == mime {
			return "data:" + mime + ";base64," + block.Content.String(), nil
		}
	}
	return "", gai.UnsupportedInputModalityErr(block.MimeType)
}
