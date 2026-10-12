package classifier

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/spachava753/gai"
	wire "github.com/spachava753/gai/internal/openai"
)

// OpenAI is a client for the OpenAI Decisions API.
// Use [NewOpenAI] to create one; the zero value cannot make requests.
type OpenAI struct {
	client *wire.Client
}

// NewOpenAI returns a client for the OpenAI Decisions API.
// If client is nil, it uses [http.DefaultClient]. It does not modify client.
//
// If baseURL is empty, NewOpenAI uses https://api.openai.com/v1. A custom URL
// must include the API prefix, such as /v1, but not /decisions. It must be an
// absolute HTTP or HTTPS URL without user information, a query, or a fragment.
// An invalid URL returns [gai.InvalidParameterErr].
//
// apiKey is the API key to send with requests. An empty or whitespace-only key
// returns [gai.ErrMissingAPIKey]. NewOpenAI does not read environment variables
// or make a network request. Set [Request.Model] when calling Classify.
func NewOpenAI(client *http.Client, baseURL, apiKey string) (*OpenAI, error) {
	config, err := newClientConfig(client, baseURL, "https://api.openai.com/v1", apiKey)
	if err != nil {
		return nil, err
	}
	generated, err := wire.NewClient(config.baseURL, wire.WithHTTPClient(config.client), wire.WithRequestEditorFn(config.authorize))
	if err != nil {
		return nil, err
	}
	return &OpenAI{client: generated}, nil
}

// Classify sends request to POST /decisions and returns the answers in question
// order. See [Classifier.Classify] for how cancellation, errors, and request
// data are handled.
//
// Every question must include Instructions. JSON objects and arrays in Data,
// Instructions, and descriptions are sent as JSON text because the API expects
// strings. Predicate criteria are appended to that question's instructions as
// a JSON object with "true" and "false" keys.
//
// Choices may use strings or booleans, with at most 255 options. Media may
// contain up to 128 PNG, JPEG, WebP, or GIF images, sent as data URLs. Other
// media formats are rejected. The server checks model-specific image limits.
//
// SafetyIdentifier is sent as safety_identifier. Confidence is copied from the
// response when present. A refusal sets [Answer.Refused] to true, rather than
// returning an error.
func (c *OpenAI) Classify(ctx context.Context, request Request) (Response, error) {
	if c == nil || c.client == nil {
		return Response{}, fmt.Errorf("openai: uninitialized classifier")
	}
	prepared, err := prepareRequest(request)
	if err != nil {
		return Response{}, err
	}
	questions, err := openAIQuestions(prepared.questions)
	if err != nil {
		return Response{}, err
	}
	input, err := openAIInput(prepared.data, request.Input.Media)
	if err != nil {
		return Response{}, err
	}
	body := wire.DecisionRequest{Model: request.Model, Input: input, Questions: questions}
	if request.SafetyIdentifier != "" {
		body.SafetyIdentifier = &request.SafetyIdentifier
	}
	httpResponse, err := c.client.CreateDecision(ctx, body, requestHeaders(request.Headers))
	data, response, err := readHTTPResponse(ProviderOpenAI, httpResponse, err)
	if err != nil {
		return response, err
	}
	return decodeOpenAI(data, response, prepared)
}

// openAIQuestions renders structured guidance into the endpoint's string fields,
// retaining typed choices and expressing predicate criteria in the instructions.
func openAIQuestions(questions []preparedQuestion) ([]json.RawMessage, error) {
	result := make([]json.RawMessage, 0, len(questions))
	for _, q := range questions {
		if q.instructions == nil {
			return nil, invalid(q.name+".Instructions", "OpenAI requires instructions")
		}
		question := map[string]any{"type": q.kind, "name": q.name, "instructions": dataText(q.instructions)}
		switch q.kind {
		case "predicate":
			if q.positive != nil || q.negative != nil {
				criteria := map[string]json.RawMessage{}
				if q.positive != nil {
					criteria["true"] = q.positive
				}
				if q.negative != nil {
					criteria["false"] = q.negative
				}
				encoded, err := json.Marshal(criteria)
				if err != nil {
					return nil, err
				}
				question["instructions"] = dataText(q.instructions) + "\n\nCriteria (true and false outcomes):\n" + string(encoded)
			}
		case "choice":
			if len(q.options) > 255 {
				return nil, invalid(q.name+".Choice", "OpenAI accepts at most 255 choices")
			}
			choices := make([]any, 0, len(q.options))
			for _, option := range q.options {
				choice := map[string]any{"value": option.value}
				if option.description != nil {
					choice["description"] = dataText(option.description)
				}
				choices = append(choices, choice)
			}
			question["choices"] = choices
		case "score":
			levels := make([]any, 0, len(q.levels))
			for _, level := range q.levels {
				item := map[string]any{"label": level.label}
				if level.description != nil {
					item["description"] = dataText(level.description)
				}
				levels = append(levels, item)
			}
			question["levels"] = levels
		}
		encoded, err := json.Marshal(question)
		if err != nil {
			return nil, err
		}
		result = append(result, encoded)
	}
	return result, nil
}

func openAIInput(data json.RawMessage, media []gai.Block) (json.RawMessage, error) {
	if len(media) == 0 {
		return json.Marshal(dataText(data))
	}
	if len(media) > 128 {
		return nil, invalid("Input.Media", "OpenAI accepts at most 128 images")
	}
	parts := []any{}
	if data != nil {
		parts = append(parts, map[string]string{"type": "input_text", "text": dataText(data)})
	}
	for _, block := range media {
		url, err := imageURL(block, "image/png", "image/jpeg", "image/webp", "image/gif")
		if err != nil {
			return nil, err
		}
		parts = append(parts, map[string]string{"type": "input_image", "image_url": url})
	}
	return json.Marshal([]any{map[string]any{"role": "user", "content": parts}})
}

// decodeOpenAI collects metadata, correlates named answers independently of wire
// order, and publishes answers only after the full response contract validates.
func decodeOpenAI(data json.RawMessage, response Response, request preparedRequest) (Response, error) {
	if err := checkJSON(data); err != nil {
		return response, responseError("", err.Error())
	}
	var metadata wire.DecisionResponseMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return response, responseError("", err.Error())
	}
	setResponseMetadata(&response, metadata.Model, metadata.AdditionalProperties)
	if usage, err := metadata.Usage.Get(); err == nil {
		if err := openAIUsage(&response, usage); err != nil {
			return response, responseError("", err.Error())
		}
	}
	var native wire.DecisionResponse
	if err := json.Unmarshal(data, &native); err != nil {
		return response, responseError("", err.Error())
	}
	if len(native.Answers) != len(request.questions) {
		return response, responseError("", "answer count does not match questions")
	}
	byName := map[string]wire.DecisionAnswer{}
	for _, answer := range native.Answers {
		if _, exists := byName[answer.Name]; exists || answer.Name == "" {
			return response, responseError("", "invalid or duplicate answer name")
		}
		byName[answer.Name] = answer
	}
	normalized := make([]Answer, 0, len(request.questions))
	for _, q := range request.questions {
		answer, err := convertOpenAIAnswer(byName[q.name], q)
		if err != nil {
			return response, responseError(q.name, err.Error())
		}
		normalized = append(normalized, answer)
	}
	response.Answers = normalized
	if err := validateResponse(response, request); err != nil {
		response.Answers = nil
		return response, err
	}
	return response, nil
}

func openAIUsage(response *Response, usage wire.DecisionUsage) error {
	if err := readUsage(response, usage, usage.InputTokens, usage.OutputTokens); err != nil {
		return err
	}
	if details, err := usage.InputTokensDetails.Get(); err == nil {
		if err := tokenMetric(response.UsageMetadata, details.CachedTokens, gai.UsageMetricCacheReadTokens); err != nil {
			return err
		}
		if err := tokenMetric(response.UsageMetadata, details.CacheWriteTokens, gai.UsageMetricCacheWriteTokens); err != nil {
			return err
		}
	}
	if details, err := usage.OutputTokensDetails.Get(); err == nil {
		return tokenMetric(response.UsageMetadata, details.ReasoningTokens, gai.UsageMetricReasoningTokens)
	}
	return nil
}

// convertOpenAIAnswer checks variant-specific presence on the generated model.
// A refusal must not contain any of the known numeric or selected-answer fields,
// including explicit nulls, which generated nullable fields keep distinct.
func convertOpenAIAnswer(native wire.DecisionAnswer, q preparedQuestion) (Answer, error) {
	answer := Answer{Name: q.name, Confidence: optionalConfidence(native.Confidence), ExtraFields: nativeFields(native.AdditionalProperties)}
	if native.Type == "refusal" {
		if native.Probability.IsSpecified() || native.Probabilities.IsSpecified() || native.Choice.IsSpecified() || native.Score.IsSpecified() {
			return answer, fmt.Errorf("refusal contains a numeric or selected answer payload")
		}
		answer.Refused = true
		return answer, nil
	}
	if native.Type != q.kind {
		return answer, fmt.Errorf("answer kind does not match question")
	}
	var err error
	switch native.Type {
	case "predicate":
		probability, err := requiredNumber(native.Probability)
		if err != nil {
			return answer, err
		}
		answer.Predicate = &PredicateAnswer{Probability: probability}
	case "choice":
		answer.Choice, err = convertOpenAIChoice(native, q)
	case "score":
		answer.Score, err = convertOpenAIScore(native, q)
	}
	return answer, err
}

// convertOpenAIChoice decodes the string/boolean outcome union and orders the
// generated probability entries by the requested choices, rejecting duplicates.
func convertOpenAIChoice(native wire.DecisionAnswer, q preparedQuestion) (*ChoiceAnswer, error) {
	answer := &ChoiceAnswer{}
	selected, err := native.Choice.Get()
	if err != nil {
		return nil, fmt.Errorf("missing or null choice")
	}
	if err := json.Unmarshal(selected, &answer.Value); err != nil {
		return nil, err
	}
	probabilities, err := native.Probabilities.Get()
	if err != nil || len(probabilities) != len(q.options) {
		return nil, fmt.Errorf("incomplete choice distribution")
	}
	byValue := map[string]ChoiceProbability{}
	for _, p := range probabilities {
		var value Value
		if err := json.Unmarshal(p.Value, &value); err != nil {
			return nil, err
		}
		key, err := value.MarshalJSON()
		if err != nil {
			return nil, err
		}
		if _, ok := byValue[string(key)]; ok {
			return nil, fmt.Errorf("duplicate choice outcome")
		}
		byValue[string(key)] = ChoiceProbability{Value: value, Probability: json.Number(p.Probability)}
	}
	for _, option := range q.options {
		p, ok := byValue[string(option.value)]
		if !ok {
			return nil, fmt.Errorf("missing declared choice outcome")
		}
		answer.Probabilities = append(answer.Probabilities, p)
	}
	return answer, nil
}

func convertOpenAIScore(native wire.DecisionAnswer, q preparedQuestion) (*ScoreAnswer, error) {
	score, err := requiredNumber(native.Score)
	if err != nil {
		return nil, err
	}
	probabilities, err := native.Probabilities.Get()
	if err != nil || len(probabilities) != len(q.levels) {
		return nil, fmt.Errorf("incomplete score distribution")
	}
	answer := &ScoreAnswer{Score: score, Probabilities: make([]json.Number, len(q.levels))}
	seen := map[int]bool{}
	for _, p := range probabilities {
		var index *int
		if err := json.Unmarshal(p.Value, &index); err != nil || index == nil || *index < 0 || *index >= len(q.levels) || seen[*index] {
			return nil, fmt.Errorf("invalid or duplicate score level")
		}
		seen[*index] = true
		answer.Probabilities[*index] = json.Number(p.Probability)
	}
	return answer, nil
}

var _ Classifier = (*OpenAI)(nil)
