package classifier

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"

	"github.com/spachava753/gai"
	wire "github.com/spachava753/gai/internal/cloudflare"
)

var clairicFieldName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]{0,63}$`)

func isClairicKind(kind string) bool {
	return kind == "extraction" || kind == "estimate" || kind == "forecast"
}

// prepareClairic checks text/media and field-name restrictions, then translates
// each typed question into the generated request, preserving server defaults.
func prepareClairic(body *wire.ClassificationRequest, request preparedRequest, media []gai.Block) error {
	if err := requireQuestionKinds(request.questions, "extraction", "estimate", "forecast"); err != nil {
		return err
	}
	var state string
	if err := json.Unmarshal(body.State, &state); err != nil {
		return invalid("Input.Data", "Clairic requires text, not structured JSON")
	}
	if state == "" && len(media) == 0 {
		return invalid("Input.Data", "Clairic requires nonempty text or images")
	}
	for _, block := range media {
		if block.ModalityType != gai.Image {
			return gai.UnsupportedInputModalityErr(block.MimeType)
		}
	}
	fields := make(map[string]wire.ClairicField, len(request.questions))
	for _, q := range request.questions {
		if !clairicFieldName.MatchString(q.name) {
			return invalid(q.name, "invalid Clairic field name")
		}
		var source any
		switch q.kind {
		case "extraction":
			source = q.extraction
		case "estimate":
			source = q.estimate
		case "forecast":
			source = q.forecast
		}
		encoded, err := json.Marshal(source)
		if err != nil {
			return err
		}
		var field wire.ClairicField
		if err := json.Unmarshal(encoded, &field); err != nil {
			return err
		}
		if q.kind != "extraction" {
			field.Type = q.kind
		}
		if q.instructions != nil {
			var instructions string
			if err := json.Unmarshal(q.instructions, &instructions); err != nil {
				return invalid(q.name+".Instructions", "Clairic requires a string")
			}
			field.Instructions = &instructions
		}
		fields[q.name] = field
	}
	body.Fields = &fields
	return nil
}

// decodeClairic runs after Cloudflare envelope and metadata checks. Answers are
// decoded with generated models and converted without semantic validation.
func decodeClairic(data json.RawMessage, response Response, request preparedRequest) (Response, error) {
	var envelope wire.ClairicEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		return response, responseError("", err.Error())
	}
	native, err := envelope.Result.Get()
	if err != nil {
		return response, responseError("", "missing Clairic result")
	}
	if len(native.Answers) != len(request.questions) {
		return response, responseError("", "answers must contain exactly the requested names")
	}
	answers := make([]Answer, 0, len(request.questions))
	for _, q := range request.questions {
		n, found := native.Answers[q.name]
		if !found {
			return response, responseError(q.name, "missing answer")
		}
		a, err := convertClairicAnswer(n, q)
		if err != nil {
			return response, responseError(q.name, err.Error())
		}
		answers = append(answers, a)
	}
	response.Answers = answers
	return response, nil
}

// convertClairicAnswer checks the returned tag and explicit value, then converts
// extraction, estimate, or forecast fields without conflating null with refusal.
func convertClairicAnswer(n wire.ClairicAnswer, q preparedQuestion) (Answer, error) {
	a := Answer{Name: q.name, Confidence: optionalConfidence(n.Confidence), ExtraFields: nativeFields(n.AdditionalProperties)}
	kind := q.kind
	if q.extraction != nil {
		kind = q.extraction.Type
	}
	if n.Type == nil || *n.Type != kind {
		return a, fmt.Errorf("answer type does not match question")
	}
	if len(n.Value) == 0 {
		return a, fmt.Errorf("missing value (explicit null is allowed)")
	}
	value, err := clairicValue(n.Value)
	if err != nil {
		return a, err
	}
	switch q.kind {
	case "extraction":
		guess, err := clairicValue(optionalRaw(n.Guess))
		if err != nil {
			return a, err
		}
		a.Extraction = &ExtractionAnswer{Value: value, Guess: guess, NullProbability: optionalConfidence(n.PNull)}
		if n.Span != nil {
			a.Extraction.Span = *n.Span
		}
		if n.Spans != nil {
			a.Extraction.Spans = *n.Spans
		}
	case "estimate":
		a.Estimate, err = convertClairicEstimate(n)
	case "forecast":
		if n.Steps == nil {
			return a, fmt.Errorf("missing forecast steps")
		}
		a.Forecast = &ForecastAnswer{Value: value, NullProbability: optionalConfidence(n.PNull)}
		for _, step := range *n.Steps {
			estimate, stepErr := convertClairicEstimate(step)
			if stepErr != nil {
				return a, stepErr
			}
			a.Forecast.Steps = append(a.Forecast.Steps, *estimate)
		}
	}
	return a, err
}

func convertClairicEstimate(n wire.ClairicAnswer) (*EstimateAnswer, error) {
	if len(n.Value) == 0 {
		return nil, fmt.Errorf("missing estimate value")
	}
	a := &EstimateAnswer{
		Confidence:       optionalConfidence(n.Confidence),
		BelowProbability: optionalConfidence(n.PBelow), AboveProbability: optionalConfidence(n.PAbove),
		NullProbability: optionalConfidence(n.PNull), ExtraFields: nativeFields(n.AdditionalProperties),
	}
	for _, field := range []struct {
		raw    json.RawMessage
		target *any
	}{
		{n.Value, &a.Value}, {optionalRaw(n.Mean), &a.Mean}, {optionalRaw(n.Median), &a.Median},
		{optionalRaw(n.P10), &a.P10}, {optionalRaw(n.P90), &a.P90},
	} {
		value, err := clairicValue(field.raw)
		if err != nil {
			return nil, err
		}
		*field.target = value
	}
	if n.Quantiles != nil {
		a.Quantiles = make([]any, 0, len(*n.Quantiles))
		for _, raw := range *n.Quantiles {
			value, err := clairicValue(raw)
			if err != nil {
				return nil, err
			}
			a.Quantiles = append(a.Quantiles, value)
		}
	}
	return a, nil
}

func clairicValue(data json.RawMessage) (any, error) {
	if len(data) == 0 {
		return nil, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	return value, nil
}
