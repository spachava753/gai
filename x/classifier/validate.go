package classifier

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math/big"
	"strconv"
	"strings"

	"github.com/spachava753/gai"
)

type preparedRequest struct {
	data      json.RawMessage
	questions []preparedQuestion
}

type preparedQuestion struct {
	name         string
	kind         string
	instructions json.RawMessage
	positive     json.RawMessage
	negative     json.RawMessage
	options      []preparedOption
	levels       []preparedLevel
	extraction   *Extraction
	estimate     *Estimate
	forecast     *Forecast
}

type preparedOption struct {
	value       json.RawMessage
	description json.RawMessage
}

type preparedLevel struct {
	label       string
	description json.RawMessage
}

func invalid(parameter, reason string) error {
	return gai.InvalidParameterErr{Parameter: parameter, Reason: reason}
}

func guidance(value any, path string) (json.RawMessage, error) {
	if value == nil {
		return nil, nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, invalid(path, "must be JSON-encodable: "+err.Error())
	}
	if len(data) == 0 || !strings.ContainsRune("\"{[", rune(data[0])) {
		return nil, invalid(path, "must encode as a string, object, or array")
	}
	return data, nil
}

// prepareRequest validates the request and encodes input, instructions, and
// descriptions once. Each provider checks its own additional requirements.
func prepareRequest(r Request) (preparedRequest, error) {
	var result preparedRequest
	if strings.TrimSpace(r.Model) == "" {
		return result, invalid("Model", "must not be empty")
	}
	data, err := guidance(r.Input.Data, "Input.Data")
	if err != nil {
		return result, err
	}
	if data == nil && len(r.Input.Media) == 0 {
		return result, invalid("Input", "requires data or media")
	}
	for i, block := range r.Input.Media {
		if err := validateMedia(block, fmt.Sprintf("Input.Media[%d]", i)); err != nil {
			return result, err
		}
	}
	if len(r.Questions) == 0 {
		return result, invalid("Questions", "requires at least one question")
	}
	result.data = data
	seen := make(map[string]bool, len(r.Questions))
	for i, question := range r.Questions {
		path := fmt.Sprintf("Questions[%d]", i)
		if strings.TrimSpace(question.Name) == "" || seen[question.Name] {
			return result, invalid(path+".Name", "must be nonempty and unique")
		}
		seen[question.Name] = true
		prepared, err := prepareQuestion(question, path)
		if err != nil {
			return result, err
		}
		result.questions = append(result.questions, prepared)
	}
	return result, nil
}

func validateMedia(block gai.Block, path string) error {
	if block.BlockType != gai.Content || block.Content == nil {
		return invalid(path, "requires an ordinary content block with a payload")
	}
	if block.ModalityType != gai.Image && block.ModalityType != gai.Audio && block.ModalityType != gai.Video {
		return invalid(path, "requires image, audio, or video media")
	}
	if block.MimeType == "" {
		return invalid(path, "requires a MIME type")
	}
	payload := block.Content.String()
	if len(payload) == 0 {
		return invalid(path, "requires nonempty base64 media")
	}
	if _, err := base64.StdEncoding.Strict().DecodeString(payload); err != nil {
		return invalid(path, "requires valid base64 media")
	}
	return nil
}

// prepareQuestion checks the exclusive variant and converts its guidance and
// finite outcome schema without rewriting missing instructions or criteria.
func prepareQuestion(q Question, path string) (preparedQuestion, error) {
	p := preparedQuestion{name: q.Name}
	count := 0
	for _, present := range []bool{q.Predicate != nil, q.Choice != nil, q.Score != nil, q.Box != nil, q.Extraction != nil, q.Estimate != nil, q.Forecast != nil} {
		if present {
			count++
		}
	}
	if count != 1 {
		return p, invalid(path, "requires exactly one question variant")
	}
	var err error
	p.instructions, err = guidance(q.Instructions, path+".Instructions")
	if err != nil {
		return p, err
	}
	switch {
	case q.Predicate != nil:
		p.kind = "predicate"
		if criteria := q.Predicate.Criteria; criteria != nil {
			p.positive, err = guidance(criteria.True, path+".Predicate.Criteria.True")
			if err != nil {
				return p, err
			}
			p.negative, err = guidance(criteria.False, path+".Predicate.Criteria.False")
			if err != nil {
				return p, err
			}
		}
		if p.instructions == nil && p.positive == nil && p.negative == nil {
			return p, invalid(path, "predicate requires instructions or criteria")
		}
	case q.Choice != nil:
		p.kind = "choice"
		p.options, err = prepareOptions(q.Choice.Options, path)
	case q.Score != nil:
		p.kind = "score"
		p.levels, err = prepareLevels(q.Score.Levels, path)
	case q.Box != nil:
		p.kind = "box"
		if p.instructions == nil {
			return p, invalid(path, "box requires instructions describing what to locate")
		}
	case q.Extraction != nil:
		p.kind, p.extraction = "extraction", q.Extraction
		err = validateExtraction(*q.Extraction)
	case q.Estimate != nil:
		p.kind, p.estimate = "estimate", q.Estimate
		err = validateEstimate(*q.Estimate)
	case q.Forecast != nil:
		p.kind, p.forecast = "forecast", q.Forecast
		err = validateForecast(*q.Forecast)
	}
	return p, err
}

func prepareOptions(options []ChoiceOption, path string) ([]preparedOption, error) {
	if len(options) < 2 {
		return nil, invalid(path+".Choice.Options", "requires at least two outcomes")
	}
	seen := make(map[string]bool, len(options))
	result := make([]preparedOption, 0, len(options))
	for i, option := range options {
		value, err := option.Value.MarshalJSON()
		if err != nil || seen[string(value)] {
			return nil, invalid(path+".Choice.Options", "requires distinct valid string/boolean values")
		}
		seen[string(value)] = true
		description, err := guidance(option.Description, fmt.Sprintf("%s.Choice.Options[%d].Description", path, i))
		if err != nil {
			return nil, err
		}
		result = append(result, preparedOption{value: value, description: description})
	}
	return result, nil
}

func prepareLevels(levels []ScoreLevel, path string) ([]preparedLevel, error) {
	if len(levels) < 2 {
		return nil, invalid(path+".Score.Levels", "requires at least two levels")
	}
	result := make([]preparedLevel, 0, len(levels))
	for i, level := range levels {
		if strings.TrimSpace(level.Label) == "" {
			return nil, invalid(path+".Score.Levels", "requires nonempty labels")
		}
		description, err := guidance(level.Description, fmt.Sprintf("%s.Score.Levels[%d].Description", path, i))
		if err != nil {
			return nil, err
		}
		result = append(result, preparedLevel{label: level.Label, description: description})
	}
	return result, nil
}

func responseError(name, reason string) error {
	return &ResponseError{Question: name, Reason: reason}
}

// boundedNumber rejects non-JSON spellings and unbounded numeric representations
// before allocating a rational for exact interval comparisons.
func boundedNumber(value json.Number, maximum int) (*big.Rat, error) {
	number, err := exactNumber(value)
	if err != nil {
		return nil, err
	}
	if number.Sign() < 0 || number.Cmp(new(big.Rat).SetInt64(int64(maximum))) > 0 {
		return nil, fmt.Errorf("number must lie in [0,%d]", maximum)
	}
	return number, nil
}

func exactNumber(value json.Number) (*big.Rat, error) {
	text := value.String()
	if len(text) == 0 || len(text) > 256 || !json.Valid([]byte(text)) || !strings.ContainsRune("-0123456789", rune(text[0])) {
		return nil, fmt.Errorf("requires a JSON number of at most 256 bytes")
	}
	if _, exponent, found := strings.Cut(strings.ToLower(text), "e"); found {
		n, err := strconv.Atoi(exponent)
		if err != nil || n < -10000 || n > 10000 {
			return nil, fmt.Errorf("numeric exponent exceeds supported magnitude 10000")
		}
	}
	number, ok := new(big.Rat).SetString(text)
	if !ok {
		return nil, fmt.Errorf("invalid JSON number")
	}
	return number, nil
}
