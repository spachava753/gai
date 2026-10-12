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
}

type preparedOption struct {
	value       json.RawMessage
	description json.RawMessage
}

type preparedLevel struct {
	label       string
	description json.RawMessage
}

// Validate checks the common request contract without contacting a provider.
// Provider-specific limits, modalities, and required instructions are checked by
// the adapter. Arbitrary Data and guidance are marshaled once during validation;
// custom JSON marshalers must be safe for concurrent use and have no side effects.
func (r Request) Validate() error {
	_, err := prepareRequest(r)
	return err
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

// prepareRequest validates shared structure and serializes evidence/guidance once,
// leaving endpoint-specific requirements to the provider adapter.
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
	for _, present := range []bool{q.Predicate != nil, q.Choice != nil, q.Score != nil} {
		if present {
			count++
		}
	}
	if count != 1 {
		return p, invalid(path, "requires exactly one of Predicate, Choice, or Score")
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

// Validate checks response names/order, variants, complete distributions, and
// numeric consistency against request. Numbers are compared using exact rational
// arithmetic, not float64. Upstream rounding is allowed up to 0.00005 per
// probability and 0.00005 for a reported score, including accumulated error in
// distribution sums and weighted means. Values are never silently renormalized.
// For resource safety, numeric tokens are limited to 256 bytes and decimal
// exponents to magnitude 10000. Missing/invalid numbers are not implicit zeros.
func (r Response) Validate(request Request) error {
	prepared, err := prepareRequest(request)
	if err != nil {
		return err
	}
	return validateResponse(r, prepared)
}

func responseError(name, reason string) error {
	return &ResponseError{Question: name, Reason: reason}
}

func validateResponse(r Response, request preparedRequest) error {
	if r.Model == "" {
		return responseError("", "missing model")
	}
	if len(r.Answers) != len(request.questions) {
		return responseError("", "answer count does not match questions")
	}
	for i, q := range request.questions {
		if err := validateAnswer(r.Answers[i], q); err != nil {
			return err
		}
	}
	return nil
}

// validateAnswer first checks identity, refusal, and confidence invariants, then
// dispatches to the matching finite-distribution validator.
func validateAnswer(a Answer, q preparedQuestion) error {
	if a.Name != q.name {
		return responseError(q.name, "answer name/order does not match question")
	}
	count := 0
	for _, present := range []bool{a.Predicate != nil, a.Choice != nil, a.Score != nil, a.Refusal != nil} {
		if present {
			count++
		}
	}
	if count != 1 {
		return responseError(q.name, "requires exactly one answer variant")
	}
	if a.Refusal != nil {
		if a.Confidence != nil {
			return responseError(q.name, "refusal cannot have confidence")
		}
		return nil
	}
	if a.Confidence != nil {
		if _, err := boundedNumber(*a.Confidence, 1); err != nil {
			return responseError(q.name, "invalid confidence: "+err.Error())
		}
	}
	switch {
	case a.Predicate != nil && q.kind == "predicate":
		if _, err := boundedNumber(a.Predicate.Probability, 1); err != nil {
			return responseError(q.name, "invalid probability: "+err.Error())
		}
		return nil
	case a.Choice != nil && q.kind == "choice":
		return validateChoice(*a.Choice, q)
	case a.Score != nil && q.kind == "score":
		return validateScore(*a.Score, q)
	default:
		return responseError(q.name, "answer kind does not match question")
	}
}

// validateChoice checks typed outcome identity/order and the selected maximum,
// then verifies that the complete distribution sums to approximately one.
func validateChoice(a ChoiceAnswer, q preparedQuestion) error {
	if len(a.Probabilities) != len(q.options) {
		return responseError(q.name, "incomplete choice distribution")
	}
	selected, err := a.Value.MarshalJSON()
	if err != nil {
		return responseError(q.name, "invalid selected value")
	}
	var selectedProbability *big.Rat
	maximum := new(big.Rat)
	numbers := make([]json.Number, len(a.Probabilities))
	for i, probability := range a.Probabilities {
		value, err := probability.Value.MarshalJSON()
		if err != nil || string(value) != string(q.options[i].value) {
			return responseError(q.name, "choice outcomes/order do not match schema")
		}
		number, err := boundedNumber(probability.Probability, 1)
		if err != nil {
			return responseError(q.name, "invalid choice probability: "+err.Error())
		}
		if number.Cmp(maximum) > 0 {
			maximum.Set(number)
		}
		if string(selected) == string(value) {
			selectedProbability = number
		}
		numbers[i] = probability.Probability
	}
	if selectedProbability == nil || !near(selectedProbability, maximum, 2) {
		return responseError(q.name, "selected value is not a highest-probability outcome")
	}
	_, err = distribution(numbers, q.name)
	return err
}

func validateScore(a ScoreAnswer, q preparedQuestion) error {
	if len(a.Probabilities) != len(q.levels) {
		return responseError(q.name, "incomplete score distribution")
	}
	numbers := make([]json.Number, len(a.Probabilities))
	for i, probability := range a.Probabilities {
		if probability.Index != i {
			return responseError(q.name, "score indices/order do not match schema")
		}
		numbers[i] = probability.Probability
	}
	mean, err := distribution(numbers, q.name)
	if err != nil {
		return err
	}
	score, err := boundedNumber(a.Score, len(q.levels)-1)
	if err != nil {
		return responseError(q.name, "invalid score: "+err.Error())
	}
	n := int64(len(numbers))
	if !near(score, mean, 1+n*(n-1)/2) {
		return responseError(q.name, "score disagrees with weighted level probabilities")
	}
	return nil
}

func distribution(numbers []json.Number, name string) (*big.Rat, error) {
	sum, mean := new(big.Rat), new(big.Rat)
	for i, value := range numbers {
		number, err := boundedNumber(value, 1)
		if err != nil {
			return nil, responseError(name, "invalid distribution: "+err.Error())
		}
		sum.Add(sum, number)
		mean.Add(mean, new(big.Rat).Mul(number, new(big.Rat).SetInt64(int64(i))))
	}
	if !near(sum, big.NewRat(1, 1), int64(len(numbers))) {
		return nil, responseError(name, "probabilities do not sum to approximately one")
	}
	return mean, nil
}

func near(a, b *big.Rat, units int64) bool {
	difference := new(big.Rat).Sub(a, b)
	difference.Abs(difference)
	return difference.Cmp(big.NewRat(units, 20000)) <= 0
}

// boundedNumber rejects non-JSON spellings and unbounded numeric representations
// before allocating a rational for exact interval comparisons.
func boundedNumber(value json.Number, maximum int) (*big.Rat, error) {
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
	if !ok || number.Sign() < 0 || number.Cmp(new(big.Rat).SetInt64(int64(maximum))) > 0 {
		return nil, fmt.Errorf("number must lie in [0,%d]", maximum)
	}
	return number, nil
}
