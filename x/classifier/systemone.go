package classifier

import (
	"encoding/json"
	"fmt"
	"strconv"

	wire "github.com/spachava753/gai/internal/typesafe"
)

// systemOneQuestions translates the shared finite schema to System One wire
// questions. Endpoint-specific required guidance and limits remain with callers.
func systemOneQuestions(questions []preparedQuestion) (map[string]json.RawMessage, error) {
	result := make(map[string]json.RawMessage, len(questions))
	for _, q := range questions {
		question := map[string]any{"type": q.kind}
		if q.instructions != nil {
			question["instructions"] = q.instructions
		}
		switch q.kind {
		case "predicate":
			question["type"] = "noul"
			criteria := map[string]json.RawMessage{}
			if q.positive != nil {
				criteria["true"] = q.positive
			}
			if q.negative != nil {
				criteria["false"] = q.negative
			}
			if len(criteria) > 0 {
				question["criteria"] = criteria
			}
		case "choice":
			criteria := make(map[string]json.RawMessage, len(q.options))
			for _, option := range q.options {
				var value string
				if err := json.Unmarshal(option.value, &value); err != nil {
					return nil, invalid(q.name+".Choice", "this endpoint supports only string choice values")
				}
				criteria[value] = option.description
			}
			question["criteria"] = criteria
		case "score":
			criteria := make([]any, 0, len(q.levels))
			for _, level := range q.levels {
				if level.description == nil {
					criteria = append(criteria, level.label)
				} else {
					criteria = append(criteria, map[string]any{"label": level.label, "description": level.description})
				}
			}
			question["criteria"] = criteria
		}
		encoded, err := json.Marshal(question)
		if err != nil {
			return nil, err
		}
		result[q.name] = encoded
	}
	return result, nil
}

func systemOneLimits(questions []preparedQuestion, requireInstructions bool) error {
	for _, q := range questions {
		if requireInstructions && q.instructions == nil {
			return invalid(q.name+".Instructions", "this endpoint requires instructions")
		}
		if len(q.options) > 255 {
			return invalid(q.name+".Choice", "this endpoint accepts at most 255 outcomes")
		}
		if len(q.levels) > 10 {
			return invalid(q.name+".Score", "this endpoint accepts at most 10 levels")
		}
	}
	return nil
}

// convertSystemOne shares answer conversion across the System One protocol.
// The provider adapters decode their own generated response models, then convert
// structurally identical generated answers to TypeSafe's model for this helper.
func convertSystemOne(answers map[string]wire.Answer, response Response, request preparedRequest) (Response, error) {
	if len(answers) != len(request.questions) {
		return response, responseError("", "answers must contain exactly the requested names")
	}
	normalized := make([]Answer, 0, len(request.questions))
	for _, q := range request.questions {
		answer, err := convertSystemOneAnswer(answers[q.name], q)
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

// convertSystemOneAnswer checks the generated answer's variant and required
// nullable fields, retaining additional fields and the optional score legend.
func convertSystemOneAnswer(native wire.Answer, q preparedQuestion) (Answer, error) {
	answer := Answer{Name: q.name, Confidence: optionalConfidence(native.Confidence)}
	fields := native.AdditionalProperties
	if native.Legend.IsSpecified() {
		legend, err := json.Marshal(native.Legend)
		if err != nil {
			return answer, err
		}
		if fields == nil {
			fields = map[string]json.RawMessage{}
		}
		fields["legend"] = legend
	}
	answer.ExtraFields = nativeFields(fields)
	var err error
	switch {
	case native.Type == "noul" && q.kind == "predicate":
		probability, err := requiredNumber(native.Noul)
		if err != nil {
			return answer, err
		}
		answer.Predicate = &PredicateAnswer{Probability: probability}
	case native.Type == "choice" && q.kind == "choice":
		answer.Choice, err = convertSystemOneChoice(native, q)
	case native.Type == "score" && q.kind == "score":
		answer.Score, err = convertSystemOneScore(native, q)
	default:
		return answer, fmt.Errorf("unexpected answer kind")
	}
	return answer, err
}

func convertSystemOneChoice(native wire.Answer, q preparedQuestion) (*ChoiceAnswer, error) {
	selected, err := native.Choice.Get()
	if err != nil {
		return nil, fmt.Errorf("missing or null choice")
	}
	if native.Probabilities == nil || len(*native.Probabilities) != len(q.options) {
		return nil, fmt.Errorf("incomplete choice distribution")
	}
	answer := &ChoiceAnswer{Value: StringValue(selected)}
	for _, option := range q.options {
		var value Value
		if err := json.Unmarshal(option.value, &value); err != nil || value.String == nil {
			return nil, fmt.Errorf("invalid string choice schema")
		}
		probability, ok := (*native.Probabilities)[*value.String]
		if !ok {
			return nil, fmt.Errorf("missing declared choice outcome")
		}
		answer.Probabilities = append(answer.Probabilities, ChoiceProbability{Value: value, Probability: json.Number(probability)})
	}
	return answer, nil
}

func convertSystemOneScore(native wire.Answer, q preparedQuestion) (*ScoreAnswer, error) {
	score, err := requiredNumber(native.Score)
	if err != nil {
		return nil, err
	}
	if native.Probabilities == nil || len(*native.Probabilities) != len(q.levels) {
		return nil, fmt.Errorf("incomplete score distribution")
	}
	answer := &ScoreAnswer{Score: score}
	for i := range q.levels {
		probability, ok := (*native.Probabilities)[strconv.Itoa(i)]
		if !ok {
			return nil, fmt.Errorf("missing declared score level")
		}
		answer.Probabilities = append(answer.Probabilities, json.Number(probability))
	}
	return answer, nil
}

func emptyState(data json.RawMessage) json.RawMessage {
	if data == nil {
		return json.RawMessage(`""`)
	}
	return data
}
