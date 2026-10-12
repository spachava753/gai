package classifier_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/spachava753/gai"
	"github.com/spachava753/gai/classifier"
)

func predicateRequest() classifier.Request {
	return classifier.Request{
		Model:     "test-model",
		Input:     classifier.Input{Data: "Please refund the duplicate charge."},
		Questions: []classifier.Question{{Name: "refund", Instructions: "Is a refund requested?", Predicate: &classifier.Predicate{}}},
	}
}

func TestRequestValidate(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(*classifier.Request)
		valid  bool
	}{
		{"text", func(r *classifier.Request) {}, true},
		{"structured state and mixed instructions", func(r *classifier.Request) {
			r.Input.Data = struct {
				ID json.Number `json:"id"`
			}{json.Number("9007199254740993")}
			r.Questions[0].Instructions = []any{"Check the request", map[string]any{"examples": []any{true, nil, json.Number("0.1234567890123456789")}}}
		}, true},
		{"media only", func(r *classifier.Request) {
			r.Input = classifier.Input{Media: []gai.Block{gai.ImageBlock([]byte("image"), "image/png")}}
		}, true},
		{"criteria only", func(r *classifier.Request) {
			r.Questions[0].Instructions = nil
			r.Questions[0].Predicate.Criteria = &classifier.PredicateCriteria{True: "Money back is requested"}
		}, true},
		{"empty model", func(r *classifier.Request) { r.Model = "" }, false},
		{"missing input", func(r *classifier.Request) { r.Input = classifier.Input{} }, false},
		{"boolean data", func(r *classifier.Request) { r.Input.Data = false }, false},
		{"null data", func(r *classifier.Request) { r.Input.Data = json.RawMessage(`null`) }, false},
		{"unencodable data", func(r *classifier.Request) { r.Input.Data = map[string]any{"bad": make(chan int)} }, false},
		{"empty questions", func(r *classifier.Request) { r.Questions = nil }, false},
		{"duplicate name", func(r *classifier.Request) { r.Questions = append(r.Questions, r.Questions[0]) }, false},
		{"empty name", func(r *classifier.Request) { r.Questions[0].Name = "" }, false},
		{"no variant", func(r *classifier.Request) { r.Questions[0].Predicate = nil }, false},
		{"two variants", func(r *classifier.Request) { r.Questions[0].Score = &classifier.Score{} }, false},
		{"unguided predicate", func(r *classifier.Request) { r.Questions[0].Instructions = nil }, false},
		{"scalar guidance", func(r *classifier.Request) { r.Questions[0].Instructions = 12 }, false},
		{"text is not media", func(r *classifier.Request) { r.Input.Media = []gai.Block{gai.TextBlock("not media")} }, false},
		{"invalid base64", func(r *classifier.Request) {
			r.Input.Media = []gai.Block{{BlockType: gai.Content, ModalityType: gai.Image, MimeType: "image/png", Content: gai.Str("!!!")}}
		}, false},
		{"tool is not media", func(r *classifier.Request) {
			r.Input.Media = []gai.Block{{BlockType: gai.ToolCall, ModalityType: gai.Image, Content: gai.Str("aGVsbG8=")}}
		}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := predicateRequest()
			test.change(&request)
			err := request.Validate()
			if (err == nil) != test.valid {
				t.Fatalf("Validate() = %v, valid = %v", err, test.valid)
			}
			if err != nil {
				var invalid gai.InvalidParameterErr
				if !errors.As(err, &invalid) {
					t.Fatalf("expected InvalidParameterErr, got %T", err)
				}
			}
		})
	}
}

func TestValueJSON(t *testing.T) {
	for _, input := range []string{`false`, `true`, `"false"`, `""`, `"billing"`} {
		t.Run(input, func(t *testing.T) {
			var value classifier.Value
			if err := json.Unmarshal([]byte(input), &value); err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(value)
			if err != nil || string(encoded) != input {
				t.Fatalf("round trip = %s, %v", encoded, err)
			}
		})
	}
	for _, input := range []string{`null`, `1`, `{}`, `[]`} {
		t.Run("reject "+input, func(t *testing.T) {
			var value classifier.Value
			if err := json.Unmarshal([]byte(input), &value); err == nil {
				t.Fatal("accepted non-string/boolean value")
			}
		})
	}
}

func TestFiniteSchema(t *testing.T) {
	for _, test := range []struct {
		name     string
		question classifier.Question
		valid    bool
	}{
		{"typed values", classifier.Question{Choice: &classifier.Choice{Options: []classifier.ChoiceOption{{Value: classifier.BooleanValue(false)}, {Value: classifier.StringValue("false")}}}}, true},
		{"duplicate values", classifier.Question{Choice: &classifier.Choice{Options: []classifier.ChoiceOption{{Value: classifier.StringValue("a")}, {Value: classifier.StringValue("a")}}}}, false},
		{"invalid value", classifier.Question{Choice: &classifier.Choice{Options: []classifier.ChoiceOption{{}, {Value: classifier.StringValue("a")}}}}, false},
		{"one choice", classifier.Question{Choice: &classifier.Choice{Options: []classifier.ChoiceOption{{Value: classifier.StringValue("a")}}}}, false},
		{"rubric", classifier.Question{Score: &classifier.Score{Levels: []classifier.ScoreLevel{{Label: "low"}, {Label: "high", Description: map[string]any{"rule": "blocks work"}}}}}, true},
		{"empty label", classifier.Question{Score: &classifier.Score{Levels: []classifier.ScoreLevel{{}, {Label: "high"}}}}, false},
		{"one level", classifier.Question{Score: &classifier.Score{Levels: []classifier.ScoreLevel{{Label: "low"}}}}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := predicateRequest()
			request.Questions[0] = test.question
			request.Questions[0].Name = "test"
			if err := request.Validate(); (err == nil) != test.valid {
				t.Fatalf("Validate() = %v, valid = %v", err, test.valid)
			}
		})
	}
}

func TestNumericValidation(t *testing.T) {
	for _, test := range []struct {
		value string
		valid bool
	}{
		{"0", true}, {"1", true}, {"1e-30", true}, {"0.12345678901234567890123456789", true},
		{"", false}, {"null", false}, {`"0.5"`, false}, {"NaN", false}, {"Inf", false},
		{"-0.0000000000000000000000000001", false}, {"1.0000000000000000000000000001", false},
		{"01", false}, {"1/2", false}, {"1e999999999", false},
	} {
		t.Run(test.value, func(t *testing.T) {
			response := classifier.Response{Model: "resolved", Answers: []classifier.Answer{{Name: "refund", Predicate: &classifier.PredicateAnswer{Probability: json.Number(test.value)}}}}
			err := response.Validate(predicateRequest())
			if (err == nil) != test.valid {
				t.Fatalf("Validate() = %v, valid = %v", err, test.valid)
			}
			if err != nil {
				var invalid *classifier.ResponseError
				if !errors.As(err, &invalid) || invalid.Question != "refund" {
					t.Fatalf("unexpected error %v", err)
				}
			}
		})
	}
}

func TestResponseContract(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(*classifier.Response)
		valid  bool
	}{
		{"zero probability", func(r *classifier.Response) {}, true},
		{"explicit zero confidence", func(r *classifier.Response) { n := json.Number("0"); r.Answers[0].Confidence = &n }, true},
		{"refusal", func(r *classifier.Response) {
			r.Answers[0].Predicate = nil
			r.Answers[0].Refusal = &classifier.RefusalAnswer{}
		}, true},
		{"missing answers", func(r *classifier.Response) { r.Answers = nil }, false},
		{"wrong name", func(r *classifier.Response) { r.Answers[0].Name = "different" }, false},
		{"missing model", func(r *classifier.Response) { r.Model = "" }, false},
		{"wrong kind", func(r *classifier.Response) {
			r.Answers[0].Predicate = nil
			r.Answers[0].Choice = &classifier.ChoiceAnswer{}
		}, false},
		{"two variants", func(r *classifier.Response) { r.Answers[0].Refusal = &classifier.RefusalAnswer{} }, false},
		{"refusal confidence", func(r *classifier.Response) {
			r.Answers[0].Predicate = nil
			r.Answers[0].Refusal = &classifier.RefusalAnswer{}
			n := json.Number("1")
			r.Answers[0].Confidence = &n
		}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := classifier.Response{Model: "resolved", Answers: []classifier.Answer{{Name: "refund", Predicate: &classifier.PredicateAnswer{Probability: "0"}}}}
			test.change(&response)
			if err := response.Validate(predicateRequest()); (err == nil) != test.valid {
				t.Fatalf("Validate() = %v, valid = %v", err, test.valid)
			}
		})
	}
}

func TestScoreDistribution(t *testing.T) {
	for _, test := range []struct {
		name          string
		probabilities []classifier.ScoreProbability
		score         json.Number
		valid         bool
	}{
		{"expected not argmax", []classifier.ScoreProbability{{Index: 0, Probability: "0.1"}, {Index: 1, Probability: "0.7"}, {Index: 2, Probability: "0.2"}}, "1.1", true},
		{"rounded", []classifier.ScoreProbability{{Index: 0, Probability: "0.3333"}, {Index: 1, Probability: "0.3333"}, {Index: 2, Probability: "0.3333"}}, "1", true},
		{"not normalized", []classifier.ScoreProbability{{Index: 0, Probability: "0.1"}, {Index: 1, Probability: "0.1"}, {Index: 2, Probability: "0.1"}}, "0.3", false},
		{"argmax is wrong score", []classifier.ScoreProbability{{Index: 0, Probability: "0.1"}, {Index: 1, Probability: "0.7"}, {Index: 2, Probability: "0.2"}}, "1", false},
		{"missing outcome", []classifier.ScoreProbability{{Index: 0, Probability: "1"}}, "0", false},
		{"wrong index", []classifier.ScoreProbability{{Index: 0, Probability: "1"}, {Index: 1, Probability: "0"}, {Index: 1, Probability: "0"}}, "0", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := predicateRequest()
			request.Questions = []classifier.Question{{Name: "severity", Score: &classifier.Score{Levels: []classifier.ScoreLevel{{Label: "low"}, {Label: "medium"}, {Label: "high"}}}}}
			response := classifier.Response{Model: "resolved", Answers: []classifier.Answer{{Name: "severity", Score: &classifier.ScoreAnswer{Score: test.score, Probabilities: test.probabilities}}}}
			if err := response.Validate(request); (err == nil) != test.valid {
				t.Fatalf("Validate() = %v, valid = %v", err, test.valid)
			}
		})
	}
}

func TestChoiceDistribution(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(*classifier.ChoiceAnswer)
		valid  bool
	}{
		{"valid", func(a *classifier.ChoiceAnswer) {}, true},
		{"wrong winner", func(a *classifier.ChoiceAnswer) { a.Value = classifier.StringValue("false") }, false},
		{"unknown winner", func(a *classifier.ChoiceAnswer) { a.Value = classifier.StringValue("other") }, false},
		{"missing mass", func(a *classifier.ChoiceAnswer) { a.Probabilities = a.Probabilities[:1] }, false},
		{"duplicate", func(a *classifier.ChoiceAnswer) { a.Probabilities[1].Value = classifier.BooleanValue(false) }, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := predicateRequest()
			request.Questions = []classifier.Question{{Name: "choice", Choice: &classifier.Choice{Options: []classifier.ChoiceOption{{Value: classifier.BooleanValue(false)}, {Value: classifier.StringValue("false")}}}}}
			answer := &classifier.ChoiceAnswer{Value: classifier.BooleanValue(false), Probabilities: []classifier.ChoiceProbability{{Value: classifier.BooleanValue(false), Probability: "0.8"}, {Value: classifier.StringValue("false"), Probability: "0.2"}}}
			test.change(answer)
			response := classifier.Response{Model: "resolved", Answers: []classifier.Answer{{Name: "choice", Choice: answer}}}
			if err := response.Validate(request); (err == nil) != test.valid {
				t.Fatalf("Validate() = %v, valid = %v", err, test.valid)
			}
		})
	}
}

func TestFuncAndSerialization(t *testing.T) {
	model := classifier.Func(func(ctx context.Context, request classifier.Request) (classifier.Response, error) {
		return classifier.Response{Model: request.Model, Answers: []classifier.Answer{{Name: "refund", Predicate: &classifier.PredicateAnswer{Probability: "0.123456789012345678901"}}}}, ctx.Err()
	})
	response, err := model.Classify(context.Background(), predicateRequest())
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(response)
	if err != nil || !strings.Contains(string(data), `"probability":0.123456789012345678901`) {
		t.Fatalf("numeric precision not preserved: %s, %v", data, err)
	}
	var decoded classifier.Response
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if err := decoded.Validate(predicateRequest()); err != nil {
		t.Fatal(err)
	}
}
