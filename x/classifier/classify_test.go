package classifier

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/spachava753/gai"
)

func predicateRequest() Request {
	return Request{
		Model:     "test-model",
		Input:     Input{Data: "Please refund the duplicate charge."},
		Questions: []Question{{Name: "refund", Instructions: "Is a refund requested?", Predicate: &Predicate{}}},
	}
}

func TestRequestPreparation(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(*Request)
		valid  bool
	}{
		{"text", func(r *Request) {}, true},
		{"structured state and mixed instructions", func(r *Request) {
			r.Input.Data = struct {
				ID json.Number `json:"id"`
			}{json.Number("9007199254740993")}
			r.Questions[0].Instructions = []any{"Check the request", map[string]any{"examples": []any{true, nil, json.Number("0.1234567890123456789")}}}
		}, true},
		{"media only", func(r *Request) {
			r.Input = Input{Media: []gai.Block{gai.ImageBlock([]byte("image"), "image/png")}}
		}, true},
		{"criteria only", func(r *Request) {
			r.Questions[0].Instructions = nil
			r.Questions[0].Predicate.Criteria = &PredicateCriteria{True: "Money back is requested"}
		}, true},
		{"empty model", func(r *Request) { r.Model = "" }, false},
		{"missing input", func(r *Request) { r.Input = Input{} }, false},
		{"boolean data", func(r *Request) { r.Input.Data = false }, false},
		{"null data", func(r *Request) { r.Input.Data = json.RawMessage(`null`) }, false},
		{"unencodable data", func(r *Request) { r.Input.Data = map[string]any{"bad": make(chan int)} }, false},
		{"empty questions", func(r *Request) { r.Questions = nil }, false},
		{"duplicate name", func(r *Request) { r.Questions = append(r.Questions, r.Questions[0]) }, false},
		{"empty name", func(r *Request) { r.Questions[0].Name = "" }, false},
		{"no variant", func(r *Request) { r.Questions[0].Predicate = nil }, false},
		{"two variants", func(r *Request) { r.Questions[0].Score = &Score{} }, false},
		{"unguided predicate", func(r *Request) { r.Questions[0].Instructions = nil }, false},
		{"scalar guidance", func(r *Request) { r.Questions[0].Instructions = 12 }, false},
		{"text is not media", func(r *Request) { r.Input.Media = []gai.Block{gai.TextBlock("not media")} }, false},
		{"invalid base64", func(r *Request) {
			r.Input.Media = []gai.Block{{BlockType: gai.Content, ModalityType: gai.Image, MimeType: "image/png", Content: gai.Str("!!!")}}
		}, false},
		{"tool is not media", func(r *Request) {
			r.Input.Media = []gai.Block{{BlockType: gai.ToolCall, ModalityType: gai.Image, Content: gai.Str("aGVsbG8=")}}
		}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := predicateRequest()
			test.change(&request)
			_, err := prepareRequest(request)
			if (err == nil) != test.valid {
				t.Fatalf("prepareRequest() = %v, valid = %v", err, test.valid)
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
			var value Value
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
			var value Value
			if err := json.Unmarshal([]byte(input), &value); err == nil {
				t.Fatal("accepted non-string/boolean value")
			}
		})
	}
}

func TestFiniteSchema(t *testing.T) {
	for _, test := range []struct {
		name     string
		question Question
		valid    bool
	}{
		{"typed values", Question{Choice: &Choice{Options: []ChoiceOption{{Value: BooleanValue(false)}, {Value: StringValue("false")}}}}, true},
		{"duplicate values", Question{Choice: &Choice{Options: []ChoiceOption{{Value: StringValue("a")}, {Value: StringValue("a")}}}}, false},
		{"invalid value", Question{Choice: &Choice{Options: []ChoiceOption{{}, {Value: StringValue("a")}}}}, false},
		{"one choice", Question{Choice: &Choice{Options: []ChoiceOption{{Value: StringValue("a")}}}}, false},
		{"rubric", Question{Score: &Score{Levels: []ScoreLevel{{Label: "low"}, {Label: "high", Description: map[string]any{"rule": "blocks work"}}}}}, true},
		{"empty label", Question{Score: &Score{Levels: []ScoreLevel{{}, {Label: "high"}}}}, false},
		{"one level", Question{Score: &Score{Levels: []ScoreLevel{{Label: "low"}}}}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := predicateRequest()
			request.Questions[0] = test.question
			request.Questions[0].Name = "test"
			if _, err := prepareRequest(request); (err == nil) != test.valid {
				t.Fatalf("prepareRequest() = %v, valid = %v", err, test.valid)
			}
		})
	}
}

func TestAnswerJSON(t *testing.T) {
	for _, test := range []struct {
		name   string
		answer Answer
		json   string
	}{
		{"refusal", Answer{Name: "refund", Refused: true}, `{"name":"refund","refused":true}`},
		{"score", Answer{Name: "severity", Score: &ScoreAnswer{Score: "1.1000", Probabilities: []json.Number{"0.1000", "0.7000", "0.2000"}}}, `{"name":"severity","score":{"score":1.1000,"probabilities":[0.1000,0.7000,0.2000]}}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			data, err := json.Marshal(test.answer)
			if err != nil || string(data) != test.json {
				t.Fatalf("Marshal = %s, %v; want %s", data, err, test.json)
			}
			var decoded Answer
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(decoded, test.answer) {
				t.Fatalf("round trip = %+v; want %+v", decoded, test.answer)
			}
		})
	}
}

func TestFuncAndSerialization(t *testing.T) {
	model := Func(func(ctx context.Context, request Request) (Response, error) {
		return Response{Model: request.Model, Answers: []Answer{{Name: "refund", Predicate: &PredicateAnswer{Probability: "0.123456789012345678901"}}}}, ctx.Err()
	})
	response, err := model.Classify(context.Background(), predicateRequest())
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(response)
	if err != nil || !strings.Contains(string(data), `"probability":0.123456789012345678901`) {
		t.Fatalf("numeric precision not preserved: %s, %v", data, err)
	}
	var decoded Response
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, response) {
		t.Fatalf("round trip = %+v, want %+v", decoded, response)
	}
}
