package classifier_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"

	"github.com/spachava753/gai"
	"github.com/spachava753/gai/x/classifier"
)

func ExampleClassifier() {
	// Applications can use a provider adapter, middleware, or an in-process model.
	model := classifier.Func(func(_ context.Context, request classifier.Request) (classifier.Response, error) {
		return classifier.Response{
			Model: request.Model,
			Answers: []classifier.Answer{{
				Name:      "refund_requested",
				Predicate: &classifier.PredicateAnswer{Probability: "0.98"},
			}},
		}, nil
	})
	response, err := model.Classify(context.Background(), classifier.Request{
		Model: "example",
		Input: classifier.Input{Data: map[string]any{
			"message": "Please refund the duplicate charge.",
			"policy":  "Duplicate charges qualify for refunds.",
		}},
		Questions: []classifier.Question{{
			Name:         "refund_requested",
			Instructions: "Does the customer explicitly request a refund?",
			Predicate:    &classifier.Predicate{},
		}},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(response.Answers[0].Predicate.Probability)
	// Output: 0.98
}

func ExampleValue() {
	values := []classifier.Value{
		classifier.StringValue("false"),
		classifier.BooleanValue(false),
	}
	data, err := json.Marshal(values)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(data))
	// Output: ["false",false]
}

func ExampleNewTypeSafe() {
	// This local endpoint makes the example runnable without credentials. Use an
	// empty base URL and your application's API key to call the hosted service.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"model":"jev-1.13.0","answers":{"refund":{"type":"noul","noul":0.9750}},"usage":{"input_tokens":32,"output_tokens":0}}`)
	}))
	defer server.Close()

	model, err := classifier.NewTypeSafe(server.Client(), server.URL, "example-key")
	if err != nil {
		fmt.Println(err)
		return
	}
	response, err := model.Classify(context.Background(), classifier.Request{
		Model: "jev-1.13.0",
		Input: classifier.Input{Data: "Please refund the duplicate charge."},
		Questions: []classifier.Question{{
			Name: "refund", Instructions: "Does the customer request a refund?",
			Predicate: &classifier.Predicate{},
		}},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(response.Answers[0].Predicate.Probability)
	if tokens, ok := gai.InputTokens(response.UsageMetadata); ok {
		fmt.Println(tokens)
	}
	// Output:
	// 0.9750
	// 32
}
