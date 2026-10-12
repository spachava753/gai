package classifier_test

import (
	"context"
	"fmt"

	"github.com/spachava753/gai/classifier"
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
