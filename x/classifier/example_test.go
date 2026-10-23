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

func ExampleTypeSafe_Classify() {
	// This local server stands in for TypeSafe; no API key or network service is used.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"model":"jev-1.14.0","answers":{"mug":{"type":"box","box":{"top":0.25,"bottom":0.82,"left":0.44,"right":0.84},"confidence":0.9}}}`)
	}))
	defer server.Close()
	client, err := classifier.NewTypeSafe(server.Client(), server.URL, "example-key")
	if err != nil {
		fmt.Println(err)
		return
	}
	// Supply real PNG bytes in an application. The named image can also be
	// referenced inside instructions or option descriptions.
	photo := gai.ImageBlock([]byte("example PNG bytes"), "image/png")
	photo.ID = "unique-kitchen-photo"
	response, err := client.Classify(context.Background(), classifier.Request{
		Model:     "jev-1.14.0",
		Input:     classifier.Input{Data: map[string]any{"photo": photo.ID}, Media: []gai.Block{photo}},
		Questions: []classifier.Question{{Name: "mug", Instructions: "Locate the mug in photo.", Box: &classifier.Box{}}},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	box := response.Answers[0].Box
	fmt.Println(box.Top, box.Bottom, box.Left, box.Right)
	// Output: 0.25 0.82 0.44 0.84
}

func ExampleCloudflare_Classify() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"success":true,"result":{"model":"clairic","answers":{"due":{"type":"date","value":"2026-11-19"},"days":{"type":"estimate","value":30,"p10":21,"p90":38}},"truncated":false,"usage":{"input_tokens":100,"output_tokens":0}}}`)
	}))
	defer server.Close()
	client, err := classifier.NewCloudflare(server.Client(), server.URL, "account", "example-key")
	if err != nil {
		fmt.Println(err)
		return
	}
	response, err := client.Classify(context.Background(), classifier.Request{
		Model: "@cf/cloudflare/clairic",
		Input: classifier.Input{Data: "Invoice issued October 20, 2026, net 30."},
		Questions: []classifier.Question{
			{Name: "due", Instructions: "Invoice due date", Extraction: &classifier.Extraction{Type: "date"}},
			{Name: "days", Instructions: "Days until payment", Estimate: &classifier.Estimate{Min: json.Number("0"), Max: json.Number("120"), Step: json.Number("1")}},
		},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(response.Answers[0].Extraction.Value)
	fmt.Println(response.Answers[1].Estimate.P10, response.Answers[1].Estimate.P90)
	// Output:
	// 2026-11-19
	// 21 38
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
