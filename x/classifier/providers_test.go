package classifier_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/spachava753/gai"
	"github.com/spachava753/gai/x/classifier"
)

type providerFixture struct {
	name     string
	model    string
	path     string
	provider gai.Provider
	new      func(*http.Client, string, string) (classifier.Classifier, error)
}

func predicateRequest() classifier.Request {
	return classifier.Request{
		Model:     "test-model",
		Input:     classifier.Input{Data: "Please refund the duplicate charge."},
		Questions: []classifier.Question{{Name: "refund", Instructions: "Is a refund requested?", Predicate: &classifier.Predicate{}}},
	}
}

func providerFixtures() []providerFixture {
	return []providerFixture{
		{"openai", "gpt-6-luna", "/decisions", classifier.ProviderOpenAI, func(c *http.Client, url, key string) (classifier.Classifier, error) {
			return classifier.NewOpenAI(c, url, key)
		}},
		{"typesafe", "jev-1.13.0", "/systemone", classifier.ProviderTypeSafe, func(c *http.Client, url, key string) (classifier.Classifier, error) {
			return classifier.NewTypeSafe(c, url, key)
		}},
		{"perplexity", "pplx-decider-v1.1-27b", "/decisions", classifier.ProviderPerplexity, func(c *http.Client, url, key string) (classifier.Classifier, error) {
			return classifier.NewPerplexity(c, url, key)
		}},
		{"cloudflare", "@cf/cloudflare/clef", "/accounts/account/ai/run/@cf/cloudflare/clef", classifier.ProviderCloudflare, func(c *http.Client, url, key string) (classifier.Classifier, error) {
			return classifier.NewCloudflare(c, url, "account", key)
		}},
	}
}

func wrapProviderResponse(provider providerFixture, response string) string {
	if provider.name == "cloudflare" {
		return `{"success":true,"result":` + response + `,"errors":[],"messages":[],"envelope_number":9007199254740993}`
	}
	return response
}

func predicateBody(provider providerFixture, number string) string {
	if provider.name == "openai" {
		return `{"model":"resolved","answers":[{"name":"refund","type":"predicate","probability":` + number + `}],"usage":{"input_tokens":12,"output_tokens":0}}`
	}
	return `{"model":"resolved","answers":{"refund":{"type":"noul","noul":` + number + `}},"usage":{"input_tokens":12,"output_tokens":0}}`
}

func mixedRequest(model string) classifier.Request {
	return classifier.Request{
		Model: model, SafetyIdentifier: "opaque-user", Headers: http.Header{"X-Session": {"one", "two"}},
		Input: classifier.Input{Data: map[string]any{"ticket": "duplicate charge", "id": json.Number("9007199254740993")}},
		Questions: []classifier.Question{
			{Name: "refund", Instructions: []any{"Does the customer request money back?", map[string]any{"example": true}}, Predicate: &classifier.Predicate{Criteria: &classifier.PredicateCriteria{True: map[string]any{"rule": "explicit refund"}, False: "billing question only"}}},
			{Name: "team", Instructions: "Which team handles this?", Choice: &classifier.Choice{Options: []classifier.ChoiceOption{{Value: classifier.StringValue("billing"), Description: map[string]any{"handles": []any{"refunds", "charges"}}}, {Value: classifier.StringValue("other")}}}},
			{Name: "severity", Instructions: "Rate customer impact.", Score: &classifier.Score{Levels: []classifier.ScoreLevel{{Label: "low"}, {Label: "medium", Description: []any{"workaround", map[string]any{"blocked": false}}}, {Label: "high"}}}},
		},
	}
}

func mixedBody(provider providerFixture) string {
	if provider.name == "openai" {
		return `{"model":"resolved","trace":9007199254740993,"answers":[
		{"name":"severity","type":"score","score":1.1000,"confidence":0.55,"probabilities":[{"value":2,"label":"high","probability":0.2},{"value":0,"label":"low","probability":0.1},{"value":1,"label":"medium","probability":0.7}]},
		{"name":"team","type":"choice","choice":"billing","confidence":0.6000,"probabilities":[{"value":"other","probability":0.2},{"value":"billing","probability":0.8}],"future":9007199254740993},
		{"name":"refund","type":"predicate","probability":0.98765432109876543210987654321}],
		"usage":{"input_tokens":123,"output_tokens":2,"total_tokens":125,"input_tokens_details":{"cached_tokens":1,"cache_write_tokens":2},"output_tokens_details":{"reasoning_tokens":0}}}`
	}
	return `{"model":"resolved","trace":9007199254740993,"answers":{
	"severity":{"type":"score","score":1.1000,"confidence":0.55,"probabilities":{"2":0.2,"0":0.1,"1":0.7},"legend":{"0":"low","1":"medium","2":"high"}},
	"team":{"type":"choice","choice":"billing","confidence":0.6000,"probabilities":{"other":0.2,"billing":0.8},"future":9007199254740993},
	"refund":{"type":"noul","noul":0.98765432109876543210987654321}},"usage":{"input_tokens":123,"output_tokens":2}}`
}

func decodedJSON(t *testing.T, data []byte) any {
	t.Helper()
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		t.Fatal(err)
	}
	return value
}

func TestProviderContracts(t *testing.T) {
	for _, provider := range providerFixtures() {
		t.Run(provider.name, func(t *testing.T) {
			var captured []byte
			var headers http.Header
			var path string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				captured, _ = io.ReadAll(r.Body)
				headers, path = r.Header.Clone(), r.URL.Path
				w.Header().Set("X-Request-ID", "request-123")
				fmt.Fprint(w, wrapProviderResponse(provider, mixedBody(provider)))
			}))
			defer server.Close()
			model, err := provider.new(server.Client(), server.URL+"/prefix/", "test-key")
			if err != nil {
				t.Fatal(err)
			}
			request := mixedRequest(provider.model)
			before, err := json.Marshal(request)
			if err != nil {
				t.Fatal(err)
			}
			response, err := model.Classify(context.Background(), request)
			if err != nil {
				t.Fatal(err)
			}
			after, _ := json.Marshal(request)
			if string(before) != string(after) {
				t.Fatal("adapter mutated request")
			}
			if path != "/prefix"+provider.path || headers.Get("Authorization") != "Bearer test-key" || !reflect.DeepEqual(headers.Values("X-Session"), []string{"one", "two"}) {
				t.Fatalf("unexpected HTTP request: %s %v", path, headers)
			}
			wire := decodedJSON(t, captured).(map[string]any)
			if provider.name == "openai" {
				expected := `{"model":"gpt-6-luna","safety_identifier":"opaque-user","input":"{\"id\":9007199254740993,\"ticket\":\"duplicate charge\"}","questions":[
				{"type":"predicate","name":"refund","instructions":"[\"Does the customer request money back?\",{\"example\":true}]\n\nCriteria (true and false outcomes):\n{\"false\":\"billing question only\",\"true\":{\"rule\":\"explicit refund\"}}"},
				{"type":"choice","name":"team","instructions":"Which team handles this?","choices":[{"value":"billing","description":"{\"handles\":[\"refunds\",\"charges\"]}"},{"value":"other"}]},
				{"type":"score","name":"severity","instructions":"Rate customer impact.","levels":[{"label":"low"},{"label":"medium","description":"[\"workaround\",{\"blocked\":false}]"},{"label":"high"}]}]}`
				if !reflect.DeepEqual(wire, decodedJSON(t, []byte(expected))) {
					t.Fatalf("OpenAI body mismatch: %s", captured)
				}
				if count, ok := gai.CacheReadTokens(response.UsageMetadata); !ok || count != 1 {
					t.Fatalf("cache usage: %v", response.UsageMetadata)
				}
			} else {
				bodyModel := provider.model
				if provider.name == "cloudflare" {
					bodyModel = "clef"
				}
				expected := `{"model":` + fmt.Sprintf("%q", bodyModel) + `,"state":{"id":9007199254740993,"ticket":"duplicate charge"},"questions":{
				"refund":{"type":"noul","instructions":["Does the customer request money back?",{"example":true}],"criteria":{"true":{"rule":"explicit refund"},"false":"billing question only"}},
				"team":{"type":"choice","instructions":"Which team handles this?","criteria":{"billing":{"handles":["refunds","charges"]},"other":null}},
				"severity":{"type":"score","instructions":"Rate customer impact.","criteria":["low",{"label":"medium","description":["workaround",{"blocked":false}]},"high"]}}}`
				if !reflect.DeepEqual(wire, decodedJSON(t, []byte(expected))) {
					t.Fatalf("System One body mismatch: %s", captured)
				}
				if _, ok := gai.CacheReadTokens(response.UsageMetadata); ok {
					t.Fatal("invented cache usage")
				}
			}
			if response.Answers[0].Predicate.Probability != "0.98765432109876543210987654321" || response.Answers[2].Score.Score != "1.1000" || *response.Answers[1].Confidence != "0.6000" {
				t.Fatalf("numeric precision lost: %+v", response.Answers)
			}
			if *response.Answers[1].Choice.Probabilities[0].Value.String != "billing" || !reflect.DeepEqual(response.Answers[2].Score.Probabilities, []json.Number{"0.1", "0.7", "0.2"}) {
				t.Fatal("distribution order was not normalized")
			}
			if n, ok := gai.InputTokens(response.UsageMetadata); !ok || n != 123 {
				t.Fatalf("input usage: %v", response.UsageMetadata)
			}
			if n, ok := gai.OutputTokens(response.UsageMetadata); !ok || n != 2 {
				t.Fatalf("output usage: %v", response.UsageMetadata)
			}
			raw := response.ExtraFields[classifier.ExtraFieldWireFields].(map[string]json.RawMessage)
			if string(raw["trace"]) != "9007199254740993" || response.ExtraFields[classifier.ExtraFieldProvider] != provider.provider {
				t.Fatalf("missing invocation metadata: %v", response.ExtraFields)
			}
			answerFields := response.Answers[1].ExtraFields[classifier.ExtraFieldWireFields].(map[string]json.RawMessage)
			if string(answerFields["future"]) != "9007199254740993" {
				t.Fatal("answer metadata lost")
			}
			*request.Questions[1].Choice.Options[0].Value.String = "changed"
			if *response.Answers[1].Choice.Probabilities[0].Value.String != "billing" {
				t.Fatal("response aliases request memory")
			}
		})
	}
}

func TestMediaTranslation(t *testing.T) {
	for _, provider := range providerFixtures() {
		t.Run(provider.name, func(t *testing.T) {
			var captured map[string]any
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				decoder := json.NewDecoder(r.Body)
				decoder.UseNumber()
				if err := decoder.Decode(&captured); err != nil {
					t.Error(err)
				}
				fmt.Fprint(w, wrapProviderResponse(provider, predicateBody(provider, "0.75")))
			}))
			defer server.Close()
			model, err := provider.new(server.Client(), server.URL, "key")
			if err != nil {
				t.Fatal(err)
			}
			request := predicateRequest()
			request.Model = provider.model
			request.Input.Media = []gai.Block{gai.ImageBlock([]byte("abc"), "image/png"), gai.ImageBlock([]byte("def"), "image/jpeg")}
			_, err = model.Classify(context.Background(), request)
			if provider.name == "typesafe" {
				var modality gai.UnsupportedInputModalityErr
				if !errors.As(err, &modality) || calls.Load() != 0 {
					t.Fatalf("Jev must reject media before HTTP: %v, calls=%d", err, calls.Load())
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			switch provider.name {
			case "openai":
				parts := captured["input"].([]any)[0].(map[string]any)["content"].([]any)
				if len(parts) != 3 || parts[1].(map[string]any)["image_url"] != "data:image/png;base64,YWJj" || parts[2].(map[string]any)["image_url"] != "data:image/jpeg;base64,ZGVm" {
					t.Fatalf("image parts: %v", parts)
				}
			case "perplexity":
				parts := captured["state"].([]any)
				if len(parts) != 3 || parts[1].(map[string]any)["image_url"].(map[string]any)["url"] != "data:image/png;base64,YWJj" {
					t.Fatalf("image state: %v", parts)
				}
			case "cloudflare":
				images := captured["images"].([]any)
				if len(images) != 2 || images[0].(map[string]any)["base64"] != "YWJj" || images[1].(map[string]any)["content_type"] != "image/jpeg" {
					t.Fatalf("images: %v", images)
				}
			}
			request.Input.Media = []gai.Block{gai.PDFBlock([]byte("pdf"), "invoice.pdf")}
			_, err = model.Classify(context.Background(), request)
			if err == nil || calls.Load() != 1 {
				t.Fatalf("PDF must not masquerade as image: %v, calls=%d", err, calls.Load())
			}
		})
	}
}

func TestOpenAIBooleanAndRefusal(t *testing.T) {
	t.Run("typed false differs from string", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			if !strings.Contains(string(body), `"value":false`) || !strings.Contains(string(body), `"value":"false"`) {
				t.Errorf("typed values lost: %s", body)
			}
			fmt.Fprint(w, `{"model":"resolved","answers":[{"name":"choice","type":"choice","choice":false,"confidence":0,"probabilities":[{"value":false,"probability":0.5},{"value":"false","probability":0.5}]}]}`)
		}))
		defer server.Close()
		model, err := classifier.NewOpenAI(server.Client(), server.URL, "key")
		if err != nil {
			t.Fatal(err)
		}
		request := predicateRequest()
		request.Questions = []classifier.Question{{Name: "choice", Instructions: "Choose", Choice: &classifier.Choice{Options: []classifier.ChoiceOption{{Value: classifier.BooleanValue(false)}, {Value: classifier.StringValue("false")}}}}}
		response, err := model.Classify(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		if response.Answers[0].Choice.Value.Boolean == nil || *response.Answers[0].Choice.Value.Boolean || response.Answers[0].Confidence == nil || *response.Answers[0].Confidence != "0" {
			t.Fatal("lost explicit false/zero")
		}
	})
	t.Run("mixed refusal and answer", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, `{"model":"resolved","answers":[{"name":"refund","type":"refusal"},{"name":"other","type":"predicate","probability":0}],"usage":{"input_tokens":9,"output_tokens":0}}`)
		}))
		defer server.Close()
		model, err := classifier.NewOpenAI(server.Client(), server.URL, "key")
		if err != nil {
			t.Fatal(err)
		}
		request := predicateRequest()
		request.Questions = append(request.Questions, classifier.Question{Name: "other", Instructions: "Another question", Predicate: &classifier.Predicate{}})
		response, err := model.Classify(context.Background(), request)
		if err != nil || len(response.Answers) != 2 || !response.Answers[0].Refused || response.Answers[1].Predicate.Probability != "0" {
			t.Fatalf("refusal treated as execution failure or negative answer: %+v, %v", response, err)
		}
	})
}

func TestRefusalPayloadRejected(t *testing.T) {
	for _, payload := range []string{`"probability":0`, `"probability":null`, `"probabilities":null`, `"choice":null`, `"score":null`, `"confidence":0`} {
		t.Run(payload, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, `{"model":"resolved","answers":[{"name":"refund","type":"refusal",`+payload+`}]}`)
			}))
			defer server.Close()
			model, err := classifier.NewOpenAI(server.Client(), server.URL, "key")
			if err != nil {
				t.Fatal(err)
			}
			response, err := model.Classify(context.Background(), predicateRequest())
			var contract *classifier.ResponseError
			if !errors.As(err, &contract) || len(response.Answers) != 0 {
				t.Fatalf("contradictory refusal payload accepted: %+v, %v", response, err)
			}
		})
	}
}

func TestInvalidNativeResponses(t *testing.T) {
	for _, provider := range providerFixtures() {
		t.Run(provider.name, func(t *testing.T) {
			for _, test := range []struct {
				name   string
				change func(string) string
			}{
				{"quoted number", func(s string) string { return strings.ReplaceAll(s, "0.75", `"0.75"`) }},
				{"null number", func(s string) string { return strings.ReplaceAll(s, "0.75", `null`) }},
				{"missing probability", func(s string) string {
					s = strings.ReplaceAll(s, `"probability":`, `"missing_probability":`)
					return strings.ReplaceAll(s, `"noul":0.75`, `"missing_noul":0.75`)
				}},
				{"out of range", func(s string) string { return strings.ReplaceAll(s, "0.75", `1.00000000000000000000000001`) }},
				{"extreme exponent", func(s string) string { return strings.ReplaceAll(s, "0.75", `1e-999999999`) }},
				{"wrong answer name", func(s string) string { return strings.ReplaceAll(s, `"refund"`, `"unknown"`) }},
				{"missing model", func(s string) string { return strings.ReplaceAll(s, `"model":"resolved",`, ``) }},
				{"duplicate JSON key", func(s string) string {
					return strings.ReplaceAll(s, `"model":"resolved"`, `"model":"a","model":"resolved"`)
				}},
				{"invalid usage", func(s string) string { return strings.ReplaceAll(s, `"input_tokens":12`, `"input_tokens":-1`) }},
				{"quoted usage", func(s string) string { return strings.ReplaceAll(s, `"input_tokens":12`, `"input_tokens":"12"`) }},
				{"trailing value", func(s string) string { return s + `{}` }},
			} {
				t.Run(test.name, func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						fmt.Fprint(w, wrapProviderResponse(provider, test.change(predicateBody(provider, "0.75"))))
					}))
					defer server.Close()
					model, err := provider.new(server.Client(), server.URL, "key")
					if err != nil {
						t.Fatal(err)
					}
					request := predicateRequest()
					request.Model = provider.model
					response, err := model.Classify(context.Background(), request)
					var contract *classifier.ResponseError
					if !errors.As(err, &contract) || len(response.Answers) != 0 {
						t.Fatalf("malformed response accepted: %+v, %v", response, err)
					}
				})
			}
		})
	}
}

func TestUnavailableUsage(t *testing.T) {
	for _, provider := range providerFixtures() {
		t.Run(provider.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body := strings.ReplaceAll(predicateBody(provider, "0"), `"usage":{"input_tokens":12,"output_tokens":0}`, `"usage":{"input_tokens":null}`)
				fmt.Fprint(w, wrapProviderResponse(provider, body))
			}))
			defer server.Close()
			model, err := provider.new(server.Client(), server.URL, "key")
			if err != nil {
				t.Fatal(err)
			}
			request := predicateRequest()
			request.Model = provider.model
			response, err := model.Classify(context.Background(), request)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := gai.InputTokens(response.UsageMetadata); ok {
				t.Fatal("invented input count")
			}
			if _, ok := gai.OutputTokens(response.UsageMetadata); ok {
				t.Fatal("invented output count")
			}
			if response.Answers[0].Confidence != nil {
				t.Fatal("unexpected synthesized confidence")
			}
		})
	}
}

func TestHTTPFailures(t *testing.T) {
	for _, provider := range providerFixtures() {
		t.Run(provider.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.Header().Set("Retry-After", "2")
				w.Header().Set("X-Request-ID", "failed")
				w.WriteHeader(http.StatusTooManyRequests)
				fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
			}))
			defer server.Close()
			model, err := provider.new(server.Client(), server.URL, "key")
			if err != nil {
				t.Fatal(err)
			}
			request := predicateRequest()
			request.Model = provider.model
			response, err := model.Classify(context.Background(), request)
			var apiErr *gai.ApiErr
			if !errors.As(err, &apiErr) || apiErr.Kind != gai.APIErrorKindRateLimit || apiErr.Provider != provider.provider || apiErr.Message != "slow down" || !apiErr.Retryable() {
				t.Fatalf("unexpected API error: %v", err)
			}
			if delay, ok := apiErr.RetryAfter(); !ok || delay != 2*time.Second {
				t.Fatalf("retry delay = %v, %v", delay, ok)
			}
			if calls.Load() != 1 || response.ExtraFields[classifier.ExtraFieldHeaders].(http.Header).Get("X-Request-ID") != "failed" {
				t.Fatal("unexpected retry or discarded headers")
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_, err = model.Classify(ctx, request)
			if !errors.Is(err, context.Canceled) || calls.Load() != 1 {
				t.Fatalf("cancellation ignored: %v", err)
			}
		})
	}
}

func TestConcurrentRequests(t *testing.T) {
	for _, provider := range providerFixtures() {
		t.Run(provider.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				var input string
				field := "state"
				if provider.name == "openai" {
					field = "input"
				}
				if err := json.Unmarshal(body[field], &input); err != nil {
					t.Error(err)
				}
				if input != r.Header.Get("X-Invocation") {
					t.Errorf("cross-call state: %s versus %s", input, r.Header.Get("X-Invocation"))
				}
				fmt.Fprint(w, wrapProviderResponse(provider, predicateBody(provider, "1")))
			}))
			defer server.Close()
			model, err := provider.new(server.Client(), server.URL, "key")
			if err != nil {
				t.Fatal(err)
			}
			var wait sync.WaitGroup
			for i := range 12 {
				wait.Go(func() {
					request := predicateRequest()
					request.Model = provider.model
					request.Input.Data = fmt.Sprintf("call-%d", i)
					request.Headers = http.Header{"X-Invocation": {request.Input.Data.(string)}}
					if _, err := model.Classify(context.Background(), request); err != nil {
						t.Error(err)
					}
				})
			}
			wait.Wait()
		})
	}
}

func TestDistributionFailures(t *testing.T) {
	for _, provider := range providerFixtures() {
		t.Run(provider.name, func(t *testing.T) {
			levelField, unknownLevel, duplicateLevel := `"2":`, `"3":`, `"0":`
			if provider.name == "openai" {
				levelField, unknownLevel, duplicateLevel = `"value":2`, `"value":3`, `"value":0`
			}
			for _, test := range []struct {
				name string
				from string
				to   string
			}{
				{"missing distributions", `"probabilities":`, `"missing":`},
				{"wrong choice maximum", `"choice":"billing"`, `"choice":"other"`},
				{"incorrect score", `"score":1.1000`, `"score":1.8`},
				{"quoted score", `"score":1.1000`, `"score":"1.1000"`},
				{"missing score", `"score":1.1000`, `"other_score":1.1000`},
				{"null score", `"score":1.1000`, `"score":null`},
				{"quoted distribution probability", `:0.2`, `:"0.2"`},
				{"null distribution probability", `:0.2`, `:null`},
				{"boolean distribution probability", `:0.2`, `:false`},
				{"missing chosen value", `"choice":"billing"`, `"other_choice":"billing"`},
				{"unknown score level", levelField, unknownLevel},
				{"duplicate score level", levelField, duplicateLevel},
				{"wrong kind", `"type":"score"`, `"type":"choice"`},
				{"unknown choice", `"billing"`, `"unknown"`},
				{"invalid confidence", `"confidence":0.6000`, `"confidence":"0.6"`},
			} {
				t.Run(test.name, func(t *testing.T) {
					body := strings.ReplaceAll(mixedBody(provider), test.from, test.to)
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						fmt.Fprint(w, wrapProviderResponse(provider, body))
					}))
					defer server.Close()
					model, err := provider.new(server.Client(), server.URL, "key")
					if err != nil {
						t.Fatal(err)
					}
					response, err := model.Classify(context.Background(), mixedRequest(provider.model))
					var contract *classifier.ResponseError
					if !errors.As(err, &contract) || len(response.Answers) != 0 {
						t.Fatalf("invalid distribution accepted: %+v, %v", response, err)
					}
					// Duplicate JSON keys make the whole body ambiguous, so usage
					// cannot be recovered from that System One response.
					if provider.name == "openai" || test.name != "duplicate score level" {
						if n, ok := gai.InputTokens(response.UsageMetadata); !ok || n != 123 {
							t.Fatal("trustworthy usage lost on malformed answer")
						}
						fields, ok := response.ExtraFields[classifier.ExtraFieldWireFields].(map[string]json.RawMessage)
						if !ok || string(fields["trace"]) != "9007199254740993" {
							t.Fatal("invocation metadata lost on malformed answer")
						}
						if _, ok := fields["answers"]; ok {
							t.Fatal("answers leaked into invocation metadata")
						}
					}
				})
			}
		})
	}
}

func TestEndpointRestrictions(t *testing.T) {
	for _, provider := range providerFixtures() {
		t.Run(provider.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				fmt.Fprint(w, wrapProviderResponse(provider, predicateBody(provider, "1")))
			}))
			defer server.Close()
			model, err := provider.new(server.Client(), server.URL, "key")
			if err != nil {
				t.Fatal(err)
			}
			request := predicateRequest()
			request.Model = provider.model
			request.Questions[0].Instructions = nil
			request.Questions[0].Predicate.Criteria = &classifier.PredicateCriteria{True: "refund requested"}
			_, err = model.Classify(context.Background(), request)
			if provider.name == "perplexity" {
				if err != nil || calls.Load() != 1 {
					t.Fatalf("criteria-only predicate rejected: %v", err)
				}
			} else if err == nil || calls.Load() != 0 {
				t.Fatalf("missing required instructions reached API: %v", err)
			}
			if provider.name != "openai" {
				before := calls.Load()
				request.Questions = []classifier.Question{{Name: "choice", Instructions: "Choose", Choice: &classifier.Choice{Options: []classifier.ChoiceOption{{Value: classifier.BooleanValue(false)}, {Value: classifier.StringValue("false")}}}}}
				if _, err := model.Classify(context.Background(), request); err == nil || calls.Load() != before {
					t.Fatalf("boolean choice was silently coerced: %v", err)
				}
			}
		})
	}
}

func TestCloudflareExtensions(t *testing.T) {
	t.Run("audio video and media-only state", func(t *testing.T) {
		var captured map[string]json.RawMessage
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/accounts/account/ai/run/@cf/cloudflare/clef-omni" {
				t.Errorf("wrong model route: %s", r.URL.Path)
			}
			if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
				t.Error(err)
			}
			fmt.Fprint(w, `{"success":true,"result":{"model":"clef-omni","answers":{"refund":{"type":"noul","noul":1}},"usage":{"input_tokens":42,"output_tokens":0}}}`)
		}))
		defer server.Close()
		model, err := classifier.NewCloudflare(server.Client(), server.URL, "account", "key")
		if err != nil {
			t.Fatal(err)
		}
		request := predicateRequest()
		request.Model = "@cf/cloudflare/clef-omni"
		request.Input = classifier.Input{Media: []gai.Block{
			gai.AudioBlock([]byte("audio"), "audio/wav"),
			{BlockType: gai.Content, ModalityType: gai.Video, MimeType: "video/mp4", Content: gai.Str("dmlkZW8=")},
		}}
		if _, err := model.Classify(context.Background(), request); err != nil {
			t.Fatal(err)
		}
		if string(captured["model"]) != `"clef-omni"` || string(captured["state"]) != `""` || len(captured["audio"]) == 0 || len(captured["videos"]) == 0 {
			t.Fatalf("incorrect extensions: %v", captured)
		}
	})
	t.Run("failed success envelope", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, `{"success":false,"result":null,"errors":[{"code":1000,"message":"model unavailable"}]}`)
		}))
		defer server.Close()
		model, err := classifier.NewCloudflare(server.Client(), server.URL, "account", "key")
		if err != nil {
			t.Fatal(err)
		}
		request := predicateRequest()
		request.Model = "@cf/cloudflare/clef"
		response, err := model.Classify(context.Background(), request)
		var apiErr *gai.ApiErr
		if !errors.As(err, &apiErr) || apiErr.Message != "model unavailable" || len(response.Answers) != 0 || response.ExtraFields[classifier.ExtraFieldEnvelope] == nil {
			t.Fatalf("failed envelope accepted or lost: %+v, %v", response, err)
		}
	})
	t.Run("path traversal rejected", func(t *testing.T) {
		if _, err := classifier.NewCloudflare(nil, "", "../account", "key"); err == nil {
			t.Fatal("accepted account traversal")
		}
		var calls atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer server.Close()
		model, err := classifier.NewCloudflare(server.Client(), server.URL, "account", "key")
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range []string{"@cf/../clef", "@cf/clef", "@cf/cloudflare/clef/extra", "@/cloudflare/clef", "@cf//clef", "@cf/cloudflare/clef%2Fextra"} {
			t.Run(name, func(t *testing.T) {
				request := predicateRequest()
				request.Model = name
				if _, err := model.Classify(t.Context(), request); err == nil {
					t.Fatal("accepted invalid model path")
				}
			})
		}
		if calls.Load() != 0 {
			t.Fatal("invalid model path reached HTTP transport")
		}
	})
	t.Run("result-only REST envelope", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, `{"result":{"model":"clef","answers":{"refund":{"type":"noul","noul":1}},"usage":{"input_tokens":1,"output_tokens":0}}}`)
		}))
		defer server.Close()
		model, err := classifier.NewCloudflare(server.Client(), server.URL, "account", "key")
		if err != nil {
			t.Fatal(err)
		}
		request := predicateRequest()
		request.Model = "@cf/cloudflare/clef"
		if _, err := model.Classify(context.Background(), request); err != nil {
			t.Fatalf("valid result-only envelope rejected: %v", err)
		}
	})
}

// Live tests exercise the same public constructors and Classify method as users.
// They make no retries, do not refresh fixtures, and skip only when live testing
// or the required credential is absent. Once enabled, API failures fail the test.
// These test API contracts, not model accuracy: any contract-valid judgment or
// refusal is acceptable, without expected labels or decision thresholds.
type liveClassifierProvider struct {
	name, keyEnv, model, baseURL string
	protocol                     gai.Provider
	router, images, auth         bool
}

func liveClassifierProviders() []liveClassifierProvider {
	return []liveClassifierProvider{
		{name: "openai", keyEnv: "OPENAI_API_KEY", model: "gpt-6-luna", protocol: classifier.ProviderOpenAI, images: true, auth: true},
		{name: "openrouter_systemone", keyEnv: "OPENROUTER_API_KEY", model: "typesafe/jev-1.13", baseURL: "https://openrouter.ai/api/v1", protocol: classifier.ProviderTypeSafe, router: true},
		{name: "openrouter_decisions_jev", keyEnv: "OPENROUTER_API_KEY", model: "typesafe/jev-1.13", baseURL: "https://openrouter.ai/api/alpha", protocol: classifier.ProviderPerplexity, router: true},
		{name: "openrouter_decisions_perplexity", keyEnv: "OPENROUTER_API_KEY", model: "perplexity/pplx-decider-v1.1-27b", baseURL: "https://openrouter.ai/api/alpha", protocol: classifier.ProviderPerplexity, router: true, images: true, auth: true},
		{name: "cloudflare_clef", keyEnv: "CLOUDFLARE_API_TOKEN", model: "@cf/cloudflare/clef", protocol: classifier.ProviderCloudflare, images: true, auth: true},
	}
}

func requireLiveAPIKey(t testing.TB, env string) string {
	t.Helper()
	if os.Getenv("LIVE_TESTS") == "" {
		t.Skip("LIVE_TESTS not set")
	}
	key := os.Getenv(env)
	if key == "" {
		t.Skipf("%s not set", env)
	}
	return key
}

func newLiveClassifier(t *testing.T, provider liveClassifierProvider) classifier.Classifier {
	t.Helper()
	key := requireLiveAPIKey(t, provider.keyEnv)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	t.Cleanup(transport.CloseIdleConnections)
	client := &http.Client{
		Transport: transport, Timeout: time.Minute,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	var model classifier.Classifier
	var err error
	switch provider.protocol {
	case classifier.ProviderOpenAI:
		model, err = classifier.NewOpenAI(client, provider.baseURL, key)
	case classifier.ProviderTypeSafe:
		model, err = classifier.NewTypeSafe(client, provider.baseURL, key)
	case classifier.ProviderPerplexity:
		model, err = classifier.NewPerplexity(client, provider.baseURL, key)
	case classifier.ProviderCloudflare:
		account := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
		if account == "" {
			t.Fatal("CLOUDFLARE_ACCOUNT_ID is required when Cloudflare live tests are enabled")
		}
		model, err = classifier.NewCloudflare(client, provider.baseURL, account, key)
	default:
		t.Fatalf("unknown test protocol %q", provider.protocol)
	}
	if err != nil {
		t.Fatalf("construct classifier: %v", err)
	}
	return model
}

// requireLiveClassification avoids logging API bodies, request URLs, or headers:
// a failed service could echo authentication data in its error response.
func requireLiveClassification(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		return
	}
	var api *gai.ApiErr
	var contract *classifier.ResponseError
	switch {
	case errors.As(err, &api):
		t.Fatalf("Classify: API status=%d kind=%s (body withheld)", api.StatusCode, api.Kind)
	case errors.As(err, &contract):
		t.Fatalf("Classify: %v", contract)
	case errors.Is(err, context.DeadlineExceeded):
		t.Fatal("Classify: deadline exceeded")
	default:
		t.Fatalf("Classify: %T (details withheld)", err)
	}
}

func liveNumber(t *testing.T, number json.Number, min, max float64) float64 {
	t.Helper()
	value, err := number.Float64()
	if !json.Valid([]byte(number)) || err != nil || math.IsNaN(value) || math.IsInf(value, 0) || value < min || value > max {
		t.Fatalf("number %q is not finite JSON in [%g, %g]", number, min, max)
	}
	return value
}

func assertLiveDistribution(t *testing.T, probabilities []json.Number) {
	t.Helper()
	var sum float64
	for _, probability := range probabilities {
		sum += liveNumber(t, probability, 0, 1)
	}
	// The public contract permits 0.00005 rounding error per entry.
	if math.Abs(sum-1) > float64(len(probabilities))*0.00005+1e-12 {
		t.Errorf("distribution sums to %.12g, want 1 within documented rounding tolerance", sum)
	}
}

func assertLiveMetadata(t *testing.T, provider liveClassifierProvider, response classifier.Response) {
	t.Helper()
	if response.Model == "" {
		t.Error("missing resolved model")
	}
	if got := response.ExtraFields[classifier.ExtraFieldProvider]; got != provider.protocol {
		t.Errorf("protocol metadata = %v, want %v", got, provider.protocol)
	}
	if headers, ok := response.ExtraFields[classifier.ExtraFieldHeaders].(http.Header); !ok || len(headers) == 0 {
		t.Error("missing HTTP response headers")
	}
	input, inputOK := gai.InputTokens(response.UsageMetadata)
	output, outputOK := gai.OutputTokens(response.UsageMetadata)
	if !inputOK || input <= 0 || !outputOK || output < 0 {
		t.Errorf("usage input=(%d, %t) output=(%d, %t), want positive input and nonnegative output", input, inputOK, output, outputOK)
	}
	usage, ok := response.ExtraFields[classifier.ExtraFieldUsage].(map[string]json.RawMessage)
	if !ok {
		t.Fatal("missing native usage object")
	}
	for field, count := range map[string]int{"input_tokens": input, "output_tokens": output} {
		var native int
		if err := json.Unmarshal(usage[field], &native); err != nil || native != count {
			t.Errorf("native %s differs from common usage count %d", field, count)
		}
	}
	if provider.router {
		// Cost is an OpenRouter extension, not a common token metric.
		liveNumber(t, json.Number(usage["cost"]), 0, math.MaxFloat64)
		fields, ok := response.ExtraFields[classifier.ExtraFieldWireFields].(map[string]json.RawMessage)
		var id string
		if !ok || json.Unmarshal(fields["id"], &id) != nil || id == "" {
			t.Error("missing OpenRouter invocation ID")
		}
	}
	if provider.protocol == classifier.ProviderCloudflare {
		fields, ok := response.ExtraFields[classifier.ExtraFieldEnvelope].(map[string]json.RawMessage)
		var success bool
		if !ok || json.Unmarshal(fields["success"], &success) != nil || !success {
			t.Error("missing successful Cloudflare envelope")
		}
		if _, present := fields["result"]; present {
			t.Error("Cloudflare result leaked into envelope metadata")
		}
	}
	t.Logf("resolved model=%s input_tokens=%d output_tokens=%d", response.Model, input, output)
}

// assertLiveAnswerShapes checks the public contract independently of the
// implementation's validator. It does not grade the model's judgment.
func assertLiveAnswerShapes(t *testing.T, questions []classifier.Question, answers []classifier.Answer) {
	t.Helper()
	if len(answers) != len(questions) {
		t.Fatalf("got %d answers, want %d", len(answers), len(questions))
	}
	for i, question := range questions {
		answer := answers[i]
		if answer.Name != question.Name {
			t.Fatalf("answer %d: name=%q, want %q", i, answer.Name, question.Name)
		}
		if answer.Refused {
			if answer.Predicate != nil || answer.Choice != nil || answer.Score != nil || answer.Confidence != nil {
				t.Errorf("%s: refusal contains a judgment or confidence", question.Name)
			}
			continue
		}
		if answer.Confidence != nil {
			liveNumber(t, *answer.Confidence, 0, 1)
		}
		switch {
		case question.Predicate != nil:
			if answer.Predicate == nil || answer.Choice != nil || answer.Score != nil {
				t.Fatalf("%s: wrong answer variant", question.Name)
			}
			liveNumber(t, answer.Predicate.Probability, 0, 1)
		case question.Choice != nil:
			if answer.Choice == nil || answer.Predicate != nil || answer.Score != nil {
				t.Fatalf("%s: wrong answer variant", question.Name)
			}
			if len(answer.Choice.Probabilities) != len(question.Choice.Options) {
				t.Fatalf("%s: incomplete choice distribution", question.Name)
			}
			var probabilities []json.Number
			selected := false
			for j, p := range answer.Choice.Probabilities {
				if !reflect.DeepEqual(p.Value, question.Choice.Options[j].Value) {
					t.Errorf("%s: outcome %d does not match request order or value type", question.Name, j)
				}
				selected = selected || reflect.DeepEqual(p.Value, answer.Choice.Value)
				probabilities = append(probabilities, p.Probability)
			}
			if !selected {
				t.Errorf("%s: selected value is not a declared option", question.Name)
			}
			assertLiveDistribution(t, probabilities)
		case question.Score != nil:
			if answer.Score == nil || answer.Predicate != nil || answer.Choice != nil {
				t.Fatalf("%s: wrong answer variant", question.Name)
			}
			if len(answer.Score.Probabilities) != len(question.Score.Levels) {
				t.Fatalf("%s: incomplete score distribution", question.Name)
			}
			assertLiveDistribution(t, answer.Score.Probabilities)
			score := liveNumber(t, answer.Score.Score, 0, float64(len(question.Score.Levels)-1))
			var mean, roundingWeight float64
			for level, probability := range answer.Score.Probabilities {
				mean += float64(level) * liveNumber(t, probability, 0, 1)
				roundingWeight += float64(level)
			}
			if math.Abs(score-mean) > (1+roundingWeight)*0.00005+1e-12 {
				t.Errorf("%s: score=%g, probability-weighted mean=%g", question.Name, score, mean)
			}
		}
	}
}

func TestLiveClassifier(t *testing.T) {
	t.Run("MixedQuestions", func(t *testing.T) {
		for _, provider := range liveClassifierProviders() {
			t.Run(provider.name, func(t *testing.T) {
				model := newLiveClassifier(t, provider)
				const ticket = "Please refund the duplicate charge on my invoice. Service is working normally."
				// Same task, distinct API input forms; no expected judgment.
				for _, test := range []struct {
					name  string
					input any
				}{
					{"text", ticket},
					{"structured", map[string]any{"ticket": ticket, "history": []any{"first report", map[string]any{"resolved": false, "assigned": nil}}}},
				} {
					t.Run(test.name, func(t *testing.T) {
						request := classifier.Request{
							Model: provider.model, Input: classifier.Input{Data: test.input},
							Questions: []classifier.Question{
								{Name: "z_refund", Instructions: "Does the customer explicitly ask for a refund or money back?", Predicate: &classifier.Predicate{}},
								{Name: "a_team", Instructions: []any{"Which team should handle this ticket?", map[string]any{"rule": "Choose based on the customer's stated problem."}}, Choice: &classifier.Choice{Options: []classifier.ChoiceOption{
									{Value: classifier.StringValue("shipping"), Description: "Parcel delivery and tracking."},
									{Value: classifier.StringValue("billing"), Description: "Invoices, duplicate charges and refunds."},
									{Value: classifier.StringValue("account"), Description: "Passwords, login and account access."},
								}}},
								{Name: "m_impact", Instructions: "Rate ability to use the service, not financial impact. Use the stated availability and workaround status.", Score: &classifier.Score{Levels: []classifier.ScoreLevel{
									{Label: "unaffected", Description: "Every feature of the service works normally."},
									{Label: "workaround", Description: "A feature is broken, but an alternative way works."},
									{Label: "blocked", Description: "Cannot use any part of the service and no workaround exists."},
								}}},
							},
						}
						if provider.protocol == classifier.ProviderOpenAI {
							// Exercise the value union in the existing request. Any
							// declared selection is valid; false and "false" must
							// remain distinct entries in the returned distribution.
							request.Questions = append(request.Questions, classifier.Question{
								Name: "typed_choice", Instructions: "Does the customer request money back? Choose true for yes, false for no, or the string 'false' if uncertain.",
								Choice: &classifier.Choice{Options: []classifier.ChoiceOption{
									{Value: classifier.BooleanValue(true)}, {Value: classifier.BooleanValue(false)}, {Value: classifier.StringValue("false")},
								}},
							})
						}
						before, err := json.Marshal(request)
						if err != nil {
							t.Fatal(err)
						}
						ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
						defer cancel()
						response, err := model.Classify(ctx, request)
						requireLiveClassification(t, err)
						assertLiveMetadata(t, provider, response)
						assertLiveAnswerShapes(t, request.Questions, response.Answers)
						after, err := json.Marshal(request)
						if err != nil || string(before) != string(after) {
							t.Error("Classify mutated the request")
						}
					})
				}
			})
		}
	})
	t.Run("Images", func(t *testing.T) {
		for _, provider := range liveClassifierProviders() {
			if !provider.images {
				continue // Explicit text-only profiles, not a model-name heuristic.
			}
			t.Run(provider.name, func(t *testing.T) {
				model := newLiveClassifier(t, provider)
				// This checks accepted PNG transport and the response schema,
				// not whether the model correctly recognizes the color.
				picture := image.NewRGBA(image.Rect(0, 0, 256, 256))
				draw.Draw(picture, picture.Bounds(), image.NewUniform(color.RGBA{R: 255, A: 255}), image.Point{}, draw.Src)
				var buffer bytes.Buffer
				if err := png.Encode(&buffer, picture); err != nil {
					t.Fatal(err)
				}
				request := classifier.Request{
					Model: provider.model,
					Input: classifier.Input{Media: []gai.Block{{BlockType: gai.Content, ModalityType: gai.Image, MimeType: "image/png", Content: gai.Str(base64.StdEncoding.EncodeToString(buffer.Bytes()))}}},
					Questions: []classifier.Question{{Name: "color", Instructions: "Which color covers most of the attached image?", Choice: &classifier.Choice{Options: []classifier.ChoiceOption{
						{Value: classifier.StringValue("green")}, {Value: classifier.StringValue("red")}, {Value: classifier.StringValue("blue")},
					}}}},
				}
				ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
				defer cancel()
				response, err := model.Classify(ctx, request)
				requireLiveClassification(t, err)
				assertLiveMetadata(t, provider, response)
				assertLiveAnswerShapes(t, request.Questions, response.Answers)
			})
		}
	})
	t.Run("Authentication", func(t *testing.T) {
		for _, provider := range liveClassifierProviders() {
			if !provider.auth {
				continue // One credential-rejection probe per service.
			}
			t.Run(provider.name, func(t *testing.T) {
				model := newLiveClassifier(t, provider)
				request := predicateRequest()
				request.Model = provider.model
				// Deliberately override the borrowed client's valid credential for
				// this call only. No credential is embedded in the fixture.
				request.Headers = http.Header{"Authorization": {"Bearer gai-classifier-invalid-test-token"}}
				ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
				defer cancel()
				response, err := model.Classify(ctx, request)
				var api *gai.ApiErr
				if !errors.As(err, &api) {
					t.Fatalf("invalid credential: got %T, want *gai.ApiErr", err)
				}
				if api.StatusCode != http.StatusUnauthorized && api.StatusCode != http.StatusForbidden {
					t.Errorf("authentication status=%d, want 401 or 403", api.StatusCode)
				}
				if api.Provider != provider.protocol || api.RawBody == "" {
					t.Error("API error lost protocol identity or raw body")
				}
				if len(response.Answers) != 0 {
					t.Error("authentication failure returned usable answers")
				}
			})
		}
	})
}

func TestInFlightCancellation(t *testing.T) {
	for _, provider := range providerFixtures() {
		t.Run(provider.name, func(t *testing.T) {
			started := make(chan struct{})
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				close(started)
				select {
				case <-r.Context().Done():
				case <-release:
				}
			}))
			defer server.Close()
			defer close(release)
			model, err := provider.new(server.Client(), server.URL, "key")
			if err != nil {
				t.Fatal(err)
			}
			request := predicateRequest()
			request.Model = provider.model
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			go func() {
				_, err := model.Classify(ctx, request)
				done <- err
			}()
			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("request never started")
			}
			cancel()
			select {
			case err := <-done:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancellation not preserved: %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("Classify did not honor cancellation")
			}
		})
	}
}
