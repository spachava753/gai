package gai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	openai "github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
	"google.golang.org/genai"
)

func TestRequestHeaders(t *testing.T) {
	t.Run("CaseInsensitiveReplacement", func(t *testing.T) {
		request := &http.Request{Header: make(http.Header)}
		request.Header.Set("X-Session", "old")
		request.Header.Set("X-Remove", "old")
		headers := http.Header{"x-session": {"one", "two"}, "x-remove": nil}
		setRequestHeaders(request, headers)
		headers["x-session"][0] = "changed"
		if got := request.Header.Values("X-SESSION"); !reflect.DeepEqual(got, []string{"one", "two"}) {
			t.Fatalf("session headers = %v", got)
		}
		if _, present := request.Header["X-Remove"]; present {
			t.Fatal("header was not removed")
		}
	})
	t.Run("Providers", func(t *testing.T) {
		constructors := []struct {
			name string
			new  func(*testing.T, *httptest.Server) Generator
		}{
			{"OpenAI", func(t *testing.T, s *httptest.Server) Generator {
				g, err := NewOpenAiGenerator(s.Client(), s.URL, "test-key")
				if err != nil {
					t.Fatal(err)
				}
				return g
			}},
			{"Cerebras", func(t *testing.T, s *httptest.Server) Generator {
				g, err := NewCerebrasGenerator(s.Client(), s.URL, "test-key")
				if err != nil {
					t.Fatal(err)
				}
				return g
			}},
			{"DeepSeek", func(t *testing.T, s *httptest.Server) Generator {
				g, err := NewDeepSeekGenerator(s.Client(), s.URL, "test-key")
				if err != nil {
					t.Fatal(err)
				}
				return g
			}},
			{"ZAI", func(t *testing.T, s *httptest.Server) Generator {
				g, err := NewZaiGenerator(s.Client(), s.URL, "test-key")
				if err != nil {
					t.Fatal(err)
				}
				return g
			}},
			{"Moonshot", func(t *testing.T, s *httptest.Server) Generator {
				g, err := NewMoonshotGenerator(s.Client(), s.URL, "test-key")
				if err != nil {
					t.Fatal(err)
				}
				return g
			}},
			{"OpenRouter", func(t *testing.T, s *httptest.Server) Generator {
				g, err := NewOpenRouterGenerator(s.Client(), s.URL, "test-key")
				if err != nil {
					t.Fatal(err)
				}
				return g
			}},
			{"Anthropic", func(t *testing.T, s *httptest.Server) Generator {
				client := anthropic.NewClient(anthropicoption.WithoutEnvironmentDefaults(), anthropicoption.WithAPIKey("test-key"), anthropicoption.WithBaseURL(s.URL), anthropicoption.WithHTTPClient(s.Client()), anthropicoption.WithMaxRetries(0))
				return NewAnthropicGenerator(&client.Messages)
			}},
			{"Responses", func(t *testing.T, s *httptest.Server) Generator {
				client := openai.NewClient(openaioption.WithAPIKey("test-key"), openaioption.WithBaseURL(s.URL), openaioption.WithHTTPClient(s.Client()), openaioption.WithMaxRetries(0))
				return NewResponsesGenerator(&client.Responses)
			}},
			{"Gemini", func(t *testing.T, s *httptest.Server) Generator {
				client, err := genai.NewClient(t.Context(), &genai.ClientConfig{APIKey: "test-key", Backend: genai.BackendGeminiAPI, HTTPClient: s.Client(), HTTPOptions: genai.HTTPOptions{BaseURL: s.URL, Headers: http.Header{"X-Default": {"default"}, "X-Replace": {"old"}}}})
				if err != nil {
					t.Fatal(err)
				}
				return NewGeminiGenerator(client)
			}},
		}
		for _, constructor := range constructors {
			t.Run(constructor.name, func(t *testing.T) {
				captured := make(chan http.Header, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if _, ok := body["headers"]; ok {
						t.Error("headers leaked into provider body")
					}
					captured <- r.Header.Clone()
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusBadRequest)
					fmt.Fprint(w, `{"error":{"message":"test failure","type":"invalid_request_error","code":400,"status":"INVALID_ARGUMENT"}}`)
				}))
				defer server.Close()
				g := constructor.new(t, server)
				operations := []struct {
					name string
					run  func(GenerationRequest)
				}{
					{"Generate", func(r GenerationRequest) {
						if _, err := g.Generate(t.Context(), r); err == nil {
							t.Error("expected HTTP error")
						}
					}},
					{"Stream", func(r GenerationRequest) {
						for range g.(StreamingGenerator).Stream(t.Context(), r) {
						}
					}},
				}
				if counter, ok := g.(TokenCounter); ok && constructor.name != "OpenAI" {
					operations = append(operations, struct {
						name string
						run  func(GenerationRequest)
					}{"Count", func(r GenerationRequest) {
						if _, err := counter.Count(t.Context(), r); err == nil {
							t.Error("expected HTTP error")
						}
					}})
				}
				for _, operation := range operations {
					t.Run(operation.name, func(t *testing.T) {
						for _, headers := range []http.Header{
							{"X-Opencode-Session": {"conversation"}, "X-Replace": {"one", "two"}, "Authorization": {"Bearer override"}, "X-Api-Key": {"override"}, "X-Goog-Api-Key": {"override"}},
							nil,
						} {
							before := headers.Clone()
							operation.run(GenerationRequest{Model: "test-model", Headers: headers, Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}})
							var got http.Header
							select {
							case got = <-captured:
							default:
								t.Fatal("request did not reach server")
							}
							if headers != nil {
								for name, want := range headers {
									if constructor.name == "Gemini" && name == "X-Goog-Api-Key" {
										want = []string{"test-key"}
									}
									if !reflect.DeepEqual(got.Values(name), want) {
										t.Errorf("%s = %v, want %v", name, got.Values(name), want)
									}
								}
							} else if got.Get("X-Opencode-Session") != "" || got.Get("Authorization") == "Bearer override" || got.Get("X-Api-Key") == "override" || got.Get("X-Goog-Api-Key") == "override" {
								t.Errorf("request headers leaked into next call: %v", got)
							}
							if constructor.name == "Gemini" {
								if got.Get("X-Default") != "default" {
									t.Error("lost client default header")
								}
								if headers == nil && got.Get("X-Replace") != "old" {
									t.Error("mutated client default header")
								}
							}
							if !reflect.DeepEqual(headers, before) {
								t.Fatal("mutated caller headers")
							}
						}
					})
				}
			})
		}
	})
	t.Run("ConcurrentSessions", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body struct {
				Model string `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if got := r.Header.Get("X-Opencode-Session"); got != body.Model {
				t.Errorf("session = %q, model = %q", got, body.Model)
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
		}))
		defer server.Close()
		g, err := NewOpenAiGenerator(server.Client(), server.URL, "test-key")
		if err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		for i := range 10 {
			wg.Go(func() {
				id := fmt.Sprintf("session-%d", i)
				_, err := g.Generate(t.Context(), GenerationRequest{Model: id, Headers: http.Header{"X-Opencode-Session": {id}}, Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}})
				if err != nil {
					t.Error(err)
				}
			})
		}
		wg.Wait()
	})
}
