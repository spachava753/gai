package gai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	oai "github.com/openai/openai-go/v3"
	oaissestream "github.com/openai/openai-go/v3/packages/ssestream"
	wire "github.com/spachava753/gai/internal/openai"
)

const openAIStockInstructions = `You are a helpful assistant that returns the price of a stock and nothing else.
Only output the price, like
<example>
435.56
</example>
<example>
3235.55
</example>
`

const openAIStockComparisonInstructions = `You are a helpful assistant that compares the price of two stocks and returns the ticker of whichever is greater.
Only mentioned the ticker and nothing else.
Only output the price, like
<example>
User: Which one is more expensive? Apple or NVidia?
Assistant: calls get_stock_price for both Apple and Nvidia
Tool Result: Apple: 123.45; Nvidia: 345.65
Assistant: Nvidia
</example>
`

func TestChatCompletionRequestControls(t *testing.T) {
	t.Run("NativeAudioOptions", func(t *testing.T) {
		modalities := []Modality{Text, Audio}
		options := NewGenerationOptions(WithOpenAIOutputModalities(modalities...), WithOpenAIAudioConfig(OpenAIAudioConfig{VoiceName: "alloy", Format: "wav"}))
		modalities[0] = Video
		request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: options}
		for _, stream := range []bool{false, true} {
			params, parsed, err := openAIRequest(request, stream)
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(params)
			if err != nil {
				t.Fatal(err)
			}
			var body map[string]any
			if err := json.Unmarshal(data, &body); err != nil {
				t.Fatal(err)
			}
			if fmt.Sprint(body["modalities"]) != "[text audio]" || body["audio"].(map[string]any)["voice"] != "alloy" || body["audio"].(map[string]any)["format"] != "wav" || parsed.AudioConfig.Format != "wav" {
				t.Fatalf("audio configuration lost: %s", data)
			}
		}
		for _, key := range []string{OpenAIGenerationOptionOutputModalities, OpenAIGenerationOptionAudioConfig} {
			request.Options = GenerationOptions{key: "wrong"}
			if _, _, err := openAIRequest(request, false); err == nil {
				t.Fatalf("accepted invalid %s", key)
			}
		}
		request.Options = GenerationOptions{"output_modalities": []Modality{Audio}, "audio_config": OpenAIAudioConfig{VoiceName: "alloy", Format: "wav"}}
		params, parsed, err := openAIRequest(request, false)
		if err != nil {
			t.Fatal(err)
		}
		data, _ := json.Marshal(params)
		var body map[string]any
		_ = json.Unmarshal(data, &body)
		if body["modalities"] != nil || body["audio"] != nil || parsed.AudioConfig.Format != "" {
			t.Fatalf("legacy generic keys still consumed: %s", data)
		}
	})
	t.Run("CommonOptions", func(t *testing.T) {
		providers := []struct {
			name string
			new  func(*http.Client, string, string) (Generator, error)
		}{
			{"openai", func(c *http.Client, u, k string) (Generator, error) { return NewOpenAiGenerator(c, u, k) }},
			{"cerebras", func(c *http.Client, u, k string) (Generator, error) { return NewCerebrasGenerator(c, u, k) }},
			{"openrouter", func(c *http.Client, u, k string) (Generator, error) { return NewOpenRouterGenerator(c, u, k) }},
			{"zai", func(c *http.Client, u, k string) (Generator, error) { return NewZaiGenerator(c, u, k) }},
			{"deepseek", func(c *http.Client, u, k string) (Generator, error) { return NewDeepSeekGenerator(c, u, k) }},
			{"moonshot", func(c *http.Client, u, k string) (Generator, error) { return NewMoonshotGenerator(c, u, k) }},
		}
		for _, provider := range providers {
			for _, streaming := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%t", provider.name, streaming), func(t *testing.T) {
					var received map[string]any
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
							t.Error(err)
						}
						if streaming {
							w.Header().Set("Content-Type", "text/event-stream")
							fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
						} else {
							w.Header().Set("Content-Type", "application/json")
							fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
						}
					}))
					defer server.Close()
					generator, err := provider.new(server.Client(), server.URL, "test-key")
					if err != nil {
						t.Fatal(err)
					}
					invoke := func(options GenerationOptions) {
						t.Helper()
						before, _ := json.Marshal(options)
						received = nil
						request := GenerationRequest{Model: "test-model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: options}
						if streaming {
							for chunk := range generator.(StreamingGenerator).Stream(t.Context(), request) {
								if chunk.Err != nil {
									t.Fatal(chunk.Err)
								}
							}
						} else if _, err := generator.Generate(t.Context(), request); err != nil {
							t.Fatal(err)
						}
						after, _ := json.Marshal(options)
						if !bytes.Equal(before, after) {
							t.Fatal("mutated caller options")
						}
					}
					invoke(nil)
					for _, field := range []string{"logit_bias", "logprobs", "top_logprobs", "parallel_tool_calls", "seed", "min_p", "top_a", "repetition_penalty"} {
						if _, present := received[field]; present {
							t.Errorf("omitted option sent: %s", field)
						}
					}
					for _, zero := range []bool{false, true} {
						bias, count, seed, minP, topA, repetition := 123.5, 30, -7, -0.5, 1.5, -2.0
						if zero {
							bias, count, seed, minP, topA, repetition = 0, 0, 0, 0, 0, 0
						}
						options := NewGenerationOptions(WithLogitBias(map[string]float64{"42": bias}), WithLogprobs(!zero), WithTopLogprobs(count), WithParallelToolCalls(!zero), WithSeed(seed), WithOpenRouterMinP(minP), WithOpenRouterTopA(topA), WithOpenRouterRepetitionPenalty(repetition))
						invoke(options)
						if received["logprobs"] != !zero || received["top_logprobs"] != float64(count) || received["parallel_tool_calls"] != !zero || received["seed"] != float64(seed) || !reflect.DeepEqual(received["logit_bias"], map[string]any{"42": bias}) {
							t.Fatalf("common options = %v", received)
						}
						for field, want := range map[string]float64{"min_p": minP, "top_a": topA, "repetition_penalty": repetition} {
							if provider.name == "openrouter" {
								if received[field] != want {
									t.Errorf("%s = %v, want %v", field, received[field], want)
								}
							} else if _, present := received[field]; present {
								t.Errorf("unsupported field sent: %s", field)
							}
						}
						explicit := map[string]json.RawMessage{"logit_bias": json.RawMessage(`{"99":-1}`), "logprobs": json.RawMessage(`false`), "top_logprobs": json.RawMessage(`2`), "parallel_tool_calls": json.RawMessage(`false`), "seed": json.RawMessage(`13`)}
						if provider.name == "openrouter" {
							explicit["min_p"] = json.RawMessage(`0.4`)
							explicit["top_a"] = json.RawMessage(`0.3`)
							explicit["repetition_penalty"] = json.RawMessage(`1.2`)
						}
						WithOpenAIExtraBody(explicit)(options)
						invoke(options)
						for field, raw := range explicit {
							var want any
							_ = json.Unmarshal(raw, &want)
							if !reflect.DeepEqual(received[field], want) {
								t.Errorf("override %s = %v, want %v", field, received[field], want)
							}
						}
					}
					// A count does not silently enable logprobs or rewrite explicit false.
					invoke(NewGenerationOptions(WithTopLogprobs(2)))
					if _, present := received["logprobs"]; present {
						t.Fatal("top-logprobs implicitly enabled logprobs")
					}
					invoke(NewGenerationOptions(WithTopLogprobs(2), WithLogprobs(false)))
					if received["logprobs"] != false {
						t.Fatal("explicit logprobs false ignored")
					}
					keys := []string{GenerationOptionLogitBias, GenerationOptionLogprobs, GenerationOptionTopLogprobs, GenerationOptionParallelToolCalls, GenerationOptionSeed}
					if provider.name == "openrouter" {
						keys = append(keys, OpenRouterGenerationOptionMinP, OpenRouterGenerationOptionTopA, OpenRouterGenerationOptionRepetitionPenalty)
					}
					for _, key := range keys {
						received = nil
						request := GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: GenerationOptions{key: struct{}{}}}
						var err error
						if streaming {
							for chunk := range generator.(StreamingGenerator).Stream(t.Context(), request) {
								err = chunk.Err
							}
						} else {
							_, err = generator.Generate(t.Context(), request)
						}
						var invalid *InvalidParameterErr
						if !errors.As(err, &invalid) || invalid.Parameter != key || received != nil {
							t.Fatalf("invalid %s: err=%v request=%v", key, err, received)
						}
					}
				})
			}
		}
	})
	for _, stream := range []bool{false, true} {
		request := GenerationRequest{Model: "model", SafetyIdentifier: "opaque-user", PromptCacheKey: "prefix", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: NewGenerationOptions(WithThinkingBudget(2048), WithOpenAIPromptCacheRetention("24h"), WithOpenAIPromptCacheOptions(json.RawMessage(`{"mode":"implicit","ttl":"30m"}`)), WithOpenAIResponseFormat(json.RawMessage(`{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"}}}`)))}
		params, _, err := openAIRequest(request, stream)
		if err != nil {
			t.Fatal(err)
		}
		data, err := json.Marshal(params)
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		if err := json.Unmarshal(data, &body); err != nil {
			t.Fatal(err)
		}
		if body["safety_identifier"] != "opaque-user" || body["prompt_cache_key"] != "prefix" || body["prompt_cache_retention"] != "24h" || body["prompt_cache_options"].(map[string]any)["ttl"] != "30m" || body["response_format"].(map[string]any)["type"] != "json_schema" {
			t.Fatalf("body: %s", data)
		}
		if body["reasoning_effort"] != nil || body["thinking_budget"] != nil {
			t.Fatal("numeric thinking budget treated as effort")
		}
		request.SafetyIdentifier = ""
		request.PromptCacheKey = ""
		request.Options = nil
		params, _, err = openAIRequest(request, stream)
		if err != nil {
			t.Fatal(err)
		}
		data, _ = json.Marshal(params)
		body = nil
		_ = json.Unmarshal(data, &body)
		if _, ok := body["safety_identifier"]; ok {
			t.Fatal("empty safety identifier sent")
		}
		if _, ok := body["prompt_cache_key"]; ok {
			t.Fatal("empty cache key sent")
		}
	}
	raw := json.RawMessage(`{"type":"json_object"}`)
	fields := map[string]json.RawMessage{"other": json.RawMessage(`true`)}
	opts := NewGenerationOptions(WithOpenAIExtraBody(fields), WithOpenAIResponseFormat(raw), WithLogprobs(true))
	raw[0] = 'x'
	if len(fields) != 1 || string(opts[OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)["response_format"]) != `{"type":"json_object"}` {
		t.Fatal("option mutated or retained mutable input")
	}
	request := GenerationRequest{Model: "model", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: NewGenerationOptions(WithOpenAIResponseFormat(json.RawMessage(`invalid`)))}
	if _, _, err := openAIRequest(request, false); err == nil {
		t.Fatal("invalid raw format accepted")
	}
}

func ExampleNewOpenAiGenerator() {
	generator, err := NewOpenAiGenerator(nil, "https://api.openai.com/v1", "application-supplied-key")
	if err != nil {
		panic(err)
	}
	options := NewGenerationOptions(WithMaxGenerationTokens(128), WithOpenAITokenLimitField("max_tokens"))
	_ = GenerationRequest{Model: "caller-selected-model", Options: options}
	fmt.Println(generator != nil, options[OpenAIGenerationOptionTokenLimitField])
	// Output: true max_tokens
}

func ExampleWithOpenAIAudioConfig() {
	options := NewGenerationOptions(
		WithOpenAIOutputModalities(Text, Audio),
		WithOpenAIAudioConfig(OpenAIAudioConfig{VoiceName: "alloy", Format: "wav"}),
	)
	// Supply these options to a Chat Completions model that supports audio.
	// The format also determines the MIME type of returned audio blocks.
	config := options[OpenAIGenerationOptionAudioConfig].(OpenAIAudioConfig)
	fmt.Println(options[OpenAIGenerationOptionOutputModalities])
	fmt.Println(config.VoiceName, config.Format)
	// Output:
	// [text audio]
	// alloy wav
}

func ExampleWithOpenAIExtraBody() {
	options := NewGenerationOptions(
		WithOpenAIExtraBody(map[string]json.RawMessage{"thinking": json.RawMessage(`{"type":"enabled","clear_thinking":false}`)}),
		WithOpenAIStreamUsage(false),
	)
	fields := options[OpenAIGenerationOptionExtraBody].(map[string]json.RawMessage)
	fmt.Println(string(fields["thinking"]))
	fmt.Println(options[OpenAIGenerationOptionStreamUsage])
	// Output:
	// {"type":"enabled","clear_thinking":false}
	// false
}

func TestToOpenAIMessage(t *testing.T) {
	tests := []struct {
		name    string
		msg     Message
		want    oai.ChatCompletionMessageParamUnion
		wantErr bool
	}{
		{
			name: "error: empty blocks",
			msg: Message{
				Role:   User,
				Blocks: []Block{},
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "error: nil blocks",
			msg: Message{
				Role:   User,
				Blocks: nil,
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "user message",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hello, how are you?"),
					},
				},
			},
			want:    oai.UserMessage("Hello, how are you?"),
			wantErr: false,
		},
		{
			name: "assistant message",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("I'm doing well, thank you!"),
					},
				},
			},
			want:    oai.AssistantMessage("I'm doing well, thank you!"),
			wantErr: false,
		},
		{
			name: "tool call",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						ID:           "call_123",
						BlockType:    ToolCall,
						ModalityType: Text,
						Content:      Str(`{"name": "get_weather", "parameters": {"location": "London"}}`),
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					ToolCalls: []oai.ChatCompletionMessageToolCallUnionParam{
						{
							OfFunction: &oai.ChatCompletionMessageFunctionToolCallParam{
								ID: "call_123",
								Function: oai.ChatCompletionMessageFunctionToolCallFunctionParam{
									Name:      "get_weather",
									Arguments: `{"location":"London"}`,
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "tool result",
			msg: Message{
				Role: ToolResult,
				Blocks: []Block{
					{
						ID:           "call_123",
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("The current temperature is 72°F"),
					},
				},
			},
			want:    oai.ToolMessage("The current temperature is 72°F", "call_123"),
			wantErr: false,
		},
		{
			name: "tool call with text",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str(`Let me get the weather for you:`),
					},
					{
						ID:           "call_123",
						BlockType:    ToolCall,
						ModalityType: Text,
						Content:      Str(`{"name": "get_weather", "parameters": {"location": "London"}}`),
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					Content: oai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: oai.String("Let me get the weather for you:"),
					},
					ToolCalls: []oai.ChatCompletionMessageToolCallUnionParam{
						{
							OfFunction: &oai.ChatCompletionMessageFunctionToolCallParam{
								ID: "call_123",
								Function: oai.ChatCompletionMessageFunctionToolCallFunctionParam{
									Name:      "get_weather",
									Arguments: `{"location":"London"}`,
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "error: video modality not supported",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Video,
						Content:      Str("fake-base64-data"),
					},
				},
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "error: invalid role",
			msg: Message{
				Role: 999,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hello"),
					},
				},
			},
			want:    oai.ChatCompletionMessageParamUnion{},
			wantErr: true,
		},
		{
			name: "multimodal user message with text and image",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What's in this image?"),
					},
					{
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "image/jpeg",
						Content:      Str("fake-image-base64-data"),
					},
				},
			},
			want: oai.UserMessage([]oai.ChatCompletionContentPartUnionParam{
				{
					OfText: &oai.ChatCompletionContentPartTextParam{
						Text: "What's in this image?",
					},
				},
				{
					OfImageURL: &oai.ChatCompletionContentPartImageParam{
						ImageURL: oai.ChatCompletionContentPartImageImageURLParam{
							URL: "data:image/jpeg;base64,fake-image-base64-data",
						},
					},
				},
			}),
			wantErr: false,
		},
		{
			name: "multimodal user message with text and audio",
			msg: Message{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What's in this audio?"),
					},
					{
						BlockType:    Content,
						ModalityType: Audio,
						MimeType:     "audio/wav",
						Content:      Str("fake-audio-base64-data"),
					},
				},
			},
			want: oai.UserMessage([]oai.ChatCompletionContentPartUnionParam{
				{
					OfText: &oai.ChatCompletionContentPartTextParam{
						Text: "What's in this audio?",
					},
				},
				{
					OfInputAudio: &oai.ChatCompletionContentPartInputAudioParam{
						InputAudio: oai.ChatCompletionContentPartInputAudioInputAudioParam{
							Data:   "fake-audio-base64-data",
							Format: "audio/wav",
						},
					},
				},
			}),
			wantErr: false,
		},
		{
			name: "assistant message with audio ID",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						ID:           "audio_abc123",
						BlockType:    Content,
						ModalityType: Audio,
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					Audio: oai.ChatCompletionAssistantMessageParamAudio{
						ID: "audio_abc123",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "assistant message with text and audio ID",
			msg: Message{
				Role: Assistant,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Here's my response:"),
					},
					{
						ID:           "audio_abc123",
						BlockType:    Content,
						ModalityType: Audio,
					},
				},
			},
			want: oai.ChatCompletionMessageParamUnion{
				OfAssistant: &oai.ChatCompletionAssistantMessageParam{
					Audio: oai.ChatCompletionAssistantMessageParamAudio{
						ID: "audio_abc123",
					},
					Content: oai.ChatCompletionAssistantMessageParamContentUnion{
						OfArrayOfContentParts: []oai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
							{
								OfText: &oai.ChatCompletionContentPartTextParam{
									Text: "Here's my response:",
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toOpenAIMessage(tt.msg)
			if (err != nil) != tt.wantErr {
				t.Errorf("toOpenAIMessage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				wantRole := ""
				switch {
				case tt.want.OfUser != nil:
					wantRole = "user"
				case tt.want.OfAssistant != nil:
					wantRole = "assistant"
				case tt.want.OfTool != nil:
					wantRole = "tool"
				case tt.want.OfSystem != nil:
					wantRole = "system"
				}
				if got.Role != wantRole {
					t.Errorf("role = %s, want %s", got.Role, wantRole)
				}
				if tt.want.OfTool != nil && (got.ToolCallId == nil || *got.ToolCallId != tt.want.OfTool.ToolCallID) {
					t.Error("tool call ID mismatch")
				}
				if tt.want.OfAssistant != nil {
					calls, _ := got.ToolCalls.Get()
					if len(calls) != len(tt.want.OfAssistant.ToolCalls) {
						t.Fatal("tool call count mismatch")
					}
					if len(calls) > 0 {
						want := tt.want.OfAssistant.ToolCalls[0].OfFunction
						if calls[0].Id == nil || *calls[0].Id != want.ID || calls[0].Function == nil || calls[0].Function.Name == nil || *calls[0].Function.Name != want.Function.Name {
							t.Error("tool call details mismatch")
						}
					}
				}
			}
		})
	}
}

func TestOpenAIAdapterScenarios(t *testing.T) {
	t.Run("EmbeddedErrors", func(t *testing.T) {
		for _, tc := range []struct {
			name, body string
			status     int
			kind       APIErrorKind
			partial    bool
		}{
			{"top level", `{"id":"failed","error":{"code":429,"message":"limited"}}`, 429, APIErrorKindRateLimit, false},
			{"choice with partial output", `{"choices":[{"index":0,"message":{"role":"assistant","content":"partial"},"finish_reason":"error","error":{"code":502,"message":"limited"}}],"usage":{"prompt_tokens":5}}`, 502, APIErrorKindServer, true},
			{"choice without message", `{"choices":[{"index":0,"error":{"code":503,"message":"limited"}}]}`, 503, APIErrorKindServiceUnavailable, false},
			{"choice with interrupted tool", `{"choices":[{"index":0,"message":{"role":"assistant","content":"partial","reasoning":"checking","tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}]},"finish_reason":"error","error":{"code":503,"message":"limited"}}]}`, 503, APIErrorKindServiceUnavailable, true},
			{"top level with interrupted tool", `{"error":{"code":503,"message":"limited"},"choices":[{"index":0,"message":{"role":"assistant","content":"partial","reasoning":"checking","tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}]},"finish_reason":"error"}]}`, 503, APIErrorKindServiceUnavailable, true},
			{"string code", `{"error":{"code":"rate_limit_exceeded","message":"limited"}}`, 200, APIErrorKindUnknown, false},
		} {
			for _, mode := range []string{"generate", "stream JSON error", "stream event error"} {
				t.Run(tc.name+"/"+mode, func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						w.Header().Set("Retry-After", "7")
						w.Header().Set("X-Request-Id", "request-1")
						if mode == "stream event error" {
							w.Header().Set("Content-Type", "text/event-stream")
							fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\ndata: %s\n\ndata: [DONE]\n\n", tc.body)
						} else {
							w.Header().Set("Content-Type", "application/json")
							fmt.Fprint(w, tc.body)
						}
					}))
					defer server.Close()
					g, err := NewOpenAiGenerator(server.Client(), server.URL, "test-key")
					if err != nil {
						t.Fatal(err)
					}
					request := GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
					if mode == "generate" {
						var response Response
						response, err = g.Generate(t.Context(), request)
						if tc.partial && (len(response.Candidates) != 1 || response.Candidates[0].Blocks[0].Content.String() != "partial") {
							t.Fatalf("partial response lost: %+v", response)
						}
						if strings.Contains(tc.name, "interrupted tool") {
							for _, block := range response.Candidates[0].Blocks {
								if block.BlockType == ToolCall {
									t.Fatal("incomplete tool call exposed")
								}
							}
							fields := response.Candidates[0].ExtraFields[OpenAIExtraFieldWireFields].(map[string]json.RawMessage)
							if string(fields["reasoning"]) != `"checking"` {
								t.Fatal("partial reasoning metadata lost")
							}
						}
						if response.ExtraFields[OpenAIResponseExtraFieldHeaders].(http.Header).Get("X-Request-Id") != "request-1" {
							t.Fatal("response headers lost")
						}
					} else {
						var text strings.Builder
						for chunk := range g.Stream(t.Context(), request) {
							if err != nil {
								t.Fatal("stream continued after error")
							}
							if chunk.Err != nil {
								err = chunk.Err
								continue
							}
							if chunk.Block.BlockType == Content {
								text.WriteString(chunk.Block.Content.String())
							}
						}
						if mode == "stream event error" && text.String() != "partial" {
							t.Fatalf("partial stream text = %q", text.String())
						}
					}
					var api *ApiErr
					if !errors.As(err, &api) {
						t.Fatalf("error = %v, want ApiErr", err)
					}
					if api.StatusCode != tc.status || api.Kind != tc.kind || api.Message != "limited" || api.RawBody != tc.body {
						t.Fatalf("API error = %+v", api)
					}
					if delay, ok := api.RetryAfter(); !ok || delay != 7*time.Second {
						t.Fatalf("retry delay = %v, %v", delay, ok)
					}
				})
			}
		}
	})
	t.Run("StreamChoiceMetadata", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null,"logprobs":{"content":[{"token":"A","logprob":-0.1,"extra":9007199254740993}]}}]}`,
				`{"choices":[{"index":0,"delta":{},"logprobs":null}]}`,
				`{"choices":[{"index":0,"delta":{"content":"B"},"logprobs":{"content":[{"token":"B","logprob":-0.2}]}}]}`,
				`{"choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":{"content":null}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cache_write_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}}`,
			} {
				fmt.Fprintf(w, "data: %s\n\n", event)
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		defer server.Close()
		g, err := NewOpenAiGenerator(server.Client(), server.URL, "test-key")
		if err != nil {
			t.Fatal(err)
		}
		response, err := (&StreamingAdapter{S: g}).Generate(t.Context(), GenerationRequest{Model: "test", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}})
		if err != nil {
			t.Fatal(err)
		}
		if len(response.Candidates) != 1 || response.Candidates[0].Blocks[0].Content.String() != "AB" {
			t.Fatalf("response = %+v", response)
		}
		fields := response.Candidates[0].ExtraFields[OpenAIExtraFieldChoice].(map[string]json.RawMessage)
		var logprobs struct {
			Content []struct {
				Token string          `json:"token"`
				Extra json.RawMessage `json:"extra"`
			} `json:"content"`
		}
		if err := json.Unmarshal(fields["logprobs"], &logprobs); err != nil {
			t.Fatal(err)
		}
		if len(logprobs.Content) != 2 || logprobs.Content[0].Token != "A" || logprobs.Content[1].Token != "B" || string(logprobs.Content[0].Extra) != "9007199254740993" {
			t.Fatalf("logprobs = %s", fields["logprobs"])
		}
		if string(fields["finish_reason"]) != `"stop"` {
			t.Fatalf("finish reason = %s", fields["finish_reason"])
		}
		if response.UsageMetadata[UsageMetricCacheWriteTokens] != 3 || response.UsageMetadata[UsageMetricReasoningTokens] != 2 {
			t.Fatalf("usage = %+v", response.UsageMetadata)
		}
	})
	t.Run("HTTP", func(t *testing.T) {
		t.Run("native options", func(t *testing.T) {
			request := GenerationRequest{Model: "m", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}}
			for _, options := range []GenerationOptions{
				NewGenerationOptions(WithOpenAIExtraBody(map[string]json.RawMessage{"messages": json.RawMessage(`[]`)})),
				NewGenerationOptions(WithOpenAIExtraBody(map[string]json.RawMessage{"thinking": json.RawMessage(`{`)})),
				{OpenAIGenerationOptionExtraBody: "wrong type"},
				{OpenAIGenerationOptionStreamUsage: "wrong type"},
			} {
				request.Options = options
				if _, _, err := openAIRequest(request, true); err == nil {
					t.Fatalf("accepted invalid native options: %v", options)
				}
			}
			fields := map[string]json.RawMessage{"thinking": json.RawMessage(`{"type":"enabled"}`), "response_format": json.RawMessage(`null`)}
			request.Options = NewGenerationOptions(WithOpenAIExtraBody(fields), WithOpenAIStreamUsage(false))
			params, _, err := openAIRequest(request, true)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(params)
			if err != nil {
				t.Fatal(err)
			}
			if params.StreamOptions.IsSpecified() || !strings.Contains(string(encoded), `"response_format":null`) {
				t.Fatalf("lost explicit native configuration: %s", encoded)
			}
			fields["thinking"][0] = '!'
			if _, err := json.Marshal(params); err != nil {
				t.Fatalf("request aliases caller JSON bytes: %v", err)
			}
			request.Options = nil
			next, _, err := openAIRequest(request, true)
			if err != nil {
				t.Fatal(err)
			}
			if len(next.AdditionalProperties) != 0 || !next.StreamOptions.IsSpecified() {
				t.Fatal("native options leaked into next request")
			}
		})
		t.Run("tool choice controls", func(t *testing.T) {
			for _, choice := range []string{"auto", "required", "none"} {
				t.Run(choice, func(t *testing.T) {
					request := GenerationRequest{Model: "m", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("Use a tool if allowed.")}}}, Options: NewGenerationOptions(WithToolChoice(choice))}
					for _, stream := range []bool{false, true} {
						params, _, err := openAIRequest(request, stream)
						if err != nil {
							t.Fatal(err)
						}
						data, err := json.Marshal(params)
						if err != nil {
							t.Fatal(err)
						}
						var body map[string]json.RawMessage
						if err := json.Unmarshal(data, &body); err != nil {
							t.Fatal(err)
						}
						if string(body["tool_choice"]) != `"`+choice+`"` {
							t.Fatalf("tool choice encoded as %s", body["tool_choice"])
						}
					}
					if _, err := convertToolsToOpenAI([]Tool{{Name: choice}}); err == nil {
						t.Fatal("accepted reserved tool name")
					}
				})
			}
		})
		t.Run("tool arguments replay as strings", func(t *testing.T) {
			for _, test := range []struct {
				name, arguments string
				wantError       bool
			}{
				{"observed string response", `"{\"n\":9007199254740993}"`, false},
				{"reject object response", `{"n":9007199254740993}`, true},
			} {
				t.Run(test.name, func(t *testing.T) {
					var message wire.ResponseMessage
					data := `{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":` + test.arguments + `}}]}`
					err := json.Unmarshal([]byte(data), &message)
					if test.wantError {
						if err == nil {
							t.Fatal("accepted object-valued tool arguments")
						}
						return
					}
					if err != nil {
						t.Fatal(err)
					}
					converted, err := openAIResponseMessage(message, "")
					if err != nil {
						t.Fatal(err)
					}
					replayed, err := toOpenAIMessage(converted)
					if err != nil {
						t.Fatal(err)
					}
					encoded, err := json.Marshal(replayed)
					if err != nil {
						t.Fatal(err)
					}
					// A string-typed destination rejects object-valued request arguments.
					var request struct {
						ToolCalls []struct {
							Function struct {
								Arguments string `json:"arguments"`
							} `json:"function"`
						} `json:"tool_calls"`
					}
					if err := json.Unmarshal(encoded, &request); err != nil {
						t.Fatalf("replayed arguments must be strings: %s: %v", encoded, err)
					}
					if len(request.ToolCalls) != 1 || request.ToolCalls[0].Function.Arguments != `{"n":9007199254740993}` {
						t.Fatalf("arguments changed during replay: %s", encoded)
					}
				})
			}
		})
		t.Run("constructor", func(t *testing.T) {
			if _, err := NewOpenAiGenerator(nil, "", ""); !errors.Is(err, ErrMissingAPIKey) {
				t.Fatalf("missing key: %v", err)
			}
			for _, base := range []string{"relative/path", "file:///tmp/client", "https://example.com?key=value"} {
				if _, err := NewOpenAiGenerator(nil, base, "key"); err == nil {
					t.Fatalf("accepted %q", base)
				}
			}
		})
		t.Run("headers errors and no retry", func(t *testing.T) {
			for _, status := range []int{400, 429, 503} {
				t.Run(fmt.Sprint(status), func(t *testing.T) {
					calls := 0
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						calls++
						if r.URL.Path != "/prefix/chat/completions" || r.Header.Get("Authorization") != "Bearer fixture" {
							t.Errorf("request: %s %v", r.URL, r.Header)
						}
						w.Header().Set("Content-Type", "text/html")
						w.Header().Set("Retry-After", "3")
						w.WriteHeader(status)
						_, _ = io.WriteString(w, "gateway failure")
					}))
					defer server.Close()
					g, err := NewOpenAiGenerator(server.Client(), server.URL+"/prefix", "fixture")
					if err != nil {
						t.Fatal(err)
					}
					request := GenerationRequest{Model: "m", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hello")}}}}
					_, err = g.Generate(t.Context(), request)
					var api *ApiErr
					if !errors.As(err, &api) || api.StatusCode != status || api.RawBody != "gateway failure" {
						t.Fatalf("error: %#v", err)
					}
					if delay, ok := api.RetryAfter(); !ok || delay != 3*time.Second {
						t.Fatalf("retry timing: %v %v", delay, ok)
					}
					if calls != 1 {
						t.Fatalf("POST retried %d times", calls)
					}
				})
			}
		})
		t.Run("wire replay and token fields", func(t *testing.T) {
			var requests []map[string]json.RawMessage
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				requests = append(requests, request)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("x-request-id", "request-1")
				_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":null,"reasoning_content":"thinking","reasoning_details":[{"type":"reasoning.encrypted","data":"cipher","signature":"sig"}],"opaque":{"n":9007199254740993},"tool_calls":[{"id":"c","type":"function","function":{"name":"lookup","arguments":"{\"n\":9007199254740993}","signature":"tool-sig"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":0,"completion_tokens":4},"provider_number":9007199254740993}`)
			}))
			defer server.Close()
			g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
			if err != nil {
				t.Fatal(err)
			}
			request := GenerationRequest{Model: "first", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: NewGenerationOptions(WithMaxGenerationTokens(8), WithOpenAITokenLimitField("max_tokens"))}
			response, err := g.Generate(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			if string(requests[0]["max_tokens"]) != "8" || requests[0]["max_completion_tokens"] != nil {
				t.Fatalf("token selection: %s", requests[0])
			}
			if response.ExtraFields[OpenAIResponseExtraFieldHeaders].(http.Header).Get("x-request-id") != "request-1" {
				t.Fatal("lost headers")
			}
			if !strings.Contains(response.Candidates[0].Blocks[1].Content.String(), "9007199254740993") {
				t.Fatalf("tool arguments rounded: %+v", response.Candidates[0])
			}
			// Persist and reload to exercise ExtraFields without Go-specific map types.
			saved, err := json.Marshal(response.Candidates[0].ExtraFields)
			if err != nil {
				t.Fatal(err)
			}
			replay := response.Candidates[0]
			replay.ExtraFields = nil
			decoder := json.NewDecoder(bytes.NewReader(saved))
			decoder.UseNumber()
			if err := decoder.Decode(&replay.ExtraFields); err != nil {
				t.Fatal(err)
			}
			request.Model = "second"
			request.Options = NewGenerationOptions(WithMaxGenerationTokens(9))
			request.Dialog = append(request.Dialog, replay)
			if _, err := g.Generate(t.Context(), request); err != nil {
				t.Fatal(err)
			}
			if requests[1]["max_tokens"] != nil || string(requests[1]["max_completion_tokens"]) != "9" {
				t.Fatalf("stale token field: %s", requests[1])
			}
			messageJSON := string(requests[1]["messages"])
			for _, value := range []string{`"reasoning_content":"thinking"`, `"signature":"sig"`, `"signature":"tool-sig"`, `"arguments":"{\"n\":9007199254740993}"`, `"content":null`, `"opaque":{"n":9007199254740993}`} {
				if !strings.Contains(messageJSON, value) {
					t.Errorf("missing %s in %s", value, messageJSON)
				}
			}
		})
	})
	t.Run("SSE", func(t *testing.T) {
		stream := ": keepalive\r\n\r\n" +
			`data: {"choices":[{"index":0,"delta":{"content":"Hi","reasoning_details":[{"index":0,"type":"reasoning.encrypted","data":"part"}],"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"second","arguments":"{\"n\":"}},{"index":0,"id":"a","type":"function","function":{"name":"first","arguments":"{"}}]}}]}` + "\r\n\r\n" +
			`data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"index":0,"data":"two","signature":"sig"}],"tool_calls":[{"index":0,"function":{"arguments":"}"}},{"index":1,"function":{"arguments":"9007199254740993}"}}]},"finish_reason":"tool_calls","usage":{"completion_tokens":7}}]}` + "\n\n" +
			"data: [DONE]\n\n" + `data: {"cost":"0","choices":[],"usage":{"completion_tokens":7}}` + "\n\n"
		calls := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls++
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, stream)
		}))
		defer server.Close()
		g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
		if err != nil {
			t.Fatal(err)
		}
		response, err := (&StreamingAdapter{S: g}).Generate(t.Context(), GenerationRequest{Model: "m", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hi")}}}})
		if err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("POST replayed %d times", calls)
		}
		if len(response.Candidates) != 1 || len(response.Candidates[0].Blocks) != 3 {
			t.Fatalf("assembled response: %+v", response)
		}
		blocks := response.Candidates[0].Blocks
		if blocks[1].ID != "a" || blocks[2].ID != "b" || !strings.Contains(blocks[2].Content.String(), "9007199254740993") {
			t.Fatalf("corrupted tools: %+v", blocks)
		}
		fields := response.ExtraFields[OpenAIResponseExtraFieldWireFields].(map[string]json.RawMessage)
		if string(fields["cost"]) != `"0"` {
			t.Fatalf("lost trailing cost: %s", fields)
		}
		candidateFields := response.Candidates[0].ExtraFields[OpenAIExtraFieldWireFields].(map[string]json.RawMessage)
		if !strings.Contains(string(candidateFields["reasoning_details"]), `"data":"parttwo"`) {
			t.Fatalf("lost reasoning fragments: %s", candidateFields)
		}
		if value, ok := OutputTokens(response.UsageMetadata); !ok || value != 7 {
			t.Fatalf("usage: %+v", response.UsageMetadata)
		}
	})
	t.Run("termination", func(t *testing.T) {
		for _, test := range []struct {
			name, body string
			want       string
		}{
			{"empty", "", "unexpected EOF"},
			{"incomplete", `data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}` + "\n\n", "unexpected EOF"},
			{"malformed", "data: {\n\n", "decode openai event"},
			{"provider error", `data: {"error":{"message":"quota"}}` + "\n\n", "quota"},
			{"length", `data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":"length"}]}` + "\n\n", "maximum generation limit"},
		} {
			t.Run(test.name, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, test.body)
				}))
				defer server.Close()
				g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
				if err != nil {
					t.Fatal(err)
				}
				var terminal error
				for chunk := range g.Stream(t.Context(), GenerationRequest{Model: "m", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hi")}}}}) {
					if chunk.Err != nil {
						terminal = chunk.Err
					}
				}
				if terminal == nil || !strings.Contains(terminal.Error(), test.want) {
					t.Fatalf("error: %v, want %s", terminal, test.want)
				}
			})
		}
		t.Run("early stop closes body", func(t *testing.T) {
			closed := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"content":"first"}}]}`+"\n\n")
				w.(http.Flusher).Flush()
				<-r.Context().Done()
				close(closed)
			}))
			defer server.Close()
			g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			defer cancel()
			for chunk := range g.Stream(ctx, GenerationRequest{Model: "m", Dialog: Dialog{Message{Role: User, Blocks: []Block{TextBlock("hi")}}}}) {
				if chunk.Err != nil {
					t.Fatal(chunk.Err)
				}
				break
			}
			select {
			case <-closed:
			case <-ctx.Done():
				t.Fatal("consumer stop did not close response")
			}
		})
	})
	t.Run("audio and candidates", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"content":"first","audio":{"id":"audio-1","data":"AA","transcript":"hel"}}},{"index":1,"delta":{"content":"second"}}]}`+"\n\n"+
				`data: {"choices":[{"index":0,"delta":{"audio":{"data":"BB","transcript":"lo"}},"finish_reason":"stop"},{"index":1,"delta":{},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
		}))
		defer server.Close()
		g, err := NewOpenAiGenerator(server.Client(), server.URL, "fixture")
		if err != nil {
			t.Fatal(err)
		}
		var audio Block
		var second bool
		for chunk := range g.Stream(t.Context(), GenerationRequest{Model: "m", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hi")}}}, Options: NewGenerationOptions(WithOpenAIOutputModalities(Text, Audio), WithOpenAIAudioConfig(OpenAIAudioConfig{VoiceName: "alloy", Format: "wav"}))}) {
			if chunk.Err != nil {
				t.Fatal(chunk.Err)
			}
			if chunk.Block.ModalityType == Audio {
				audio = chunk.Block
			}
			if chunk.CandidatesIndex == 1 && chunk.Block.Content != nil && chunk.Block.Content.String() == "second" {
				second = true
			}
		}
		if !second || audio.ID != "audio-1" || audio.Content == nil || audio.Content.String() != "AABB" || audio.MimeType != "audio/wav" {
			t.Fatalf("candidates/audio lost: %v %+v", second, audio)
		}
	})
	t.Run("OpenAIErrorMappingUsesHTTPStatus", func(t *testing.T) {
		var cause oai.Error
		if err := json.Unmarshal([]byte(`{"code":"invalid_api_key","message":"Invalid API key"}`), &cause); err != nil {
			t.Fatalf("unmarshal SDK error: %v", err)
		}
		cause.StatusCode = http.StatusServiceUnavailable
		cause.Response = &http.Response{Header: http.Header{"Retry-After": []string{"9"}}}

		got := mapOpenAISDKError(ProviderOpenAI, &cause)
		if got == nil {
			t.Fatal("mapOpenAISDKError() = nil, want *ApiErr")
		}
		if got.Kind != APIErrorKindServiceUnavailable {
			t.Fatalf("Kind = %q, want %q", got.Kind, APIErrorKindServiceUnavailable)
		}
		if delay, ok := got.RetryAfter(); !ok || delay != 9*time.Second {
			t.Fatalf("RetryAfter() = (%s, %t), want (9s, true)", delay, ok)
		}
		if !errors.Is(got, &cause) {
			t.Fatal("mapped error does not wrap the SDK error")
		}
	})
	t.Run("OpenAIGenerateReturnsContentPolicyErrorForContentFilter", func(t *testing.T) {
		client := &mockChatCompletionService{response: &oai.ChatCompletion{
			Choices: []oai.ChatCompletionChoice{{FinishReason: "content_filter"}},
		}}
		generator := newTestOpenAIGenerator(t, client)

		response, err := generator.Generate(context.Background(), GenerationRequest{
			Model:  "gpt-5",
			Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
		})
		if response.FinishReason != ContentPolicyViolation {
			t.Fatalf("FinishReason = %v, want ContentPolicyViolation", response.FinishReason)
		}
		var policyErr ContentPolicyErr
		if !errors.As(err, &policyErr) {
			t.Fatalf("Generate error = %T %v, want ContentPolicyErr", err, err)
		}
		if !strings.Contains(policyErr.Error(), "content policy violation detected") {
			t.Fatalf("Generate error = %q, want content filter fallback", policyErr)
		}
	})
	t.Run("OpenAIGenerateReturnsContentPolicyErrorForRefusal", func(t *testing.T) {
		client := &mockChatCompletionService{response: &oai.ChatCompletion{
			Choices: []oai.ChatCompletionChoice{{
				FinishReason: "stop",
				Message:      oai.ChatCompletionMessage{Refusal: "I cannot help with that."},
			}},
		}}
		generator := newTestOpenAIGenerator(t, client)

		response, err := generator.Generate(context.Background(), GenerationRequest{
			Model:  "gpt-5",
			Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
		})
		if response.FinishReason != ContentPolicyViolation {
			t.Fatalf("FinishReason = %v, want ContentPolicyViolation", response.FinishReason)
		}
		var policyErr ContentPolicyErr
		if !errors.As(err, &policyErr) {
			t.Fatalf("Generate error = %T %v, want ContentPolicyErr", err, err)
		}
		if !strings.Contains(policyErr.Error(), "I cannot help with that.") {
			t.Fatalf("Generate error = %q, want refusal message", policyErr)
		}
	})
	t.Run("OpenAIStreamReturnsContentPolicyErrorForRefusal", func(t *testing.T) {
		client := &mockChatCompletionService{streamEvents: []oaissestream.Event{{
			Data: []byte(`{"id":"chatcmpl_123","object":"chat.completion.chunk","created":0,"model":"gpt-5","choices":[{"index":0,"delta":{"refusal":"I cannot help with that."},"finish_reason":""}]}`),
		}}}
		generator := newTestOpenAIGenerator(t, client)

		var gotErr error
		for chunk := range generator.Stream(context.Background(), GenerationRequest{
			Model:  "gpt-5",
			Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("unsafe request")}}},
		}) {
			if chunk.Err != nil {
				gotErr = chunk.Err
				break
			}
		}

		var policyErr ContentPolicyErr
		if !errors.As(gotErr, &policyErr) {
			t.Fatalf("Stream error = %T %v, want ContentPolicyErr", gotErr, gotErr)
		}
		if !strings.Contains(policyErr.Error(), "I cannot help with that.") {
			t.Fatalf("Stream error = %q, want refusal message", policyErr)
		}
	})
	t.Run("OpenAiGeneratorUsesRequestScopedState", func(t *testing.T) {
		client := &mockChatCompletionService{response: &oai.ChatCompletion{
			Choices: []oai.ChatCompletionChoice{{
				FinishReason: "stop",
				Message:      oai.ChatCompletionMessage{Role: "assistant", Content: "ok"},
			}},
		}}
		generator := newTestOpenAIGenerator(t, client)
		dialog := Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}

		_, err := generator.Generate(context.Background(), GenerationRequest{
			Model:        "first-model",
			Instructions: SystemMessage(TextBlock("first instructions")),
			Dialog:       dialog,
			Tools:        []Tool{{Name: "lookup"}},
		})
		if err != nil {
			t.Fatalf("first Generate() error = %v", err)
		}
		_, err = generator.Generate(context.Background(), GenerationRequest{
			Model:  "second-model",
			Dialog: dialog,
		})
		if err != nil {
			t.Fatalf("second Generate() error = %v", err)
		}

		if len(client.requests) != 2 {
			t.Fatalf("captured requests = %d, want 2", len(client.requests))
		}
		if got := string(client.requests[0].Model); got != "first-model" {
			t.Fatalf("first model = %q", got)
		}
		if got := string(client.requests[1].Model); got != "second-model" {
			t.Fatalf("second model = %q", got)
		}
		if got := len(client.requests[0].Tools); got != 1 {
			t.Fatalf("first request tools = %d, want 1", got)
		}
		if got := len(client.requests[1].Tools); got != 0 {
			t.Fatalf("second request tools = %d, want 0", got)
		}
		if got := len(client.requests[0].Messages); got != 2 {
			t.Fatalf("first request messages = %d, want system + user", got)
		}
		if got := len(client.requests[1].Messages); got != 1 {
			t.Fatalf("second request messages = %d, want only user", got)
		}
	})
	t.Run("OpenAiGenerator/Count", func(t *testing.T) {
		// Mock the tokenizer with a simple generator
		g := OpenAiGenerator{}

		// Create a simple dialog with text and image
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hello!"),
					},
					{
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "image/jpeg",
						Content:      Str("base64-content"),
						ExtraFields: map[string]interface{}{
							"width":  1024,
							"height": 1024,
						},
					},
				},
			},
		}

		// This is more of an integration test, as it relies on the tiktoken library
		// We're just checking that it runs without errors
		_, err := g.Count(context.Background(), GenerationRequest{
			Model:        "gpt-4o",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Errorf("Error counting tokens: %v", err)
		}
	})
	t.Run("OpenAiGenerator/Count/Example", func(t *testing.T) {
		// Create an OpenAI client
		client := &OpenAiGenerator{}
		generator := client
		// Create a dialog with a user message
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is the capital of France?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        oai.ChatModelGPT4o,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		}
		// Count tokens in the dialog
		tokenCount, err := generator.Count(context.Background(), request)
		if err != nil {
			t.Fatalf("count tokens: %v", err)
		}
		if tokenCount == 0 {
			t.Fatal("expected non-zero token count")
		}
		// Add a response to the dialog
		dialog = append(dialog, Message{
			Role: Assistant,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("The capital of France is Paris. It's known as the 'City of Light' and is famous for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."),
				},
			},
		})
		request.Dialog = dialog
		// Count tokens in the updated dialog
		tokenCount, err = generator.Count(context.Background(), request)
		if err != nil {
			t.Fatalf("count tokens: %v", err)
		}
		if tokenCount == 0 {
			t.Fatal("expected non-zero token count")
		}
	})
	t.Run("OpenAiGenerator/Generate", func(t *testing.T) {
		// Create an OpenAI client
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := newLiveOpenAIGenerator(t, "", apiKey)
		// Instantiate a OpenAI Generator
		gen := client
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hi!"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        oai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
		}
		// Generate a response
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// The exact response text may vary, so we'll just print a placeholder
		// Customize generation parameters
		request.Options = NewGenerationOptions(
			WithTopK(10),
			WithCandidateCount(2),
			WithMaxGenerationTokens(1024),
		)
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := len(resp.Candidates); got == 0 {
			t.Fatal("expected at least one item")
		}
	})
	t.Run("OpenAiGenerator/Generate/audio", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		audioBytes, err := os.ReadFile("sample.wav")
		if err != nil {
			t.Skip("could not open sample.wav")
			return
		}
		// Encode as base64 for inline audio usage
		audioBase64 := Str(base64.StdEncoding.EncodeToString(audioBytes))
		client := newLiveOpenAIGenerator(t, "", apiKey)
		gen := client
		// Using inline audio data
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Audio,
						MimeType:     "audio/wav",
						Content:      audioBase64,
					},
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("In this audio, a person is introducing themselves. What is the name of person in the greeting in this audio? Return a one word response of the name"),
					},
				},
			},
		}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        oai.ChatModelGPT4oAudioPreview,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(128)),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Blocks) > 0 {
			if got := strings.ToLower(resp.Candidates[0].Blocks[0].Content.String()); !strings.Contains(got, "friday") {
				t.Fatalf("content = %q, want it to contain friday", got)
			}
		}
	})
	t.Run("OpenAiGenerator/Generate/image", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		imgBytes, err := os.ReadFile("sample.jpg")
		if err != nil {
			t.Skip("could not open sample.jpg")
			return
		}
		imgBase64 := Str(base64.StdEncoding.EncodeToString(imgBytes))
		client := newLiveOpenAIGenerator(t, "", apiKey)
		gen := client
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Image,
						MimeType:     "image/jpeg",
						Content:      imgBase64,
					},
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is in this image? (Hint, it's a character from The Croods, a DreamWorks animated movie.)"),
					},
				},
			},
		}
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        oai.ChatModelGPT4o,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(512)),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(resp.Candidates) != 1 {
			t.Fatalf("candidates = %d, want 1", len(resp.Candidates))
		}
		if len(resp.Candidates[0].Blocks) != 1 {
			t.Fatalf("blocks = %d, want 1", len(resp.Candidates[0].Blocks))
		}
		if !strings.Contains(resp.Candidates[0].Blocks[0].Content.String(), "Crood") {
			t.Fatalf("content does not contain Crood")
		}
	})
	t.Run("OpenAiGenerator/Generate/openRouter", func(t *testing.T) {
		// Create an OpenAI client for open router
		client := newLiveOpenAIGenerator(t, "https://openrouter.ai/api/v1/", requireLiveAPIKey(t, "OPENROUTER_API_KEY"))
		// Instantiate a OpenAI Generator
		gen := client
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hi!"),
					},
				},
			},
		}
		// Generate a response
		resp, err := gen.Generate(context.Background(), GenerationRequest{
			Model:        "google/gemini-2.5-pro-preview-03-25",
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithMaxGenerationTokens(1024)),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// The exact response text may vary, so we'll just print a placeholder
		if got := len(resp.Candidates); got == 0 {
			t.Fatal("expected at least one item")
		}
	})
	t.Run("OpenAiGenerator/Generate/pdf", func(t *testing.T) {
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		pdfBytes, err := os.ReadFile("sample.pdf")
		if err != nil {
			t.Skip("could not open sample.wav")
			return
		}
		client := newLiveOpenAIGenerator(t, "", apiKey)
		gen := client
		// Create a dialog with PDF content
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					TextBlock("What is the title of this PDF? Just output the title and nothing else"),
					PDFBlock(pdfBytes, "sample.pdf"),
				},
			},
		}
		// Generate a response
		ctx := context.Background()
		response, err := gen.Generate(ctx, GenerationRequest{
			Model:        oai.ChatModelGPT4_1,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant.")),
			Dialog:       dialog,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// The response would contain the model's analysis of the PDF
		if len(response.Candidates) > 0 && len(response.Candidates[0].Blocks) > 0 {
			if got := response.Candidates[0].Blocks[0].Content.String(); got == "" {
				t.Fatal("expected non-empty content")
			}
		}
	})
	t.Run("OpenAiGenerator/Generate/thinking", func(t *testing.T) {
		requireLiveAPIKey(t, "OPENAI_API_KEY")
		// Create an OpenAI client
		client := newLiveOpenAIGenerator(t)
		// Instantiate a OpenAI Generator
		gen := client
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hi!"),
					},
				},
			},
		}
		// Customize generation parameters
		request := GenerationRequest{
			Model:        oai.ChatModelO3Mini,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options: NewGenerationOptions(
				WithMaxGenerationTokens(4096),
				WithReasoningEffort("low"),
				WithTemperature(1.0),
			),
		}
		// Generate a response
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// The exact response text may vary, so we'll just print a placeholder
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: User,
			Blocks: []Block{
				{
					BlockType:    Content,
					ModalityType: Text,
					Content:      Str("What can you do?"),
				},
			},
		})
		request.Dialog = dialog
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := len(resp.Candidates); got == 0 {
			t.Fatal("expected at least one item")
		}
	})
	t.Run("OpenAiGenerator/RequestTools", func(t *testing.T) {
		requireLiveAPIKey(t, "OPENAI_API_KEY")
		// Create an OpenAI client
		client := newLiveOpenAIGenerator(t, "https://gateway.ai.cloudflare.com/v1/4eee6dd2fdc8cebc7802c5a638f460fe/cpe/openai/", os.Getenv("OPENAI_API_KEY"))
		// Instantiate a OpenAI Generator
		gen := client
		tickerTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is the price of Apple stock?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        oai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock(openAIStockInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options:      NewGenerationOptions(WithToolChoice("get_stock_price")),
		}
		// Generate a response
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           resp.Candidates[0].Blocks[0].ID,
					ModalityType: Text,
					Content:      Str("123.45"),
				},
			},
		})
		request.Dialog = dialog
		request.Options = nil
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
	})
	t.Run("OpenAiGenerator/RequestTools/openRouter", func(t *testing.T) {
		// Define request tools
		tickerTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		// Create an OpenAI client for open router
		client := newLiveOpenAIGenerator(t, "https://openrouter.ai/api/v1/", requireLiveAPIKey(t, "OPENROUTER_API_KEY"))
		// Instantiate a OpenAI Generator
		gen := client
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("What is the price of Apple stock?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        "google/gemini-2.5-pro-preview-03-25",
			Instructions: SystemMessage(TextBlock(openAIStockInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
			Options:      NewGenerationOptions(WithToolChoice("get_stock_price")),
		}
		// Generate a response
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           resp.Candidates[0].Blocks[0].ID,
					ModalityType: Text,
					Content:      Str("123.45"),
				},
			},
		})
		request.Dialog = dialog
		request.Options = nil
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
	})
	t.Run("OpenAiGenerator/RequestTools/openRouterParallelToolUse", func(t *testing.T) {
		// Create an OpenAI client
		client := newLiveOpenAIGenerator(t, "https://openrouter.ai/api/v1/", requireLiveAPIKey(t, "OPENROUTER_API_KEY"))
		// Define request tools
		tickerTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		// Instantiate a OpenAI Generator
		gen := client
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Which stock, Apple vs. Microsoft, is more expensive?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        "google/gemini-2.5-pro-preview-03-25",
			Instructions: SystemMessage(TextBlock(openAIStockComparisonInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
		}
		// Generate a response
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		if got := resp.Candidates[0].Blocks[1].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           resp.Candidates[0].Blocks[0].ID,
					ModalityType: Text,
					Content:      Str("123.45"),
				},
			},
		}, Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           resp.Candidates[0].Blocks[1].ID,
					ModalityType: Text,
					Content:      Str("678.45"),
				},
			},
		})
		request.Dialog = dialog
		request.Options = nil
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
	})
	t.Run("OpenAiGenerator/RequestTools/parallelToolUse", func(t *testing.T) {
		requireLiveAPIKey(t, "OPENAI_API_KEY")
		// Create an OpenAI client
		client := newLiveOpenAIGenerator(t)
		// Define request tools
		tickerTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		// Instantiate a OpenAI Generator
		gen := client
		tickerTool.Description += "\nYou can call this tool in parallel"
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Which stock, Apple vs. Microsoft, is more expensive?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        oai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock(openAIStockComparisonInstructions)),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
		}
		// Generate a response
		resp, err := gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		if got := resp.Candidates[0].Blocks[1].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
		dialog = append(dialog, resp.Candidates[0], Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           resp.Candidates[0].Blocks[0].ID,
					ModalityType: Text,
					Content:      Str("123.45"),
				},
			},
		}, Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           resp.Candidates[0].Blocks[1].ID,
					ModalityType: Text,
					Content:      Str("678.45"),
				},
			},
		})
		request.Dialog = dialog
		request.Options = nil
		resp, err = gen.Generate(context.Background(), request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := resp.Candidates[0].Blocks[0].Content.String(); got == "" {
			t.Fatal("expected non-empty content")
		}
	})
	t.Run("OpenAiGenerator/Stream", func(t *testing.T) {
		// Create an OpenAI client
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := newLiveOpenAIGenerator(t, "", apiKey)
		// Instantiate a OpenAI Generator
		gen := client
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Hi!"),
					},
				},
			},
		}
		// Stream a response
		blocks := make([][]Block, 2)
		for chunk := range gen.Stream(context.Background(), GenerationRequest{
			Model:        oai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Options:      NewGenerationOptions(WithCandidateCount(2)),
		}) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			blocks[chunk.CandidatesIndex] = append(blocks[chunk.CandidatesIndex], chunk.Block)
		}
		if len(blocks) == 2 && len(blocks[0]) > 1 && len(blocks[1]) > 1 {
		}
	})
	t.Run("OpenAiGenerator/Stream/parallelToolUse", func(t *testing.T) {
		// Create an OpenAI client
		apiKey := requireLiveAPIKey(t, "OPENAI_API_KEY")
		client := newLiveOpenAIGenerator(t, "", apiKey)
		// Define request tools
		tickerTool := Tool{
			Name:        "get_stock_price",
			Description: "Get the current stock price for a given ticker symbol.",
			InputSchema: func() *jsonschema.Schema {
				schema, err := GenerateSchema[struct {
					Ticker string `json:"ticker" jsonschema:"required" jsonschema_description:"The stock ticker symbol, e.g. AAPL for Apple Inc."`
				}]()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return schema
			}(),
		}
		// Instantiate a OpenAI Generator
		gen := client
		tickerTool.Description += "\nYou can call this tool in parallel"
		dialog := Dialog{
			{
				Role: User,
				Blocks: []Block{
					{
						BlockType:    Content,
						ModalityType: Text,
						Content:      Str("Which stock, Apple vs. Microsoft, is more expensive?"),
					},
				},
			},
		}
		request := GenerationRequest{
			Model:        oai.ChatModelGPT4oMini,
			Instructions: SystemMessage(TextBlock("You are a helpful assistant")),
			Dialog:       dialog,
			Tools:        []Tool{tickerTool},
		}
		// Stream a response
		var blocks []Block
		for chunk := range gen.Stream(context.Background(), request) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			blocks = append(blocks, chunk.Block)
		}
		if len(blocks) > 1 {
		}
		// collect the blocks
		var prevToolCallId string
		var toolCalls []Block
		var toolcallArgs string
		var toolCallInput ToolCallInput
		for _, block := range blocks {
			// Skip metadata blocks
			if block.BlockType == MetadataBlockType {
				continue
			}
			if block.ID != "" && block.ID != prevToolCallId {
				if toolcallArgs != "" {
					// Parse the arguments string into a map
					if err := json.Unmarshal([]byte(toolcallArgs), &toolCallInput.Parameters); err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					// Marshal back to JSON for consistent representation
					toolUseJSON, err := json.Marshal(toolCallInput)
					if err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					toolCalls[len(toolCalls)-1].Content = Str(toolUseJSON)
					toolCallInput = ToolCallInput{}
					toolcallArgs = ""
				}
				prevToolCallId = block.ID
				toolCalls = append(toolCalls, Block{
					ID:           block.ID,
					BlockType:    ToolCall,
					ModalityType: Text,
					MimeType:     "text/plain",
				})
				toolCallInput.Name = block.Content.String()
			} else {
				toolcallArgs += block.Content.String()
			}
		}
		if toolcallArgs != "" {
			// Parse the arguments string into a map
			if err := json.Unmarshal([]byte(toolcallArgs), &toolCallInput.Parameters); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// Marshal back to JSON for consistent representation
			toolUseJSON, err := json.Marshal(toolCallInput)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			toolCalls[len(toolCalls)-1].Content = Str(toolUseJSON)
			toolCallInput = ToolCallInput{}
		}
		if got := len(toolCalls); got == 0 {
			t.Fatal("expected at least one item")
		}
		dialog = append(dialog, Message{
			Role:   Assistant,
			Blocks: toolCalls,
		}, Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           toolCalls[0].ID,
					ModalityType: Text,
					Content:      Str("123.45"),
				},
			},
		}, Message{
			Role: ToolResult,
			Blocks: []Block{
				{
					ID:           toolCalls[1].ID,
					ModalityType: Text,
					Content:      Str("678.45"),
				},
			},
		})
		// Stream a response
		blocks = nil
		request.Dialog = dialog
		for chunk := range gen.Stream(context.Background(), request) {
			if chunk.Err != nil {
				t.Fatalf("stream returned error: %v", chunk.Err)
			}
			blocks = append(blocks, chunk.Block)
		}
		if len(blocks) > 1 {
		}
	})
	t.Run("OpenAiGenerator/calculateImageTokens", func(t *testing.T) {
		// Create an OpenAI generator
		g := OpenAiGenerator{}

		// Create a test image block with dimensions in ExtraFields
		block := Block{
			BlockType:    Content,
			ModalityType: Image,
			MimeType:     "image/jpeg",
			Content:      Str("base64-content"),
			ExtraFields: map[string]interface{}{
				"width":  1024,
				"height": 1024,
			},
		}

		// Test high detail image
		tokens, err := g.calculateImageTokens(block, "gpt-4o")
		if err != nil {
			t.Errorf("Error calculating image tokens: %v", err)
		}
		expectedTokens := 765 // 85 + (170 * 4)
		if tokens != expectedTokens {
			t.Errorf("Expected %d tokens for 1024x1024 image with gpt-4o, got %d", expectedTokens, tokens)
		}

		// Test low detail image
		block.ExtraFields["detail"] = "low"
		tokens, err = g.calculateImageTokens(block, "gpt-4o")
		if err != nil {
			t.Errorf("Error calculating image tokens: %v", err)
		}
		expectedTokens = 85 // Just base tokens for low detail
		if tokens != expectedTokens {
			t.Errorf("Expected %d tokens for low detail image with gpt-4o, got %d", expectedTokens, tokens)
		}

		// Test minimal model
		block.ExtraFields["detail"] = "high" // Reset to high detail
		tokens, err = g.calculateImageTokens(block, "gpt-4.1-mini")
		if err != nil {
			t.Errorf("Error calculating image tokens: %v", err)
		}
		// 32x32 patches for 1024x1024 image = 32x32 = 1024 patches
		// With 1.62 multiplier for gpt-4.1-mini = 1024 * 1.62 = 1658 (rounded)
		expectedTokens = 1658
		if tokens != expectedTokens {
			t.Errorf("Expected %d tokens for 1024x1024 image with gpt-4.1-mini, got %d", expectedTokens, tokens)
		}

		// Test with actual image data
		// Create a small 2x2 pixel JPEG image in base64
		miniJpegBase64 := "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4QBaRXhpZgAASUkqAAgAAAABAGEAAAEBAAEAAAABAAIAAAECAAEAAAABAAIAAAEDAAEAAAABAAIAAAABBAABAAAAYAAAAP/bAEMAAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAf/bAEMBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAf/AABEIAAIAAQMBEQACEQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEFEiExQQYTUWEHInEUMoGRoQgjQrHBFVLR8CQzYnKCCQoWFxgZGiUmJygpKjQ1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4eLj5OXm5+jp6vHy8/T19vf4+fr/xAAfAQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgv/xAC1EQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGhscEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2gAMAwEAAhEDEQA/AO72j0GM9fvHn8K/6N3Jvkd9rOytdfDv8+lj/gvqxhzy5dLbv4drIlSGWOKKJmfKLhsD7u0YJz/L8a/VOFcvwOX5fhcHlWDw+AwtChRwlHCZbQ+owYehToxVPCZbQkpewowhbD4WjCTs74TCy93+YszxOYYrF4jG43E4jHY3FV6+Kxmb5nWdfMMdi8RWlVr4zMcXWlUqYnFV6k5Vq1Z1atSTtUeq+TQ/yrb9T5D+o3P/AGj/AB3P/9k="
		// Remove the data URL prefix
		miniJpegBase64 = strings.Replace(miniJpegBase64, "data:image/jpeg;base64,", "", 1)

		// Decode the base64 string to ensure it's valid
		_, err = base64.StdEncoding.DecodeString(miniJpegBase64)
		if err != nil {
			t.Fatalf("Failed to decode test image base64: %v", err)
		}

		block = Block{
			BlockType:    Content,
			ModalityType: Image,
			MimeType:     "image/jpeg",
			Content:      Str(miniJpegBase64),
		}

		tokens, err = g.calculateImageTokens(block, "gpt-4.1-mini")
		if err != nil {
			t.Errorf("Error calculating image tokens from actual image data: %v", err)
		}
		if tokens <= 0 {
			t.Errorf("Expected positive token count for actual image data, got %d", tokens)
		}

		// Test case for image without dimensions
		blockWithoutDimensions := Block{
			BlockType:    Content,
			ModalityType: Image,
			MimeType:     "image/jpeg",
			Content:      Str("invalid-base64-content"),
		}

		// This should now return an error
		_, err = g.calculateImageTokens(blockWithoutDimensions, "gpt-4.1-mini")
		if err == nil {
			t.Errorf("Expected error when calculating tokens for image without dimensions, but got none")
		}
	})
}
