package gai

import (
	"encoding/json"
	"errors"
	"testing"

	a "github.com/anthropics/anthropic-sdk-go"
)

func TestAnthropicRequestControls(t *testing.T) {
	t.Run("parallel tool calls", func(t *testing.T) {
		for _, allow := range []bool{false, true} {
			for _, choice := range []a.ToolChoiceUnionParam{
				{}, {OfAuto: &a.ToolChoiceAutoParam{}}, {OfAny: &a.ToolChoiceAnyParam{}}, {OfTool: &a.ToolChoiceToolParam{Name: "lookup"}},
			} {
				applyAnthropicParallelToolCalls(&choice, &allow, true)
				data, err := json.Marshal(choice)
				if err != nil {
					t.Fatal(err)
				}
				var body map[string]any
				if err := json.Unmarshal(data, &body); err != nil {
					t.Fatal(err)
				}
				if body["disable_parallel_tool_use"] != !allow {
					t.Fatalf("choice: %s", data)
				}
			}
		}
		for _, stream := range []bool{false, true} {
			for _, allow := range []bool{false, true} {
				service := &mockAnthropicSvc{response: &a.Message{StopReason: a.StopReasonEndTurn}}
				generator := NewAnthropicGenerator(service)
				request := GenerationRequest{Model: "model", Tools: []Tool{{Name: "lookup"}}, Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: NewGenerationOptions(WithParallelToolCalls(allow))}
				if stream {
					for chunk := range generator.Stream(t.Context(), request) {
						if chunk.Err != nil {
							t.Fatal(chunk.Err)
						}
					}
				} else {
					if _, err := generator.Generate(t.Context(), request); err != nil {
						t.Fatal(err)
					}
				}
				if service.lastParams.ToolChoice.OfAuto == nil || service.lastParams.ToolChoice.OfAuto.DisableParallelToolUse.Value != !allow {
					t.Fatalf("choice: %#v", service.lastParams.ToolChoice)
				}
				if len(request.Options) != 1 || request.Options[GenerationOptionParallelToolCalls] != allow {
					t.Fatal("mutated request options")
				}
			}
		}
		allow := false
		choice := a.ToolChoiceUnionParam{}
		applyAnthropicParallelToolCalls(&choice, &allow, false)
		if choice.OfAuto != nil {
			t.Fatal("introduced tool choice without tools")
		}
		choice.OfNone = &a.ToolChoiceNoneParam{}
		applyAnthropicParallelToolCalls(&choice, &allow, true)
		if choice.OfAuto != nil {
			t.Fatal("changed none choice")
		}
		if _, err := parseAnthropicGenerationOptions(GenerationOptions{GenerationOptionParallelToolCalls: "false"}); err == nil {
			t.Fatal("accepted non-bool")
		}
	})
	for _, value := range []any{"5", int(5), ^uint(0)} {
		if _, err := parseAnthropicGenerationOptions(GenerationOptions{GenerationOptionTopK: value}); err == nil {
			t.Fatalf("accepted top_k %#v", value)
		}
	}
	defaults, err := parseAnthropicGenerationOptions(nil)
	if err != nil || defaults.TopK != nil {
		t.Fatalf("default top_k: %#v, %v", defaults, err)
	}
	for _, stream := range []bool{false, true} {
		for _, budget := range []bool{false, true} {
			service := &mockAnthropicSvc{response: &a.Message{StopReason: a.StopReasonEndTurn}}
			generator := NewAnthropicGenerator(service)
			option := WithReasoningEffort("high")
			if budget {
				option = WithThinkingBudget(2048)
			}
			request := GenerationRequest{Model: "model", SafetyIdentifier: "opaque-user", PromptCacheKey: "ignored", Dialog: Dialog{{Role: User, Blocks: []Block{TextBlock("hello")}}}, Options: NewGenerationOptions(option, WithTopK(0))}
			if stream {
				for chunk := range generator.Stream(t.Context(), request) {
					if chunk.Err != nil {
						t.Fatal(chunk.Err)
					}
				}
			} else {
				if _, err := generator.Generate(t.Context(), request); err != nil {
					t.Fatal(err)
				}
			}
			data, err := json.Marshal(service.lastParams)
			if err != nil {
				t.Fatal(err)
			}
			var body map[string]any
			if err := json.Unmarshal(data, &body); err != nil {
				t.Fatal(err)
			}
			if body["top_k"] != float64(0) {
				t.Fatalf("top_k zero: %s", data)
			}
			metadata := body["metadata"].(map[string]any)
			if metadata["user_id"] != "opaque-user" || body["prompt_cache_key"] != nil || body["safety_identifier"] != nil {
				t.Fatalf("metadata: %s", data)
			}
			thinking := body["thinking"].(map[string]any)
			if budget {
				if thinking["budget_tokens"] != float64(2048) || thinking["type"] != "enabled" {
					t.Fatalf("budget: %s", data)
				}
			} else if thinking["type"] != "adaptive" || body["output_config"].(map[string]any)["effort"] != "high" {
				t.Fatalf("effort: %s", data)
			}
		}
	}
}

func TestApplyAnthropicThinkingConfig(t *testing.T) {
	tests := []struct {
		name             string
		budget           string
		wantAdaptive     bool
		wantDisabled     bool
		wantBudgetTokens int64
		wantEffort       a.OutputConfigEffort
		wantInvalidParam bool
	}{
		{
			name:         "adaptive",
			budget:       "adaptive",
			wantAdaptive: true,
		},
		{
			name:         "disabled",
			budget:       "disabled",
			wantDisabled: true,
		},
		{
			name:             "numeric token budget",
			budget:           "5000",
			wantBudgetTokens: 5000,
		},
		{
			name:         "effort level enables adaptive thinking",
			budget:       "high",
			wantAdaptive: true,
			wantEffort:   a.OutputConfigEffortHigh,
		},
		{
			name:         "max effort level",
			budget:       "max",
			wantAdaptive: true,
			wantEffort:   a.OutputConfigEffortMax,
		},
		{
			name:         "normalizes level casing and whitespace",
			budget:       " XHIGH ",
			wantAdaptive: true,
			wantEffort:   a.OutputConfigEffortXhigh,
		},
		{
			name:             "invalid budget",
			budget:           "larger",
			wantInvalidParam: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var params a.MessageNewParams
			err := applyAnthropicThinkingConfig(&params, tt.budget)
			if tt.wantInvalidParam {
				var invalid *InvalidParameterErr
				if !errors.As(err, &invalid) {
					t.Fatalf("expected InvalidParameterErr, got %T: %v", err, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got := params.Thinking.OfAdaptive != nil; got != tt.wantAdaptive {
				t.Fatalf("adaptive mismatch: got %v, want %v", got, tt.wantAdaptive)
			}
			if got := params.Thinking.OfDisabled != nil; got != tt.wantDisabled {
				t.Fatalf("disabled mismatch: got %v, want %v", got, tt.wantDisabled)
			}
			if tt.wantBudgetTokens != 0 {
				if params.Thinking.OfEnabled == nil {
					t.Fatal("expected enabled thinking config")
				}
				if got := params.Thinking.OfEnabled.BudgetTokens; got != tt.wantBudgetTokens {
					t.Fatalf("budget tokens mismatch: got %d, want %d", got, tt.wantBudgetTokens)
				}
			}
			if got := params.OutputConfig.Effort; got != tt.wantEffort {
				t.Fatalf("effort mismatch: got %q, want %q", got, tt.wantEffort)
			}
		})
	}
}
