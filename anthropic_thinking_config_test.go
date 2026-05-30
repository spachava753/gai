package gai

import (
	"errors"
	"testing"

	a "github.com/anthropics/anthropic-sdk-go"
)

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
