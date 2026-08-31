package gai

import (
	"testing"
	"time"
)

func TestBackoffPolicyScenarios(t *testing.T) {
	t.Run("ExponentialBackoffAppliesDownwardJitter", func(t *testing.T) {
		minimumJitter := func(int64) int64 { return 0 }
		backoff := exponentialBackoff(minimumJitter)

		got, retry := backoff(4)
		if !retry || got != 2*time.Second {
			t.Fatalf("backoff(4) = (%s, %t), want (2s, true)", got, retry)
		}
	})
	t.Run("ExponentialBackoffProgressionAndCap", func(t *testing.T) {
		maximumJitter := func(limit int64) int64 { return limit - 1 }
		backoff := exponentialBackoff(maximumJitter)
		want := []time.Duration{
			500 * time.Millisecond,
			time.Second,
			2 * time.Second,
			4 * time.Second,
			8 * time.Second,
			15 * time.Second,
			15 * time.Second,
		}

		for i, wantDelay := range want {
			got, retry := backoff(uint(i + 1))
			if !retry || got != wantDelay {
				t.Fatalf("backoff(%d) = (%s, %t), want (%s, true)", i+1, got, retry, wantDelay)
			}
		}
	})
}
