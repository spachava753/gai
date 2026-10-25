package gai

import (
	"fmt"
	"time"
)

// promptCacheTTL validates the shared type/positivity contract and formats the
// duration without rounding. The API validates supported lifetimes.
// An absent option is distinct from an explicitly supplied zero duration.
func promptCacheTTL(options GenerationOptions) (string, error) {
	ttl, present, err := generationOption[time.Duration](options, GenerationOptionPromptCacheTTL)
	if err != nil || !present {
		return "", err
	}
	if ttl <= 0 {
		return "", &InvalidParameterErr{
			Parameter: GenerationOptionPromptCacheTTL,
			Reason:    "must be a positive time.Duration",
		}
	}
	if ttl%time.Hour == 0 {
		return fmt.Sprintf("%dh", ttl/time.Hour), nil
	}
	if ttl%time.Minute == 0 {
		return fmt.Sprintf("%dm", ttl/time.Minute), nil
	}
	return ttl.String(), nil
}
