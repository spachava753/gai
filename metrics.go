package gai

// Metadata stores usage measurements in [Response.UsageMetadata] and
// [MetadataBlock]. Common keys have typed accessors; provider-specific keys
// document their concrete value types and can be read with [GetMetric].
type Metadata map[string]any

const (
	// UsageMetricInputTokens is the int [Metadata] key read by [InputTokens].
	UsageMetricInputTokens = "input_tokens"

	// UsageMetricGenerationTokens is the int [Metadata] key read by
	// [OutputTokens].
	UsageMetricGenerationTokens = "gen_tokens"

	// UsageMetricCacheReadTokens is the int [Metadata] key read by
	// [CacheReadTokens].
	UsageMetricCacheReadTokens = "cache_read_tokens"

	// UsageMetricCacheWriteTokens is the int [Metadata] key read by
	// [CacheWriteTokens].
	UsageMetricCacheWriteTokens = "cache_write_tokens"

	// UsageMetricReasoningTokens is the int [Metadata] key for provider-reported
	// reasoning tokens. Read it with [GetMetric].
	UsageMetricReasoningTokens = "reasoning_tokens"
)

// InputTokens reads [UsageMetricInputTokens]. It returns false when the key is
// absent and panics when the stored value is not an int.
func InputTokens(m Metadata) (int, bool) {
	return GetMetric[int](m, UsageMetricInputTokens)
}

// OutputTokens reads [UsageMetricGenerationTokens]. It returns false when the
// key is absent and panics when the stored value is not an int.
func OutputTokens(m Metadata) (int, bool) {
	return GetMetric[int](m, UsageMetricGenerationTokens)
}

// CacheReadTokens reads [UsageMetricCacheReadTokens]. It returns false when
// the key is absent and panics when the stored value is not an int.
func CacheReadTokens(m Metadata) (int, bool) {
	return GetMetric[int](m, UsageMetricCacheReadTokens)
}

// CacheWriteTokens reads [UsageMetricCacheWriteTokens]. It returns false when
// the key is absent and panics when the stored value is not an int.
func CacheWriteTokens(m Metadata) (int, bool) {
	return GetMetric[int](m, UsageMetricCacheWriteTokens)
}

// GetMetric reads key from m and asserts its value to T. It returns the zero
// value and false when key is absent. It panics when a present value is not T;
// use the concrete type documented by the metric key.
func GetMetric[T any](m Metadata, key string) (T, bool) {
	var metric T
	metricVal, ok := m[key]
	if !ok {
		return metric, false
	}
	metric = metricVal.(T)
	return metric, true
}
