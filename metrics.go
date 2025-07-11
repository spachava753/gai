package gai

// Metrics represents a collection of metrics returned by a Generator in a Response.
// The map's keys are metric names, and values can be of any type, though typically
// they are numeric types like int or float64.
//
// Two common metrics that a Generator typically returns are:
//   - [UsageMetricInputTokens]: The number of tokens in the input Dialog
//   - [UsageMetricGenerationTokens]: The number of tokens generated in the Response
//
// A Generator may return additional implementation-specific metrics.
type Metrics map[string]any

const (
	// UsageMetricInputTokens is a metric key representing the number of tokens in the input Dialog.
	// The value associated with this key is expected to be of type int.
	UsageMetricInputTokens = "input_tokens"

	// UsageMetricGenerationTokens is a metric key representing the number of tokens generated
	// in the Response. The value associated with this key is expected to be of type int.
	UsageMetricGenerationTokens = "gen_tokens"
)

// InputTokens returns the number of tokens in the input Dialog from the metrics.
// The first return value is the number of input tokens, and the second indicates
// whether the metric was present in the map.
//
// If the metric is not present, returns (0, false).
// If the metric is present, returns (tokens, true).
//
// Panics if the value in the metrics map cannot be type asserted to int.
func InputTokens(m Metrics) (int, bool) {
	return GetMetric[int](m, UsageMetricInputTokens)
}

// OutputTokens returns the number of tokens generated in the Response from the metrics.
// The first return value is the number of generated tokens, and the second indicates
// whether the metric was present in the map.
//
// If the metric is not present, returns (0, false).
// If the metric is present, returns (tokens, true).
//
// Panics if the value in the metrics map cannot be type asserted to int.
func OutputTokens(m Metrics) (int, bool) {
	return GetMetric[int](m, UsageMetricGenerationTokens)
}

// GetMetric is a generic function that retrieves a metric value of type T from the metrics map.
// The first return value is the metric value of type T, and the second indicates whether
// the metric was present in the map.
//
// If the metric is not present, returns (zero value of T, false).
// If the metric is present, returns (metric value, true).
//
// Panics if the value in the metrics map cannot be type asserted to T.
//
// Example usage:
//
//	// Get a float64 metric
//	if cost, ok := GetMetric[float64](metrics, "cost"); ok {
//	    fmt.Printf("Request cost: $%.2f\n", cost)
//	}
//
//	// Get a string metric
//	if model, ok := GetMetric[string](metrics, "model"); ok {
//	    fmt.Printf("Model used: %s\n", model)
//	}
func GetMetric[T any](m Metrics, key string) (T, bool) {
	var metric T
	metricVal, ok := m[key]
	if !ok {
		return metric, false
	}
	metric = metricVal.(T)
	return metric, true
}
