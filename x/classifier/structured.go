package classifier

import "encoding/json"

// Box asks for the location of an object in an image. Instructions describe
// what to locate. TypeSafe currently supports this experimental question type.
type Box struct{}

// BoxAnswer contains normalized image coordinates, measured from the top left.
// Coordinates are between zero and one; Top < Bottom and Left < Right.
type BoxAnswer struct {
	Top    json.Number `json:"top"`
	Bottom json.Number `json:"bottom"`
	Left   json.Number `json:"left"`
	Right  json.Number `json:"right"`
}

// Extraction asks for a value found or computed from the input. Cloudflare
// Clairic currently supports extraction. A missing value is not a refusal.
type Extraction struct {
	// Type is text, number, date, time, datetime, duration, or list.
	Type string `json:"type"`
	// Items is required for lists: text, number, or date. Otherwise leave it empty.
	Items string `json:"items,omitempty"`
	// Unit labels numeric values. For durations it selects seconds, minutes,
	// hours, days, weeks, months, or years; omission uses the server default.
	Unit string `json:"unit,omitempty"`
	// Integer requests integral numbers or durations.
	Integer bool `json:"integer,omitempty"`
	// Decimals requests zero to six decimal places for numbers, durations,
	// or number lists. Nil leaves the server default unchanged.
	Decimals *int `json:"decimals,omitempty"`
	// NullThreshold is the probability of absence above which Value becomes
	// nil. It must be greater than zero and at most one. Nil uses the server default.
	NullThreshold *json.Number `json:"null_threshold,omitempty"`
}

// Estimate asks for a numeric or date estimate on a bounded grid. Cloudflare
// Clairic currently supports estimates. Bounds do not imply certainty: the
// answer can report probability below or above the grid.
type Estimate struct {
	// Min and Max are both json.Number values or both ISO dates (YYYY-MM-DD).
	// Min must be less than Max. Floating-point Go values are not accepted.
	Min any `json:"min"`
	Max any `json:"max"`
	// Step is a positive json.Number, or "month" or "year" for date grids.
	// A numeric date step is measured in days. Nil leaves the server default.
	Step any `json:"step,omitempty"`
	// Scale is "linear" or "log". Empty leaves the server default unchanged.
	Scale string `json:"scale,omitempty"`
	// NullThreshold has the same meaning as Extraction.NullThreshold.
	NullThreshold *json.Number `json:"null_threshold,omitempty"`
}

// Forecast asks for a sequence of estimates. Cloudflare Clairic currently
// supports forecasts. Each step uses the same grid.
type Forecast struct {
	Estimate
	// Horizon is the number of steps, from 1 to 64.
	Horizon int `json:"horizon"`
	// Interval is step, minute, hour, day, week, month, quarter, or year.
	Interval string `json:"interval"`
}

// ExtractionAnswer contains an extracted or computed value. Value is nil when
// the input does not determine it. This differs from Answer.Refused.
type ExtractionAnswer struct {
	// Value is a string for text/date/time/datetime, a json.Number for
	// number/duration, or []any of strings or json.Number for a list.
	Value any `json:"value"`
	// Guess is the provider's candidate before applying its null threshold.
	// It uses the same types as Value; nil means absent or null.
	Guess any `json:"guess,omitempty"`
	// NullProbability is the probability that the input does not determine a value.
	NullProbability *json.Number `json:"null_probability,omitempty"`
	// Span and Spans are provider character offsets into the input text.
	// Each span is a start/end pair. Spans corresponds to list entries.
	// Cloudflare's returned state_text, when requested, is retained in
	// Response.ExtraFields under ExtraFieldWireFields.
	Span  []int   `json:"span,omitempty"`
	Spans [][]int `json:"spans,omitempty"`
}

// EstimateAnswer contains the provider's estimate and uncertainty summaries.
// Values are json.Number or ISO date strings, according to the question's grid.
// Nil means absent or null; no missing statistic is inferred.
type EstimateAnswer struct {
	// Confidence is the provider's optional confidence, including for forecast steps.
	Confidence       *json.Number `json:"confidence,omitempty"`
	Value            any          `json:"value"`
	Mean             any          `json:"mean,omitempty"`
	Median           any          `json:"median,omitempty"`
	P10              any          `json:"p10,omitempty"`
	P90              any          `json:"p90,omitempty"`
	BelowProbability *json.Number `json:"below_probability,omitempty"`
	AboveProbability *json.Number `json:"above_probability,omitempty"`
	NullProbability  *json.Number `json:"null_probability,omitempty"`
	// Quantiles retains the provider's percentile entries without converting
	// numbers through float64. Entry structure is provider-defined.
	Quantiles []any `json:"quantiles,omitempty"`
	// ExtraFields retains other provider fields under ExtraFieldWireFields.
	ExtraFields map[string]any `json:"extra_fields,omitempty"`
}

// ForecastAnswer contains one estimate per requested step, in provider order.
// Value retains the provider's aggregate value (including null); its structure
// is provider-defined, and numbers use json.Number.
type ForecastAnswer struct {
	// NullProbability is the provider's probability of an undetermined forecast.
	NullProbability *json.Number     `json:"null_probability,omitempty"`
	Value           any              `json:"value"`
	Steps           []EstimateAnswer `json:"steps"`
}
