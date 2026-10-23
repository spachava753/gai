package classifier

import (
	"encoding/json"
	"slices"
	"time"
)

func requireQuestionKinds(questions []preparedQuestion, kinds ...string) error {
	for _, q := range questions {
		if !slices.Contains(kinds, q.kind) {
			return invalid(q.name, "question type is not supported by this endpoint")
		}
	}
	return nil
}

func validateNullThreshold(value *json.Number) error {
	if value != nil {
		n, err := boundedNumber(*value, 1)
		if err != nil || n.Sign() == 0 {
			return invalid("NullThreshold", "must be greater than zero and at most one")
		}
	}
	return nil
}

// validateExtraction checks the extraction tag and dependent list, precision,
// and duration options without inventing a model capability table.
func validateExtraction(e Extraction) error {
	if !slices.Contains([]string{"text", "number", "date", "time", "datetime", "duration", "list"}, e.Type) {
		return invalid("Extraction.Type", "unsupported extraction type")
	}
	if e.Type == "list" {
		if !slices.Contains([]string{"text", "number", "date"}, e.Items) {
			return invalid("Extraction.Items", "list requires text, number, or date items")
		}
	} else if e.Items != "" {
		return invalid("Extraction.Items", "only lists accept Items")
	}
	if e.Integer && e.Type != "number" && e.Type != "duration" {
		return invalid("Extraction.Integer", "only numbers and durations accept Integer")
	}
	if e.Decimals != nil && (*e.Decimals < 0 || *e.Decimals > 6 || (e.Type != "number" && e.Type != "duration" && !(e.Type == "list" && e.Items == "number"))) {
		return invalid("Extraction.Decimals", "requires number, duration, or number list and zero to six places")
	}
	if e.Type == "duration" && e.Unit != "" && !slices.Contains([]string{"seconds", "minutes", "hours", "days", "weeks", "months", "years"}, e.Unit) {
		return invalid("Extraction.Unit", "invalid duration unit")
	}
	return validateNullThreshold(e.NullThreshold)
}

// validateEstimate compares exact numeric or ISO-date bounds, validates the
// grid step and scale, and checks the optional null threshold.
func validateEstimate(e Estimate) error {
	switch low := e.Min.(type) {
	case json.Number:
		high, ok := e.Max.(json.Number)
		if !ok {
			return invalid("Estimate.Max", "numeric bounds require json.Number")
		}
		l, le := exactNumber(low)
		h, he := exactNumber(high)
		if le != nil || he != nil || l.Cmp(h) >= 0 {
			return invalid("Estimate", "requires valid Min < Max")
		}
		if e.Scale == "log" && l.Sign() <= 0 {
			return invalid("Estimate.Min", "logarithmic grids require positive bounds")
		}
	case string:
		high, ok := e.Max.(string)
		l, le := time.Parse(time.DateOnly, low)
		h, he := time.Parse(time.DateOnly, high)
		if !ok || le != nil || he != nil || !l.Before(h) {
			return invalid("Estimate", "requires ISO date bounds with Min < Max")
		}
	default:
		return invalid("Estimate.Min", "requires json.Number or an ISO date")
	}
	if e.Scale != "" && e.Scale != "linear" && e.Scale != "log" {
		return invalid("Estimate.Scale", "requires linear or log")
	}
	if e.Step != nil {
		switch step := e.Step.(type) {
		case json.Number:
			n, err := exactNumber(step)
			if err != nil || n.Sign() <= 0 {
				return invalid("Estimate.Step", "requires a positive number")
			}
		case string:
			_, dateGrid := e.Min.(string)
			if !dateGrid || (step != "month" && step != "year") {
				return invalid("Estimate.Step", "month/year steps require date bounds")
			}
		default:
			return invalid("Estimate.Step", "requires json.Number or month/year")
		}
	}
	return validateNullThreshold(e.NullThreshold)
}

func validateForecast(f Forecast) error {
	if f.Horizon < 1 || f.Horizon > 64 {
		return invalid("Forecast.Horizon", "requires 1 to 64 steps")
	}
	if !slices.Contains([]string{"step", "minute", "hour", "day", "week", "month", "quarter", "year"}, f.Interval) {
		return invalid("Forecast.Interval", "invalid forecast interval")
	}
	return validateEstimate(f.Estimate)
}
