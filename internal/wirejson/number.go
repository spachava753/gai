// Package wirejson provides JSON types used by generated provider clients.
package wirejson

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Number is a JSON number that preserves its original spelling. Unlike
// json.Number, it rejects quoted numbers when decoding. Nullable fields in
// generated clients handle null separately.
type Number json.Number

// UnmarshalJSON sets n to an unquoted JSON number. On error, n is unchanged.
func (n *Number) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || !bytes.ContainsAny(data[:1], "-0123456789") || !json.Valid(data) {
		return fmt.Errorf("expected an unquoted JSON number")
	}
	*n = Number(data)
	return nil
}

// MarshalJSON returns n as a JSON number without converting it to float64.
func (n Number) MarshalJSON() ([]byte, error) {
	if n == "" {
		return nil, fmt.Errorf("missing JSON number")
	}
	return json.Marshal(json.Number(n))
}
