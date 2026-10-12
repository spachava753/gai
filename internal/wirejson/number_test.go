package wirejson

import (
	"encoding/json"
	"testing"
)

func TestNumber(t *testing.T) {
	t.Run("missing number cannot be encoded as zero", func(t *testing.T) {
		if _, err := json.Marshal(Number("")); err == nil {
			t.Fatal("encoded a missing number")
		}
	})
	for _, test := range []struct {
		input string
		valid bool
	}{
		{"0", true}, {"-0", true}, {"1.1000", true}, {"9007199254740993", true},
		{"0.123456789012345678901", true}, {"1e-10000", true},
		{`"0.75"`, false}, {"null", false}, {"true", false}, {"[]", false}, {"{}", false},
		{"", false}, {"NaN", false}, {"01", false},
	} {
		t.Run(test.input, func(t *testing.T) {
			value := Number("42")
			err := json.Unmarshal([]byte(test.input), &value)
			if (err == nil) != test.valid {
				t.Fatalf("Unmarshal = %v, valid = %v", err, test.valid)
			}
			if !test.valid {
				if value != "42" {
					t.Fatal("invalid input changed the number")
				}
				return
			}
			data, err := json.Marshal(value)
			if err != nil || string(data) != test.input {
				t.Fatalf("round trip = %s, %v", data, err)
			}
		})
	}
}
