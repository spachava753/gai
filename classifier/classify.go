// Package classifier evaluates shared evidence against typed questions using
// probabilistic classifier models. It does not generate messages or execute
// decisions. Applications own thresholds, authorization, and subsequent actions.
//
// A Classifier stores execution dependencies, not a model, question schema, or
// history. Every Request supplies those inputs. Requests are borrowed read-only
// until Classify returns; implementations must not retain or mutate them.
package classifier

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/spachava753/gai"
)

// Classifier evaluates questions against shared evidence. A successful response
// contains one answer per question in request order, including explicit refusals.
// Confidence is implementation-specific. The interface does not promise that
// adding a question leaves other predictions unchanged, or that probabilities
// are calibrated on a particular workload.
type Classifier interface {
	// Classify honors cancellation and treats request as borrowed read-only data.
	// On error, usage and diagnostic metadata may be returned, but no answers
	// are actionable. Implementations document supported media and native options.
	Classify(context.Context, Request) (Response, error)
}

// Func adapts an ordinary function to Classifier, for middleware and test fixtures.
type Func func(context.Context, Request) (Response, error)

// Classify calls f.
func (f Func) Classify(ctx context.Context, request Request) (Response, error) {
	return f(ctx, request)
}

// Request contains all semantic inputs to one classification invocation.
type Request struct {
	// Headers overrides ordinary HTTP request headers without changing the client.
	// Values may contain secrets; do not log requests without redaction.
	Headers http.Header `json:"headers,omitempty"`
	// Model selects the provider model for this call.
	Model string `json:"model"`
	// SafetyIdentifier is an opaque end-user identity, mapped only where supported.
	SafetyIdentifier string `json:"safety_identifier,omitempty"`
	// Input is shared evidence, not a system prompt or a conversation to continue.
	Input Input `json:"input"`
	// Questions defines named judgments and their finite outcome schemas.
	Questions []Question `json:"questions"`
	// Options holds native, namespaced controls. Unknown keys are ignored.
	Options Options `json:"options,omitempty"`
}

// Input combines textual or structured evidence with optional media attachments.
// Attachments are not an interleaved conversation: providers choose their native
// placement relative to Data. Unsupported media must cause an error, never be
// discarded or processed through hidden OCR, uploads, or URL fetching.
type Input struct {
	// Data must marshal as a JSON string, object, or array. Structs, maps, slices,
	// and json.RawMessage are accepted. Nested values can be any JSON value;
	// use json.Number to preserve numeric representations. Nil is allowed only
	// with media, and adapters supply an empty textual state where required.
	Data any `json:"data,omitempty"`
	// Media contains ordinary image/audio/video Content blocks, not text,
	// thinking, tools, or stream metadata. Acceptance depends on the adapter.
	// Order within each modality is preserved; cross-modality ordering is not
	// part of this attachment contract.
	Media []gai.Block `json:"media,omitempty"`
}

// Options holds provider-specific controls. There are no shared sampling or
// generation options. Recognized values must have their documented Go types.
type Options map[string]any

// Question has exactly one of Predicate, Choice, or Score populated. Name is a
// correlation key, not implicit model instructions. Instructions and descriptions
// are guidance, not JSON Schemas, executable commands, or output fields.
type Question struct {
	// Name is a nonempty, unique identifier within the request.
	Name string `json:"name"`
	// Instructions accepts a string or a value marshaling to a JSON object or
	// array, including mixed nested values. Some providers require it. When
	// omitted, the question must supply criteria that define the judgment.
	Instructions any `json:"instructions,omitempty"`
	// Predicate estimates the probability that a proposition is true.
	Predicate *Predicate `json:"predicate,omitempty"`
	// Choice selects one of the supplied categorical outcomes.
	Choice *Choice `json:"choice,omitempty"`
	// Score evaluates an ordered rubric.
	Score *Score `json:"score,omitempty"`
}

// Predicate specifies a binary judgment, answered by P(true), not a boolean
// chosen using an implicit threshold.
type Predicate struct {
	// Criteria optionally defines what true and false mean.
	Criteria *PredicateCriteria `json:"criteria,omitempty"`
}

// PredicateCriteria gives additional guidance for either side of a predicate.
type PredicateCriteria struct {
	// True is optional string/object/array guidance for a positive answer.
	True any `json:"true,omitempty"`
	// False is optional string/object/array guidance for a negative answer.
	False any `json:"false,omitempty"`
}

// Choice declares at least two distinct, mutually exclusive outcomes. Use
// several predicates, not one Choice, for independent tags or multi-label tasks.
type Choice struct {
	// Options is the ordered list of allowed outcomes.
	Options []ChoiceOption `json:"options"`
}

// ChoiceOption defines one allowed categorical outcome.
type ChoiceOption struct {
	// Value is the supplied outcome, not generated text. String values are the
	// portable subset; some providers also accept booleans.
	Value Value `json:"value"`
	// Description is optional string/object/array guidance explaining the value.
	Description any `json:"description,omitempty"`
}

// Value represents a string or boolean outcome, with exactly one field populated.
// Its JSON representation is the scalar itself. False and "false" are distinct.
// Use StringValue or BooleanValue instead of allocating pointers manually.
type Value struct {
	// String contains a string outcome, including an explicitly empty string.
	String *string `json:"-"`
	// Boolean contains a boolean outcome, including explicit false.
	Boolean *bool `json:"-"`
}

// StringValue constructs a string outcome.
func StringValue(value string) Value { return Value{String: &value} }

// BooleanValue constructs a boolean outcome. Not all providers support it.
func BooleanValue(value bool) Value { return Value{Boolean: &value} }

// MarshalJSON encodes a valid Value as a JSON string or boolean.
func (v Value) MarshalJSON() ([]byte, error) {
	switch {
	case v.String != nil && v.Boolean == nil:
		return json.Marshal(*v.String)
	case v.Boolean != nil && v.String == nil:
		return json.Marshal(*v.Boolean)
	default:
		return nil, fmt.Errorf("classifier value requires exactly one string or boolean")
	}
}

// UnmarshalJSON decodes a scalar string or boolean without conflating their types.
func (v *Value) UnmarshalJSON(data []byte) error {
	var scalar any
	if err := json.Unmarshal(data, &scalar); err != nil {
		return err
	}
	switch value := scalar.(type) {
	case string:
		*v = StringValue(value)
	case bool:
		*v = BooleanValue(value)
	default:
		return fmt.Errorf("classifier value must be a string or boolean")
	}
	return nil
}

// Score declares an ordinal rubric with at least two levels. Its answer is the
// expected zero-based level index, not an argmax or arbitrary numeric regression.
type Score struct {
	// Levels is the rubric, ordered from lowest to highest.
	Levels []ScoreLevel `json:"levels"`
}

// ScoreLevel defines one ordinal outcome; its numeric value is its slice index.
type ScoreLevel struct {
	// Label names this level.
	Label string `json:"label"`
	// Description is optional string/object/array guidance for this level.
	Description any `json:"description,omitempty"`
}

// Response contains normalized answers and invocation accounting. There is no
// generated assistant message, finish reason, or tool execution implied here.
type Response struct {
	// Model is the provider-returned model name, which may merely echo an alias.
	Model string `json:"model"`
	// Answers follows request order. Refusals are successful nonnumeric outcomes.
	Answers []Answer `json:"answers"`
	// UsageMetadata holds provider-reported measurements. Input/output counts
	// use gai's common int-valued keys; absent counts are not invented as zero.
	// Fractional native measurements use json.Number, not float64.
	UsageMetadata gai.Metadata `json:"usage_metadata,omitempty"`
	// ExtraFields holds invocation-level native data under documented keys.
	ExtraFields map[string]any `json:"extra_fields,omitempty"`
}

// Answer contains exactly one typed result or refusal.
type Answer struct {
	// Name identifies its question.
	Name string `json:"name"`
	// Predicate contains the probability that the proposition is true.
	Predicate *PredicateAnswer `json:"predicate,omitempty"`
	// Choice contains the selected value and full categorical distribution.
	Choice *ChoiceAnswer `json:"choice,omitempty"`
	// Score contains the expected index and full ordinal distribution.
	Score *ScoreAnswer `json:"score,omitempty"`
	// Refusal means the implementation declined to answer, not uncertainty.
	Refusal *RefusalAnswer `json:"refusal,omitempty"`
	// Confidence is optional and implementation-specific: it may be forwarded
	// or calculated locally. It lies in [0,1], is absent for refusals, and is
	// not universally the probability that the answer is correct.
	Confidence *json.Number `json:"confidence,omitempty"`
	// ExtraFields holds native answer data, such as an echoed score legend.
	ExtraFields map[string]any `json:"extra_fields,omitempty"`
}

// PredicateAnswer is a Bernoulli probability; applications choose thresholds.
type PredicateAnswer struct {
	// Probability is P(true), in [0,1]. Zero is a valid negative prediction.
	Probability json.Number `json:"probability"`
}

// ChoiceAnswer preserves the selected value and every allowed outcome.
type ChoiceAnswer struct {
	// Value is a highest-probability supplied outcome; native tie selection is kept.
	Value Value `json:"value"`
	// Probabilities follows declared option order, including zero probabilities.
	Probabilities []ChoiceProbability `json:"probabilities"`
}

// ChoiceProbability assigns probability to one supplied outcome.
type ChoiceProbability struct {
	// Value identifies the outcome, preserving its scalar type.
	Value Value `json:"value"`
	// Probability lies in [0,1].
	Probability json.Number `json:"probability"`
}

// ScoreAnswer preserves the reported weighted-average level index. The score is
// approximately sum(index * probability); serialization rounding may differ.
type ScoreAnswer struct {
	// Score lies between zero and the highest level index; it can be fractional.
	Score json.Number `json:"score"`
	// Probabilities follows numeric level order, with complete coverage.
	Probabilities []ScoreProbability `json:"probabilities"`
}

// ScoreProbability assigns probability to a zero-based rubric level.
type ScoreProbability struct {
	// Index identifies the level in the question's Levels slice.
	Index int `json:"index"`
	// Probability lies in [0,1].
	Probability json.Number `json:"probability"`
}

// RefusalAnswer is a nonnumeric result; it is not a transport failure and should
// not automatically trigger retry or provider fallback.
type RefusalAnswer struct {
	// Reason is empty when the API provides no explanation.
	Reason string `json:"reason,omitempty"`
}

// ResponseError reports a malformed or incomplete classifier response. Answers
// returned alongside this error must not be acted upon.
type ResponseError struct {
	// Question identifies the affected question, or is empty for envelope errors.
	Question string
	// Reason describes the violated response contract.
	Reason string
}

// Error describes the invalid response without embedding evidence or credentials.
func (e *ResponseError) Error() string {
	return fmt.Sprintf("classifier response %q: %s", e.Question, e.Reason)
}
