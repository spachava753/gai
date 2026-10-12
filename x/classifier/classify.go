// Package classifier provides clients for classifying and scoring input with AI models.
//
// A [Request] contains input and one or more questions about it. A [Predicate]
// asks a yes-or-no question, a [Choice] asks the model to select an option, and
// a [Score] asks it to rate the input on a scale. The [Response] contains an
// answer for each question, with probabilities rather than generated text.
// Applications decide how to use those answers.
//
// Use [NewTypeSafe] for Jev, [NewOpenAI] for OpenAI Decisions, [NewCloudflare]
// for hosted Clef, or [NewPerplexity] for Perplexity Decisions. These clients
// are safe for concurrent use. They do not keep conversation history or retry
// failed calls. The supplied HTTP client's transport and redirect policies
// still apply.
//
// Probabilities, scores, and confidence values use [json.Number] to preserve
// the numbers returned by the provider. Confidence has a different meaning
// for each implementation; see [Answer.Confidence].
//
// This package is experimental; its API may change.
package classifier

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/spachava753/gai"
)

// A Classifier answers questions about input. Each call supplies its own model,
// input, and questions.
//
// All questions in a request use the same input. They do not use each other's
// answers, but adding or removing a question may change the model's predictions.
type Classifier interface {
	// Classify returns one answer for each question, in request order.
	// A model's refusal sets [Answer.Refused] to true; it is not an error.
	//
	// Classify must honor context cancellation. It must not modify request
	// or retain its data after returning. Callers must not modify request,
	// including its maps, slices, and pointed-to values, until the call returns.
	// Returned answers must not share mutable data with request.
	//
	// On error, the response may contain usage counts and diagnostic fields,
	// but callers must not use its answers. The clients in this package return
	// [gai.InvalidParameterErr] for invalid request fields, *[gai.ApiErr] for
	// HTTP API errors, and *[ResponseError] for invalid responses. Unsupported
	// media may return [gai.UnsupportedInputModalityErr].
	Classify(context.Context, Request) (Response, error)
}

// Func implements [Classifier] by calling a function with the same signature.
type Func func(context.Context, Request) (Response, error)

// Classify calls f.
func (f Func) Classify(ctx context.Context, request Request) (Response, error) {
	return f(ctx, request)
}

// A Request specifies a model, input, and questions for one call to Classify.
//
// The clients check requests before sending them, including encoding Data,
// Instructions, and descriptions with [json.Marshal]. Custom marshalers must
// not modify the request and must be safe for concurrent use if their values
// are shared.
type Request struct {
	// Headers contains HTTP headers to set for this call. Each entry replaces
	// the client's default values for that header, including Authorization.
	// A nil or empty value slice removes the header. The client is not modified.
	// Headers may contain credentials; avoid logging them.
	Headers http.Header `json:"headers,omitempty"`

	// Model is the name of the model to use. It must not be empty.
	Model string `json:"model"`

	// SafetyIdentifier identifies the end user to the provider. OpenAI sends it
	// as safety_identifier. The other clients in this package ignore it.
	SafetyIdentifier string `json:"safety_identifier,omitempty"`

	// Input is the content to evaluate. All questions use this input.
	Input Input `json:"input"`

	// Questions contains the questions to answer. It must contain at least one
	// question, and each question must have a different name.
	Questions []Question `json:"questions"`
}

// Input is the content a model evaluates. It can contain text, structured data,
// media attachments, or a combination of these. At least one of Data or Media
// must be supplied.
type Input struct {
	// Data contains text or structured data. It must encode as a JSON string,
	// object, or array when passed to [json.Marshal]. For example:
	//
	//	Input{Data: "Please refund the duplicate charge."}
	//	Input{Data: map[string]any{"message": "Please refund it.", "amount": 25}}
	//
	// Objects and arrays may contain any JSON values. Use [json.Number] for
	// numbers that must not be rounded to float64. Data may be nil if Media is
	// not empty. An empty string is valid; a value that encodes as null is not.
	Data any `json:"data,omitempty"`

	// Media contains image, audio, or video attachments. Each block must have
	// BlockType gai.Content, a MIME type, and nonempty base64-encoded content.
	// The provider determines which formats are supported. Unsupported formats
	// cause an error; clients do not fetch URLs or convert attachments.
	//
	// Attachments of the same kind keep their order. A provider may group
	// images, audio, and video separately and choose where to place Data.
	Media []gai.Block `json:"media,omitempty"`
}

// A Question tells the model what to determine about the input.
// Exactly one of Predicate, Choice, or Score must be non-nil.
type Question struct {
	// Name identifies the question and its answer. It must be nonempty and
	// unique within the request. It is not a substitute for Instructions.
	Name string `json:"name"`

	// Instructions tells the model what to decide, for example,
	// "Does the customer request a refund?" It can also include context or
	// examples needed for this question.
	//
	// Instructions accepts the same JSON types as [Input.Data]. Object keys
	// are caller-defined; an array describes one question, not separate calls.
	// It may be nil when the options, levels, or predicate criteria define the
	// question, but some providers require instructions in every question.
	Instructions any `json:"instructions,omitempty"`

	// Predicate asks a yes-or-no question.
	Predicate *Predicate `json:"predicate,omitempty"`

	// Choice asks the model to select one of the supplied options.
	Choice *Choice `json:"choice,omitempty"`

	// Score asks the model to rate the input using the supplied levels.
	Score *Score `json:"score,omitempty"`
}

// A Predicate asks a yes-or-no question. Its answer is the probability of yes,
// not a boolean. The caller chooses a threshold if it needs a boolean result.
type Predicate struct {
	// Criteria describes when the answer should be yes or no.
	// It may be nil if the question has instructions.
	Criteria *PredicateCriteria `json:"criteria,omitempty"`
}

// PredicateCriteria describes the two possible outcomes of a [Predicate].
// Either field may be nil. Non-nil fields accept the same JSON types as
// [Question.Instructions].
type PredicateCriteria struct {
	// True describes when the answer should be yes.
	True any `json:"true,omitempty"`

	// False describes when the answer should be no.
	False any `json:"false,omitempty"`
}

// A Choice asks the model to select one option from a list. Its answer includes
// the selected value and the probability of each option. To ask whether several
// labels apply, use a separate [Predicate] for each label.
type Choice struct {
	// Options lists the allowed choices. There must be at least two, with
	// distinct values. The answer's probabilities follow this order.
	Options []ChoiceOption `json:"options"`
}

// A ChoiceOption is one of the options in a [Choice].
type ChoiceOption struct {
	// Value is the value returned when this option is selected. All clients
	// support strings; only OpenAI also supports booleans.
	Value Value `json:"value"`

	// Description explains when to select this option. It may be nil;
	// otherwise it accepts the same JSON types as [Question.Instructions].
	Description any `json:"description,omitempty"`
}

// A Value is a string or boolean used in a [Choice]. Exactly one field must be
// non-nil; the zero Value is invalid. Use [StringValue] or [BooleanValue] to
// construct a Value.
//
// A Value encodes as a JSON string or boolean, not an object. Boolean false and
// the string "false" are different values.
type Value struct {
	// String holds the value when it is a string. The string may be empty.
	String *string `json:"-"`

	// Boolean holds the value when it is a boolean.
	Boolean *bool `json:"-"`
}

// StringValue returns a [Value] containing value as a string.
func StringValue(value string) Value { return Value{String: &value} }

// BooleanValue returns a [Value] containing value as a boolean.
// Not all providers accept boolean choices.
func BooleanValue(value bool) Value { return Value{Boolean: &value} }

// MarshalJSON returns v encoded as a JSON string or boolean.
// It returns an error unless exactly one of v.String or v.Boolean is non-nil.
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

// UnmarshalJSON sets v to the JSON string or boolean in data.
// Other JSON types, including null, are errors. On error, v is unchanged.
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

// A Score asks the model to rate input on an ordered scale, such as low, medium,
// and high. Levels have numeric values 0, 1, 2, and so on. The answer is the
// probability-weighted average of those values; it may fall between levels.
type Score struct {
	// Levels lists at least two levels, from lowest to highest.
	Levels []ScoreLevel `json:"levels"`
}

// A ScoreLevel is one level in a [Score]. Its numeric value is its index in Levels.
type ScoreLevel struct {
	// Label names the level. It must not be empty.
	Label string `json:"label"`

	// Description explains when this level applies. It may be nil;
	// otherwise it accepts the same JSON types as [Question.Instructions].
	Description any `json:"description,omitempty"`
}

// A Response contains answers and usage information from a call to Classify.
//
// The built-in clients check that answers match the questions and include a
// probability for every option or level. Invalid responses cause Classify to
// return a *[ResponseError]. Callers do not need to validate responses themselves.
//
// Numbers are checked without converting to float64. The checks allow a rounding
// error of 0.00005 per probability and per reported score. These allowances add
// up when checking a sum or weighted average. For example, three probabilities
// may differ from a sum of 1 by up to 0.00015. Each probability and confidence
// must still be between 0 and 1, inclusive.
//
// Each number must be a valid JSON number of at most 256 bytes, with an exponent
// between -10000 and 10000. An empty or invalid number is an error, not zero.
type Response struct {
	// Model is the model name returned by the provider. It may be an alias,
	// rather than the name of a specific model version.
	Model string `json:"model"`

	// Answers contains one answer per question, in request order.
	// An answer may be a refusal; see [Answer.Refused].
	Answers []Answer `json:"answers"`

	// UsageMetadata contains usage counts reported by the provider. Use
	// [gai.InputTokens] and [gai.OutputTokens] to read token counts. A missing
	// count means it was not reported, not that it was zero.
	//
	// Counts cover the entire call, including questions and attachments, not
	// just Input.Data. Token counts use int; fractional measurements, if
	// supplied by an implementation, use [json.Number].
	UsageMetadata gai.Metadata `json:"usage_metadata,omitempty"`

	// ExtraFields contains provider-specific response data. The built-in
	// clients use [ExtraFieldHeaders], [ExtraFieldProvider], [ExtraFieldUsage],
	// [ExtraFieldWireFields], and, for Cloudflare, [ExtraFieldEnvelope].
	ExtraFields map[string]any `json:"extra_fields,omitempty"`
}

// An Answer is the result for one question. If Refused is false, exactly one
// of Predicate, Choice, or Score is non-nil and matches the question type.
// If Refused is true, all three and Confidence must be nil.
type Answer struct {
	// Name is the name of the question this answer belongs to.
	Name string `json:"name"`

	// Predicate contains the answer to a yes-or-no question.
	Predicate *PredicateAnswer `json:"predicate,omitempty"`

	// Choice contains the selected option and each option's probability.
	Choice *ChoiceAnswer `json:"choice,omitempty"`

	// Score contains the numeric score and each level's probability.
	Score *ScoreAnswer `json:"score,omitempty"`

	// Refused is true when the model declines to answer. A refusal is not
	// a negative answer or a low-confidence prediction.
	Refused bool `json:"refused,omitempty"`

	// Confidence measures how certain the implementation is about the answer.
	// It is between 0 and 1, inclusive. Nil means unavailable, not zero.
	// It is always nil for a refusal.
	//
	// Each implementation defines what confidence means. It may return the
	// provider's value, calculate one, or omit it. The built-in clients return
	// the provider's value when present. Do not assume it is the probability
	// that the answer is correct or compare it across implementations.
	Confidence *json.Number `json:"confidence,omitempty"`

	// ExtraFields contains provider-specific data for this answer.
	// See [ExtraFieldWireFields].
	ExtraFields map[string]any `json:"extra_fields,omitempty"`
}

// A PredicateAnswer gives the probability that the answer to a [Predicate] is yes.
type PredicateAnswer struct {
	// Probability is the probability of yes, between 0 and 1, inclusive.
	// For example, "0.8" means an 80% probability of yes.
	Probability json.Number `json:"probability"`
}

// A ChoiceAnswer contains the selected option and the probability of every option.
type ChoiceAnswer struct {
	// Value is the option selected by the model. It has the highest reported
	// probability, allowing for rounding. The model breaks ties.
	Value Value `json:"value"`

	// Probabilities contains one entry per option, in the order given by
	// [Choice.Options]. Options with zero probability are included.
	// The probabilities sum to 1, allowing for rounding; see [Response].
	Probabilities []ChoiceProbability `json:"probabilities"`
}

// A ChoiceProbability gives the probability of one option in a [Choice].
type ChoiceProbability struct {
	// Value is the option's value, with the same string or boolean type
	// used in the question.
	Value Value `json:"value"`

	// Probability is the probability of this option, between 0 and 1, inclusive.
	Probability json.Number `json:"probability"`
}

// A ScoreAnswer contains a score and the probability of each level.
// The score is the sum of each level's index multiplied by its probability.
// For example, probabilities 0.2, 0.3, and 0.5 for levels 0, 1, and 2 give:
//
//	0*0.2 + 1*0.3 + 2*0.5 = 1.3
//
// Reported scores may differ slightly due to rounding; see [Response].
type ScoreAnswer struct {
	// Score is the numeric score returned by the model. It is between zero
	// and the highest index in the question's [Score.Levels], inclusive.
	// It need not be an integer.
	Score json.Number `json:"score"`

	// Probabilities contains one probability per level, in the order given by
	// [Score.Levels], including levels with zero probability. Probabilities[i]
	// is the probability of level i. Each value is between 0 and 1, inclusive,
	// and the values sum to 1, allowing for rounding.
	Probabilities []json.Number `json:"probabilities"`
}

// A ResponseError describes an invalid or incomplete response from a classifier.
// Callers must not use answers returned with this error.
type ResponseError struct {
	// Question is the name of the affected question. It is empty if the error
	// applies to the response as a whole.
	Question string

	// Reason describes what is wrong with the response.
	Reason string
}

// Error returns a description of the error, including the question name and reason.
func (e *ResponseError) Error() string {
	return fmt.Sprintf("classifier response %q: %s", e.Question, e.Reason)
}
