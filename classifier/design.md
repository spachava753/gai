# Classifier design

Package `classifier` provides a public sibling to `gai.Generator`. It evaluates shared evidence against typed questions; it does not generate messages, execute tools, or choose application actions.

```go
type Classifier interface {
    Classify(context.Context, Request) (Response, error)
}
```

Every request contains its model, evidence, question schema, headers, and options. Implementations retain only execution dependencies such as HTTP clients and credentials. Request slices, maps, pointers, and custom JSON marshalers are borrowed read-only during the invocation. Returned answers must not alias mutable request values. Clients and user-provided marshalers must support concurrent use when shared.

## Evidence and guidance

`Input.Data` accepts a string or a Go value that marshals to a JSON object or array. `Input.Media` holds ordinary GAI image/audio/video content blocks. Nil data is permitted with media; an explicit empty string is also representable. Media are attachments, not an ordered chat transcript: adapters preserve order within a modality but make no cross-modality placement promise. Unsupported media must be rejected rather than discarded, fetched, uploaded, or processed through hidden OCR.

Each named `Question` has one of three typed variants:

- `Predicate` asks for P(true); thresholding belongs to the application.
- `Choice` specifies mutually exclusive string or boolean outcomes. Boolean outcomes are not supported by every provider. Boolean false is different from the string `"false"`.
- `Score` specifies ordered levels. The result is the expected zero-based level index, not the most probable category or arbitrary regression.

Instructions, option descriptions, level descriptions, and predicate criteria accept string/object/array guidance. Nested arrays can contain mixed JSON values. They are not output schemas or executable steps. Evidence is the shared material being evaluated; instructions define each judgment and may contain question-local examples or records. No request-wide system instruction field is needed.

Adapters marshal evidence and guidance before the network request. Use `json.Number` for numbers whose precision matters. Values already rounded by conversion to float64 cannot be recovered. A question name is a correlation key, not implicit instructions.

Some APIs can infer a task from criteria alone; others require explicit instructions. Common validation preserves omission, and adapters enforce their own requirements. Questions do not consume earlier answers. Their predictions are not guaranteed independent or invariant when another question is added.

## Results and validation

A successful response has one answer per question in request order. Each answer has exactly one typed result or a refusal. Choice and score distributions cover every supplied outcome, including zeros, in schema order. A refusal has no numeric payload and is not an execution error. Low confidence is likewise not an error.

Probabilities and scores are `json.Number`. Optional `Answer.Confidence` is a `*json.Number`; nil means unavailable, not zero. Its source and interpretation are implementation-specific: implementations may forward it, calculate it, or omit it. No common calibration or probability-of-correctness claim is made.

`Request.Validate` and `Response.Validate` are available to custom implementations. Response validation uses rational arithmetic for bounds, distributions, selected outcomes, and expected indices. It allows 0.00005 rounding error per probability and 0.00005 for the reported score, with accumulated bounds for sums and weighted means. It does not rewrite values. Numeric tokens are limited to 256 bytes and exponents to magnitude 10000 to bound validation resources. Applications may explicitly convert to float64 if acceptable for their calculations.

Execution failures use Go errors. Malformed responses use `ResponseError`. On error, response usage and diagnostic metadata may be available, but no answers are actionable. Transport retry and fallback are separate middleware concerns; implementations must not silently retry model POSTs or execute decisions.

## Usage and native metadata

Reuse `gai.Metadata`: input/output counts use the existing int-valued keys and typed accessors. Missing counts remain absent. Counts describe provider accounting, not comparable compute across models. Questions, criteria, and media contribute to input accounting. Output tokens may be reported even when output is free and no prose is generated. No per-question allocation or cost estimation is invented.

Optional cache and reasoning counts use existing GAI keys when reported. Fractional native metrics use `json.Number`. Invocation IDs, response headers, backend identity, native usage objects, and opaque fields belong in response metadata; answer-specific fields belong on the answer. Preserve opaque JSON without a float64 round trip. Metadata may be sensitive; the package does not log payloads.

## Scope

There is no streaming, tool execution, automatic generic-struct schema derivation, or generator-to-classifier confidence conversion. Provider capability limits belong in adapters, not model-name heuristics in shared code. A label-only model does not satisfy the complete-distribution contract unless an explicit scoring implementation obtains the missing probabilities.
