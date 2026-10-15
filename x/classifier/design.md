# Classifier design

Experimental package `github.com/spachava753/gai/x/classifier` provides clients for classifying and scoring input with AI models. Its API may change.

```go
type Classifier interface {
    Classify(context.Context, Request) (Response, error)
}
```

Each request specifies a model, input, and questions. A client keeps its HTTP client and credentials, but no conversation history. It returns answers; the application decides what to do with them. Classification is separate from `gai.Generator`, which returns generated messages.

## Input and questions

`Input.Data` contains text or structured data. It accepts a Go value that encodes as a JSON string, object, or array. For example, it can hold a customer message alone or an object containing both the message and a refund policy. Nested values may be any JSON type. Use `json.Number` when a number must not be rounded to float64.

`Input.Media` contains GAI image, audio, or video blocks. Data may be nil when media is supplied. Attachments of the same kind keep their order, but a provider may group images, audio, and video separately and choose where to put the text. Clients reject unsupported media rather than fetch URLs, upload files, or convert them behind the caller's back.

All questions in a request use the same input. Each question has a unique name and one of three types:

- `Predicate` asks a yes-or-no question and returns the probability of yes. The application chooses a threshold if it needs a boolean.
- `Choice` asks the model to select one option and returns the probability of every option. Options can have string or boolean values, although not all providers accept booleans. Boolean false and the string `"false"` are different values.
- `Score` asks the model to rate the input on an ordered scale. Levels have values 0, 1, 2, and so on. The result is their probability-weighted average. For example, probabilities 0.2, 0.3, and 0.5 give a score of `0*0.2 + 1*0.3 + 2*0.5 = 1.3`, not the most likely level, 2.

A question's `Instructions` tells the model what to decide and can include context or examples for that question. Instructions and option descriptions accept strings, objects, or arrays. Object keys are caller-defined. An array describes one question, not a sequence of calls. These values explain the task; they are not JSON Schemas.

Some providers allow instructions to be omitted when the options, levels, or predicate criteria define the question. Others require instructions every time. The question name identifies its answer; it is not a substitute for instructions.

Questions do not use earlier answers. Their predictions are not necessarily independent: adding or removing a question may affect other answers.

## Answers and validation

A successful call returns one answer per question, in request order. Choice and score answers include a probability for every option or level, including those with zero probability. Score probabilities are an ordered `[]json.Number`: entry `i` gives the probability of level `i`. A model may instead refuse to answer, setting `Answer.Refused` to true. A refusal has no probability, score, or confidence and does not by itself make the call fail.

Probabilities and scores use `json.Number`. `Answer.Confidence` is a `*json.Number`; nil means unavailable, not zero. Each implementation defines what confidence means. It may return the provider's value, calculate one, or omit it. The built-in clients return the provider's value when present. Confidence is not necessarily the probability that an answer is correct, and values from different implementations may not be comparable.

The built-in clients check input and question definitions before sending a request. Before returning answers, they check that each answer matches its question, includes all required probabilities, and has consistent numbers. These checks are internal; callers only need to call `Classify` and check its error.

Response validation compares numbers without converting them to float64. It allows a rounding error of 0.00005 per probability and per reported score, adding these allowances when checking sums and weighted averages. For example, three probabilities may sum to a value within 0.00015 of 1. Validation does not change returned values. To limit resource use, each number may have at most 256 bytes and an exponent between -10000 and 10000.

Invalid responses return `ResponseError`. On any error, usage counts and diagnostic fields may still be available, but callers must not use the answers. Retrying a failed call or trying another provider is up to the caller.

## Data ownership

Classify does not modify requests or retain their data after returning. Callers must not change a request's maps, slices, pointers, or other data while a call is running. Returned answers do not share mutable data with the request.

Built-in clients are safe for concurrent use. Custom JSON marshalers must also support concurrent calls if their values are shared. They must not modify the request when encoding it.

## Providers

Constructors take an HTTP client, an optional API base URL, and credentials. They do not read environment variables or make network requests. A nil client means `http.DefaultClient`. Cloudflare also requires an account ID. The model is specified in each request, not in the constructor.

| Client | Default endpoint | Input and instructions | Choice values |
| --- | --- | --- | --- |
| `NewTypeSafe` (Jev) | `https://api.typesafe.ai/v1/systemone` | Text or JSON; no media | Strings |
| `NewOpenAI` (Decisions) | `https://api.openai.com/v1/decisions` | JSON objects and arrays are sent as text; PNG/JPEG/WebP/GIF images | Strings and booleans |
| `NewCloudflare` (Clef) | `https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}` | Text or JSON; PNG/JPEG/WebP images; Clef-Omni audio/video fields | Strings |
| `NewPerplexity` (Decisions) | `https://api.perplexity.ai/v1/decisions` | Text or JSON; PNG/JPEG/WebP images | Strings |

A custom base URL includes its API prefix, such as `/v1`, but not the endpoint suffix. A service must support the chosen protocol; supporting Chat Completions does not imply support for Decisions. The Cloudflare client supports hosted Workers AI, not self-hosted Clef. Its `Request.Model` is a full model name such as `@cf/cloudflare/clef`, with exactly three components: namespace, author, and model.

OpenRouter exposes compatible endpoints without requiring another classifier type:

- Use `NewTypeSafe(client, "https://openrouter.ai/api/v1", key)` for its [System One endpoint](https://openrouter.ai/docs/guides/community/typesafe-sdk), with a model such as `typesafe/jev-1.13`.
- Use `NewPerplexity(client, "https://openrouter.ai/api/alpha", key)` for its [Decisions endpoint](https://openrouter.ai/docs/guides/community/jev-tutorial). The live suite covers `typesafe/jev-1.13` and `perplexity/pplx-decider-v1.1-27b` through this route.

These clients retain their protocol's error and metadata provider identifiers, not an OpenRouter identifier. Native OpenRouter invocation IDs, resolved providers, and costs remain in the corresponding extra fields. Supporting these routes does not establish compatibility with every model in OpenRouter's catalog.

OpenAI, TypeSafe, and Cloudflare require instructions for every question. Perplexity allows criteria-only questions. Provider methods document their limits on question counts, choices, levels, and attachments. The server checks other model-specific constraints, such as image dimensions. Cloudflare places images before the text and may truncate long text to fit the context limit.

OpenAI has no separate field for predicate criteria, so its client appends a JSON object describing the true and false outcomes to the question's instructions. System One score criteria include each level's label and description together. OpenAI's explicit refusal response sets `Answer.Refused` to true; the other clients do not interpret a negative or uncertain answer as a refusal.

`SafetyIdentifier` is sent to OpenAI as `safety_identifier`. The other clients ignore it.

## Generated HTTP clients

Provider HTTP clients must be generated, not handwritten. OpenAI Decisions uses the `/decisions` operation in `internal/openai`; TypeSafe, Perplexity, and Cloudflare have generated packages under `internal/` with those names. Each package has a maintained OpenAPI schema, an oapi-codegen configuration, and a `go:generate` command. Run `go generate ./internal/...` after changing a schema. Never edit `client.gen.go` directly.

Adapters build generated request types and call generated operations. Input and question unions remain raw JSON so nested numbers, nulls, and provider-specific shapes are preserved. Authentication and per-call headers use the generated request editors. Cloudflare account, model namespace, author, and name are separate path parameters encoded by its generated client. Model-route slashes must remain path separators; passing the entire model route as one parameter produces encoded slashes that Cloudflare rejects.

Adapters read bounded raw HTTP responses, reject malformed JSON and duplicate keys, then decode the generated response models. They convert typed answers, usage, and Cloudflare envelopes rather than decoding known fields from raw JSON maps. Generated metadata projections preserve usage and invocation details when answer decoding fails. Generated error models supply API error messages; unrecognized error bodies remain available in full.

Generated classifier models use `internal/wirejson.Number` for fractional values. Unlike decoding directly into `json.Number`, this type rejects quoted numbers. Nullable fields distinguish missing values, null, and zero. Adapters convert valid numbers to the public `json.Number` fields without floating-point conversion. `http.go` contains this response handling and constructor/header helpers; it does not construct URLs, create HTTP requests, or call `http.Client.Do`. Provider-specific paths and HTTP execution belong to generated code. Existing provider SDKs elsewhere in GAI remain supported; handwritten provider clients are not an alternative.

## HTTP errors and usage

Clients honor context cancellation and set request headers without modifying the HTTP client. They do not add retries. The supplied HTTP client's redirect and transport policies still apply. HTTP API errors return `gai.ApiErr`, including the provider, status code, response body, and Retry-After value when available.

Responses are limited to 32 MiB. Duplicate JSON keys, trailing JSON values, and malformed or inconsistent answers return `ResponseError`. No answers are returned until the entire response has passed validation.

Token counts cover the entire call, including input, questions, descriptions, and attachments. They do not measure comparable computation across providers. An output-token count may be reported even if output is free. The clients do not estimate missing counts, per-question usage, or costs.

| Provider data | Where to read it |
| --- | --- |
| Input and output token counts | `gai.InputTokens` and `gai.OutputTokens` on `Response.UsageMetadata`; absent or null counts remain absent |
| OpenAI cache-read, cache-write, and reasoning token counts | The corresponding GAI usage accessors |
| Original usage object, including additional fields | `ExtraFieldUsage` in `Response.ExtraFields` |
| HTTP response headers | `ExtraFieldHeaders` in `Response.ExtraFields` |
| Provider identity | `ExtraFieldProvider` in `Response.ExtraFields` |
| Additional response or answer fields | `ExtraFieldWireFields` in `Response.ExtraFields` or `Answer.ExtraFields` |
| Cloudflare wrapper fields, such as success and errors | `ExtraFieldEnvelope` in `Response.ExtraFields` |

Original JSON objects are stored as `map[string]json.RawMessage`. When decoding them into `any`, use `Decoder.UseNumber` to avoid rounding numbers to float64. Common token counts remain `int` values. Metadata may contain sensitive data; the package does not log requests or responses.

## Tests and scope

The compiled examples show basic classification, string and boolean choices, and use of a provider client. Each generated client package checks offline regeneration, JSON round trips, and raw HTTP behavior. Offline provider tests use local HTTP servers and representative documented responses, not live captures. They cover structured input, all three question types, numeric precision, refusals, usage, media, cancellation, concurrent calls, and invalid responses. The Cloudflare HTTP regression checks the escaped path, not just Go's decoded `URL.Path`.

`TestLiveClassifier` in `providers_test.go` exercises the public constructors and `Classify` against real services. Ordinary tests skip it. Set `LIVE_TESTS=1` and the credentials for the selected profiles:

| Profile | Model | Required environment variables |
| --- | --- | --- |
| `openai` | `gpt-6-luna` | `OPENAI_API_KEY` |
| `openrouter_systemone` | `typesafe/jev-1.13` | `OPENROUTER_API_KEY` |
| `openrouter_decisions_jev` | `typesafe/jev-1.13` | `OPENROUTER_API_KEY` |
| `openrouter_decisions_perplexity` | `perplexity/pplx-decider-v1.1-27b` | `OPENROUTER_API_KEY` |
| `cloudflare_clef` | `@cf/cloudflare/clef` | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` |

Supply credentials through the environment; tests do not access password managers or local CLI authentication. The Cloudflare read-only token used for verification could both list models with `cf` and run these inference requests; no new resources or tokens were created. Use a token authorized for Workers AI inference on the selected account.

```sh
# All configured profiles; unconfigured profiles explicitly skip.
LIVE_TESTS=1 go test ./x/classifier -run '^TestLiveClassifier$' -count=1 -timeout=10m -v

# Only Cloudflare mixed questions.
LIVE_TESTS=1 go test ./x/classifier -run '^TestLiveClassifier$/MixedQuestions/cloudflare_' -count=1 -v

# Explicitly offline, even if credentials are already set.
LIVE_TESTS= go test ./...
```

Each call has a one-minute deadline and no added retry. Tests run sequentially and refuse redirects. Authentication, quota, model availability, and response-contract failures fail enabled tests; they are not converted into skips. Missing keys skip a profile, but a configured Cloudflare token without an account ID fails configuration. API error bodies and authentication headers are not printed.

The suite tests API compatibility, not model accuracy. With all profiles configured it makes 16 calls:

- Ten mixed-question calls: text and structured JSON through each of the five profiles. Both input forms describe the same task. Predicate, choice, and score questions are batched; OpenAI also includes boolean and string choice values in these requests.
- Three image calls: one locally generated PNG through OpenAI, OpenRouter/Perplexity, and Cloudflare. These check accepted image requests and typed responses, not visual recognition.
- Three authentication failures: one per service, rather than one per model or compatible route.

Assertions cover named answer ordering, answer variants, complete distributions, numeric bounds and internal consistency, string/boolean identity, request immutability, usage, OpenRouter cost/ID metadata, Cloudflare envelopes, and typed API errors. Any declared choice and any contract-valid probability or score is acceptable. A well-formed refusal is also acceptable; the tests do not demand that a model answer a particular input. Weighted-score checks compare the score with its returned distribution, not with a predetermined rating. No assertion requires a correct classification, a particular color, or a decision threshold. Exact numeric precision and serialization edge cases remain deterministic offline tests.

Initial live probes on 2026-10-14 passed for these routes and also for Cloudflare Clef-Flash and Clef-Omni. The reduced API-contract suite does not repeat the same protocol tests across those Cloudflare model variants. The revised 16-call API-contract suite passed live on 2026-10-14 with no skips or retries, including the batched OpenAI boolean choices. Direct TypeSafe and Perplexity hosting, audio/video input, and explicit refusal scenarios remain live-unverified. A successful live probe establishes compatibility for that request and route at that time, not model accuracy or permanent provider availability.

There is no streaming, tool execution, automatic question derivation from Go structs, or conversion of generator output into confidence values. A model that only returns a label cannot implement the complete Choice answer without also obtaining probabilities for every option.

Protocol references:

- [OpenAI Decisions](https://developers.openai.com/api/reference/resources/decisions/methods/create)
- [TypeSafe API](https://docs.typesafe.ai/api)
- [Perplexity Decisions](https://docs.perplexity.ai/api-reference/decisions-post)
- [Clef input schema](https://developers.cloudflare.com/workers-ai/models/clef/schema-input.json), [output schema](https://developers.cloudflare.com/workers-ai/models/clef/schema-output.json), and [Clef-Omni input schema](https://developers.cloudflare.com/workers-ai/models/clef-omni/schema-input.json)
