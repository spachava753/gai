# Generator design

This document defines the generator interfaces, shared request and response types, state ownership, and design rationale for GAI.

## Goals

The generator API has these goals:

- A generation request contains every semantic input needed to reproduce a model call.
- Generator values are safe to share because they do not contain mutable request configuration.
- Provider adapters translate one shared request into provider SDK types.
- Callers own conversation history and tool execution.
- Common generation parameters are easy to set without closing the option set to provider-specific features.
- Streaming, token counting, retry, fallback, and preprocessing compose around the same request value.

## State boundary

A generator stores immutable execution dependencies. These include provider clients, HTTP transports, credentials, and endpoint configuration. They determine where and how a request is sent, not what the model is asked to do.

A `GenerationRequest` stores semantic state:

```go
type GenerationRequest struct {
    Model        string
    Instructions Message
    Dialog       Dialog
    Tools        []Tool
    Options      GenerationOptions
}
```

The request contains the model, system instructions, complete dialog, available tools, common generation parameters, and provider-specific generation parameters. A caller can log, queue, authorize, copy, transform, or replay this value without reading generator fields.

Generators must not mutate a request or retain references to request data after an invocation ends. Callers must not mutate a request concurrently with an invocation. Go slices and maps remain reference-backed values, so immutability is a contract rather than a property enforced by the language.

Invocation-local data such as retry counters, provider message conversions, tool-call ID maps, and streaming assembly buffers belongs to the invocation. Wrapper policy such as retry timing and fallback ordering belongs to the wrapper and is not model request state.

## Core interfaces

### Generation

```go
type Generator interface {
    Generate(ctx context.Context, request GenerationRequest) (Response, error)
}
```

`context.Context` carries cancellation and deadlines. Generation parameters belong to `GenerationRequest`, not to context values.

The single request argument prevents wrappers from accidentally forwarding only part of a call. Retry and preprocessing forward a request value. Fallback forwards the same request to each target unless an explicit request transformation is part of the fallback policy.

### Streaming

```go
type StreamingGenerator interface {
    Stream(
        ctx context.Context,
        request GenerationRequest,
    ) iter.Seq[StreamChunk]
}
```

`StreamingGenerator` is a separate capability because not every provider offers streaming. The standard-library `iter.Seq` type lets consumers stop iteration without a separate channel. A chunk with a non-nil `Err` reports a terminal stream failure. The producer yields that error chunk once and then returns.

`StreamingAdapter` collects a stream into a `Response`. It reconstructs tool calls, joins content deltas, preserves logical block boundaries, extracts usage metadata, and supports candidate index zero.

### Token counting

```go
type TokenCounter interface {
    Count(ctx context.Context, request GenerationRequest) (uint, error)
}
```

Counting receives the same request as generation because model, instructions, dialog, and tools can all affect input token usage. A provider may ignore output-only generation options while counting.

Token counting remains a separate capability because some providers do not expose it and implementations may use either a local tokenizer or a provider API.

## System instructions

`GenerationRequest.Instructions` is a `Message`. A non-empty instruction message must use the `System` role:

```go
type Role uint

const (
    User Role = iota
    Assistant
    ToolResult
    System
)
```

A helper creates the canonical value:

```go
func SystemMessage(blocks ...Block) Message
```

Using `Message` gives instructions the same ordered block representation and provider metadata placement rules as dialog content. It also leaves room for APIs to accept image, audio, document, or other instruction modalities without changing `GenerationRequest`.

Provider adapters enforce their actual instruction capabilities. Every integrated provider accepts text instructions, so adapters accept `Content` blocks with `Text` modality. An adapter returns an unsupported-modality error when an instruction contains a modality that its API cannot accept. It returns an invalid-parameter error for instruction block types that cannot represent system content, such as tool calls or thinking blocks.

An empty instruction message means no system instructions. `System` messages do not belong in `Dialog`; the dedicated field preserves the provider distinction between instructions and conversation turns.

Providers that accept multiple text instruction parts preserve block order. Providers that accept one string join text blocks with a blank line so block boundaries do not concatenate words.

## Dialog and content

```go
type Dialog []Message

type Message struct {
    Role            Role
    Blocks          []Block
    ToolResultError bool
    ExtraFields     map[string]interface{}
}
```

The caller supplies the complete dialog on every invocation. Generators do not append messages or retain conversation history. This supports stateless provider calls and lets applications own persistence, truncation, redaction, and context-window policy.

A `Message` groups ordered blocks under one role. `ToolResultError` distinguishes an error returned to the model from a successful tool result. `ExtraFields` stores provider data whose scope is the complete message.

```go
type Block struct {
    ID           string
    BlockType    string
    ModalityType Modality
    MimeType     string
    Content      fmt.Stringer
    ExtraFields  map[string]interface{}
}
```

A block is the shared container for text, media, thinking, tool calls, stream metadata, and stream separators. Known block types use string discriminators so provider additions do not require a closed enum. `Modality` is a numeric enum for the shared text, image, audio, and video set.

`Content` implements `fmt.Stringer`. Provider adapters consume `Content.String()`, allowing text and base64-encoded media to use the same field. Package helpers such as `TextBlock`, `ImageBlock`, `AudioBlock`, `PDFBlock`, and `ToolCallBlock` construct canonical values.

`ExtraFields` stores provider-specific replay data at the narrowest useful scope. Thinking signatures and image hints belong to blocks. Metadata that applies to an entire assistant message belongs to `Message.ExtraFields`.

## Tools

```go
type Tool struct {
    Name        string
    Description string
    InputSchema *jsonschema.Schema
}

type ToolCallInput struct {
    Name       string
    Parameters map[string]any
}
```

Tools are request data. `GenerationRequest.Tools` is the complete tool set available to that invocation. Generators do not expose a registration method and do not retain converted tool definitions.

Each provider validates and converts tools while building its request. Conversion is intentionally invocation-scoped. An immutable prepared-tool optimization can be added if measurement shows conversion cost is material, but it must not introduce mutable generator configuration.

JSON Schema is the provider-independent tool definition format. Provider adapters enforce their supported subset. `GenerateSchema[T]` derives a schema from a Go type and disallows unknown object properties by default.

Generators return tool calls to the application. Applications own authorization, execution, retries, persistence, and the follow-up generation request. `ToolCallback` and `ToolCallBackFunc` help implement application callbacks but are not generator state.

## Generation options

Generation options use one extensible map:

```go
type GenerationOptions map[string]any
```

Presence in the map distinguishes an explicit zero value from an omitted parameter. Recognized keys have canonical value types. A provider validates the type and value of every recognized key it uses. Unknown keys are ignored, allowing one request to carry options for more than one fallback provider.

Common keys are exported constants:

```go
const (
    GenerationOptionTemperature         = "temperature"
    GenerationOptionTopP                = "top_p"
    GenerationOptionTopK                = "top_k"
    GenerationOptionFrequencyPenalty    = "frequency_penalty"
    GenerationOptionPresencePenalty     = "presence_penalty"
    GenerationOptionCandidateCount      = "candidate_count"
    GenerationOptionMaxGenerationTokens = "max_generation_tokens"
    GenerationOptionToolChoice          = "tool_choice"
    GenerationOptionStopSequences       = "stop_sequences"
    GenerationOptionOutputModalities    = "output_modalities"
    GenerationOptionAudioConfig         = "audio_config"
    GenerationOptionThinkingBudget      = "thinking_budget"
)
```

The canonical value types are `float64` for temperature, top-p, frequency penalty, and presence penalty; `uint` for top-k and candidate count; `int` for maximum generation tokens; `string` for tool choice and thinking budget; `[]string` for stop sequences; `[]Modality` for output modalities; and `AudioConfig` for audio output configuration.

Functional options provide typed construction for common parameters:

```go
type GenerationOption func(GenerationOptions)

func NewGenerationOptions(options ...GenerationOption) GenerationOptions
func WithTemperature(value float64) GenerationOption
func WithTopP(value float64) GenerationOption
func WithTopK(value uint) GenerationOption
func WithFrequencyPenalty(value float64) GenerationOption
func WithPresencePenalty(value float64) GenerationOption
func WithCandidateCount(value uint) GenerationOption
func WithMaxGenerationTokens(value int) GenerationOption
func WithToolChoice(value string) GenerationOption
func WithStopSequences(values ...string) GenerationOption
func WithOutputModalities(values ...Modality) GenerationOption
func WithAudioConfig(value AudioConfig) GenerationOption
func WithThinkingBudget(value string) GenerationOption
```

Callers can use functional options, map literals, or both:

```go
options := NewGenerationOptions(
    WithTemperature(0.2),
    WithToolChoice(ToolChoiceAuto),
)
options[ResponsesServiceTierParam] = "priority"
```

Provider-specific exported constants identify custom keys. Provider-specific values live directly in `GenerationOptions`; there is no nested extra-arguments map.

A map is chosen because provider parameters change more often than the shared request structure. Functional options recover type safety and discoverability for common parameters while direct map entries keep provider work independent.

## Responses

```go
type Response struct {
    Candidates    []Message
    FinishReason  FinishReason
    UsageMetadata Metadata
}
```

Candidates are assistant messages and use the same block representation as dialog input. The slice supports providers that return more than one candidate.

`FinishReason` normalizes provider stop reasons into `Unknown`, `EndTurn`, `StopSequence`, `MaxGenerationLimit`, `ToolUse`, and `ContentPolicyViolation`.

`Metadata` is `map[string]any`. Common keys cover input, generation, cache read, cache write, and reasoning token counts. The map permits provider metrics without expanding `Response`; typed helpers retrieve common values.

Provider failures use `ApiErr`, which retains provider details and adds a provider-independent `APIErrorKind` for retry and fallback decisions. Stable local conditions use sentinel or structured error types.

## Streaming data

```go
type StreamChunk struct {
    Block              Block
    MessageExtraFields map[string]interface{}
    CandidatesIndex    int
    Err                error
}
```

Streams reuse partial blocks instead of defining another content hierarchy. Content and thinking chunks carry string fragments. Tool calls begin with a block containing the call ID and tool name, followed by blocks containing JSON argument fragments. Separator blocks preserve provider block boundaries. Metadata blocks carry usage.

`MessageExtraFields` carries message-level metadata discovered during streaming. `StreamingAdapter` merges those fields and rejects conflicting values. `Err` is excluded from serialized forms. A non-nil `Err` is mutually exclusive with stream payload fields and terminates the stream. Providers normalize validation, request, transport, decoding, context, and in-band API errors into this terminal chunk.

## Provider adapters

Provider generators retain only execution dependencies:

| Provider generator | Stored dependencies | Capabilities |
| --- | --- | --- |
| `OpenAiGenerator` | completion service | generate, stream, count |
| `AnthropicGenerator` | message service | generate, stream, count |
| `GeminiGenerator` | Gemini client | generate, stream, count |
| `CerebrasGenerator` | HTTP client, endpoint, API key | generate |
| `OpenRouterGenerator` | completion service | generate |
| `ResponsesGenerator` | Responses service | generate, stream |
| `ZaiGenerator` | generated clients and API transport | generate, stream |

Model, instructions, tools, and thinking settings come from `GenerationRequest`. Constructors establish transport dependencies and return generators ready for concurrent requests.

`ResponsesGenerator` sets the upstream Responses API `store` option to false. Provider-side conversation storage and local generator state are separate concerns; both are stateless in this adapter.

## Wrappers

```go
type GeneratorWrapper struct {
    Inner Generator
}

type WrapperFunc func(Generator) Generator
```

`GeneratorWrapper` delegates generation and performs runtime checks for optional streaming and counting capabilities. Embedding it lets middleware override selected methods. `Wrap` applies wrapper functions so the first supplied wrapper is the outermost call layer.

`RetryGenerator` stores retry policy and an inner generator. Attempt counters and timers are invocation-local. It retries generation failures and streaming failures that occur without emitted output.

`FallbackGenerator` stores ordered generators and fallback policy. It forwards the request unchanged. A fallback that needs provider-specific model names uses an explicit request transformation policy rather than mutating a generator.

`PreprocessingGenerator` rewrites parallel tool-result messages in the request dialog for providers that require consolidated results. It copies the request value and replaces only the dialog passed inward.

`StreamingAdapter` stores a streaming generator and collects its stream into a normal response.

`AnthropicServiceWrapper` operates below the generator and applies immutable provider request modifiers such as cache controls.

## Concurrency and ownership

Generator values contain no mutable semantic configuration. Concurrent calls may use different models, instructions, tools, dialogs, and options through the same generator. Safety also depends on the supplied provider client supporting concurrent use.

Request maps and slices belong to the caller. Provider adapters read them and create provider-specific values without modifying them. Functional option constructors clone slice arguments so later caller mutation does not change an option value unexpectedly.

Callbacks stored in retry policy, fallback policy, or service middleware may run concurrently. Their documentation must state this requirement.

## Rationale

A self-contained request makes call behavior explicit. Logging, policy checks, replay, testing, routing, and queueing can operate on one value.

A `Message` for instructions reuses ordered multimodal blocks and metadata scoping while letting each provider reject unsupported instruction content. The common request does not need a structural change when a provider adds multimodal instructions.

Request-scoped tools remove configuration ordering and data races around registration. They also allow authorization to produce a different tool set for every call.

Map-backed options avoid a permanent split between common struct fields and provider escape hatches. Exported keys and functional options give common settings stable names and canonical Go types.

Immutable transport dependencies remain on generators because passing clients, credentials, and endpoints in every request would expose secrets as semantic data and make wrapper composition cumbersome. Statelessness means no hidden or mutable model request state, not a fieldless Go value.
