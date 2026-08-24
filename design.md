# Current generator design

This document records the generator design implemented at `v0.42.0` (`1b09d9f`). It describes the current contracts, the types that carry data across those contracts, where state lives, and the tradeoffs created by those choices. It is a baseline for a later redesign, not a proposal for the replacement API.

## Design goals expressed by the current code

The current design has a small provider-independent generation interface and adds optional behavior through separate capability interfaces. Callers own conversation history and pass the complete dialog on every call. Provider adapters translate the shared types into provider SDK types and normalize provider responses back into shared types.

The design favors:

- one core interface that is easy to implement, wrap, and fake in tests;
- shared request and response types across providers;
- explicit caller ownership of conversation history and tool execution;
- optional capabilities for tools, streaming, and token counting;
- escape hatches for provider data that does not fit the common types;
- wrapper composition for retry, fallback, preprocessing, and application middleware.

These goals explain most of the current type shapes. They also produce several tensions discussed below.

## Core interfaces

### Generator

```go
type Generator interface {
    Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error)
}
```

`Generator` is intentionally small. A provider only needs one method to participate in normal generation and in wrappers such as retry and fallback.

The arguments have separate responsibilities:

- `context.Context` carries cancellation, deadlines, and request-scoped values. The public documentation permits provider-specific values in the context, but the current provider generators do not read generation parameters from it.
- `Dialog` contains the complete caller-managed conversation supplied to this invocation.
- `*GenOpts` contains common and provider-specific generation parameters. It may be `nil`.

The return values separate a successful provider-normalized `Response` from Go errors. Some model stop conditions become `Response.FinishReason`; invalid input, transport failures, provider failures, and selected stop conditions become errors.

### Tool registration

```go
type ToolRegister interface {
    Register(tool Tool) error
}

type ToolCallingGenerator interface {
    Generator
    ToolRegister
}
```

Tool support is modeled as a capability because providers do not all support the same features. `ToolCallingGenerator` embeds `Generator`, so every value satisfying it can both register tools and generate.

`Register` mutates the generator. Provider implementations validate and convert the shared `Tool` into a provider SDK type, then store that converted value in a map or slice for later calls. Duplicate names are rejected.

Registration only supplies the tool definition. The generator returns tool calls to the application, and the application owns authorization, execution, persistence, and the follow-up generation loop. `ToolCallback` and `ToolCallBackFunc` help applications implement callbacks, but callbacks are not part of `Tool` and are not stored or executed by a generator.

### Streaming

```go
type StreamingGenerator interface {
    Stream(
        ctx context.Context,
        dialog Dialog,
        options *GenOpts,
    ) iter.Seq2[StreamChunk, error]
}
```

`StreamingGenerator` does not embed `Generator`. A stream-only implementation is therefore valid, although the provider implementations generally expose both methods.

The standard-library `iter.Seq2` type gives the consumer pull-based iteration and a way to stop early by returning `false` from `yield`. Stream failures travel through the second sequence value rather than through a separate channel or terminal result.

`StreamingAdapter` converts a `StreamingGenerator` into a `Generator`. It collects chunks, reconstructs tool calls, joins text and thinking deltas, extracts usage metadata, and creates a normal `Response`. It currently supports only candidate index zero.

### Token counting

```go
type TokenCounter interface {
    Count(ctx context.Context, dialog Dialog) (uint, error)
}
```

Counting is separate because it is not available from every provider and because its implementation varies. OpenAI counts locally with a tokenizer, while Anthropic and Gemini call provider APIs.

`Count` accepts only a dialog. Model, system instructions, and registered tools come from generator state. It cannot describe a count for a different system prompt or tool set without constructing or mutating another generator.

## Where request inputs live

There is no single generation request type. Inputs are split across construction, mutation, call arguments, and one escape-hatch map.

| Input | Current location | Can vary per call? |
| --- | --- | --- |
| Provider client and credentials | Provider generator fields | No |
| Base URL or endpoint | Provider generator or client fields | No |
| Model | Provider generator field | No |
| System instructions | Provider generator field | No |
| Tool definitions | Provider generator map or slice, populated by `Register` | Only by mutating the generator |
| Z.AI thinking defaults | Provider generator fields set by constructor options | No |
| Dialog | `Generate`, `Stream`, and `Count` argument | Yes |
| Common generation parameters | `*GenOpts` argument | Yes |
| Selected provider parameters | `GenOpts.ExtraArgs` | Yes |
| Cancellation and deadline | `context.Context` | Yes |

At present, `GenOpts.ExtraArgs` is consumed by `ResponsesGenerator` for Responses-specific options. Other providers primarily use the named `GenOpts` fields.

This split makes a configured generator convenient for repeated calls to the same model, system prompt, and tool set. It does not make the complete request visible in one value.

## Generation options

```go
type GenOpts struct {
    Temperature         *float64
    TopP                *float64
    TopK                *uint
    FrequencyPenalty    *float64
    PresencePenalty     *float64
    N                   *uint
    MaxGenerationTokens *int
    ToolChoice          string
    StopSequences       []string
    OutputModalities    []Modality
    AudioConfig         AudioConfig
    ThinkingBudget      string
    ExtraArgs           map[string]any
}
```

Scalar numeric options mostly use pointers. This preserves the difference between an omitted value and an explicit zero, which matters because omission asks the provider to apply its default. The options object itself is also a pointer, so `nil` means no options were supplied.

Some choices are looser:

- `ToolChoice` is a string so it can hold `"auto"`, `"required"`, or a tool name without a tagged union.
- `ThinkingBudget` is a string because providers accept different concepts, including token counts, disabled or adaptive modes, and named effort levels.
- `ExtraArgs` is `map[string]any` so a provider feature can ship without first becoming part of the shared API.
- unsupported common options may be ignored by a provider, as documented on individual fields.

This shape keeps the common API broad and easy to extend. The cost is runtime validation, provider-dependent behavior, and limited compile-time help for provider-specific settings.

## Conversation and content types

### Dialog and message

```go
type Dialog []Message

type Message struct {
    Role            Role
    Blocks          []Block
    ToolResultError bool
    ExtraFields     map[string]interface{}
}
```

A dialog is a slice rather than an object with an ID or hidden cursor. The caller owns the full history and decides what to retain, redact, cache, or persist. This also lets generators call stateless provider endpoints by reconstructing each upstream request from the supplied dialog.

`Message` groups blocks under one role. `ToolResultError` preserves the distinction between a successful tool result and an error returned to the model. `ExtraFields` carries provider data that applies to the whole message, such as the OpenAI Responses message phase.

`Role` is a numeric enum with `User` as its zero value, followed by `Assistant` and `ToolResult`. A system role is not present because system instructions currently live on each provider generator rather than in the dialog.

### Block

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

A block is the common container for text, media, thinking, tool calls, stream metadata, and stream separators. This avoids a different message type for every combination of content a provider can return.

The fields make different tradeoffs:

- `BlockType` is a string. Known values are `content`, `thinking`, `tool_call`, `metadata`, and `separator`. A string leaves room for additional block kinds without changing an enum.
- `Modality` is a numeric enum because the shared set is small: text, image, audio, and video. Text is the zero value.
- `Content` is `fmt.Stringer`. Provider adapters consume `Content.String()`, so text and base64-encoded media can use one field. The package's `Str` type is the normal implementation.
- `ExtraFields` stores provider-specific replay data and hints at the narrowest relevant scope. Thinking signatures belong to blocks; Responses phases belong to messages.
- tool call input is encoded as JSON in `Content`, with the shared logical shape `ToolCallInput{Name, Parameters}`. This keeps `Block` uniform but moves tool-call type checking to runtime.

The comments describe content and `text/plain` as semantic defaults, but a raw zero-value `Block` has an empty `BlockType`, empty `MimeType`, and nil `Content`. Constructors such as `TextBlock`, `ImageBlock`, `PDFBlock`, and `ToolCallBlock` populate canonical fields and should be preferred.

Provider-specific metadata in `ExtraFields` is not incidental. Some providers require signatures, encrypted reasoning data, IDs, or phases to be sent back on later turns. Retaining that data in the dialog makes replay possible without hidden conversation state in the generator.

### Tools

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

The package uses `jsonschema.Schema` instead of a provider SDK schema. JSON Schema is the shared interchange format accepted, with different subsets, by all tool-capable providers. `GenerateSchema[T]` derives a schema from a Go type and disallows unknown object properties by default.

Provider adapters still validate and translate schemas because support differs. For example, Gemini accepts a narrower set of `anyOf` forms than OpenAI or Anthropic.

## Response types

```go
type Response struct {
    Candidates    []Message
    FinishReason  FinishReason
    UsageMetadata Metadata
}
```

`Candidates` is a slice to support `GenOpts.N`; the common default is one candidate. Each candidate is a normal assistant `Message`, so generated content uses the same role, block, modality, and provider-metadata representation as dialog input.

`FinishReason` normalizes provider stop reasons into `Unknown`, `EndTurn`, `StopSequence`, `MaxGenerationLimit`, `ToolUse`, and `ContentPolicyViolation`. It is attached to the whole response rather than to each candidate.

`Metadata` is `map[string]any`. Common keys cover input, generation, cache read, cache write, and reasoning token counts. The map admits provider metrics without repeatedly changing `Response`, but callers need runtime type assertions. Helpers such as `InputTokens` and `OutputTokens` centralize assertions for common keys.

The current error model combines:

- sentinel errors for stable conditions such as an empty dialog or exceeded context;
- string and struct error types for invalid modalities, parameters, and tool choices;
- `ApiErr`, which retains provider details while adding a shared `APIErrorKind` used by retry and fallback policies.

## Streaming data model

```go
type StreamChunk struct {
    Block              Block
    MessageExtraFields map[string]interface{}
    CandidatesIndex    int
}
```

A stream reuses partial `Block` values instead of defining a second content hierarchy. Content and thinking chunks carry string fragments. A tool call starts with a block containing its ID and tool name, followed by blocks containing JSON parameter fragments. Separator blocks retain provider block boundaries. A final metadata block carries usage as JSON.

`MessageExtraFields` exists separately because message-level metadata can become known while streaming even though each event otherwise contains a block. `StreamingAdapter` merges those maps and rejects conflicting values.

The format gives direct stream consumers access to provider-normalized events and lets the adapter reconstruct the non-streaming response shape. Its cost is a protocol with ordering rules that implementations and wrappers must preserve.

## Provider capability matrix

The table describes methods implemented by pointers to the concrete provider types. `G` is `Generator`, `S` is `StreamingGenerator`, `R` is `ToolRegister`, and `C` is `TokenCounter`.

| Provider generator | G | S | R | C | Stored request configuration |
| --- | --- | --- | --- | --- | --- |
| `OpenAiGenerator` | Yes | Yes | Yes | Yes | client, model, system instructions, converted tools |
| `AnthropicGenerator` | Yes | Yes | Yes | Yes | client, model, system instructions, converted tools |
| `GeminiGenerator` | Yes | Yes | Yes | Yes | client, model, system instructions, converted tools |
| `CerebrasGenerator` | Yes | No | Yes | No | HTTP client, endpoint, API key, model, system instructions, converted tools |
| `OpenRouterGenerator` | Yes | No | Yes | No | client, model, system instructions, converted tools |
| `ResponsesGenerator` | Yes | Yes | Yes | No | client, model, system instructions, converted tools |
| `ZaiGenerator` | Yes | Yes | Yes | No | clients, model, system instructions, converted tools, thinking defaults |

Anthropic and Gemini constructors wrap their concrete adapters in `PreprocessingGenerator` and return an anonymous interface containing generation, streaming, registration, and counting. OpenAI returns a concrete value even though its interface methods have pointer receivers. Other constructors return concrete pointers or values. Constructor return shapes are therefore inconsistent.

`ResponsesGenerator` is described in its code as stateless because it sets the upstream Responses API's `store` option to false and replays encrypted reasoning data through the dialog. The Go object is still configured and mutable: it stores model, system instructions, and registered tools. Upstream conversation statelessness and local object statelessness are separate properties.

## Wrappers and adapters

`GeneratorWrapper` stores an `Inner Generator` and implements `Generate`, `Count`, `Register`, and `Stream`. It delegates optional methods after runtime interface checks. Embedding it lets a wrapper override selected methods.

```go
type GeneratorWrapper struct {
    Inner Generator
}

type WrapperFunc func(Generator) Generator
```

`Wrap` applies functions in reverse order so the first wrapper supplied is the outermost call layer. `WrapperFunc` accepts and returns only `Generator`; optional capabilities remain available through the returned value's dynamic method set, not through its static return type.

Current wrappers have these shapes:

| Wrapper or adapter | Behavior and stored state |
| --- | --- |
| `RetryGenerator` | Stores an inner generator and immutable retry configuration. Attempt counters and timers are local to each call. It retries `Generate` and only pre-output `Stream` failures. |
| `FallbackGenerator` | Stores a slice of generators and fallback policy. It implements only `Generator`, so streaming, registration, and counting are not available through it. |
| `PreprocessingGenerator` | Stores an inner generator and rewrites parallel tool-result messages before generation or streaming. Other methods delegate. |
| `StreamingAdapter` | Stores a streaming generator and exposes collected `Generate`; `Register` and `Count` perform runtime capability checks. |
| `AnthropicServiceWrapper` | Wraps the provider service below the generator and stores parameter modifier functions, mainly for caching and request transformation. |

There is a subtle cost to the delegation design. `GeneratorWrapper`, `RetryGenerator`, and `StreamingAdapter` have optional capability methods even when their inner value lacks the capability. They therefore satisfy interfaces such as `TokenCounter` or `StreamingGenerator` at compile time and return an error at runtime. Interface satisfaction does not always prove that the operation is supported.

## State and concurrency

"Stateful" has several meanings in the current implementation:

- Conversation state is caller-owned. Generators do not append to or retain the dialog between calls.
- Upstream conversation state depends on the provider path. `ResponsesGenerator` explicitly disables upstream storage.
- Configuration state is generator-owned. Clients, model, system instructions, endpoints, credentials, and provider defaults live on generator objects.
- Tool state is mutable. `Register` changes a provider-specific map or slice used by later calls.
- Invocation state is local. Retry attempt counts, stream assembly, and per-call tool ID mappings are allocated inside each invocation.
- Wrapper state contains dependencies and policies such as inner generators, fallback lists, and retry functions.

No provider generator synchronizes tool registration. Concurrent generation is reasonable only after configuration and registration have stopped, assuming the supplied provider client is itself safe for concurrent use. Calling `Register` concurrently with `Generate`, `Stream`, `Count`, or another `Register` can race on provider tool maps or slices.

The configured-object model also makes ordering part of correctness: callers must construct a generator, register all tools, and only then share it. Tests that need another model, prompt, or tool set must construct and configure another object.

## What the current choices buy us

The small `Generator` interface keeps provider adapters and higher-order generators simple. A retry or fallback layer can forward the same dialog and options without understanding provider SDK request types.

Explicit dialogs keep conversation policy outside the library. Applications can persist history, remove thinking, enforce context limits, and run tools without a hidden session inside a generator.

Blocks and metadata maps absorb provider differences. New reasoning metadata or a provider metric can pass through the library without creating provider-specific response classes throughout application code.

Construction-time model and system configuration reduce repeated call-site data. Registration-time tool conversion validates schemas once and avoids converting an unchanged tool set on every request.

Separate capabilities reflect real provider differences and prevent the core generation interface from requiring unsupported operations.

## Costs and redesign pressure

The complete logical request is not represented by one value. Model, system instructions, tools, dialog, and generation parameters enter through different APIs at different times. That makes requests harder to log, compare, queue, replay, hash, authorize, or route as a unit.

Mutable registration complicates sharing. A generator has a configuration phase and a use phase, but the type system does not distinguish them and the implementation does not synchronize them.

A caller cannot vary a system prompt or tool set per request without mutation or another generator. This works against request-scoped tenancy, dynamic tool authorization, and fallback calls that should receive exactly the same logical request.

Token counting has a narrower input shape than generation. It depends on hidden generator configuration and cannot accept the same complete input that `Generate` and `Stream` would use.

Optional capability preservation is inconsistent. Some wrappers erase capabilities, while others advertise methods that can fail because an inner generator does not implement them.

Escape hatches keep the API moving but defer mistakes to runtime. `ExtraArgs`, `ExtraFields`, string discriminators, `fmt.Stringer`, and JSON-encoded tool calls all trade compile-time guarantees for provider flexibility.

Provider-specific converted tools are cached in mutable generator state. Moving tools into a request will require a deliberate choice: convert them on every call, introduce immutable prepared requests or tool sets, or cache conversion outside the semantic generator state.

These pressures motivate discussing a request-oriented design in which system instructions, tools, model selection, and generation parameters are explicit invocation inputs. That discussion should separately decide what "stateless generator" means for transport clients, immutable provider configuration, conversion caches, middleware policy, and upstream conversation storage. This document does not make those decisions.
