# Generator design

This document describes the request model, generator interfaces, and provider adapters used by GAI. It also describes the rules that wrappers rely on when they retry, fall back, preprocess dialogs, or collect a stream.

A generator wraps a connection to a model provider. It stores the provider client and, for direct HTTP adapters, the endpoint and credentials. It does not store a model, system instructions, tools, or conversation history. Those values belong to `GenerationRequest` and are supplied on every call.

The library does not append to the conversation or execute tool calls. The application owns both jobs.

A generation call follows this path:

```text
GenerationRequest -> wrappers -> provider adapter -> provider API
                                              |
Response          <- normalization <----------+
```

For example, one OpenAI generator can handle requests with different models and instructions:

```go
client := openai.NewClient()
generator := NewOpenAiGenerator(&client.Chat.Completions)

request := GenerationRequest{
    Model:        "gpt-4o-mini",
    Instructions: SystemMessage(TextBlock("Answer in one sentence.")),
    Dialog: Dialog{{
        Role:   User,
        Blocks: []Block{TextBlock("Why is the sky blue?")},
    }},
    Options: NewGenerationOptions(WithTemperature(0.2)),
}

response, err := generator.Generate(ctx, request)
```

The model and instructions appear in the request because they describe this call, not the connection to OpenAI. A later call can reuse `generator` with a different request.

## The request

```go
type GenerationRequest struct {
    Model        string
    Instructions Message
    Dialog       Dialog
    Tools        []Tool
    Options      GenerationOptions
}
```

The fields have the following meanings:

| Field | Meaning |
| --- | --- |
| `Model` | Provider model name for this call. |
| `Instructions` | Optional system message. |
| `Dialog` | Complete conversation presented to the model. |
| `Tools` | Complete set of tools available during this call. |
| `Options` | Common and provider-specific generation settings. |

A retry can repeat the same request, and middleware can inspect or replace any part of it without reading generator fields. Fallback passes the same value to each target.

`context.Context` remains a separate argument because it controls the execution of the call, not the model's behavior. It carries cancellation and deadlines.

Generators treat a request as read-only. Copying `GenerationRequest` is shallow because its messages, tools, and options contain slices or maps. The caller must not modify those values while a call is in progress, and a generator must not retain them after the call returns.

Data created while handling a request stays local to that call. Examples include converted provider messages, tool-call ID maps, retry counters, and stream assembly buffers.

## Interfaces

Every provider implements `Generator`:

```go
type Generator interface {
    Generate(ctx context.Context, request GenerationRequest) (Response, error)
}
```

`Generate` waits for the provider to finish and returns one normalized `Response`.

Streaming and token counting are optional because not every provider supports them:

```go
type StreamingGenerator interface {
    Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk]
}

type TokenCounter interface {
    Count(ctx context.Context, request GenerationRequest) (uint, error)
}
```

All three methods receive the same request. In particular, token counting includes the model, instructions, dialog, and tools instead of relying on configuration hidden in the generator. OpenAI counts locally with `tiktoken`; Anthropic and Gemini call their token-counting APIs.

Cerebras and OpenRouter implement generation only. The other provider capabilities are listed below.

## Messages and instructions

```go
type Dialog []Message

type Role uint

const (
    User Role = iota
    Assistant
    ToolResult
    System
)

type Message struct {
    Role            Role
    Blocks          []Block
    ToolResultError bool
    ExtraFields     map[string]interface{}
}
```

`Dialog` is the complete conversation sent to the provider. The generator neither adds the returned assistant message nor removes old turns. Applications that need persistence, truncation, or redaction perform those operations before constructing the next request.

A message has one role and an ordered list of blocks. `ToolResultError` distinguishes a failed tool result from a successful one. `ExtraFields` holds provider metadata that applies to the message as a whole.

System instructions use the same `Message` representation but live in a separate request field:

```go
request := GenerationRequest{
    Instructions: SystemMessage(TextBlock("Answer as a Go programmer.")),
    Dialog: Dialog{
        {Role: User, Blocks: []Block{TextBlock("What does append return?")}},
        {Role: Assistant, Blocks: []Block{TextBlock("It returns the updated slice.")}},
        {Role: User, Blocks: []Block{TextBlock("Can it reuse the backing array?")}},
    },
}
```

A non-empty instruction message must have the `System` role. An empty message means no instructions. System messages are not valid dialog turns because provider APIs handle instructions separately from conversation history.

`Instructions` is a `Message`, rather than a string, so it can preserve ordered blocks and provider metadata. This also leaves a place for images or documents if provider APIs later allow them in system instructions.

**Current implementation.** Provider adapters accept text content blocks in `Instructions`. They reject other media with `UnsupportedInputModalityErr` and reject tool-call or thinking blocks with `InvalidParameterErr`. An adapter that needs one instruction string joins text blocks with a blank line.

## Blocks

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

A `Message` contains one or more blocks. `BlockType` identifies text or media content, model thinking, a tool call, usage metadata, or an internal stream separator. It is a string so an adapter can carry a provider-specific block type without changing the shared enum first.

`ModalityType` describes the data as text, image, audio, or video. `MimeType` gives the concrete media format. `Content` is a `fmt.Stringer`, and adapters read it through `Content.String()`. The package constructors `TextBlock`, `ImageBlock`, `AudioBlock`, `PDFBlock`, and `ToolCallBlock` fill these fields consistently.

`ExtraFields` carries provider data that must survive a later turn. For example, an Anthropic thinking signature belongs to the thinking block that produced it, while an OpenAI Responses phase belongs to `Message.ExtraFields` because it applies to the whole assistant message.

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

`GenerationRequest.Tools` is the complete tool list for one call. A tool schema can be written directly or derived from a Go type:

```go
type WeatherInput struct {
    Location string `json:"location"`
}

schema, err := GenerateSchema[WeatherInput]()
if err != nil {
    return err
}

request.Tools = []Tool{{
    Name:        "get_weather",
    Description: "Return the current weather for a location.",
    InputSchema: schema,
}}
```

JSON Schema is the interchange format because every supported tool API accepts a subset of it. `GenerateSchema[T]` rejects unknown object properties by default. Each provider adapter checks the schema again and reports constructs that its API cannot represent.

The adapter converts tools while it builds the provider request. It does not retain the converted definitions, and there is no registration method. As a result, authorization code can choose a different tool list for each request without mutating a shared generator.

A response may contain a `ToolCall` block whose content decodes to `ToolCallInput`. GAI does not execute that call. The application validates and runs it, appends a `ToolResult` message to the dialog, and sends another generation request. `ToolCallback` and `ToolCallBackFunc` are helpers for this application-owned loop.

**Implementation note.** Converting an unchanged schema on every request costs more than registration-time conversion. The current design favors immutable generators. If conversion becomes measurable, it can be cached in a separate immutable tool-set value.

## Generation options

```go
type GenerationOptions map[string]any

type GenerationOption func(GenerationOptions)
```

Common settings have typed helpers:

```go
options := NewGenerationOptions(
    WithTemperature(0.2),
    WithMaxGenerationTokens(500),
    WithToolChoice(ToolChoiceAuto),
)
```

The helpers write entries into the map using exported keys such as `GenerationOptionTemperature`. Provider-specific settings use their own exported keys and the same map:

```go
options[ResponsesServiceTierParam] = "priority"
```

There is no nested `ExtraArgs` map. An adapter reads the settings it supports, checks their Go types, and ignores unknown keys. Ignoring unknown keys allows a request used for fallback to contain settings for more than one provider.

Omission and zero have different meanings. If `GenerationOptionTemperature` is absent, the provider chooses its default. If the map contains `GenerationOptionTemperature: float64(0)`, the adapter sends zero.

The common keys use these value types:

| Keys | Type |
| --- | --- |
| `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` | `float64` |
| `top_k`, `candidate_count` | `uint` |
| `max_generation_tokens` | `int` |
| `tool_choice`, `thinking_budget` | `string` |
| `stop_sequences` | `[]string` |
| `output_modalities` | `[]Modality` |
| `audio_config` | `AudioConfig` |

`GenerationOptions` is a map rather than a struct because the supported parameters are the union of several provider APIs and that union changes often. Typed helpers keep common settings discoverable, while direct entries allow a provider option to be added without changing the shared request type.

## Responses

```go
type Response struct {
    Candidates    []Message
    FinishReason  FinishReason
    UsageMetadata Metadata
}
```

Each candidate is an assistant `Message`, so generated content can go back into a later dialog without conversion. The slice also matches providers that return several candidates.

`FinishReason` translates provider stop reasons into the package values `Unknown`, `EndTurn`, `StopSequence`, `MaxGenerationLimit`, `ToolUse`, and `ContentPolicyViolation`.

`Metadata` is `map[string]any`. Helpers read the common token counts, while the map leaves room for provider usage fields that GAI does not know about.

Provider API failures use `ApiErr`. It keeps the provider details and adds an `APIErrorKind` that retry and fallback policies can inspect. Local validation failures use sentinel errors or small error structs.

## Streaming

A streaming provider yields `StreamChunk` values through `iter.Seq`:

```go
type StreamChunk struct {
    Block              Block
    MessageExtraFields map[string]interface{}
    CandidatesIndex    int
    Err                error
}
```

Each value is either a data chunk or the terminal error. A data chunk has a nil `Err`. An error chunk has no payload; after yielding it, the producer returns. This gives callers one ordinary iterator to range over:

```go
for chunk := range generator.Stream(ctx, request) {
    if chunk.Err != nil {
        return chunk.Err
    }
    consume(chunk)
}
```

Adapters convert all failures to this form, whether the provider sent an error event, the SDK returned an iterator error, the connection failed, or the context was cancelled. `Err` is excluded from JSON and YAML because it is part of the live stream, not generated content.

Data chunks use the same `Block` type as messages. The stream protocol adds a few ordering rules:

- Text and thinking blocks contain fragments that may be joined with adjacent blocks of the same type.
- A tool call starts with a block containing the call ID and tool name. Later blocks contain JSON argument fragments.
- A separator ends a provider block. It prevents adjacent fragments from being joined and is omitted from the final response.
- If the provider reports usage, the final data chunk contains a metadata block.

`MessageExtraFields` carries metadata for the assistant message under construction. `StreamingAdapter` merges these maps and rejects conflicting values.

`CandidatesIndex` identifies the generated candidate. OpenAI Chat Completions reports this index when `candidate_count` is greater than one.

**Current limitations.** `StreamingAdapter` returns an error for candidate indexes above zero. The Gemini API supports multiple streamed candidates, but `GeminiGenerator.Stream` does not yet handle them.

## Provider generators

A provider generator keeps only what it needs to send a request:

| Generator | Stored fields | Interfaces |
| --- | --- | --- |
| `OpenAiGenerator` | completion service | `Generator`, `StreamingGenerator`, `TokenCounter` |
| `AnthropicGenerator` | message service | `Generator`, `StreamingGenerator`, `TokenCounter` |
| `GeminiGenerator` | Gemini client | `Generator`, `StreamingGenerator`, `TokenCounter` |
| `CerebrasGenerator` | HTTP client, endpoint, API key | `Generator` |
| `OpenRouterGenerator` | completion service | `Generator` |
| `ResponsesGenerator` | Responses service | `Generator`, `StreamingGenerator` |
| `ZaiGenerator` | generated clients and API transport | `Generator`, `StreamingGenerator` |

Constructors set up those connections. They do not choose a model, install tools, or store instructions. Every call gets that data from `GenerationRequest`.

`ResponsesGenerator` also sets the upstream `store` option to false. The OpenAI service does not retain the conversation, and the Go generator does not retain request state.

## Wrappers

```go
type GeneratorWrapper struct {
    Inner Generator
}

type WrapperFunc func(Generator) Generator
```

`GeneratorWrapper` provides default delegation for middleware that needs to change only one operation. A wrapper embeds it and overrides `Generate`, `Stream`, or `Count` as needed.

`Wrap` builds a middleware stack. The first function is the outermost wrapper and receives the call first:

```go
generator := Wrap(
    base,
    WithRetry(retryConfig),
    WithPreprocessing(),
)
```

`GeneratorWrapper` defines `Stream` and `Count` even when its inner `Generator` does not implement the corresponding optional interface. An interface assertion on the wrapper therefore does not prove that the operation is available. `Count` returns an unsupported error, and `Stream` yields one terminal error chunk.

`RetryGenerator` retries ordinary generation failures according to its policy. For streaming, it retries only failures that occur before the caller receives a data chunk. Retrying after partial output would duplicate content.

`FallbackGenerator` tries generators in order and forwards the request unchanged.

**Current limitation.** Fallback also forwards `Model` unchanged. Targets that use different model names need a request-rewrite step outside `FallbackGenerator`; the package does not provide a rewrite hook.

`PreprocessingGenerator` makes a shallow copy of the request and replaces its dialog with one that combines parallel tool results. Anthropic and Gemini require this form.

`StreamingAdapter` collects a `StreamingGenerator` and exposes it as a normal `Generator`.

`AnthropicServiceWrapper` wraps the Anthropic SDK service before it reaches `AnthropicGenerator`. It applies provider request changes such as cache controls.

## Ownership and concurrency

A generator can handle concurrent calls with different models, instructions, dialogs, tools, and options. The provider client stored inside it must also support concurrent use.

The caller owns request slices and maps. Adapters read them and build provider request values without changing them. Functional option helpers clone slice arguments, so changing the source slice later does not change the stored option.

Retry notifications, fallback policy functions, and service middleware may run concurrently. Code passed into those hooks must be safe for that use.

Here, "stateless generator" has a narrow meaning. A generator has no hidden model request settings and no mutable tool registry. It still has fields for clients, credentials, endpoints, and wrapper policy.
