# Generator design

This document describes the request model, generator interfaces, and provider adapters used by GAI. It also describes the rules that wrappers rely on when they retry, fall back, preprocess dialogs, or collect a stream.

A generator wraps a connection to a model provider. It stores the provider client and, for direct HTTP adapters, the endpoint and credentials. It does not store a model, system instructions, tools, or conversation history. Those values belong to `GenerationRequest` and are supplied on every call.

The root `gai` package does not append to the conversation or execute tool calls. Applications can own that loop directly or use the public [`agent` package](agent/design.md), which executes tools and manages a run's active dialog. Persistence remains outside the generator layer.

This document describes the implemented generator layer, not a proposed migration. See [README.md](README.md) for setup and usage, [doc.go](doc.go) for the package API map, and [agent/design.md](agent/design.md) for the agent loop.

A generation call follows this path:

```text
GenerationRequest -> wrappers -> provider adapter -> provider API
                                              |
Response          <- normalization <----------+
```

For example, one OpenAI generator can handle requests with different models and instructions:

```go
generator, err := NewOpenAiGenerator(nil, "", apiKey)
if err != nil {
    return err
}

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
    Headers          http.Header
    Model            string
    SafetyIdentifier string
    PromptCacheKey   string
    Instructions     Message
    Dialog           Dialog
    Tools            []Tool
    Options          GenerationOptions
}
```

The fields have the following meanings:

| Field | Meaning |
| --- | --- |
| `Headers` | Per-invocation HTTP header overrides, not provider JSON fields. |
| `SafetyIdentifier` | Provider end-user identity where supported. |
| `PromptCacheKey` | Cache routing hint where supported. |
| `Model` | Provider model name for this call. |
| `Instructions` | Optional system message. |
| `Dialog` | Complete conversation presented to the model. |
| `Tools` | Complete set of tools available during this call. |
| `Options` | Common and provider-specific generation settings. |

A retry can repeat the same request, and middleware can inspect or replace any part of it without reading generator fields. Fallback passes the same value to each target.

`context.Context` remains a separate argument because it controls the execution of the call, not the model's behavior. It carries cancellation and deadlines.

Generators treat a request as read-only. Copying `GenerationRequest` is shallow because its headers, messages, tools, and options contain slices or maps. The caller must not modify those values while a call is in progress, and a generator must not retain them after the call returns.

Data created while handling a request stays local to that call. Examples include converted provider messages, tool-call ID maps, retry counters, and stream assembly buffers.

### Request headers

`Headers` is an `http.Header` field, not a generation-option key or part of the provider JSON body. It applies to generation, streaming, and remote token counting. Supplied header values replace matching names; unrelated generator/client headers remain intact. Multiple values are preserved. Use `Header.Set` or `Header.Add` to canonicalize names, and do not supply duplicate names with different capitalization.

Headers are applied per invocation, never by mutating a shared client. The Chat Completions client uses a request editor; Anthropic and Responses use SDK request options; Gemini uses per-call `HTTPOptions`; OpenRouter delegates to the shared Chat Completions request editor. Z.AI and Moonshot apply headers to their native counting requests as well. SDK-managed authentication can take precedence: Gemini sets `X-Goog-Api-Key` from its client credentials.

Request headers can contain secrets. Serializing or logging a `GenerationRequest` also includes its headers, so callers must redact them where necessary. Fallback forwards headers unchanged, including credentials if the caller supplied them; only use that with trusted targets.

### Identity and cache routing

`SafetyIdentifier` and `PromptCacheKey` have different meanings. Neither is derived from the other or inserted into prompt content.

- `SafetyIdentifier` maps to `safety_identifier` for OpenAI Chat Completions/Responses and Moonshot, `metadata.user_id` for Anthropic, `user_id` for Z.AI/DeepSeek, and `user` for Cerebras/OpenRouter. DeepSeek also uses its identity field for cache and scheduling isolation.
- `PromptCacheKey` maps to `prompt_cache_key` for OpenAI Chat Completions/Responses, Moonshot, Cerebras, and OpenRouter. An explicit OpenRouter session ID takes routing precedence. It does not enable caching, select a retention policy, or identify a stored cache resource.
- Unsupported adapters ignore these fields. In particular, Gemini does not translate either into safety settings or `cachedContent`.

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

All three methods receive the same request rather than relying on model or dialog settings hidden in the generator. Counting support is deliberately narrower than generation support:

- `OpenAiGenerator.Count` uses `tiktoken` and local estimates. It is not a native counter for arbitrary compatible endpoints. PDF and audio inputs are unsupported by this counter.
- Anthropic, Gemini, Z.AI, and Moonshot use remote counting APIs. Z.AI counts messages and tools; Moonshot rejects tool-bearing requests because its endpoint does not document tool accounting.
- Cerebras, DeepSeek, OpenRouter, and Responses do not implement `TokenCounter`.

Every provider adapter listed below implements generation and streaming. Optional interfaces describe the adapter's capabilities, not a guarantee that every model accepts every input.

## Messages and instructions

```go
type Dialog []Message

type Role uint

const (
    RoleUnknown Role = iota
    User
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

Initial system instructions use the same `Message` representation but live in a separate request field:

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

`RoleUnknown` is the zero value and catches populated messages whose caller forgot to choose a role. It is invalid in dialogs, inputs, responses, and tool results. The exact empty message uses `RoleUnknown` only to mean that optional `Instructions` are absent. Explicit instructions, including a message with zero blocks, must have the `System` role. A dialog can also contain `System` messages on providers that support inline instructions. These messages retain their position and are not merged into `Instructions` or rewritten as user messages. Chat Completions and Responses serialize them as ordered native messages. Native Gemini rejects inline system messages.

GAI's Anthropic adapter accepts only text content in inline system messages. In GAI's user/function-tool history, a system section must follow a user turn (including the complete tool-result batch) and precede an assistant turn or end the dialog. Consecutive system messages form one section; the server validates placement and model support. Ordinary text messages need no beta header. The [native Anthropic rules](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages#limitations) also allow text after an assistant turn ending in a server tool result, which GAI cannot currently represent, and native tool-change blocks with separate beta requirements, which this adapter does not expose.

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

A `Message` contains an ordered slice of blocks; whether an empty message is valid depends on its role and the operation. `BlockType` identifies text or media content, model thinking, a tool call, usage metadata, or an internal stream separator. It is a string so an adapter can carry a provider-specific block type without changing the shared enum first.

`ModalityType` describes the data as text, image, audio, or video. `MimeType` gives the concrete media format; `PDFBlock` uses the `Image` modality with `application/pdf`, not a separate PDF modality. `Content` is a `fmt.Stringer`, and adapters read it through `Content.String()`. The package constructors `TextBlock`, `ImageBlock`, `AudioBlock`, `PDFBlock`, and `ToolCallBlock` fill these fields consistently.

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

JSON Schema is the interchange format because every supported tool API accepts a subset of it. `GenerateSchema[T]` produces closed object schemas by default. Adapters validate tool declarations and perform the conversion their protocol requires. This is not a universal schema validator: the shared Chat Completions adapter preserves schema JSON and leaves provider/model restrictions to the server.

The adapter converts tools while it builds the provider request. It does not retain the converted definitions, and there is no registration method. As a result, authorization code can choose a different tool list for each request without mutating a shared generator.

A response may contain a `ToolCall` block whose content decodes to `ToolCallInput`. The generator does not execute that call. The application, or the separate `agent` package, validates and runs it, appends a `ToolResult` message to the dialog, and sends another generation request. `ToolCallback` and `ToolCallBackFunc` support application-owned dispatch.

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

The helpers write entries into the map using exported keys such as `GenerationOptionTemperature`. Established provider-specific settings have helpers that return the same `GenerationOption` type:

```go
options := NewGenerationOptions(
    WithResponsesServiceTier("priority"),
)
```

Provider helpers use exported, namespaced keys such as `ResponsesServiceTierParam`, or populate a shared native-body map. For example, `WithOpenAIResponseFormat` adds fields to `OpenAIGenerationOptionExtraBody`. Prediction and response formats keep their provider-specific APIs. `WithOpenAIExtraBody` accepts a `map[string]json.RawMessage` for explicit wire fields.

There is no nested `ExtraArgs` map. An adapter reads the settings it supports, checks their Go types, and ignores unknown keys. Ignoring unknown keys allows a request used for fallback to contain settings for more than one provider.

Omission and zero have different meanings. If `GenerationOptionTemperature` is absent, the provider chooses its default. If the map contains `GenerationOptionTemperature: float64(0)`, the adapter sends zero.

The common keys use these value types:

| Keys | Type |
| --- | --- |
| `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` | `float64` |
| `top_k`, `candidate_count` | `uint` |
| `max_generation_tokens`, `seed`, `top_logprobs` | `int` |
| `logprobs`, `parallel_tool_calls` | `bool` |
| `logit_bias` | `map[string]float64` |
| `tool_choice`, `reasoning_effort` | `string` |
| `thinking_budget` | `int` |
| `stop_sequences` | `[]string` |

`WithReasoningEffort(string)` is separate from `WithThinkingBudget(int)`. The former carries an effort label; the latter carries a positive numeric thinking-token budget. Anthropic, Gemini, and OpenRouter support the budget and reject combining the two. Gemini checks int32 representability; its zero/automatic budget sentinels are not shared-option values.

Shared Chat Completions wrappers preserve explicit options rather than inferring capabilities from model names. They validate Go value types and serialization; the server validates native values, ranges, and supported combinations. Other protocol adapters still perform their own local validation. Parallel tool calling, seeds, log probabilities, and logit bias use common option keys instead of duplicated provider helpers. Gemini maps seed, logprob controls, presence/frequency penalties, and numeric thinking budgets; Responses maps logprob controls and parallel tool calls; Anthropic maps top-k and parallel tool calls by negating its disable flag. Min-p, top-a, and repetition penalty use OpenRouter-native helpers. Output modalities and audio configuration use OpenAI-native helpers and keys; Gemini has its own output-modality helper, with returned inline image/audio bytes represented as base64 blocks carrying their MIME types and thought signatures. Streaming collection retains separate media payloads rather than concatenating their base64 strings. Anthropic and Responses do not parse output-modality options. Service tiers, routing, prediction, and response formats remain provider-specific. Forwarding a shared field through an OpenAI-compatible wrapper does not imply endpoint support; README distinguishes documented controls, fixed/ignored settings, and undocumented forwarded fields. `WithTopLogprobs` selects a count rather than implicitly enabling `WithLogprobs`. Explicit false and zero values are retained; unsupported controls are ignored by adapters that do not implement them.

`GenerationOptions` is a map rather than a struct because the supported parameters are the union of several provider APIs and that union changes often. Typed helpers keep common settings discoverable, while direct entries allow a provider option to be added without changing the shared request type.

## Responses

```go
type Response struct {
    Candidates    []Message
    FinishReason  FinishReason
    UsageMetadata Metadata
    ExtraFields   map[string]interface{}
}
```

Each candidate is an assistant `Message`, so generated content can go back into a later dialog without conversion. The slice also matches providers that return several candidates.

`FinishReason` translates provider stop reasons into the package values `Unknown`, `EndTurn`, `StopSequence`, `MaxGenerationLimit`, `ToolUse`, and `ContentPolicyViolation`.

Response data is placed at the narrowest scope that owns it:

- `UsageMetadata` contains measurements such as token counts, cache usage, cost, and timing.
- `Response.ExtraFields` contains invocation-level provider data such as completion IDs, model names, timestamps, fingerprints, and service tiers.
- `Message.ExtraFields` contains data that belongs to one candidate.
- `Block.ExtraFields` contains data tied to one content item, including metadata that must be replayed with that block on a later turn.

`Metadata` and `ExtraFields` are heterogeneous maps with value types documented on their keys. They can include `json.RawMessage`, typed maps, or HTTP headers, not just untyped decoded JSON. Providers export constants for stable native keys. Callers commonly append a returned candidate to the next dialog without retaining its enclosing `Response`, so adapters keep all replay-critical data on the message or block rather than in `Response.ExtraFields`.

Mapped provider API failures use `ApiErr`. It keeps provider details and adds an `APIErrorKind` that retry and fallback policies can inspect. Context, transport, decoding, and local validation failures can use other error types. Some generation failures return a partial `Response` alongside an error; callers must inspect the method's contract rather than assume every error implies empty output.

## Streaming

A streaming provider yields `StreamChunk` values through `iter.Seq`:

```go
type StreamChunk struct {
    Block               Block
    MessageExtraFields  map[string]interface{}
    ResponseExtraFields map[string]interface{}
    CandidatesIndex     int
    Err                 error
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
- A tool call is represented by a header block containing the call ID and tool name, followed by argument blocks. These may be incremental fragments or buffered complete arguments, depending on the adapter.
- A separator ends a provider block. It prevents adjacent fragments from being joined and is omitted from the final response.
- If the provider reports usage, the final data chunk contains a metadata block.

`MessageExtraFields` carries metadata for the assistant message under construction. `ResponseExtraFields` carries invocation-level metadata discovered while streaming. `StreamingAdapter` merges both maps across chunks and rejects conflicting values.

`CandidatesIndex` identifies the generated candidate. OpenAI Chat Completions reports this index when `candidate_count` is greater than one.

**Current limitations.** `StreamingAdapter` returns an error for candidate indexes above zero. It returns an empty response on a stream error rather than exposing partially collected output. It infers `ToolUse` when the collected blocks contain a tool call and `EndTurn` otherwise; it does not carry every native finish reason. It also converts numeric values from metadata blocks to integers, so it is not suitable for preserving fractional native usage metrics. The Gemini API supports multiple streamed candidates, but `GeminiGenerator.Stream` does not yet handle them.

### Shared Chat Completions streaming

The shared parser reads SSE events and tracks each candidate and tool call separately. Parallel tool-call fragments are assembled by index, checked, and emitted in index order after reading the stream.

Callers display reasoning by reading `Thinking` blocks. These can come from `reasoning_content`, `reasoning`, or the text and string summaries inside `reasoning_details`. Encrypted data, unknown detail types, and summaries that are not strings stay in metadata; they are not presented as readable thoughts. When a later fragment omits its type, the adapter uses the type already received for that item.

Some providers return the same reasoning in several fields. The adapter avoids showing that text twice:

- Receiving `"First step"` after `"First "` adds only `"step"` to the stream.
- Receiving `"First step"` again adds nothing.
- Receiving `"A different approach"` adds a separate thinking block. Previously streamed text cannot be replaced.

Each candidate has its own reasoning state. Chunks from separate answers do not share displayed text or saved reasoning details, even when they use the same reasoning-detail index. Callers handling multiple candidates must consume `Stream` directly; `StreamingAdapter` currently accepts only candidate 0. Within an answer, readable details are joined in item order. There is no special handling for an earlier item's text resuming after a later item starts: neither the provider documentation checked nor the live GLM/DeepSeek probes established that behavior. If separate reasoning fields contain different text with a shared beginning, both are kept, so that beginning may appear twice.

To continue a conversation, append the returned assistant message to the dialog, including all message and block `ExtraFields`. The adapter sends the original reasoning data back to the provider; it does not reconstruct signatures or encrypted data from the displayed text. `OpenAIExtraFieldReasoningSource` records which response field supplied each thinking block so the text is not sent a second time under another field name.

Editing only a thinking block's text does not replace the original reasoning kept in message metadata. If that original field is removed, the block's text is sent through `reasoning_content` instead, unless it is already present at the start of a retained reasoning string. Caller-created thinking blocks also use `reasoning_content`; an empty or null `reasoning_details` field does not discard them. These conversions do not modify the caller's message.

`[DONE]` does not immediately end reading: the parser consumes through EOF to retain trailing metadata such as gateway cost. Callers should supply a context deadline. EOF without `[DONE]` or a finish reason is treated as incomplete output. There are no automatic POST retries or stream reconnects in this adapter; retry policy belongs to a wrapper.

## Provider generators

A provider generator keeps only what it needs to send a request:

| Generator | Stored fields | Interfaces |
| --- | --- | --- |
| `OpenAiGenerator` | generated Chat Completions HTTP client | `Generator`, `StreamingGenerator`, `TokenCounter` |
| `AnthropicGenerator` | message service | `Generator`, `StreamingGenerator`, `TokenCounter` |
| `GeminiGenerator` | Gemini client | `Generator`, `StreamingGenerator`, `TokenCounter` |
| `CerebrasGenerator` | private `OpenAiGenerator` delegate | `Generator`, `StreamingGenerator` |
| `OpenRouterGenerator` | private `*OpenAiGenerator` delegate | `Generator`, `StreamingGenerator` |
| `ResponsesGenerator` | Responses service | `Generator`, `StreamingGenerator` |
| `ZaiGenerator` | private `*OpenAiGenerator` and native counting HTTP client | `Generator`, `StreamingGenerator`, `TokenCounter` |
| `DeepSeekGenerator` | private `*OpenAiGenerator` | `Generator`, `StreamingGenerator` |
| `MoonshotGenerator` | private `*OpenAiGenerator` and native counting HTTP client | `Generator`, `StreamingGenerator`, `TokenCounter` |

Constructors set up those connections. Generated clients under `internal/` never appear in public constructor parameters or return types. The generated-client adapters accept a standard `*http.Client`, base URL, and API key, then construct their private client internally. An empty base URL selects the exported provider default, including its API prefix. The API key must be non-empty; constructors do not read credentials from environment variables or other global state. Constructors return client setup errors instead of retaining an invalid client.

Constructors do not choose a model, install tools, or store instructions. Every call gets that data from `GenerationRequest`.

The Cerebras, OpenRouter, Z.AI, DeepSeek, and Moonshot adapters hold private `*OpenAiGenerator` fields, not embedded generic wrappers. Each provider's own file implements request translation and Generate/Stream delegation, copying request option maps to apply its defaults and preserve explicit native overrides. Native counting endpoints, payloads, response schemas, and limitations also belong to the corresponding provider. Only error relabeling and authenticated JSON HTTP mechanics are shared in `compatible.go`; that file does not dispatch on provider identity. This shares wire conversion and stream assembly without accidentally exposing OpenAI's local token counting. Z.AI and Moonshot have native counters; Moonshot rejects tool-bearing count requests rather than claiming undocumented accounting.

### Shared client and provider defaults

Only one generated client remains under `internal/`: `internal/openai` uses oapi-codegen. The separate OpenRouter, Cerebras, DeepSeek, Z.AI, and OpenCode clients have been removed. The shared Chat Completions schema preserves opaque JSON, nullable fields, and numeric values without routing provider-specific behavior through a central provider switch. Regenerate it with `go generate ./internal/openai`; its tests check regeneration and codec/HTTP contracts.

| Adapter | Default output-limit field | Stream usage request |
| --- | --- | --- |
| OpenAI, Cerebras, Moonshot | `max_completion_tokens` | Enabled |
| DeepSeek, OpenRouter | `max_tokens` | Enabled |
| Z.AI | `max_tokens` | Omitted |

`WithOpenAITokenLimitField` and `WithOpenAIStreamUsage` override these defaults. Translated native options are merged with explicit `WithOpenAIExtraBody` fields, with explicit fields taking precedence. Cerebras and OpenRouter use the shared request fields for identity and cache routing, not separate native helpers. OpenRouter translates effort to `reasoning.effort` and numeric budgets to `reasoning.max_tokens`, rejecting their combination. Native capability and range validation belongs to the server.

The shared adapters use OpenAI-named metadata keys even when the endpoint is another provider:

- `OpenAIExtraFieldWireFields` retains message/block fields needed for replay.
- `OpenAIExtraFieldChoice` retains candidate-level details such as logprobs.
- `OpenAIResponseExtraFieldWireFields` retains invocation fields, including IDs, timing, service tiers, and gateway cost.
- `OpenAIResponseExtraFieldUsage` retains native usage breakdowns alongside normalized common metrics.
- `OpenAIResponseExtraFieldHeaders` retains response headers.

Thinking blocks no longer carry a generator-name marker. Replay depends on the actual provider wire fields, reasoning IDs, signatures, and encrypted content. Replay metadata remains on messages/blocks so appending a candidate does not require retaining the enclosing response. It is not necessarily portable to a different backend.

The shared adapters encode image bytes as data URLs and PDFs as inline file bodies where accepted by the backend. Schema coverage is not a claim that every provider supports every modality or native tool. Recorded request tests exercise public adapters against captured SDK conversations; normal tests never refresh captures or call live APIs.

### Protocol-specific adapters and gateways

Responses, Anthropic Messages, and Gemini retain separate implementations because their request and streaming protocols differ from Chat Completions. The Anthropic and Responses constructors accept SDK service interfaces; Gemini accepts a `*genai.Client`. Their underlying SDK configuration remains the caller's responsibility.

`ResponsesGenerator` sets `store: false` in its upstream request. This opts out of storing the response through that API option; it is not a guarantee about every form of provider logging or retention. The Go generator does not retain conversation state.

OpenCode has no generator of its own. Callers select Chat Completions, Responses, or Anthropic Messages according to the model and configure that adapter's endpoint and credentials. For OpenCode Go, the Chat Completions/Responses base URL is `https://opencode.ai/zen/go/v1`; the Anthropic SDK base URL is `https://opencode.ai/zen/go/`. Set `Headers` with a stable `X-Opencode-Session` value for the conversation, including retries and tool follow-ups. Errors and metadata use the selected protocol adapter's labels and keys.

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

`FallbackGenerator` implements only `Generator`. It tries generators in order while its fallback policy permits, forwarding the request unchanged. It does not provide streaming failover or token counting.

**Current limitation.** Fallback also forwards `Model` unchanged. Targets that use different model names need a request-rewrite step outside `FallbackGenerator`; the package does not provide a rewrite hook.

`PreprocessingGenerator` makes a shallow copy of the request and replaces its dialog with one that combines consecutive parallel tool results for generation and streaming. Anthropic and Gemini require this form. Other fields, including headers, are forwarded unchanged. Its inherited `Count` method delegates without preprocessing.

`StreamingAdapter` collects a `StreamingGenerator` and exposes it as a normal `Generator`. Its `Count` method delegates when the inner stream generator also implements `TokenCounter`; otherwise it returns an unsupported error.

`AnthropicServiceWrapper` wraps the Anthropic SDK service before it reaches `AnthropicGenerator`. It applies provider request changes such as cache controls.

## Ownership and concurrency

A generator can handle concurrent calls with different headers, models, instructions, dialogs, tools, and options. The provider client stored inside it must also support concurrent use.

The caller owns request slices and maps. Adapters read them and build provider request values without changing them. Option helpers copy data according to their documented contracts; some copies are shallow. Nested maps, schemas, and block content should be treated as borrowed unless the corresponding helper promises a deeper copy.

Retry notifications, fallback policy functions, and service middleware may run concurrently. Code passed into those hooks must be safe for that use.

Here, "stateless generator" has a narrow meaning. A generator has no hidden model request settings and no mutable tool registry. It still has fields for clients, credentials, endpoints, and wrapper policy.

## TODOs

This section collects unresolved design work. These items do not describe the current API.

- **Provider-native tools.** Define how requests enable provider-native tools such as hosted web search, computer use, text editing, and apply-patch. The design must cover execution ownership, normalized call and result blocks, and replay-safe provider metadata.
