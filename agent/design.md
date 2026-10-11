# Extensible agent loop design

## Responsibility and composition

The `agent` package runs a model-and-tool loop using GAI's common requests, responses, messages, and tools. The root `gai` package owns provider normalization and composable generators. The agent package owns the ordering of generation, hooks, tool execution, and acceptance into the active dialog.

An `Agent` is reusable configuration and behavior. Each `Run` has its own active dialog, counters, and usage. The application owns the conversation across runs, persistence, external resources, and any state captured by its callbacks.

The design has three extension points:

- **Generators** supply model behavior. Applications can compose GAI generators to add retries, fallback, or preprocessing without changing the loop.
- **Tools** pair model-visible declarations with application-provided handlers. Handlers perform domain work and return results for the model.
- **Hooks** let applications make decisions at defined points in the loop or consume provisional streaming output.

Hooks and handlers use interfaces with function adapters. An implementation can be a stateful object or an ordinary function. Each hook has one configuration slot; applications combine behaviors in ordinary Go code, choosing their order and error policy. Composition does not require a plugin registry, hook dispatcher, or application framework.

## Configuration and one run

Configuration contains a generator, model, optional system instructions, fixed executable tools, and optional hooks. It is borrowed and remains unchanged during the Agent's lifetime. Changing that configuration uses a newly configured Agent. `BeforeGeneration` can select a request's model, instructions, dialog, and options while preserving the registered tool declarations.

`Run` receives a complete, nonempty dialog and generation options. The caller appends a user message when starting a new turn. The loop always starts by preparing another generation from that dialog; it does not infer that an earlier run has already completed from a trailing assistant message.

The dialog may include inline `System` messages when the selected provider supports them. `BeforeGeneration` handles dialog compaction, retrieval, and instruction changes alongside other request settings. A validated replacement dialog becomes the conversation the loop continues from; accepted responses and tool results are appended to it. Inline instructions remain separate from the request's `Instructions` field. Provider-specific model and placement rules remain the caller's responsibility. System messages cannot interrupt unresolved historical tool calls; supply the complete tool-result batch before appending instructions.

The loop validates the supplied conversation, prepares a request, generates one assistant response, and processes any requested tools sequentially in response order. It repeats until the model produces a terminal response, a hook requests a normal stop, or an error occurs. Streaming is preferred when the generator supports it, but the loop accepts only complete responses into the dialog.

`RunResult` contains the active dialog, stop information, and standard usage accepted during the invocation. It is the caller's starting point for the next turn or for handling a failure. The Agent retains no conversation between invocations.

## Hooks and their purposes

| Hook | Purpose | Position and effect |
|---|---|---|
| `BeforeGeneration` | Compaction, retrieval, instructions, generation settings, or budget checks | Receives the base request and can replace it or stop before generation. A validated replacement's dialog becomes the active conversation. Tool declarations remain fixed. |
| `GenerationChunk` | Live display or incremental consumption | Receives raw streaming chunks synchronously. Output is provisional and cannot replace the eventual response. |
| `AfterGeneration` | Review or transformation of a complete model response | Runs after initial response validation and before acceptance. A replacement must preserve the provider's finish reason and usage. |
| `BeforeTool` | Permission checks, argument changes, or controlled rejection | Runs for a known tool with valid arguments. It can replace arguments, reject execution with a model-visible result, or request a stop after the batch. |
| `AfterTool` | Transforming a handler result or requesting a review boundary | Runs after a handler returns a valid result, before acceptance. It can replace that result or request a stop after the batch. |

There is one request-change hook. `BeforeGeneration` supplies the request sent to the generator, and its validated dialog is adopted before generation. Later base requests combine the current dialog with the configured model, instructions, fixed tools, and `RunRequest.Options`; selecting other request fields does not mutate the Agent or the caller's run options. `RunResult.Dialog` reports the conversation used by the loop. The application decides what to store, remove, or reuse across runs; the loop does not classify dialog changes as persisted or temporary.

After-hooks are decision points before acceptance. Their invocation does not establish that their output entered the dialog, and they cannot repair outputs that failed initial validation. `AfterTool` also runs for a valid model-visible failure returned by a handler, but not for handler Go errors or synthetic results produced for rejected, unknown, or invalid calls.

`GenerationChunk` is the only notification-only hook because it exposes output unavailable from the complete-response hooks or `Run`'s return. Application code surrounds `Run` for lifecycle handling and records context replacements in the hook that produces them. Synthetic tool-error results are available in the dialog, without a separate completion notification.

## Replacement and stopping semantics

All replacement hooks follow one rule: **nil preserves the existing value; a supplied replacement is validated and adopted**. Request, response, and result replacements use pointers; argument replacements use maps. To replace a dialog, copy the generation request and set its `Dialog` before returning it from `BeforeGeneration`. A non-nil replacement request must contain a complete, nonempty dialog. An explicitly empty replacement is distinct from an omitted replacement and must satisfy the relevant validity rules.

An empty decision is therefore a no-op. Replacement and control decisions are independent: an `AfterTool` hook can request a batch stop while leaving its result unchanged.

```go
return agent.AfterToolDecision{
    StopAfterBatch: "review_required",
}, nil
```

A normal hook stop returns a nil Go error and an application-defined stop reason. `StopReasonModel` identifies a terminal model response. A `BeforeGeneration` stop skips the model call and ignores any replacement request.

A tool hook's stop takes effect after the entire current batch has results. This preserves a conversation that can be sent to the model later. The first stop reason wins in hook execution order; an error stops work immediately and takes precedence over a planned normal stop. Rejecting one tool call is different from stopping: rejection creates a result the model can respond to, and the batch continues.

Applications define limits using hooks, context deadlines, and their own policy. The loop imposes no default generation, tool, or token budget.

## Accepted work and failures

Acceptance means that validated data has entered the active state returned by `Run`. A `BeforeGeneration` request's dialog is adopted after validation and before calling the generator, so a subsequent generation failure returns that dialog without inventing a response. The current response or tool result and its associated standard usage are not yet accepted while its after-hook executes. Hook request changes do not add usage to the run's generation totals; applications account for any external work done by their hooks.

`RunStatus.AcceptedGenerations` counts accepted model responses. `AcceptedToolResults` counts accepted handler results, including model-visible failures, but excludes synthetic results for rejected or invalid calls. Counters and usage cover only the current invocation. They do not count attempted external operations or establish whether side effects occurred. Provider-specific metadata remains available through the complete response supplied to `AfterGeneration` or through generator wrappers.

There are two error paths:

- **Model-visible failures** are valid tool results marked as errors. A handler can return one for an expected problem. The loop also creates them for unknown tools, invalid arguments, and hook rejection, allowing the model to respond.
- **Execution failures** are Go errors from generation, handlers, hooks, cancellation, or validation. They stop the run and return a `RunError` with a phase and an underlying cause accessible through `errors.Is` and `errors.As`.

Input validation fails before callbacks run and returns a zero result. Later failures return the dialog and usage accepted so far. The loop does not roll back completed external work or invent a successful tool result. For example, a handler may succeed and its after-hook may fail, leaving the assistant's tool call in the returned dialog without the handler's result.

## Cancellation, borrowing, and concurrency

Every hook receives the Run context. The loop checks cancellation immediately before invoking each hook, including each chunk notification, and before executing a tool handler. Already-visible cancellation prevents invocation and returns a cancellation-phase `RunError`. Cancellation can arrive after that check, so callbacks must cooperate while running. It cannot undo an external effect.

Configuration, requests, hook inputs, and chunk data are borrowed. Implementations must treat referenced inputs as read-only and copy or serialize data they need to retain beyond its documented lifetime. The library does not deep-copy requests.

A struct copy shares its referenced maps, slices, and pointers. Callers copy only the parts they intend to change:

```go
next := r.Request
next.Model = "another-model" // Only the copied struct changes.
next.Options = maps.Clone(r.Request.Options)
if next.Options == nil {
    next.Options = make(gai.GenerationOptions)
}
// Map entries can now change independently; nested referenced values are shared.
return agent.BeforeGenerationDecision{Request: &next}, nil
```

The same rule applies to message blocks and argument maps. Slice append can reuse shared capacity. Replacement data and successful handler outputs belong to the loop after return and must not be modified afterward. Unchanged references may be passed through, and `RunResult` may share storage with inputs or callback outputs.

Hooks and tools run sequentially within one invocation. Separate invocations can call the same configured dependencies concurrently. Applications synchronize shared callback state and own resource cleanup. Context values can carry invocation-specific identity or application state.

## Continuation and consumer-owned durability

A supplied dialog must contain results for every historical tool call. The loop rejects unresolved calls before any hook, generation, or handler runs. It never executes tools from supplied history. Appending another user message does not resolve an unanswered call.

That boundary is intentional: a missing result cannot distinguish a tool that never started from an external action that succeeded before its response was lost. The consumer owns storage and reconciliation, supplies confirmed results or explicitly chosen model-visible failures, and then calls `Run` with the completed dialog. Generator wrappers and external services may have their own retry policies; the loop does not automatically repeat ambiguous effects or promise exactly-once execution.

Continuation supplies conversation context for further generation. It does not replay interrupted hooks or reconstruct callback state. Applications restore their own dependencies, configuration, and business state. No separate persistence or resume framework is required by this package.

## API reference and examples

[Hook contracts](hooks.go), [Agent and run types](agent.go), [tools](tool.go), and [errors](errors.go) contain field-level documentation. [Compiled examples](example_test.go) demonstrate ordinary runs, request and dialog changes, tools, streaming, and reconciled continuation. `agent/agenttest` provides scripted generators for exercising application behavior without live providers.
