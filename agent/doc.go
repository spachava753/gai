// Package agent runs reusable tool-using model loops on top of GAI.
//
// Construct an [Agent] with [New], then call [Agent.Run] with a complete dialog
// ready for another model call. Append user messages in application code. Every
// historical tool call must have a result; the loop never repeats historical
// tools or reconstructs interrupted callbacks. Persistence and reconciliation
// belong to the application.
//
// [PrepareDialogHook], [BeforeGenerationHook], [AfterGenerationHook],
// [BeforeToolHook], and [AfterToolHook] return explicit decisions. A nil
// replacement keeps the existing value. [GenerationChunkHook] receives borrowed,
// provisional streaming output and returns only an error. All hooks have
// function adapters and receive the Run context. Cancellation is checked before
// each hook invocation; callbacks must handle cancellation during their work.
//
// Inputs are borrowed and read-only. Struct copies share referenced maps,
// slices, and pointers; callers copy only the parts they need to change.
// Replacement data belongs to the loop after a hook returns. Failures return
// accepted dialog and usage alongside a [RunError] identifying the failing phase.
//
// Package agenttest provides scripted generators for deterministic tests.
// See design.md for ordering, ownership, and continuation contracts.
package agent
