// Package agent runs reusable tool-using model loops on top of GAI.
//
// Construct an [Agent] with [New], then call [Agent.Run] with a prior dialog and
// one new User message. The Agent prepares the active dialog, builds generation
// requests, prefers streaming when available, executes its fixed
// tools, appends results, and continues until the model or a hook stops.
//
// [PrepareDialogHook], [BeforeGenerationHook], [AfterGenerationHook],
// [BeforeToolHook], and [AfterToolHook] are ordered decision points. Hook input
// values are borrowed and read-only. Values returned for use by the loop belong
// to the loop after the hook returns. [Observer] receives synchronous, borrowed
// events that are valid only during [Observer.Observe]. Run failures return the
// dialog and standard usage collected before the failing operation.
//
// Package agenttest provides scripted generators and a recording observer for
// deterministic tests. See design.md for the complete behavior and rationale.
package agent
