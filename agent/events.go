package agent

import (
	"context"

	"github.com/spachava753/gai"
)

// Observer receives ordered, read-only events from one Agent.Run call. Event
// values borrow the loop's data and are valid only until Observe returns. An
// observer that needs to retain an event must serialize or copy the needed data.
type Observer interface {
	// Observe handles one event before the run continues.
	Observe(context.Context, Event) error
}

// ObserverFunc adapts a function to Observer.
type ObserverFunc func(context.Context, Event) error

// Observe calls f.
func (f ObserverFunc) Observe(ctx context.Context, event Event) error {
	return f(ctx, event)
}

// Event is one sequenced update from an Agent run.
type Event struct {
	// Sequence is zero-based and gapless within a run.
	Sequence uint64
	// Payload contains one package-defined event value.
	Payload EventPayload
}

// Kind returns the payload kind. A nil payload returns the empty EventKind.
func (e Event) Kind() EventKind {
	if e.Payload == nil {
		return ""
	}
	return e.Payload.Kind()
}

// EventPayload is implemented by the package-defined event payloads.
type EventPayload interface {
	// Kind identifies the concrete payload.
	Kind() EventKind
	eventPayload()
}

// EventKind identifies one point in an Agent run.
type EventKind string

const (
	// EventKindRunStarted is sent after validation and before any hook.
	EventKindRunStarted EventKind = "run_started"
	// EventKindDialogReplaced is sent after PrepareDialog replaces the active dialog.
	EventKindDialogReplaced EventKind = "dialog_replaced"
	// EventKindGenerationStarted is sent immediately before a generator call.
	EventKindGenerationStarted EventKind = "generation_started"
	// EventKindGenerationChunk is sent for every streamed chunk.
	EventKindGenerationChunk EventKind = "generation_chunk"
	// EventKindGenerationCompleted is sent after a response enters run state.
	EventKindGenerationCompleted EventKind = "generation_completed"
	// EventKindToolStarted is sent immediately before tool handling continues.
	EventKindToolStarted EventKind = "tool_started"
	// EventKindToolCompleted is sent after a tool result enters the dialog.
	EventKindToolCompleted EventKind = "tool_completed"
	// EventKindRunCompleted is sent immediately before a successful return.
	EventKindRunCompleted EventKind = "run_completed"
	// EventKindRunFailed is the final best-effort event after a started run fails.
	EventKindRunFailed EventKind = "run_failed"
)

// RunStartedEvent records the validated inputs and fixed Agent configuration.
type RunStartedEvent struct {
	// Model is the configured model.
	Model string
	// Instructions are the configured system instructions.
	Instructions gai.Message
	// Dialog is the prior active dialog before Input is appended.
	Dialog gai.Dialog
	// Input is the new user message for this run.
	Input gai.Message
	// Tools are the fixed provider-visible tool definitions.
	Tools []gai.Tool
	// Options are the run's read-only generation options.
	Options gai.GenerationOptions
}

// Kind returns EventKindRunStarted.
func (RunStartedEvent) Kind() EventKind { return EventKindRunStarted }
func (RunStartedEvent) eventPayload()   {}

// DialogReplacedEvent records one lasting active-dialog replacement.
type DialogReplacedEvent struct {
	// Generation is the zero-based index of the possible next model call.
	Generation uint
	// Before is the active dialog before replacement.
	Before gai.Dialog
	// After is the adopted active dialog.
	After gai.Dialog
	// Usage is the standard usage reported by PrepareDialog.
	Usage gai.Metadata
}

// Kind returns EventKindDialogReplaced.
func (DialogReplacedEvent) Kind() EventKind { return EventKindDialogReplaced }
func (DialogReplacedEvent) eventPayload()   {}

// GenerationStartedEvent records the exact request about to be sent.
type GenerationStartedEvent struct {
	// Generation is the zero-based model-call index.
	Generation uint
	// Request is the request passed to the generator.
	Request gai.GenerationRequest
}

// Kind returns EventKindGenerationStarted.
func (GenerationStartedEvent) Kind() EventKind { return EventKindGenerationStarted }
func (GenerationStartedEvent) eventPayload()   {}

// GenerationChunkEvent records one raw streaming chunk.
type GenerationChunkEvent struct {
	// Generation is the zero-based model-call index.
	Generation uint
	// Chunk is the raw chunk returned by the streaming generator.
	Chunk gai.StreamChunk
}

// Kind returns EventKindGenerationChunk.
func (GenerationChunkEvent) Kind() EventKind { return EventKindGenerationChunk }
func (GenerationChunkEvent) eventPayload()   {}

// GenerationCompletedEvent records the response accepted into run state.
type GenerationCompletedEvent struct {
	// Generation is the zero-based model-call index.
	Generation uint
	// Response is the final response after AfterGeneration.
	Response gai.Response
}

// Kind returns EventKindGenerationCompleted.
func (GenerationCompletedEvent) Kind() EventKind { return EventKindGenerationCompleted }
func (GenerationCompletedEvent) eventPayload()   {}

// ToolStartedEvent records one requested call before the handler or error-result
// path begins.
type ToolStartedEvent struct {
	// Generation is the zero-based index of the model call that requested the tool.
	Generation uint
	// ToolIndex is the zero-based index within the response's tool calls.
	ToolIndex uint
	// Block is the original tool-call block.
	Block gai.Block
	// Call is nil when the block could not be decoded.
	Call *gai.ToolCallInput
	// Definition is nil when the requested tool is unknown.
	Definition *gai.Tool
	// WillExecute reports whether the handler will run after event delivery.
	WillExecute bool
	// Reason explains why the handler will not run.
	Reason string
}

// Kind returns EventKindToolStarted.
func (ToolStartedEvent) Kind() EventKind { return EventKindToolStarted }
func (ToolStartedEvent) eventPayload()   {}

// ToolCompletedEvent records the final tool result added to the dialog.
type ToolCompletedEvent struct {
	// Generation is the zero-based index of the model call that requested the tool.
	Generation uint
	// ToolIndex is the zero-based index within the response's tool calls.
	ToolIndex uint
	// Block is the original tool-call block.
	Block gai.Block
	// Call is nil when the block could not be decoded.
	Call *gai.ToolCallInput
	// Definition is nil when the requested tool is unknown.
	Definition *gai.Tool
	// Result is the final model-visible tool result.
	Result gai.Message
	// Executed reports whether the handler ran.
	Executed bool
}

// Kind returns EventKindToolCompleted.
func (ToolCompletedEvent) Kind() EventKind { return EventKindToolCompleted }
func (ToolCompletedEvent) eventPayload()   {}

// RunCompletedEvent records the result returned by a successful run.
type RunCompletedEvent struct {
	// Result is the complete successful result.
	Result RunResult
}

// Kind returns EventKindRunCompleted.
func (RunCompletedEvent) Kind() EventKind { return EventKindRunCompleted }
func (RunCompletedEvent) eventPayload()   {}

// RunFailedEvent records a started run's final state and error.
type RunFailedEvent struct {
	// Result contains the dialog and usage collected before failure.
	Result RunResult
	// Phase identifies the operation that failed.
	Phase RunPhase
	// Err is the run error.
	Err error
}

// Kind returns EventKindRunFailed.
func (RunFailedEvent) Kind() EventKind { return EventKindRunFailed }
func (RunFailedEvent) eventPayload()   {}

// RunPhase identifies the operation in which a started run failed.
type RunPhase string

const (
	// RunPhasePrepareDialogHook identifies PrepareDialog failures.
	RunPhasePrepareDialogHook RunPhase = "prepare_dialog_hook"
	// RunPhaseBeforeGenerationHook identifies BeforeGeneration failures.
	RunPhaseBeforeGenerationHook RunPhase = "before_generation_hook"
	// RunPhaseGeneration identifies generator or stream failures.
	RunPhaseGeneration RunPhase = "generation"
	// RunPhaseAfterGenerationHook identifies AfterGeneration failures.
	RunPhaseAfterGenerationHook RunPhase = "after_generation_hook"
	// RunPhaseValidation identifies invalid generator responses.
	RunPhaseValidation RunPhase = "validation"
	// RunPhaseBeforeToolHook identifies BeforeTool failures.
	RunPhaseBeforeToolHook RunPhase = "before_tool_hook"
	// RunPhaseTool identifies tool-handler failures.
	RunPhaseTool RunPhase = "tool"
	// RunPhaseAfterToolHook identifies AfterTool failures.
	RunPhaseAfterToolHook RunPhase = "after_tool_hook"
	// RunPhaseObserver identifies observer delivery failures.
	RunPhaseObserver RunPhase = "observer"
	// RunPhaseCancellation identifies context cancellation.
	RunPhaseCancellation RunPhase = "cancellation"
	// RunPhaseInternal identifies unexpected package failures.
	RunPhaseInternal RunPhase = "internal"
)
