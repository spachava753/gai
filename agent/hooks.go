package agent

import (
	"context"

	"github.com/spachava753/gai"
)

// RunStatus reports work that has already entered the active dialog.
type RunStatus struct {
	// GenerationCalls is the number of accepted model responses.
	GenerationCalls uint
	// ToolExecutions is the number of handler results added to the dialog.
	ToolExecutions uint
	// Usage contains the standard usage accumulated so far.
	Usage gai.Metadata
}

// PrepareDialogHook may replace the active dialog before a generation request
// is built. Request fields are borrowed read-only values. A returned replacement
// belongs to the loop and must not be changed after PrepareDialog returns.
type PrepareDialogHook interface {
	// PrepareDialog returns a possible lasting dialog replacement.
	PrepareDialog(context.Context, PrepareDialogRequest) (PrepareDialogDecision, error)
}

// PrepareDialogFunc adapts a function to PrepareDialogHook.
type PrepareDialogFunc func(context.Context, PrepareDialogRequest) (PrepareDialogDecision, error)

// PrepareDialog calls f.
func (f PrepareDialogFunc) PrepareDialog(ctx context.Context, request PrepareDialogRequest) (PrepareDialogDecision, error) {
	return f(ctx, request)
}

// PrepareDialogRequest describes the base request before one possible model
// call.
type PrepareDialogRequest struct {
	// Generation is the zero-based index of the possible next model call.
	Generation uint
	// Request contains the configured model and instructions, active dialog,
	// fixed tools, and run options before any BeforeGeneration changes.
	Request gai.GenerationRequest
	// Status reports work accepted before this hook call.
	Status RunStatus
}

// PrepareDialogDecision describes a possible lasting dialog replacement.
type PrepareDialogDecision struct {
	// Dialog replaces the active dialog when non-nil. A non-nil empty dialog is
	// invalid.
	Dialog gai.Dialog
	// Usage contains standard usage incurred while producing Dialog. It must be
	// empty when Dialog is nil.
	Usage gai.Metadata
}

// BeforeGenerationHook may change one generation request or stop the run
// normally before the provider call. Request fields are borrowed read-only
// values. A returned request belongs to the loop until its generator call ends.
type BeforeGenerationHook interface {
	// BeforeGeneration returns the request for one provider call or a stop reason.
	BeforeGeneration(context.Context, BeforeGenerationRequest) (BeforeGenerationDecision, error)
}

// BeforeGenerationFunc adapts a function to BeforeGenerationHook.
type BeforeGenerationFunc func(context.Context, BeforeGenerationRequest) (BeforeGenerationDecision, error)

// BeforeGeneration calls f.
func (f BeforeGenerationFunc) BeforeGeneration(ctx context.Context, request BeforeGenerationRequest) (BeforeGenerationDecision, error) {
	return f(ctx, request)
}

// BeforeGenerationRequest describes one generation request before it is sent.
type BeforeGenerationRequest struct {
	// Generation is the zero-based model-call index.
	Generation uint
	// Request is the base request assembled from Agent and run state.
	Request gai.GenerationRequest
	// Status reports work accepted before this hook call.
	Status RunStatus
}

// BeforeGenerationDecision supplies one request or asks the loop to stop.
type BeforeGenerationDecision struct {
	// Request is sent once when StopReason is empty.
	Request gai.GenerationRequest
	// StopReason asks the loop to return normally without another provider call.
	StopReason StopReason
}

// AfterGenerationHook may change a complete response before it enters the
// active dialog. Request fields are borrowed read-only values. The returned
// response belongs to the loop and must preserve finish reason and usage.
type AfterGenerationHook interface {
	// AfterGeneration returns the response that the loop should accept.
	AfterGeneration(context.Context, AfterGenerationRequest) (gai.Response, error)
}

// AfterGenerationFunc adapts a function to AfterGenerationHook.
type AfterGenerationFunc func(context.Context, AfterGenerationRequest) (gai.Response, error)

// AfterGeneration calls f.
func (f AfterGenerationFunc) AfterGeneration(ctx context.Context, request AfterGenerationRequest) (gai.Response, error) {
	return f(ctx, request)
}

// AfterGenerationRequest describes a complete provider response before it is
// added to the dialog.
type AfterGenerationRequest struct {
	// Generation is the zero-based model-call index.
	Generation uint
	// Request is the exact request sent to the generator.
	Request gai.GenerationRequest
	// Response is the complete provider response.
	Response gai.Response
	// Status reports work accepted before this response.
	Status RunStatus
}

// BeforeToolHook may replace validated arguments, reject a tool call, or stop
// normally after the current tool batch. Request fields are borrowed read-only
// values. Non-nil replacement parameters belong to the loop and must not be
// changed after BeforeTool returns.
type BeforeToolHook interface {
	// BeforeTool returns the decision for one valid, known tool call.
	BeforeTool(context.Context, BeforeToolRequest) (BeforeToolDecision, error)
}

// BeforeToolFunc adapts a function to BeforeToolHook.
type BeforeToolFunc func(context.Context, BeforeToolRequest) (BeforeToolDecision, error)

// BeforeTool calls f.
func (f BeforeToolFunc) BeforeTool(ctx context.Context, request BeforeToolRequest) (BeforeToolDecision, error) {
	return f(ctx, request)
}

// BeforeToolRequest describes a known tool call with validated arguments.
type BeforeToolRequest struct {
	// Generation is the zero-based index of the model call that requested the tool.
	Generation uint
	// ToolIndex is the zero-based index within the response's tool calls.
	ToolIndex uint
	// Block is the original provider-produced tool-call block.
	Block gai.Block
	// Call contains the decoded and validated arguments.
	Call gai.ToolCallInput
	// Definition is the Agent's fixed tool definition.
	Definition gai.Tool
	// Status reports work accepted before this tool call.
	Status RunStatus
}

// BeforeToolDecision controls one tool call before handler execution.
type BeforeToolDecision struct {
	// Parameters replaces the original argument map when non-nil.
	Parameters map[string]any
	// Reject prevents handler execution and creates a failed tool result.
	Reject bool
	// Reason is shown to the model when Reject is true.
	Reason string
	// StopAfterBatch asks the loop to stop after every call has a result.
	StopAfterBatch StopReason
}

// AfterToolHook may replace a handler result or stop normally after the current
// tool batch. Request fields are borrowed read-only values. The returned result
// belongs to the loop, which assigns the original call ID to its blocks.
type AfterToolHook interface {
	// AfterTool returns the final result for one successfully executed handler.
	AfterTool(context.Context, AfterToolRequest) (AfterToolDecision, error)
}

// AfterToolFunc adapts a function to AfterToolHook.
type AfterToolFunc func(context.Context, AfterToolRequest) (AfterToolDecision, error)

// AfterTool calls f.
func (f AfterToolFunc) AfterTool(ctx context.Context, request AfterToolRequest) (AfterToolDecision, error) {
	return f(ctx, request)
}

// AfterToolRequest describes a model-visible handler result before it enters
// the dialog.
type AfterToolRequest struct {
	// Generation is the zero-based index of the model call that requested the tool.
	Generation uint
	// ToolIndex is the zero-based index within the response's tool calls.
	ToolIndex uint
	// Block is the original provider-produced tool-call block.
	Block gai.Block
	// Call contains the arguments passed to the handler.
	Call gai.ToolCallInput
	// Definition is the Agent's fixed tool definition.
	Definition gai.Tool
	// Result is the validated result returned by the handler.
	Result gai.Message
	// Status reports work accepted before this tool result.
	Status RunStatus
}

// AfterToolDecision supplies the final tool result and an optional batch stop.
type AfterToolDecision struct {
	// Result is added to the active dialog after validation.
	Result gai.Message
	// StopAfterBatch asks the loop to stop after every call has a result.
	StopAfterBatch StopReason
}
