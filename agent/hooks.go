package agent

import (
	"context"

	"github.com/spachava753/gai"
)

// RunStatus reports work that has already entered the active dialog.
type RunStatus struct {
	// AcceptedGenerations is the number of accepted model responses.
	AcceptedGenerations uint
	// AcceptedToolResults counts handler results added to the dialog, excluding
	// synthetic errors for unknown, invalid, or rejected calls.
	AcceptedToolResults uint
	// Usage contains the standard usage accumulated so far.
	Usage gai.Metadata
}

// BeforeGenerationHook may replace a generation request or stop the run before
// the provider call. Use it for dialog compaction, retrieval, inline [gai.System]
// instructions, or generation settings. A validated replacement's dialog becomes
// the conversation the loop continues from; see [BeforeGenerationDecision.Request].
// The application decides what to store outside the run.
//
// Request fields are borrowed read-only values. Copy the parts you change. A
// returned replacement belongs to the loop and must not be changed until Run
// returns.
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
	// Request replaces the request sent to the generator when non-nil and valid.
	// Replacements are complete requests, not patches: copy the base request to
	// retain its fields. Dialog must be nonempty with every tool call resolved.
	// Its Dialog becomes the active conversation before generation; accepted
	// responses and tool results are appended there. Later base requests and
	// [RunResult.Dialog] use that conversation, even if generation fails.
	// Other base fields are rebuilt from [Config] and [RunRequest.Options] for
	// each generation. Tools must still match the Agent's fixed tool definitions.
	// The application decides which state to store or reuse after Run.
	// Nil keeps the base request. Request is ignored when StopReason is nonempty.
	Request *gai.GenerationRequest
	// StopReason asks the loop to return normally without another provider call.
	StopReason StopReason
}

// AfterGenerationHook may change a complete response before it enters the
// active dialog. Request fields are borrowed read-only values. The returned
// replacement belongs to the loop and must preserve finish reason and usage.
type AfterGenerationHook interface {
	// AfterGeneration returns a replacement, or nil to keep the original response.
	AfterGeneration(context.Context, AfterGenerationRequest) (*gai.Response, error)
}

// AfterGenerationFunc adapts a function to AfterGenerationHook.
type AfterGenerationFunc func(context.Context, AfterGenerationRequest) (*gai.Response, error)

// AfterGeneration calls f.
func (f AfterGenerationFunc) AfterGeneration(ctx context.Context, request AfterGenerationRequest) (*gai.Response, error) {
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
	// Result replaces the handler result when non-nil. Nil keeps that result.
	Result *gai.Message
	// StopAfterBatch asks the loop to stop after every call has a result.
	StopAfterBatch StopReason
}

// GenerationChunkHook receives provisional streaming output synchronously.
// Event data is borrowed and read-only, valid only during the callback. Copy or
// serialize anything retained. Returning an error stops streaming and the run;
// incomplete output never enters the active dialog. Calls are serial within a
// run, but concurrent runs may invoke the same hook concurrently.
type GenerationChunkHook interface {
	GenerationChunk(context.Context, GenerationChunkEvent) error
}

// GenerationChunkFunc adapts a function to GenerationChunkHook.
type GenerationChunkFunc func(context.Context, GenerationChunkEvent) error

// GenerationChunk calls f.
func (f GenerationChunkFunc) GenerationChunk(ctx context.Context, event GenerationChunkEvent) error {
	return f(ctx, event)
}

// GenerationChunkEvent contains one raw chunk, including possible provider errors.
// It is not an accepted response or a durable completion record.
type GenerationChunkEvent struct {
	// Generation is the zero-based model-call index within this Run invocation.
	Generation uint
	Chunk      gai.StreamChunk
}
