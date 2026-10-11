package agent

import "fmt"

// RunError identifies the failing operation. Run returns the dialog and usage
// accepted before that failure alongside this error. Phase describes the local
// failure, not whether an external effect occurred or is safe to retry.
// Constructor errors are returned directly, without RunError.
type RunError struct {
	// Phase identifies the outer operation that failed.
	Phase RunPhase
	// Err is the underlying error, available through Unwrap.
	Err error
}

func (e *RunError) Error() string { return fmt.Sprintf("agent %s: %v", e.Phase, e.Err) }

// Unwrap preserves errors.Is and errors.As for the underlying failure.
func (e *RunError) Unwrap() error { return e.Err }

// RunPhase identifies the operation in which a run failed, including input validation.
type RunPhase string

const (
	// RunPhaseBeforeGenerationHook identifies BeforeGeneration failures.
	RunPhaseBeforeGenerationHook RunPhase = "before_generation_hook"
	// RunPhaseGeneration identifies generator or stream failures.
	RunPhaseGeneration RunPhase = "generation"
	// RunPhaseAfterGenerationHook identifies AfterGeneration failures.
	RunPhaseAfterGenerationHook RunPhase = "after_generation_hook"
	// RunPhaseValidation identifies invalid run inputs or generator responses.
	RunPhaseValidation RunPhase = "validation"
	// RunPhaseBeforeToolHook identifies BeforeTool failures.
	RunPhaseBeforeToolHook RunPhase = "before_tool_hook"
	// RunPhaseTool identifies tool-handler failures.
	RunPhaseTool RunPhase = "tool"
	// RunPhaseAfterToolHook identifies AfterTool failures.
	RunPhaseAfterToolHook RunPhase = "after_tool_hook"
	// RunPhaseGenerationChunkHook identifies GenerationChunk hook failures.
	RunPhaseGenerationChunkHook RunPhase = "generation_chunk_hook"
	// RunPhaseCancellation identifies context cancellation.
	RunPhaseCancellation RunPhase = "cancellation"
)
