package agent

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/spachava753/gai"
)

const defaultToolRejectionReason = "tool call rejected"

type chunkHookError struct {
	err error
}

func (e *chunkHookError) Error() string { return e.err.Error() }

type runState struct {
	agent           *Agent
	ctx             context.Context
	instructions    gai.Message
	toolDefinitions []gai.Tool
	dialog          gai.Dialog
	usage           gai.Metadata
	status          RunStatus
	seenCallIDs     map[string]struct{}
}

// Run executes model and tool rounds until the model or a hook stops, the
// context is canceled, or an operation fails. It continues generation from the
// supplied dialog without repeating historical tool calls or restoring prior
// hook state. Failures return a *RunError; invalid inputs return a zero result.
func (a *Agent) Run(ctx context.Context, request RunRequest) (RunResult, error) {
	if a == nil {
		return RunResult{}, &RunError{Phase: RunPhaseValidation, Err: errors.New("agent is nil")}
	}
	if ctx == nil {
		return RunResult{}, &RunError{Phase: RunPhaseValidation, Err: errors.New("context is nil")}
	}
	seen, err := validateReadyDialog(request.Dialog)
	if err != nil {
		return RunResult{}, &RunError{Phase: RunPhaseValidation, Err: gai.InvalidParameterErr{Parameter: "dialog", Reason: err.Error()}}
	}
	if err := validateOptions(request.Options); err != nil {
		return RunResult{}, &RunError{Phase: RunPhaseValidation, Err: err}
	}

	state := &runState{
		agent:           a,
		ctx:             ctx,
		instructions:    a.instructions,
		toolDefinitions: a.toolDefinitions,
		dialog:          request.Dialog,
		seenCallIDs:     seen,
	}

	for {
		if err := ctx.Err(); err != nil {
			return state.result(), &RunError{Phase: RunPhaseCancellation, Err: err}
		}
		generation := state.status.AcceptedGenerations
		if err := state.prepareDialogForGeneration(generation, request.Options); err != nil {
			return state.result(), err
		}

		baseRequest := gai.GenerationRequest{
			Model:        a.model,
			Instructions: state.instructions,
			Dialog:       state.dialog,
			Tools:        state.toolDefinitions,
			Options:      request.Options,
		}
		generationRequest, stopReason, err := state.beforeGenerationRequest(generation, baseRequest)
		if err != nil {
			return state.result(), err
		}
		if stopReason != "" {
			return state.complete(stopReason, gai.Unknown)
		}

		response, err := state.generate(generation, generationRequest)
		if err != nil {
			return state.result(), err
		}
		response, err = state.afterGenerationResponse(generation, generationRequest, response)
		if err != nil {
			return state.result(), err
		}
		if err := state.acceptGeneration(response); err != nil {
			return state.result(), err
		}

		if response.FinishReason != gai.ToolUse {
			return state.complete(StopReasonModel, response.FinishReason)
		}
		stopReason, err = state.handleToolBatch(generation, response.Candidates[0])
		if err != nil {
			return state.result(), err
		}
		if stopReason != "" {
			return state.complete(stopReason, gai.Unknown)
		}
	}
}

func (s *runState) prepareDialogForGeneration(generation uint, options gai.GenerationOptions) error {
	// prepareDialogForGeneration applies only explicit, validated replacements.
	if s.agent.prepareDialog == nil {
		return nil
	}
	if err := s.ctx.Err(); err != nil {
		return &RunError{Phase: RunPhaseCancellation, Err: err}
	}
	decision, err := s.agent.prepareDialog.PrepareDialog(s.ctx, PrepareDialogRequest{
		Generation: generation,
		Request: gai.GenerationRequest{
			Model:        s.agent.model,
			Instructions: s.instructions,
			Dialog:       s.dialog,
			Tools:        s.toolDefinitions,
			Options:      options,
		},
		Status: s.status,
	})
	if err != nil {
		return &RunError{Phase: phaseForContext(s.ctx, RunPhasePrepareDialogHook, err), Err: err}
	}
	if decision.Dialog == nil {
		if len(decision.Usage) != 0 {
			return &RunError{Phase: RunPhasePrepareDialogHook, Err: errors.New("PrepareDialog usage requires a dialog replacement")}
		}
		return nil
	}
	if len(decision.Dialog) == 0 {
		return &RunError{Phase: RunPhasePrepareDialogHook, Err: fmt.Errorf("invalid PrepareDialog replacement: %w", gai.ErrEmptyDialog)}
	}
	replacementIDs, err := validateReadyDialog(decision.Dialog)
	if err != nil {
		return &RunError{Phase: RunPhasePrepareDialogHook, Err: fmt.Errorf("invalid PrepareDialog replacement: %w", err)}
	}
	usage, err := addStandardUsage(s.usage, decision.Usage)
	if err != nil {
		return &RunError{Phase: RunPhasePrepareDialogHook, Err: fmt.Errorf("invalid PrepareDialog usage: %w", err)}
	}
	for id := range replacementIDs {
		s.seenCallIDs[id] = struct{}{}
	}
	s.dialog = decision.Dialog
	s.usage = usage
	s.status.Usage = usage
	return nil
}

func (s *runState) beforeGenerationRequest(generation uint, base gai.GenerationRequest) (gai.GenerationRequest, StopReason, error) {
	// beforeGenerationRequest applies call-only changes without replacing run state.
	if s.agent.beforeGeneration == nil {
		return base, "", nil
	}
	if err := s.ctx.Err(); err != nil {
		return gai.GenerationRequest{}, "", &RunError{Phase: RunPhaseCancellation, Err: err}
	}
	decision, err := s.agent.beforeGeneration.BeforeGeneration(s.ctx, BeforeGenerationRequest{
		Generation: generation,
		Request:    base,
		Status:     s.status,
	})
	if err != nil {
		return gai.GenerationRequest{}, "", &RunError{Phase: phaseForContext(s.ctx, RunPhaseBeforeGenerationHook, err), Err: err}
	}
	if decision.StopReason != "" {
		if err := validateStopReason(decision.StopReason); err != nil {
			return gai.GenerationRequest{}, "", &RunError{Phase: RunPhaseBeforeGenerationHook, Err: err}
		}
		return gai.GenerationRequest{}, decision.StopReason, nil
	}
	if decision.Request == nil {
		return base, "", nil
	}
	if err := validateGenerationRequest(*decision.Request, s.agent.toolDefinitions); err != nil {
		return gai.GenerationRequest{}, "", &RunError{Phase: RunPhaseBeforeGenerationHook, Err: fmt.Errorf("invalid BeforeGeneration request: %w", err)}
	}
	return *decision.Request, "", nil
}

func (s *runState) generate(generation uint, request gai.GenerationRequest) (gai.Response, error) {
	// generate prefers streaming and forwards every raw chunk before assembly.
	if err := s.ctx.Err(); err != nil {
		return gai.Response{}, &RunError{Phase: RunPhaseCancellation, Err: err}
	}
	var response gai.Response
	var err error
	if s.agent.streaming != nil {
		adapter := gai.StreamingAdapter{S: hookedStreamingGenerator{
			source:     s.agent.streaming,
			state:      s,
			generation: generation,
		}}
		response, err = adapter.Generate(s.ctx, request)
	} else {
		response, err = s.agent.generator.Generate(s.ctx, request)
	}
	if err != nil {
		var eventErr *chunkHookError
		if errors.As(err, &eventErr) {
			return gai.Response{}, &RunError{Phase: phaseForContext(s.ctx, RunPhaseGenerationChunkHook, eventErr.err), Err: eventErr.err}
		}
		return gai.Response{}, &RunError{Phase: phaseForContext(s.ctx, RunPhaseGeneration, err), Err: err}
	}
	return response, nil
}

func (s *runState) afterGenerationResponse(generation uint, request gai.GenerationRequest, response gai.Response) (gai.Response, error) {
	// afterGenerationResponse validates provider and hook responses before acceptance.
	if err := validateResponse(response, s.seenCallIDs); err != nil {
		return gai.Response{}, &RunError{Phase: RunPhaseValidation, Err: fmt.Errorf("invalid generator response: %w", err)}
	}
	if s.agent.afterGeneration == nil {
		return response, nil
	}
	if err := s.ctx.Err(); err != nil {
		return gai.Response{}, &RunError{Phase: RunPhaseCancellation, Err: err}
	}
	changed, err := s.agent.afterGeneration.AfterGeneration(s.ctx, AfterGenerationRequest{
		Generation: generation,
		Request:    request,
		Response:   response,
		Status:     s.status,
	})
	if err != nil {
		return gai.Response{}, &RunError{Phase: phaseForContext(s.ctx, RunPhaseAfterGenerationHook, err), Err: err}
	}
	if changed == nil {
		return response, nil
	}
	if err := validateAfterGenerationResponse(response, *changed, s.seenCallIDs); err != nil {
		return gai.Response{}, &RunError{Phase: RunPhaseAfterGenerationHook, Err: fmt.Errorf("invalid AfterGeneration response: %w", err)}
	}
	return *changed, nil
}

func (s *runState) acceptGeneration(response gai.Response) error {
	// acceptGeneration adopts a validated response and its usage.
	usage, err := addStandardUsage(s.usage, response.UsageMetadata)
	if err != nil {
		return &RunError{Phase: RunPhaseValidation, Err: fmt.Errorf("add response usage: %w", err)}
	}
	s.dialog = append(s.dialog, response.Candidates[0])
	s.usage = usage
	s.status.AcceptedGenerations++
	s.status.Usage = usage
	for _, block := range response.Candidates[0].Blocks {
		if block.BlockType == gai.ToolCall {
			s.seenCallIDs[block.ID] = struct{}{}
		}
	}
	return nil
}

// handleToolBatch classifies calls, applies hooks, executes handlers, and appends ordered results.
func (s *runState) handleToolBatch(generation uint, message gai.Message) (StopReason, error) {
	var batchStop StopReason
	toolIndex := uint(0)
	for _, rawBlock := range message.Blocks {
		if rawBlock.BlockType != gai.ToolCall {
			continue
		}
		if err := s.ctx.Err(); err != nil {
			return "", &RunError{Phase: RunPhaseCancellation, Err: err}
		}

		block := rawBlock
		call, tool, reason := s.classifyToolCall(block)
		willExecute := reason == ""
		if willExecute && s.agent.beforeTool != nil {
			if err := s.ctx.Err(); err != nil {
				return "", &RunError{Phase: RunPhaseCancellation, Err: err}
			}
			decision, err := s.agent.beforeTool.BeforeTool(s.ctx, BeforeToolRequest{
				Generation: generation,
				ToolIndex:  toolIndex,
				Block:      block,
				Call:       *call,
				Definition: tool.definition,
				Status:     s.status,
			})
			if err != nil {
				return "", &RunError{Phase: phaseForContext(s.ctx, RunPhaseBeforeToolHook, err), Err: err}
			}
			if decision.Reject && decision.Parameters != nil {
				return "", &RunError{Phase: RunPhaseBeforeToolHook, Err: errors.New("BeforeTool cannot reject and replace parameters together")}
			}
			if err := validateStopReason(decision.StopAfterBatch); err != nil {
				return "", &RunError{Phase: RunPhaseBeforeToolHook, Err: err}
			}
			if batchStop == "" {
				batchStop = decision.StopAfterBatch
			}
			if decision.Parameters != nil {
				if err := validateToolArguments(*tool, decision.Parameters); err != nil {
					return "", &RunError{Phase: RunPhaseBeforeToolHook, Err: fmt.Errorf("invalid BeforeTool parameters: %w", err)}
				}
				call.Parameters = decision.Parameters
			}
			if decision.Reject {
				willExecute = false
				reason = decision.Reason
				if reason == "" {
					reason = defaultToolRejectionReason
				}
			}
		}

		result, executed, stopReason, err := s.executeOrBuildToolResult(generation, toolIndex, block, call, tool, willExecute, reason)
		if err != nil {
			return "", err
		}
		if batchStop == "" {
			batchStop = stopReason
		}
		for i := range result.Blocks {
			result.Blocks[i].ID = block.ID
		}
		s.dialog = append(s.dialog, result)
		if executed {
			s.status.AcceptedToolResults++
		}
		toolIndex++
	}
	return batchStop, nil
}

func (s *runState) classifyToolCall(block gai.Block) (*gai.ToolCallInput, *executableTool, string) {
	call, err := decodeToolCall(block)
	if err != nil {
		return nil, nil, fmt.Sprintf("invalid tool call: %v", err)
	}
	index, ok := s.agent.toolIndexes[call.Name]
	if !ok {
		return &call, nil, fmt.Sprintf("unknown tool %q", call.Name)
	}
	tool := &s.agent.tools[index]
	if err := validateToolArguments(*tool, call.Parameters); err != nil {
		return &call, tool, err.Error()
	}
	return &call, tool, ""
}

// executeOrBuildToolResult synthesizes recoverable failures or executes a handler
// and its optional after-hook. It checks cancellation at each invocation boundary
// and validates outputs before returning a result for acceptance.
func (s *runState) executeOrBuildToolResult(
	generation uint,
	toolIndex uint,
	block gai.Block,
	call *gai.ToolCallInput,
	tool *executableTool,
	willExecute bool,
	reason string,
) (gai.Message, bool, StopReason, error) {
	if !willExecute {
		result := gai.ToolResultMessage(block.ID, gai.TextBlock(reason))
		result.ToolResultError = true
		return result, false, "", nil
	}
	if err := s.ctx.Err(); err != nil {
		return gai.Message{}, false, "", &RunError{Phase: RunPhaseCancellation, Err: err}
	}
	result, err := tool.handler.Execute(s.ctx, ToolRequest{
		Block: block,
		Call:  *call,
	})
	if err != nil {
		return gai.Message{}, false, "", &RunError{Phase: phaseForContext(s.ctx, RunPhaseTool, err), Err: err}
	}
	if err := validateToolResult(result); err != nil {
		return gai.Message{}, false, "", &RunError{Phase: RunPhaseTool, Err: fmt.Errorf("invalid handler result: %w", err)}
	}
	if s.agent.afterTool == nil {
		return result, true, "", nil
	}
	if err := s.ctx.Err(); err != nil {
		return gai.Message{}, false, "", &RunError{Phase: RunPhaseCancellation, Err: err}
	}
	decision, err := s.agent.afterTool.AfterTool(s.ctx, AfterToolRequest{
		Generation: generation,
		ToolIndex:  toolIndex,
		Block:      block,
		Call:       *call,
		Definition: tool.definition,
		Result:     result,
		Status:     s.status,
	})
	if err != nil {
		return gai.Message{}, false, "", &RunError{Phase: phaseForContext(s.ctx, RunPhaseAfterToolHook, err), Err: err}
	}
	if err := validateStopReason(decision.StopAfterBatch); err != nil {
		return gai.Message{}, false, "", &RunError{Phase: RunPhaseAfterToolHook, Err: err}
	}
	if decision.Result == nil {
		return result, true, decision.StopAfterBatch, nil
	}
	if err := validateToolResult(*decision.Result); err != nil {
		return gai.Message{}, false, "", &RunError{Phase: RunPhaseAfterToolHook, Err: fmt.Errorf("invalid AfterTool result: %w", err)}
	}
	return *decision.Result, true, decision.StopAfterBatch, nil
}

func (s *runState) complete(reason StopReason, finishReason gai.FinishReason) (RunResult, error) {
	result := s.result()
	result.StopReason = reason
	result.ModelFinishReason = finishReason
	return result, nil
}

func (s *runState) result() RunResult {
	return RunResult{
		Dialog:            s.dialog,
		ModelFinishReason: gai.Unknown,
		Usage:             s.usage,
	}
}

func phaseForContext(ctx context.Context, fallback RunPhase, err error) RunPhase {
	if ctx.Err() != nil && errors.Is(err, ctx.Err()) {
		return RunPhaseCancellation
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return RunPhaseCancellation
	}
	return fallback
}

type hookedStreamingGenerator struct {
	source     gai.StreamingGenerator
	state      *runState
	generation uint
}

func (g hookedStreamingGenerator) Stream(ctx context.Context, request gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
	return func(yield func(gai.StreamChunk) bool) {
		for chunk := range g.source.Stream(ctx, request) {
			if hook := g.state.agent.generationChunk; hook != nil {
				if err := ctx.Err(); err != nil {
					yield(gai.StreamChunk{Err: errors.Join(err, chunk.Err)})
					return
				}
				if err := hook.GenerationChunk(ctx, GenerationChunkEvent{Generation: g.generation, Chunk: chunk}); err != nil {
					if chunk.Err != nil {
						err = errors.Join(chunk.Err, err)
					}
					yield(gai.StreamChunk{Err: &chunkHookError{err: err}})
					return
				}
			}
			if !yield(chunk) {
				return
			}
		}
	}
}
