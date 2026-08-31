package agent

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/spachava753/gai"
)

const defaultToolRejectionReason = "tool call rejected"

type phaseError struct {
	phase RunPhase
	err   error
}

func (e *phaseError) Error() string { return e.err.Error() }

type observerDeliveryError struct {
	err error
}

func (e *observerDeliveryError) Error() string { return e.err.Error() }

type runState struct {
	agent           *Agent
	ctx             context.Context
	observer        Observer
	sequence        uint64
	instructions    gai.Message
	toolDefinitions []gai.Tool
	dialog          gai.Dialog
	usage           gai.Metadata
	status          RunStatus
	seenCallIDs     map[string]struct{}
}

// Run executes model and tool rounds until the model or a hook stops, the
// context is canceled, or an operation fails.
func (a *Agent) Run(ctx context.Context, request RunRequest, observer Observer) (RunResult, error) {
	// Run validates all caller input before sending RunStartedEvent.
	if a == nil {
		return RunResult{}, errors.New("agent is nil")
	}
	if ctx == nil {
		return RunResult{}, errors.New("context is nil")
	}
	if request.Input.Role != gai.User {
		return RunResult{}, gai.InvalidParameterErr{
			Parameter: "input",
			Reason:    fmt.Sprintf("must use the user role, got %s", request.Input.Role),
		}
	}
	if err := validateMessage(request.Input); err != nil {
		return RunResult{}, gai.InvalidParameterErr{Parameter: "input", Reason: err.Error()}
	}
	if err := validateDialog(request.Dialog); err != nil {
		return RunResult{}, gai.InvalidParameterErr{Parameter: "dialog", Reason: err.Error()}
	}
	if err := validateOptions(request.Options); err != nil {
		return RunResult{}, err
	}

	activeDialog := append(request.Dialog, request.Input)
	if isNilInterface(observer) {
		observer = nil
	}

	state := &runState{
		agent:           a,
		ctx:             ctx,
		observer:        observer,
		instructions:    a.instructions,
		toolDefinitions: a.toolDefinitions,
		dialog:          activeDialog,
		seenCallIDs:     make(map[string]struct{}),
	}
	if err := state.emit(RunStartedEvent{
		Model:        a.model,
		Instructions: state.instructions,
		Dialog:       request.Dialog,
		Input:        request.Input,
		Tools:        state.toolDefinitions,
		Options:      request.Options,
	}); err != nil {
		return state.fail(RunPhaseObserver, err)
	}

	for {
		if err := state.contextError(); err != nil {
			return state.fail(RunPhaseCancellation, err)
		}
		generation := state.status.GenerationCalls
		if err := state.prepareDialogForGeneration(generation, request.Options); err != nil {
			return state.failFrom(err)
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
			return state.failFrom(err)
		}
		if stopReason != "" {
			return state.complete(stopReason, gai.Unknown)
		}

		if err := state.emit(GenerationStartedEvent{Generation: generation, Request: generationRequest}); err != nil {
			return state.fail(RunPhaseObserver, err)
		}
		response, err := state.generate(generation, generationRequest)
		if err != nil {
			return state.failFrom(err)
		}
		response, err = state.afterGenerationResponse(generation, generationRequest, response)
		if err != nil {
			return state.failFrom(err)
		}
		if err := state.acceptGeneration(generation, response); err != nil {
			return state.failFrom(err)
		}

		if response.FinishReason != gai.ToolUse {
			return state.complete(StopReasonModel, response.FinishReason)
		}
		stopReason, err = state.handleToolBatch(generation, response.Candidates[0])
		if err != nil {
			return state.failFrom(err)
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
	if err := s.contextError(); err != nil {
		return &phaseError{phase: RunPhaseCancellation, err: err}
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
		return &phaseError{phase: phaseForContext(s.ctx, RunPhasePrepareDialogHook, err), err: err}
	}
	if decision.Dialog == nil {
		if len(decision.Usage) != 0 {
			return &phaseError{phase: RunPhasePrepareDialogHook, err: errors.New("PrepareDialog usage requires a dialog replacement")}
		}
		return nil
	}
	if len(decision.Dialog) == 0 {
		return &phaseError{phase: RunPhasePrepareDialogHook, err: fmt.Errorf("invalid PrepareDialog replacement: %w", gai.ErrEmptyDialog)}
	}
	if err := validateDialog(decision.Dialog); err != nil {
		return &phaseError{phase: RunPhasePrepareDialogHook, err: fmt.Errorf("invalid PrepareDialog replacement: %w", err)}
	}
	usage, err := addStandardUsage(s.usage, decision.Usage)
	if err != nil {
		return &phaseError{phase: RunPhasePrepareDialogHook, err: fmt.Errorf("invalid PrepareDialog usage: %w", err)}
	}
	before := s.dialog
	s.dialog = decision.Dialog
	s.usage = usage
	s.status.Usage = usage
	if err := s.emit(DialogReplacedEvent{
		Generation: generation,
		Before:     before,
		After:      s.dialog,
		Usage:      decision.Usage,
	}); err != nil {
		return &phaseError{phase: RunPhaseObserver, err: err}
	}
	return nil
}

func (s *runState) beforeGenerationRequest(generation uint, base gai.GenerationRequest) (gai.GenerationRequest, StopReason, error) {
	// beforeGenerationRequest applies call-only changes without replacing run state.
	if s.agent.beforeGeneration == nil {
		return base, "", nil
	}
	if err := s.contextError(); err != nil {
		return gai.GenerationRequest{}, "", &phaseError{phase: RunPhaseCancellation, err: err}
	}
	decision, err := s.agent.beforeGeneration.BeforeGeneration(s.ctx, BeforeGenerationRequest{
		Generation: generation,
		Request:    base,
		Status:     s.status,
	})
	if err != nil {
		return gai.GenerationRequest{}, "", &phaseError{phase: phaseForContext(s.ctx, RunPhaseBeforeGenerationHook, err), err: err}
	}
	if decision.StopReason != "" {
		if err := validateStopReason(decision.StopReason); err != nil {
			return gai.GenerationRequest{}, "", &phaseError{phase: RunPhaseBeforeGenerationHook, err: err}
		}
		return gai.GenerationRequest{}, decision.StopReason, nil
	}
	if err := validateGenerationRequest(decision.Request, s.agent.toolDefinitions); err != nil {
		return gai.GenerationRequest{}, "", &phaseError{phase: RunPhaseBeforeGenerationHook, err: fmt.Errorf("invalid BeforeGeneration request: %w", err)}
	}
	return decision.Request, "", nil
}

func (s *runState) generate(generation uint, request gai.GenerationRequest) (gai.Response, error) {
	// generate prefers streaming and forwards every raw chunk before assembly.
	if err := s.contextError(); err != nil {
		return gai.Response{}, &phaseError{phase: RunPhaseCancellation, err: err}
	}
	var response gai.Response
	var err error
	if s.agent.streaming != nil {
		adapter := gai.StreamingAdapter{S: observedStreamingGenerator{
			source:     s.agent.streaming,
			state:      s,
			generation: generation,
		}}
		response, err = adapter.Generate(s.ctx, request)
	} else {
		response, err = s.agent.generator.Generate(s.ctx, request)
	}
	if err != nil {
		var observerErr *observerDeliveryError
		if errors.As(err, &observerErr) {
			return gai.Response{}, &phaseError{phase: RunPhaseObserver, err: observerErr.err}
		}
		return gai.Response{}, &phaseError{phase: phaseForContext(s.ctx, RunPhaseGeneration, err), err: err}
	}
	return response, nil
}

func (s *runState) afterGenerationResponse(generation uint, request gai.GenerationRequest, response gai.Response) (gai.Response, error) {
	// afterGenerationResponse validates provider and hook responses before acceptance.
	if err := validateResponse(response, s.seenCallIDs); err != nil {
		return gai.Response{}, &phaseError{phase: RunPhaseValidation, err: fmt.Errorf("invalid generator response: %w", err)}
	}
	if s.agent.afterGeneration == nil {
		return response, nil
	}
	if err := s.contextError(); err != nil {
		return gai.Response{}, &phaseError{phase: RunPhaseCancellation, err: err}
	}
	changed, err := s.agent.afterGeneration.AfterGeneration(s.ctx, AfterGenerationRequest{
		Generation: generation,
		Request:    request,
		Response:   response,
		Status:     s.status,
	})
	if err != nil {
		return gai.Response{}, &phaseError{phase: phaseForContext(s.ctx, RunPhaseAfterGenerationHook, err), err: err}
	}
	if err := validateAfterGenerationResponse(response, changed, s.seenCallIDs); err != nil {
		return gai.Response{}, &phaseError{phase: RunPhaseAfterGenerationHook, err: fmt.Errorf("invalid AfterGeneration response: %w", err)}
	}
	return changed, nil
}

func (s *runState) acceptGeneration(generation uint, response gai.Response) error {
	// acceptGeneration updates dialog and usage before the completed event.
	usage, err := addStandardUsage(s.usage, response.UsageMetadata)
	if err != nil {
		return &phaseError{phase: RunPhaseValidation, err: fmt.Errorf("add response usage: %w", err)}
	}
	s.dialog = append(s.dialog, response.Candidates[0])
	s.usage = usage
	s.status.GenerationCalls++
	s.status.Usage = usage
	for _, block := range response.Candidates[0].Blocks {
		if block.BlockType == gai.ToolCall {
			s.seenCallIDs[block.ID] = struct{}{}
		}
	}
	if err := s.emit(GenerationCompletedEvent{Generation: generation, Response: response}); err != nil {
		return &phaseError{phase: RunPhaseObserver, err: err}
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
		if err := s.contextError(); err != nil {
			return "", &phaseError{phase: RunPhaseCancellation, err: err}
		}

		block := rawBlock
		call, tool, reason := s.classifyToolCall(block)
		willExecute := reason == ""
		if willExecute && s.agent.beforeTool != nil {
			decision, err := s.agent.beforeTool.BeforeTool(s.ctx, BeforeToolRequest{
				Generation: generation,
				ToolIndex:  toolIndex,
				Block:      block,
				Call:       *call,
				Definition: s.toolDefinition(tool),
				Status:     s.status,
			})
			if err != nil {
				return "", &phaseError{phase: phaseForContext(s.ctx, RunPhaseBeforeToolHook, err), err: err}
			}
			if decision.Reject && decision.Parameters != nil {
				return "", &phaseError{phase: RunPhaseBeforeToolHook, err: errors.New("BeforeTool cannot reject and replace parameters together")}
			}
			if err := validateStopReason(decision.StopAfterBatch); err != nil {
				return "", &phaseError{phase: RunPhaseBeforeToolHook, err: err}
			}
			if batchStop == "" {
				batchStop = decision.StopAfterBatch
			}
			if decision.Parameters != nil {
				if err := validateToolArguments(*tool, decision.Parameters); err != nil {
					return "", &phaseError{phase: RunPhaseBeforeToolHook, err: fmt.Errorf("invalid BeforeTool parameters: %w", err)}
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

		started := ToolStartedEvent{
			Generation:  generation,
			ToolIndex:   toolIndex,
			Block:       block,
			Call:        call,
			Definition:  s.toolDefinitionPointer(tool),
			WillExecute: willExecute,
			Reason:      reason,
		}
		if err := s.emit(started); err != nil {
			return "", &phaseError{phase: RunPhaseObserver, err: err}
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
			s.status.ToolExecutions++
		}
		if err := s.emit(ToolCompletedEvent{
			Generation: generation,
			ToolIndex:  toolIndex,
			Block:      block,
			Call:       call,
			Definition: s.toolDefinitionPointer(tool),
			Result:     result,
			Executed:   executed,
		}); err != nil {
			return "", &phaseError{phase: RunPhaseObserver, err: err}
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

func (s *runState) executeOrBuildToolResult(
	generation uint,
	toolIndex uint,
	block gai.Block,
	call *gai.ToolCallInput,
	tool *executableTool,
	willExecute bool,
	reason string,
) (gai.Message, bool, StopReason, error) {
	// executeOrBuildToolResult separates handler failures from model-visible errors.
	if !willExecute {
		result := gai.ToolResultMessage(block.ID, gai.TextBlock(reason))
		result.ToolResultError = true
		return result, false, "", nil
	}
	result, err := tool.handler.Execute(s.ctx, ToolRequest{
		Block: block,
		Call:  *call,
	})
	if err != nil {
		return gai.Message{}, false, "", &phaseError{phase: phaseForContext(s.ctx, RunPhaseTool, err), err: err}
	}
	if err := validateToolResult(result); err != nil {
		return gai.Message{}, false, "", &phaseError{phase: RunPhaseTool, err: fmt.Errorf("invalid handler result: %w", err)}
	}
	if s.agent.afterTool == nil {
		return result, true, "", nil
	}
	decision, err := s.agent.afterTool.AfterTool(s.ctx, AfterToolRequest{
		Generation: generation,
		ToolIndex:  toolIndex,
		Block:      block,
		Call:       *call,
		Definition: s.toolDefinition(tool),
		Result:     result,
		Status:     s.status,
	})
	if err != nil {
		return gai.Message{}, false, "", &phaseError{phase: phaseForContext(s.ctx, RunPhaseAfterToolHook, err), err: err}
	}
	if err := validateStopReason(decision.StopAfterBatch); err != nil {
		return gai.Message{}, false, "", &phaseError{phase: RunPhaseAfterToolHook, err: err}
	}
	if err := validateToolResult(decision.Result); err != nil {
		return gai.Message{}, false, "", &phaseError{phase: RunPhaseAfterToolHook, err: fmt.Errorf("invalid AfterTool result: %w", err)}
	}
	return decision.Result, true, decision.StopAfterBatch, nil
}

func (s *runState) toolDefinition(tool *executableTool) gai.Tool {
	return s.toolDefinitions[s.agent.toolIndexes[tool.definition.Name]]
}

func (s *runState) toolDefinitionPointer(tool *executableTool) *gai.Tool {
	if tool == nil {
		return nil
	}
	return &s.toolDefinitions[s.agent.toolIndexes[tool.definition.Name]]
}

func (s *runState) emit(payload EventPayload) error {
	if s.observer == nil {
		return nil
	}
	event := Event{
		Sequence: s.sequence,
		Payload:  payload,
	}
	s.sequence++
	return s.observer.Observe(s.ctx, event)
}

func (s *runState) complete(reason StopReason, finishReason gai.FinishReason) (RunResult, error) {
	result := s.result()
	result.StopReason = reason
	result.ModelFinishReason = finishReason
	if err := s.emit(RunCompletedEvent{Result: result}); err != nil {
		return s.failResult(result, RunPhaseObserver, err)
	}
	return result, nil
}

func (s *runState) failFrom(err error) (RunResult, error) {
	var failed *phaseError
	if errors.As(err, &failed) {
		return s.fail(failed.phase, failed.err)
	}
	return s.fail(RunPhaseInternal, err)
}

func (s *runState) fail(phase RunPhase, err error) (RunResult, error) {
	return s.failResult(s.result(), phase, err)
}

func (s *runState) failResult(result RunResult, phase RunPhase, err error) (RunResult, error) {
	if eventErr := s.emit(RunFailedEvent{Result: result, Phase: phase, Err: err}); eventErr != nil {
		err = errors.Join(err, fmt.Errorf("report run failure: %w", eventErr))
	}
	return result, err
}

func (s *runState) result() RunResult {
	return RunResult{
		Dialog:            s.dialog,
		ModelFinishReason: gai.Unknown,
		Usage:             s.usage,
	}
}

func (s *runState) contextError() error {
	return s.ctx.Err()
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

type observedStreamingGenerator struct {
	source     gai.StreamingGenerator
	state      *runState
	generation uint
}

func (g observedStreamingGenerator) Stream(ctx context.Context, request gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
	return func(yield func(gai.StreamChunk) bool) {
		for chunk := range g.source.Stream(ctx, request) {
			if err := g.state.emit(GenerationChunkEvent{Generation: g.generation, Chunk: chunk}); err != nil {
				if chunk.Err != nil {
					err = errors.Join(chunk.Err, err)
				}
				yield(gai.StreamChunk{Err: &observerDeliveryError{err: err}})
				return
			}
			if !yield(chunk) {
				return
			}
		}
	}
}
