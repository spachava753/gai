package agent

import (
	"errors"
	"fmt"

	"github.com/spachava753/gai"
)

// StopReason identifies why an Agent run returned without an error. Hooks may
// use any non-empty application-defined value except StopReasonModel. It is
// empty in a RunResult returned with an error.
type StopReason string

const (
	// StopReasonModel means a terminal model response ended the run.
	StopReasonModel StopReason = "model"
)

// Config contains Agent dependencies and behavior. New borrows all referenced
// values for the Agent's lifetime.
type Config struct {
	// Generator performs model generation and may also implement streaming.
	Generator gai.Generator
	// Model is the provider model identifier used for base generation requests.
	Model string
	// Instructions are optional. The zero Message means absent; any populated
	// value must use the System role.
	Instructions gai.Message
	// Tools are the complete fixed executable tool set.
	Tools []Tool
	// PrepareDialog may replace the lasting active dialog before each model call.
	PrepareDialog PrepareDialogHook
	// BeforeGeneration may change one request or stop before a provider call.
	BeforeGeneration BeforeGenerationHook
	// AfterGeneration may change one complete response before it enters the dialog.
	AfterGeneration AfterGenerationHook
	// BeforeTool may replace arguments, reject a call, or request a batch stop.
	BeforeTool BeforeToolHook
	// AfterTool may replace a result or request a batch stop.
	AfterTool AfterToolHook
}

// Agent is a reusable definition of one tool-using model loop. It does not
// mutate its fields after New returns. Referenced configuration remains borrowed.
type Agent struct {
	generator        gai.Generator
	streaming        gai.StreamingGenerator
	model            string
	instructions     gai.Message
	tools            []executableTool
	toolDefinitions  []gai.Tool
	toolIndexes      map[string]int
	prepareDialog    PrepareDialogHook
	beforeGeneration BeforeGenerationHook
	afterGeneration  AfterGenerationHook
	beforeTool       BeforeToolHook
	afterTool        AfterToolHook
}

// New validates config and returns an Agent safe for concurrent runs when callers
// leave borrowed configuration unchanged and its dependencies support concurrency.
func New(config Config) (*Agent, error) {
	if isNilInterface(config.Generator) {
		return nil, errors.New("agent generator is required")
	}
	if config.Model == "" {
		return nil, gai.InvalidParameterErr{Parameter: "model", Reason: "cannot be empty"}
	}
	if err := validateInstructions(config.Instructions); err != nil {
		return nil, gai.InvalidParameterErr{Parameter: "instructions", Reason: err.Error()}
	}

	tools, definitions, err := prepareExecutableTools(config.Tools)
	if err != nil {
		return nil, fmt.Errorf("prepare agent tools: %w", err)
	}
	indexes := make(map[string]int, len(tools))
	for i := range tools {
		indexes[tools[i].definition.Name] = i
	}

	agent := &Agent{
		generator:        config.Generator,
		model:            config.Model,
		instructions:     config.Instructions,
		tools:            tools,
		toolDefinitions:  definitions,
		toolIndexes:      indexes,
		prepareDialog:    config.PrepareDialog,
		beforeGeneration: config.BeforeGeneration,
		afterGeneration:  config.AfterGeneration,
		beforeTool:       config.BeforeTool,
		afterTool:        config.AfterTool,
	}
	if streaming, ok := config.Generator.(gai.StreamingGenerator); ok && !isNilInterface(streaming) {
		agent.streaming = streaming
	}
	if isNilInterface(agent.prepareDialog) {
		agent.prepareDialog = nil
	}
	if isNilInterface(agent.beforeGeneration) {
		agent.beforeGeneration = nil
	}
	if isNilInterface(agent.afterGeneration) {
		agent.afterGeneration = nil
	}
	if isNilInterface(agent.beforeTool) {
		agent.beforeTool = nil
	}
	if isNilInterface(agent.afterTool) {
		agent.afterTool = nil
	}
	return agent, nil
}

// RunRequest supplies borrowed values for one run. Callers must not change the
// prior dialog, input message, options, or anything they reference until Run
// returns.
type RunRequest struct {
	// Dialog is the prior active dialog.
	Dialog gai.Dialog
	// Input is one non-empty User-role message appended before the first hook.
	Input gai.Message
	// Options are reused as read-only data in every base generation request.
	Options gai.GenerationOptions
}

// RunResult contains the active dialog and standard usage when a run returns.
// Its referenced data may share storage with inputs and dependency outputs.
type RunResult struct {
	// Dialog is the active dialog to pass to a later run.
	Dialog gai.Dialog
	// StopReason identifies a normal model or hook stop and is empty on failure.
	StopReason StopReason
	// ModelFinishReason is meaningful only when StopReason is StopReasonModel.
	ModelFinishReason gai.FinishReason
	// Usage contains additive standard usage accumulated during the run.
	Usage gai.Metadata
}
