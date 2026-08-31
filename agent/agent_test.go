package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/spachava753/gai"
	"github.com/spachava753/gai/agent"
	"github.com/spachava753/gai/agent/agenttest"
)

type echoArguments struct {
	Value string `json:"value"`
}

type nilMapToolHandler map[string]string

func (nilMapToolHandler) Execute(context.Context, agent.ToolRequest) (gai.Message, error) {
	return gai.ToolResultMessage("", gai.TextBlock("ok")), nil
}

func TestTerminalResponseCompletesRun(t *testing.T) {
	generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{
		Check: func(request gai.GenerationRequest) error {
			if request.Model != "test-model" {
				return fmt.Errorf("model = %q", request.Model)
			}
			if textOf(request.Instructions) != "system" {
				return fmt.Errorf("instructions = %q", textOf(request.Instructions))
			}
			if len(request.Dialog) != 2 || textOf(request.Dialog[1]) != "new" {
				return fmt.Errorf("dialog = %#v", request.Dialog)
			}
			return nil
		},
		Response: textResponse("done", gai.EndTurn, gai.Metadata{
			gai.UsageMetricInputTokens:      4,
			gai.UsageMetricGenerationTokens: 2,
			"provider_detail":               "kept on event only",
		}),
	})
	configuredInstructions := gai.SystemMessage(gai.TextBlock("system"))
	a, err := agent.New(agent.Config{
		Generator:    generator,
		Model:        "test-model",
		Instructions: configuredInstructions,
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()
	starting := gai.Dialog{userMessage("prior")}
	input := userMessage("new")

	result, err := a.Run(context.Background(), agent.RunRequest{
		Dialog: starting,
		Input:  input,
	}, observer)
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != agent.StopReasonModel || result.ModelFinishReason != gai.EndTurn {
		t.Fatalf("stop = %q, finish = %v", result.StopReason, result.ModelFinishReason)
	}
	if got := dialogTexts(result.Dialog); !reflect.DeepEqual(got, []string{"prior", "new", "done"}) {
		t.Fatalf("dialog texts = %v", got)
	}
	if got, _ := gai.InputTokens(result.Usage); got != 4 {
		t.Fatalf("input tokens = %d", got)
	}
	if _, ok := result.Usage["provider_detail"]; ok {
		t.Fatal("provider-specific usage entered RunResult")
	}
	assertEventKinds(t, observer.Events(),
		agent.EventKindRunStarted,
		agent.EventKindGenerationStarted,
		agent.EventKindGenerationCompleted,
		agent.EventKindRunCompleted,
	)
	if len(starting) != 1 || textOf(starting[0]) != "prior" || textOf(input) != "new" {
		t.Fatal("caller input was modified")
	}
}

func TestLastingDialogReplacementPrecedesTemporaryRequestChange(t *testing.T) {
	generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{
		Check: func(request gai.GenerationRequest) error {
			if request.Model != "one-call-model" {
				return fmt.Errorf("model = %q", request.Model)
			}
			if got := dialogTexts(request.Dialog); !reflect.DeepEqual(got, []string{"summary", "retrieved"}) {
				return fmt.Errorf("request dialog = %v", got)
			}
			return nil
		},
		Response: textResponse("answer", gai.EndTurn, gai.Metadata{gai.UsageMetricGenerationTokens: 5}),
	})
	replacement := gai.Dialog{userMessage("summary")}
	prepareCalls := 0
	beforeCalls := 0
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "base-model",
		PrepareDialog: agent.PrepareDialogFunc(func(_ context.Context, request agent.PrepareDialogRequest) (agent.PrepareDialogDecision, error) {
			prepareCalls++
			if request.Request.Model != "base-model" {
				return agent.PrepareDialogDecision{}, fmt.Errorf("prepare model = %q", request.Request.Model)
			}
			if got := dialogTexts(request.Request.Dialog); !reflect.DeepEqual(got, []string{"prior", "question"}) {
				return agent.PrepareDialogDecision{}, fmt.Errorf("prepare dialog = %v", got)
			}
			return agent.PrepareDialogDecision{
				Dialog: replacement,
				Usage:  gai.Metadata{gai.UsageMetricInputTokens: 3},
			}, nil
		}),
		BeforeGeneration: agent.BeforeGenerationFunc(func(_ context.Context, request agent.BeforeGenerationRequest) (agent.BeforeGenerationDecision, error) {
			beforeCalls++
			changed := request.Request
			changed.Model = "one-call-model"
			changed.Dialog = append(gai.Dialog(nil), changed.Dialog...)
			changed.Dialog = append(changed.Dialog, userMessage("retrieved"))
			changed.Options = gai.GenerationOptions{"retrieval": true}
			return agent.BeforeGenerationDecision{Request: changed}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	recording := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{
		Dialog: gai.Dialog{userMessage("prior")},
		Input:  userMessage("question"),
	}, recording)
	if err != nil {
		t.Fatal(err)
	}
	if prepareCalls != 1 || beforeCalls != 1 {
		t.Fatalf("prepare calls = %d, before calls = %d", prepareCalls, beforeCalls)
	}
	if got := dialogTexts(result.Dialog); !reflect.DeepEqual(got, []string{"summary", "answer"}) {
		t.Fatalf("lasting dialog = %v", got)
	}
	if got, _ := gai.InputTokens(result.Usage); got != 3 {
		t.Fatalf("input tokens = %d", got)
	}
	if got, _ := gai.OutputTokens(result.Usage); got != 5 {
		t.Fatalf("output tokens = %d", got)
	}
	assertEventKinds(t, recording.Events(),
		agent.EventKindRunStarted,
		agent.EventKindDialogReplaced,
		agent.EventKindGenerationStarted,
		agent.EventKindGenerationCompleted,
		agent.EventKindRunCompleted,
	)
}

func TestCostBudgetHookStopsWithoutProviderCall(t *testing.T) {
	generator := agenttest.NewScriptedGenerator()
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		BeforeGeneration: agent.BeforeGenerationFunc(func(context.Context, agent.BeforeGenerationRequest) (agent.BeforeGenerationDecision, error) {
			return agent.BeforeGenerationDecision{StopReason: "cost_budget"}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("hello")}, observer)
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "cost_budget" || result.ModelFinishReason != gai.Unknown {
		t.Fatalf("stop = %q, finish = %v", result.StopReason, result.ModelFinishReason)
	}
	if len(generator.Requests()) != 0 {
		t.Fatal("generator was called")
	}
	assertEventKinds(t, observer.Events(), agent.EventKindRunStarted, agent.EventKindRunCompleted)
}

func TestToolRoundUsesHookArgumentsAndResult(t *testing.T) {
	schema, err := gai.GenerateSchema[echoArguments]()
	if err != nil {
		t.Fatal(err)
	}
	callBlock, err := gai.ToolCallBlock("call-1", "echo", map[string]any{"value": "original"})
	if err != nil {
		t.Fatal(err)
	}
	generator := agenttest.NewScriptedGenerator(
		agenttest.GenerateStep{Response: toolResponse(callBlock)},
		agenttest.GenerateStep{
			Check: func(request gai.GenerationRequest) error {
				if len(request.Dialog) != 3 {
					return fmt.Errorf("dialog length = %d", len(request.Dialog))
				}
				result := request.Dialog[2]
				if result.Role != gai.ToolResult || result.ToolResultError || textOf(result) != "after" {
					return fmt.Errorf("tool result = %#v", result)
				}
				if result.Blocks[0].ID != "call-1" {
					return fmt.Errorf("tool result ID = %q", result.Blocks[0].ID)
				}
				return nil
			},
			Response: textResponse("finished", gai.EndTurn, nil),
		},
	)
	handlerCalls := 0
	beforeStatus := agent.RunStatus{}
	afterStatus := agent.RunStatus{}
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "echo", InputSchema: schema},
			Handler: agent.ToolHandlerFunc(func(_ context.Context, request agent.ToolRequest) (gai.Message, error) {
				handlerCalls++
				if request.Call.Parameters["value"] != "changed" {
					return gai.Message{}, fmt.Errorf("parameters = %#v", request.Call.Parameters)
				}
				return gai.ToolResultMessage("wrong-id", gai.TextBlock("handler")), nil
			}),
		}},
		BeforeTool: agent.BeforeToolFunc(func(_ context.Context, request agent.BeforeToolRequest) (agent.BeforeToolDecision, error) {
			beforeStatus = request.Status
			return agent.BeforeToolDecision{Parameters: map[string]any{"value": "changed"}}, nil
		}),
		AfterTool: agent.AfterToolFunc(func(_ context.Context, request agent.AfterToolRequest) (agent.AfterToolDecision, error) {
			afterStatus = request.Status
			return agent.AfterToolDecision{Result: gai.ToolResultMessage("also-wrong", gai.TextBlock("after"))}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("use tool")}, observer)
	if err != nil {
		t.Fatal(err)
	}
	if handlerCalls != 1 {
		t.Fatalf("handler calls = %d", handlerCalls)
	}
	if beforeStatus.GenerationCalls != 1 || beforeStatus.ToolExecutions != 0 {
		t.Fatalf("before status = %#v", beforeStatus)
	}
	if afterStatus.GenerationCalls != 1 || afterStatus.ToolExecutions != 0 {
		t.Fatalf("after status = %#v", afterStatus)
	}
	if got := dialogTexts(result.Dialog); !reflect.DeepEqual(got, []string{"use tool", "echo", "after", "finished"}) {
		t.Fatalf("dialog = %v", got)
	}
	events := observer.Events()
	assertEventKinds(t, events,
		agent.EventKindRunStarted,
		agent.EventKindGenerationStarted,
		agent.EventKindGenerationCompleted,
		agent.EventKindToolStarted,
		agent.EventKindToolCompleted,
		agent.EventKindGenerationStarted,
		agent.EventKindGenerationCompleted,
		agent.EventKindRunCompleted,
	)
	toolStarted := events[3].Payload.(agent.ToolStartedEvent)
	if toolStarted.Generation != 0 || toolStarted.ToolIndex != 0 {
		t.Fatalf("tool position = generation %d, index %d", toolStarted.Generation, toolStarted.ToolIndex)
	}
	secondGeneration := events[5].Payload.(agent.GenerationStartedEvent)
	if secondGeneration.Generation != 1 {
		t.Fatalf("second generation index = %d", secondGeneration.Generation)
	}
}

func TestRecoverableToolErrorsReturnResultsToModel(t *testing.T) {
	schema, err := gai.GenerateSchema[echoArguments]()
	if err != nil {
		t.Fatal(err)
	}
	validDefinition := gai.Tool{Name: "echo", InputSchema: schema}

	tests := []struct {
		name       string
		block      gai.Block
		definition gai.Tool
	}{
		{
			name:  "unknown tool",
			block: mustToolCallBlock(t, "unknown-id", "missing", map[string]any{}),
		},
		{
			name: "malformed call",
			block: gai.Block{
				ID:           "malformed-id",
				BlockType:    gai.ToolCall,
				ModalityType: gai.Text,
				MimeType:     "text/plain",
				Content:      gai.Str("{"),
			},
			definition: validDefinition,
		},
		{
			name:       "invalid arguments",
			block:      mustToolCallBlock(t, "invalid-id", "echo", map[string]any{"value": 42}),
			definition: validDefinition,
		},
		{
			name:       "arguments for parameterless tool",
			block:      mustToolCallBlock(t, "parameterless-id", "plain", map[string]any{"unexpected": true}),
			definition: gai.Tool{Name: "plain"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator := agenttest.NewScriptedGenerator(
				agenttest.GenerateStep{Response: toolResponse(test.block)},
				agenttest.GenerateStep{
					Check: func(request gai.GenerationRequest) error {
						result := request.Dialog[len(request.Dialog)-1]
						if result.Role != gai.ToolResult || !result.ToolResultError {
							return fmt.Errorf("result = %#v", result)
						}
						if result.Blocks[0].ID != test.block.ID {
							return fmt.Errorf("result ID = %q", result.Blocks[0].ID)
						}
						return nil
					},
					Response: textResponse("recovered", gai.EndTurn, nil),
				},
			)
			handlerCalls := 0
			beforeCalls := 0
			var tools []agent.Tool
			if test.definition.Name != "" {
				tools = []agent.Tool{{
					Definition: test.definition,
					Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
						handlerCalls++
						return gai.ToolResultMessage("", gai.TextBlock("unexpected")), nil
					}),
				}}
			}
			a, newErr := agent.New(agent.Config{
				Generator: generator,
				Model:     "test-model",
				Tools:     tools,
				BeforeTool: agent.BeforeToolFunc(func(context.Context, agent.BeforeToolRequest) (agent.BeforeToolDecision, error) {
					beforeCalls++
					return agent.BeforeToolDecision{}, nil
				}),
			})
			if newErr != nil {
				t.Fatal(newErr)
			}

			result, runErr := a.Run(context.Background(), agent.RunRequest{Input: userMessage("go")}, nil)
			if runErr != nil {
				t.Fatal(runErr)
			}
			if handlerCalls != 0 || beforeCalls != 0 {
				t.Fatalf("handler calls = %d, before calls = %d", handlerCalls, beforeCalls)
			}
			if result.StopReason != agent.StopReasonModel {
				t.Fatalf("stop reason = %q", result.StopReason)
			}
		})
	}
}

func TestRejectedBatchCompletesBeforeHookStop(t *testing.T) {
	first := mustToolCallBlock(t, "first", "one", map[string]any{})
	second := mustToolCallBlock(t, "second", "two", map[string]any{})
	generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{Response: toolResponse(first, second)})
	handlerCalls := 0
	tools := []agent.Tool{
		{Definition: gai.Tool{Name: "one"}, Handler: countingHandler(&handlerCalls)},
		{Definition: gai.Tool{Name: "two"}, Handler: countingHandler(&handlerCalls)},
	}
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		Tools:     tools,
		BeforeTool: agent.BeforeToolFunc(func(_ context.Context, request agent.BeforeToolRequest) (agent.BeforeToolDecision, error) {
			return agent.BeforeToolDecision{
				Reject:         true,
				Reason:         fmt.Sprintf("denied %s", request.Call.Name),
				StopAfterBatch: "permission_denied",
			}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("run both")}, observer)
	if err != nil {
		t.Fatal(err)
	}
	if handlerCalls != 0 {
		t.Fatalf("handler calls = %d", handlerCalls)
	}
	if result.StopReason != "permission_denied" || result.ModelFinishReason != gai.Unknown {
		t.Fatalf("stop = %q, finish = %v", result.StopReason, result.ModelFinishReason)
	}
	if len(result.Dialog) != 4 {
		t.Fatalf("dialog length = %d", len(result.Dialog))
	}
	for _, message := range result.Dialog[2:] {
		if !message.ToolResultError {
			t.Fatalf("result is not marked as error: %#v", message)
		}
	}
	assertEventKinds(t, observer.Events(),
		agent.EventKindRunStarted,
		agent.EventKindGenerationStarted,
		agent.EventKindGenerationCompleted,
		agent.EventKindToolStarted,
		agent.EventKindToolCompleted,
		agent.EventKindToolStarted,
		agent.EventKindToolCompleted,
		agent.EventKindRunCompleted,
	)
}

func TestFinishReasonAndUsageCannotBeChangedByHook(t *testing.T) {
	tests := []struct {
		name   string
		change func(gai.Response) gai.Response
	}{
		{
			name: "finish reason",
			change: func(response gai.Response) gai.Response {
				response.FinishReason = gai.StopSequence
				return response
			},
		},
		{
			name: "usage",
			change: func(response gai.Response) gai.Response {
				response.UsageMetadata = gai.Metadata{gai.UsageMetricInputTokens: 99}
				return response
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{
				Response: textResponse("answer", gai.EndTurn, gai.Metadata{gai.UsageMetricInputTokens: 1}),
			})
			a, err := agent.New(agent.Config{
				Generator: generator,
				Model:     "test-model",
				AfterGeneration: agent.AfterGenerationFunc(func(_ context.Context, request agent.AfterGenerationRequest) (gai.Response, error) {
					return test.change(request.Response), nil
				}),
			})
			if err != nil {
				t.Fatal(err)
			}
			observer := agenttest.NewRecordingObserver()

			result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("question")}, observer)
			if err == nil {
				t.Fatal("expected hook validation error")
			}
			if len(result.Dialog) != 1 {
				t.Fatalf("dialog length = %d", len(result.Dialog))
			}
			failed := lastFailure(t, observer.Events())
			if failed.Phase != agent.RunPhaseAfterGenerationHook {
				t.Fatalf("failure phase = %q", failed.Phase)
			}
		})
	}
}

func TestDuplicateToolCallIDFailsBeforeSecondResponseIsAccepted(t *testing.T) {
	first := mustToolCallBlock(t, "same-id", "echo", map[string]any{})
	second := mustToolCallBlock(t, "same-id", "echo", map[string]any{})
	generator := agenttest.NewScriptedGenerator(
		agenttest.GenerateStep{Response: toolResponse(first)},
		agenttest.GenerateStep{Response: toolResponse(second)},
	)
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "echo"},
			Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
				return gai.ToolResultMessage("", gai.TextBlock("ok")), nil
			}),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("go")}, observer)
	if err == nil {
		t.Fatal("expected duplicate ID error")
	}
	if len(result.Dialog) != 3 {
		t.Fatalf("dialog length = %d", len(result.Dialog))
	}
	if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseValidation {
		t.Fatalf("phase = %q", phase)
	}
}

func TestRequestValidationPrecedesRunEvents(t *testing.T) {
	tests := []struct {
		name    string
		request agent.RunRequest
	}{
		{name: "empty input", request: agent.RunRequest{}},
		{name: "input without role", request: agent.RunRequest{
			Input: gai.Message{Blocks: []gai.Block{gai.TextBlock("missing role")}},
		}},
		{name: "assistant input", request: agent.RunRequest{Input: assistantMessage("wrong")}},
		{name: "candidate count", request: agent.RunRequest{
			Input:   userMessage("hello"),
			Options: gai.NewGenerationOptions(gai.WithCandidateCount(2)),
		}},
		{name: "system in dialog", request: agent.RunRequest{
			Dialog: gai.Dialog{gai.SystemMessage(gai.TextBlock("wrong"))},
			Input:  userMessage("hello"),
		}},
		{name: "empty assistant in dialog", request: agent.RunRequest{
			Dialog: gai.Dialog{{Role: gai.Assistant}},
			Input:  userMessage("hello"),
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator := agenttest.NewScriptedGenerator()
			a, err := agent.New(agent.Config{Generator: generator, Model: "test-model"})
			if err != nil {
				t.Fatal(err)
			}
			observer := agenttest.NewRecordingObserver()

			result, err := a.Run(context.Background(), test.request, observer)
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !reflect.DeepEqual(result, agent.RunResult{}) {
				t.Fatalf("result = %#v", result)
			}
			if len(observer.Events()) != 0 {
				t.Fatal("observer received pre-run event")
			}
		})
	}
}

func TestStreamingForwardsChunksBeforeCollectingResponse(t *testing.T) {
	generator := agenttest.NewScriptedStreamingGenerator(agenttest.StreamStep{Chunks: []gai.StreamChunk{
		{
			Block:               gai.Block{BlockType: gai.Content, ModalityType: gai.Text, MimeType: "text/plain", Content: gai.Str("hel")},
			MessageExtraFields:  map[string]any{"message_id": "msg-1"},
			ResponseExtraFields: map[string]any{"response_id": "resp-1"},
		},
		{Block: gai.Block{BlockType: gai.Content, ModalityType: gai.Text, MimeType: "text/plain", Content: gai.Str("lo")}},
		{Block: gai.MetadataBlock(gai.Metadata{
			gai.UsageMetricInputTokens:      2,
			gai.UsageMetricGenerationTokens: 1,
		})},
	}})
	a, err := agent.New(agent.Config{Generator: generator, Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("hello")}, observer)
	if err != nil {
		t.Fatal(err)
	}
	if got := textOf(result.Dialog[len(result.Dialog)-1]); got != "hello" {
		t.Fatalf("assistant text = %q", got)
	}
	if got, _ := gai.InputTokens(result.Usage); got != 2 {
		t.Fatalf("input tokens = %d", got)
	}
	assertEventKinds(t, observer.Events(),
		agent.EventKindRunStarted,
		agent.EventKindGenerationStarted,
		agent.EventKindGenerationChunk,
		agent.EventKindGenerationChunk,
		agent.EventKindGenerationChunk,
		agent.EventKindGenerationCompleted,
		agent.EventKindRunCompleted,
	)
	completed := observer.Events()[5].Payload.(agent.GenerationCompletedEvent)
	if completed.Response.ExtraFields["response_id"] != "resp-1" {
		t.Fatalf("response extra fields = %#v", completed.Response.ExtraFields)
	}
	if completed.Response.Candidates[0].ExtraFields["message_id"] != "msg-1" {
		t.Fatalf("message extra fields = %#v", completed.Response.Candidates[0].ExtraFields)
	}
}

func TestStreamErrorLeavesChunksOutOfActiveDialog(t *testing.T) {
	streamErr := errors.New("stream failed")
	generator := agenttest.NewScriptedStreamingGenerator(agenttest.StreamStep{Chunks: []gai.StreamChunk{
		{Block: gai.Block{BlockType: gai.Content, ModalityType: gai.Text, MimeType: "text/plain", Content: gai.Str("partial")}},
		{Err: streamErr},
	}})
	a, err := agent.New(agent.Config{Generator: generator, Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("hello")}, observer)
	if !errors.Is(err, streamErr) {
		t.Fatalf("error = %v", err)
	}
	if len(result.Dialog) != 1 {
		t.Fatalf("dialog length = %d", len(result.Dialog))
	}
	assertEventKinds(t, observer.Events(),
		agent.EventKindRunStarted,
		agent.EventKindGenerationStarted,
		agent.EventKindGenerationChunk,
		agent.EventKindGenerationChunk,
		agent.EventKindRunFailed,
	)
	if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseGeneration {
		t.Fatalf("phase = %q", phase)
	}
}

func TestGenerationErrorLeavesResponseOutOfActiveDialog(t *testing.T) {
	generationErr := errors.New("generation failed")
	generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{
		Response: textResponse("partial", gai.EndTurn, gai.Metadata{gai.UsageMetricInputTokens: 4}),
		Err:      generationErr,
	})
	a, err := agent.New(agent.Config{Generator: generator, Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("hello")}, observer)
	if !errors.Is(err, generationErr) {
		t.Fatalf("error = %v", err)
	}
	if len(result.Dialog) != 1 || len(result.Usage) != 0 {
		t.Fatalf("result = %#v", result)
	}
	if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseGeneration {
		t.Fatalf("phase = %q", phase)
	}
}

func TestObserverFailureStopsLaterWorkAndReturnsCurrentState(t *testing.T) {
	observerErr := errors.New("observer failed")
	tests := []struct {
		name             string
		failSequence     uint64
		toolRound        bool
		wantDialogSize   int
		wantCalls        int
		wantHandlerCalls int
	}{
		{name: "run started", failSequence: 0, wantDialogSize: 1, wantCalls: 0},
		{name: "generation started", failSequence: 1, wantDialogSize: 1, wantCalls: 0},
		{name: "generation completed", failSequence: 2, wantDialogSize: 2, wantCalls: 1},
		{name: "run completed", failSequence: 3, wantDialogSize: 2, wantCalls: 1},
		{name: "tool started", failSequence: 3, toolRound: true, wantDialogSize: 2, wantCalls: 1},
		{name: "tool completed", failSequence: 4, toolRound: true, wantDialogSize: 3, wantCalls: 1, wantHandlerCalls: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := textResponse("done", gai.EndTurn, nil)
			if test.toolRound {
				response = toolResponse(mustToolCallBlock(t, "call-1", "echo", map[string]any{}))
			}
			generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{Response: response})
			handlerCalls := 0
			config := agent.Config{Generator: generator, Model: "test-model"}
			if test.toolRound {
				config.Tools = []agent.Tool{{
					Definition: gai.Tool{Name: "echo"},
					Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
						handlerCalls++
						return gai.ToolResultMessage("", gai.TextBlock("ok")), nil
					}),
				}}
			}
			a, err := agent.New(config)
			if err != nil {
				t.Fatal(err)
			}
			observer := agenttest.NewFailingObserver(test.failSequence, observerErr)

			result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("hello")}, observer)
			if !errors.Is(err, observerErr) {
				t.Fatalf("error = %v", err)
			}
			if len(result.Dialog) != test.wantDialogSize {
				t.Fatalf("dialog length = %d", len(result.Dialog))
			}
			if len(generator.Requests()) != test.wantCalls {
				t.Fatalf("generator calls = %d", len(generator.Requests()))
			}
			if handlerCalls != test.wantHandlerCalls {
				t.Fatalf("handler calls = %d", handlerCalls)
			}
			if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseObserver {
				t.Fatalf("phase = %q", phase)
			}
		})
	}
}

func TestHandlerErrorStopsWithoutInventingToolResult(t *testing.T) {
	handlerErr := errors.New("database unavailable")
	call := mustToolCallBlock(t, "call-1", "lookup", map[string]any{})
	generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{Response: toolResponse(call)})
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "lookup"},
			Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
				return gai.Message{}, handlerErr
			}),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("lookup")}, observer)
	if !errors.Is(err, handlerErr) {
		t.Fatalf("error = %v", err)
	}
	if len(result.Dialog) != 2 {
		t.Fatalf("dialog length = %d", len(result.Dialog))
	}
	if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseTool {
		t.Fatalf("phase = %q", phase)
	}
	assertEventKinds(t, observer.Events(),
		agent.EventKindRunStarted,
		agent.EventKindGenerationStarted,
		agent.EventKindGenerationCompleted,
		agent.EventKindToolStarted,
		agent.EventKindRunFailed,
	)
}

func TestInvalidLastingDialogDecisionsAreRejected(t *testing.T) {
	tests := []struct {
		name     string
		decision agent.PrepareDialogDecision
	}{
		{name: "empty replacement", decision: agent.PrepareDialogDecision{Dialog: gai.Dialog{}}},
		{name: "usage without replacement", decision: agent.PrepareDialogDecision{
			Usage: gai.Metadata{gai.UsageMetricInputTokens: 1},
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator := agenttest.NewScriptedGenerator()
			a, err := agent.New(agent.Config{
				Generator: generator,
				Model:     "test-model",
				PrepareDialog: agent.PrepareDialogFunc(func(context.Context, agent.PrepareDialogRequest) (agent.PrepareDialogDecision, error) {
					return test.decision, nil
				}),
			})
			if err != nil {
				t.Fatal(err)
			}
			observer := agenttest.NewRecordingObserver()

			result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("hello")}, observer)
			if err == nil {
				t.Fatal("expected invalid replacement error")
			}
			if len(result.Dialog) != 1 || len(generator.Requests()) != 0 {
				t.Fatalf("result = %#v, calls = %d", result, len(generator.Requests()))
			}
			if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhasePrepareDialogHook {
				t.Fatalf("phase = %q", phase)
			}
		})
	}
}

func TestProviderResponseValidationPrecedesDialogUpdate(t *testing.T) {
	call := mustToolCallBlock(t, "call", "tool", map[string]any{})
	missingID := call
	missingID.ID = ""
	duplicate := call

	tests := []struct {
		name     string
		response gai.Response
	}{
		{name: "no candidates", response: gai.Response{FinishReason: gai.EndTurn}},
		{name: "two candidates", response: gai.Response{
			Candidates:   []gai.Message{assistantMessage("one"), assistantMessage("two")},
			FinishReason: gai.EndTurn,
		}},
		{name: "user candidate", response: gai.Response{
			Candidates:   []gai.Message{userMessage("wrong")},
			FinishReason: gai.EndTurn,
		}},
		{name: "empty assistant candidate", response: gai.Response{
			Candidates:   []gai.Message{{Role: gai.Assistant}},
			FinishReason: gai.EndTurn,
		}},
		{name: "tool finish without call", response: textResponse("none", gai.ToolUse, nil)},
		{name: "call with terminal finish", response: gai.Response{
			Candidates:   []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{call}}},
			FinishReason: gai.EndTurn,
		}},
		{name: "missing call ID", response: toolResponse(missingID)},
		{name: "duplicate call ID", response: toolResponse(call, duplicate)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{Response: test.response})
			a, err := agent.New(agent.Config{Generator: generator, Model: "test-model"})
			if err != nil {
				t.Fatal(err)
			}
			observer := agenttest.NewRecordingObserver()

			result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("hello")}, observer)
			if err == nil {
				t.Fatal("expected response validation error")
			}
			if len(result.Dialog) != 1 {
				t.Fatalf("dialog length = %d", len(result.Dialog))
			}
			if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseValidation {
				t.Fatalf("phase = %q", phase)
			}
		})
	}
}

func TestConstructorRejectsInvalidConfiguration(t *testing.T) {
	generator := agenttest.NewScriptedGenerator()
	handler := countingHandler(new(int))
	tests := []struct {
		name   string
		config agent.Config
	}{
		{name: "nil generator", config: agent.Config{Model: "model"}},
		{name: "empty model", config: agent.Config{Generator: generator}},
		{name: "instructions without role", config: agent.Config{
			Generator:    generator,
			Model:        "model",
			Instructions: gai.Message{Blocks: []gai.Block{gai.TextBlock("missing role")}},
		}},
		{name: "user instructions", config: agent.Config{
			Generator:    generator,
			Model:        "model",
			Instructions: userMessage("wrong"),
		}},
		{name: "empty tool name", config: agent.Config{
			Generator: generator,
			Model:     "model",
			Tools:     []agent.Tool{{Handler: handler}},
		}},
		{name: "reserved tool name", config: agent.Config{
			Generator: generator,
			Model:     "model",
			Tools:     []agent.Tool{{Definition: gai.Tool{Name: gai.ToolChoiceAuto}, Handler: handler}},
		}},
		{name: "duplicate tool", config: agent.Config{
			Generator: generator,
			Model:     "model",
			Tools: []agent.Tool{
				{Definition: gai.Tool{Name: "same"}, Handler: handler},
				{Definition: gai.Tool{Name: "same"}, Handler: handler},
			},
		}},
		{name: "nil handler", config: agent.Config{
			Generator: generator,
			Model:     "model",
			Tools:     []agent.Tool{{Definition: gai.Tool{Name: "tool"}}},
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := agent.New(test.config); err == nil {
				t.Fatal("expected constructor error")
			}
		})
	}
}

func TestInterfaceNilHandling(t *testing.T) {
	t.Run("nil pointer generator is absent", func(t *testing.T) {
		var generator *blockingGenerator
		if _, err := agent.New(agent.Config{Generator: generator, Model: "model"}); err == nil {
			t.Fatal("expected constructor error")
		}
	})

	t.Run("nil function handler is absent", func(t *testing.T) {
		generator := agenttest.NewScriptedGenerator()
		if _, err := agent.New(agent.Config{
			Generator: generator,
			Model:     "model",
			Tools: []agent.Tool{{
				Definition: gai.Tool{Name: "tool"},
				Handler:    agent.ToolHandlerFunc(nil),
			}},
		}); err == nil {
			t.Fatal("expected constructor error")
		}
	})

	t.Run("nil function hooks and observer are absent", func(t *testing.T) {
		generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{
			Response: textResponse("done", gai.EndTurn, nil),
		})
		a, err := agent.New(agent.Config{
			Generator:        generator,
			Model:            "model",
			Instructions:     gai.SystemMessage(),
			PrepareDialog:    agent.PrepareDialogFunc(nil),
			BeforeGeneration: agent.BeforeGenerationFunc(nil),
			AfterGeneration:  agent.AfterGenerationFunc(nil),
			BeforeTool:       agent.BeforeToolFunc(nil),
			AfterTool:        agent.AfterToolFunc(nil),
		})
		if err != nil {
			t.Fatal(err)
		}
		var observer agent.Observer = agent.ObserverFunc(nil)
		if _, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("go")}, observer); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("nil map implementation remains valid", func(t *testing.T) {
		generator := agenttest.NewScriptedGenerator()
		if _, err := agent.New(agent.Config{
			Generator: generator,
			Model:     "model",
			Tools: []agent.Tool{{
				Definition: gai.Tool{Name: "tool"},
				Handler:    nilMapToolHandler(nil),
			}},
		}); err != nil {
			t.Fatal(err)
		}
	})
}

func TestCancellationDuringGenerationReturnsCancellationPhase(t *testing.T) {
	generator := &blockingGenerator{started: make(chan struct{})}
	a, err := agent.New(agent.Config{Generator: generator, Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()
	ctx, cancel := context.WithCancel(context.Background())
	resultChannel := make(chan agent.RunResult, 1)
	errorChannel := make(chan error, 1)
	go func() {
		result, runErr := a.Run(ctx, agent.RunRequest{Input: userMessage("wait")}, observer)
		resultChannel <- result
		errorChannel <- runErr
	}()

	<-generator.started
	cancel()
	result := <-resultChannel
	runErr := <-errorChannel
	if !errors.Is(runErr, context.Canceled) {
		t.Fatalf("error = %v", runErr)
	}
	if len(result.Dialog) != 1 {
		t.Fatalf("dialog length = %d", len(result.Dialog))
	}
	if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseCancellation {
		t.Fatalf("phase = %q", phase)
	}
}

func TestContextErrorsFromOperationsUseCancellationPhase(t *testing.T) {
	toolCall := func(id string) gai.Response {
		return toolResponse(mustToolCallBlock(t, id, "echo", map[string]any{}))
	}
	tests := []struct {
		name   string
		config func() agent.Config
	}{
		{
			name: "prepare dialog",
			config: func() agent.Config {
				return agent.Config{
					Generator: agenttest.NewScriptedGenerator(),
					Model:     "model",
					PrepareDialog: agent.PrepareDialogFunc(func(context.Context, agent.PrepareDialogRequest) (agent.PrepareDialogDecision, error) {
						return agent.PrepareDialogDecision{}, context.Canceled
					}),
				}
			},
		},
		{
			name: "before generation",
			config: func() agent.Config {
				return agent.Config{
					Generator: agenttest.NewScriptedGenerator(),
					Model:     "model",
					BeforeGeneration: agent.BeforeGenerationFunc(func(context.Context, agent.BeforeGenerationRequest) (agent.BeforeGenerationDecision, error) {
						return agent.BeforeGenerationDecision{}, context.Canceled
					}),
				}
			},
		},
		{
			name: "streaming generation",
			config: func() agent.Config {
				return agent.Config{
					Generator: agenttest.NewScriptedStreamingGenerator(agenttest.StreamStep{Chunks: []gai.StreamChunk{{Err: context.Canceled}}}),
					Model:     "model",
				}
			},
		},
		{
			name: "after generation",
			config: func() agent.Config {
				return agent.Config{
					Generator: agenttest.NewScriptedGenerator(agenttest.GenerateStep{Response: textResponse("done", gai.EndTurn, nil)}),
					Model:     "model",
					AfterGeneration: agent.AfterGenerationFunc(func(context.Context, agent.AfterGenerationRequest) (gai.Response, error) {
						return gai.Response{}, context.Canceled
					}),
				}
			},
		},
		{
			name: "before tool",
			config: func() agent.Config {
				return agent.Config{
					Generator: agenttest.NewScriptedGenerator(agenttest.GenerateStep{Response: toolCall("before")}),
					Model:     "model",
					Tools: []agent.Tool{{
						Definition: gai.Tool{Name: "echo"},
						Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
							return gai.ToolResultMessage("", gai.TextBlock("ok")), nil
						}),
					}},
					BeforeTool: agent.BeforeToolFunc(func(context.Context, agent.BeforeToolRequest) (agent.BeforeToolDecision, error) {
						return agent.BeforeToolDecision{}, context.Canceled
					}),
				}
			},
		},
		{
			name: "tool handler",
			config: func() agent.Config {
				return agent.Config{
					Generator: agenttest.NewScriptedGenerator(agenttest.GenerateStep{Response: toolCall("handler")}),
					Model:     "model",
					Tools: []agent.Tool{{
						Definition: gai.Tool{Name: "echo"},
						Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
							return gai.Message{}, context.Canceled
						}),
					}},
				}
			},
		},
		{
			name: "after tool",
			config: func() agent.Config {
				return agent.Config{
					Generator: agenttest.NewScriptedGenerator(agenttest.GenerateStep{Response: toolCall("after")}),
					Model:     "model",
					Tools: []agent.Tool{{
						Definition: gai.Tool{Name: "echo"},
						Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
							return gai.ToolResultMessage("", gai.TextBlock("ok")), nil
						}),
					}},
					AfterTool: agent.AfterToolFunc(func(context.Context, agent.AfterToolRequest) (agent.AfterToolDecision, error) {
						return agent.AfterToolDecision{}, context.Canceled
					}),
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a, err := agent.New(test.config())
			if err != nil {
				t.Fatal(err)
			}
			observer := agenttest.NewRecordingObserver()

			_, err = a.Run(context.Background(), agent.RunRequest{Input: userMessage("go")}, observer)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("error = %v", err)
			}
			if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseCancellation {
				t.Fatalf("phase = %q", phase)
			}
		})
	}
}

func TestImmutableAgentSupportsConcurrentRuns(t *testing.T) {
	a, err := agent.New(agent.Config{Generator: echoGenerator{}, Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	const runs = 24
	errorsChannel := make(chan error, runs)
	for i := range runs {
		go func() {
			input := fmt.Sprintf("input-%d", i)
			result, runErr := a.Run(context.Background(), agent.RunRequest{Input: userMessage(input)}, nil)
			if runErr != nil {
				errorsChannel <- runErr
				return
			}
			if got := textOf(result.Dialog[len(result.Dialog)-1]); got != "reply:"+input {
				errorsChannel <- fmt.Errorf("reply = %q", got)
				return
			}
			errorsChannel <- nil
		}()
	}
	for range runs {
		if err := <-errorsChannel; err != nil {
			t.Fatal(err)
		}
	}
}

type blockingGenerator struct {
	started chan struct{}
}

func (g *blockingGenerator) Generate(ctx context.Context, _ gai.GenerationRequest) (gai.Response, error) {
	close(g.started)
	<-ctx.Done()
	return gai.Response{}, ctx.Err()
}

type echoGenerator struct{}

func (echoGenerator) Generate(_ context.Context, request gai.GenerationRequest) (gai.Response, error) {
	input := textOf(request.Dialog[len(request.Dialog)-1])
	return textResponse("reply:"+input, gai.EndTurn, nil), nil
}

func TestCallOnlyGenerationChangesResetBeforeNextRound(t *testing.T) {
	call := mustToolCallBlock(t, "call-1", "echo", map[string]any{})
	originalOptions := gai.GenerationOptions{"base": true}
	generator := agenttest.NewScriptedGenerator(
		agenttest.GenerateStep{
			Check: func(request gai.GenerationRequest) error {
				if request.Model != "temporary-model" || textOf(request.Instructions) != "temporary instructions" {
					return fmt.Errorf("temporary request = %#v", request)
				}
				if request.Options["temporary"] != true || len(request.Tools) != 1 {
					return fmt.Errorf("temporary options/tools = %#v/%#v", request.Options, request.Tools)
				}
				if got := dialogTexts(request.Dialog); !reflect.DeepEqual(got, []string{"question", "temporary context"}) {
					return fmt.Errorf("temporary dialog = %v", got)
				}
				return nil
			},
			Response: toolResponse(call),
		},
		agenttest.GenerateStep{
			Check: func(request gai.GenerationRequest) error {
				if request.Model != "base-model" || textOf(request.Instructions) != "base instructions" {
					return fmt.Errorf("base request = %#v", request)
				}
				if request.Options["base"] != true || request.Options["temporary"] != nil {
					return fmt.Errorf("base options = %#v", request.Options)
				}
				if len(request.Tools) != 1 {
					return fmt.Errorf("tools = %#v", request.Tools)
				}
				if got := dialogTexts(request.Dialog); !reflect.DeepEqual(got, []string{"question", "echo", "ok"}) {
					return fmt.Errorf("base dialog = %v", got)
				}
				return nil
			},
			Response: textResponse("done", gai.EndTurn, nil),
		},
	)
	a, err := agent.New(agent.Config{
		Generator:    generator,
		Model:        "base-model",
		Instructions: gai.SystemMessage(gai.TextBlock("base instructions")),
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "echo"},
			Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
				return gai.ToolResultMessage("", gai.TextBlock("ok")), nil
			}),
		}},
		BeforeGeneration: agent.BeforeGenerationFunc(func(_ context.Context, request agent.BeforeGenerationRequest) (agent.BeforeGenerationDecision, error) {
			if request.Generation != 0 {
				return agent.BeforeGenerationDecision{Request: request.Request}, nil
			}
			changed := request.Request
			changed.Model = "temporary-model"
			changed.Instructions = gai.SystemMessage(gai.TextBlock("temporary instructions"))
			changed.Dialog = append(gai.Dialog(nil), changed.Dialog...)
			changed.Dialog = append(changed.Dialog, userMessage("temporary context"))
			changed.Options = gai.GenerationOptions{"temporary": true}
			return agent.BeforeGenerationDecision{Request: changed}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}

	result, err := a.Run(context.Background(), agent.RunRequest{
		Input:   userMessage("question"),
		Options: originalOptions,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := dialogTexts(result.Dialog); !reflect.DeepEqual(got, []string{"question", "echo", "ok", "done"}) {
		t.Fatalf("result dialog = %v", got)
	}
	if !reflect.DeepEqual(originalOptions, gai.GenerationOptions{"base": true}) {
		t.Fatalf("original options = %#v", originalOptions)
	}
}

func TestCandidateFieldsCanBeChangedByGenerationHook(t *testing.T) {
	generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{
		Response: textResponse("provider", gai.EndTurn, gai.Metadata{gai.UsageMetricInputTokens: 2}),
	})
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		AfterGeneration: agent.AfterGenerationFunc(func(_ context.Context, request agent.AfterGenerationRequest) (gai.Response, error) {
			changed := request.Response
			changed.Candidates = append([]gai.Message(nil), changed.Candidates...)
			changed.Candidates[0].Blocks = append([]gai.Block(nil), changed.Candidates[0].Blocks...)
			changed.Candidates[0].Blocks[0] = gai.TextBlock("changed")
			changed.Candidates[0].ExtraFields = map[string]any{"reviewed": true}
			changed.ExtraFields = map[string]any{"hook": "after"}
			return changed, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("hello")}, observer)
	if err != nil {
		t.Fatal(err)
	}
	if got := textOf(result.Dialog[len(result.Dialog)-1]); got != "changed" {
		t.Fatalf("assistant text = %q", got)
	}
	completed := observer.Events()[2].Payload.(agent.GenerationCompletedEvent)
	if completed.Response.ExtraFields["hook"] != "after" || completed.Response.Candidates[0].ExtraFields["reviewed"] != true {
		t.Fatalf("completed response = %#v", completed.Response)
	}
	if got, _ := gai.InputTokens(result.Usage); got != 2 {
		t.Fatalf("input tokens = %d", got)
	}
}

func TestConflictingToolDecisionFailsBeforeStartedEvent(t *testing.T) {
	call := mustToolCallBlock(t, "call-1", "echo", map[string]any{})
	generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{Response: toolResponse(call)})
	handlerCalls := 0
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "echo"},
			Handler:    countingHandler(&handlerCalls),
		}},
		BeforeTool: agent.BeforeToolFunc(func(context.Context, agent.BeforeToolRequest) (agent.BeforeToolDecision, error) {
			return agent.BeforeToolDecision{Reject: true, Parameters: map[string]any{}}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("go")}, observer)
	if err == nil {
		t.Fatal("expected conflicting decision error")
	}
	if handlerCalls != 0 || len(result.Dialog) != 2 {
		t.Fatalf("handler calls = %d, dialog = %#v", handlerCalls, result.Dialog)
	}
	assertEventKinds(t, observer.Events(),
		agent.EventKindRunStarted,
		agent.EventKindGenerationStarted,
		agent.EventKindGenerationCompleted,
		agent.EventKindRunFailed,
	)
	if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseBeforeToolHook {
		t.Fatalf("phase = %q", phase)
	}
}

func TestUsageOverflowLeavesPriorTotalUnchanged(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	call := mustToolCallBlock(t, "call-1", "echo", map[string]any{})
	first := toolResponse(call)
	first.UsageMetadata = gai.Metadata{gai.UsageMetricGenerationTokens: maxInt}
	second := textResponse("too much", gai.EndTurn, gai.Metadata{
		gai.UsageMetricInputTokens:      5,
		gai.UsageMetricGenerationTokens: 1,
	})
	generator := agenttest.NewScriptedGenerator(
		agenttest.GenerateStep{Response: first},
		agenttest.GenerateStep{Response: second},
	)
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "echo"},
			Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
				return gai.ToolResultMessage("", gai.TextBlock("ok")), nil
			}),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("go")}, observer)
	if err == nil {
		t.Fatal("expected usage overflow")
	}
	if _, ok := gai.InputTokens(result.Usage); ok {
		t.Fatalf("partially added input usage: %#v", result.Usage)
	}
	if output, ok := gai.OutputTokens(result.Usage); !ok || output != maxInt {
		t.Fatalf("output usage = %d, %v", output, ok)
	}
	if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseValidation {
		t.Fatalf("phase = %q", phase)
	}
}

func TestRunStatusAndContextReachEveryHook(t *testing.T) {
	call := mustToolCallBlock(t, "call-1", "echo", map[string]any{})
	firstResponse := toolResponse(call)
	firstResponse.UsageMetadata = gai.Metadata{gai.UsageMetricInputTokens: 3}
	baseGenerator := agenttest.NewScriptedGenerator(
		agenttest.GenerateStep{Response: firstResponse},
		agenttest.GenerateStep{Response: textResponse("done", gai.EndTurn, nil)},
	)
	key := contextKey{}
	value := &struct{ name string }{name: "same context"}
	generator := contextCheckingGenerator{Generator: baseGenerator, Key: key, Value: value}
	statuses := make(map[string]agent.RunStatus)
	checkContext := func(ctx context.Context) error {
		if ctx.Value(key) != value {
			return errors.New("context value did not propagate")
		}
		return nil
	}
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "echo"},
			Handler: agent.ToolHandlerFunc(func(ctx context.Context, _ agent.ToolRequest) (gai.Message, error) {
				if err := checkContext(ctx); err != nil {
					return gai.Message{}, err
				}
				return gai.ToolResultMessage("", gai.TextBlock("ok")), nil
			}),
		}},
		PrepareDialog: agent.PrepareDialogFunc(func(ctx context.Context, request agent.PrepareDialogRequest) (agent.PrepareDialogDecision, error) {
			if err := checkContext(ctx); err != nil {
				return agent.PrepareDialogDecision{}, err
			}
			statuses[fmt.Sprintf("prepare-%d", request.Generation)] = request.Status
			return agent.PrepareDialogDecision{}, nil
		}),
		BeforeGeneration: agent.BeforeGenerationFunc(func(ctx context.Context, request agent.BeforeGenerationRequest) (agent.BeforeGenerationDecision, error) {
			if err := checkContext(ctx); err != nil {
				return agent.BeforeGenerationDecision{}, err
			}
			statuses[fmt.Sprintf("before-generation-%d", request.Generation)] = request.Status
			return agent.BeforeGenerationDecision{Request: request.Request}, nil
		}),
		AfterGeneration: agent.AfterGenerationFunc(func(ctx context.Context, request agent.AfterGenerationRequest) (gai.Response, error) {
			if err := checkContext(ctx); err != nil {
				return gai.Response{}, err
			}
			statuses[fmt.Sprintf("after-generation-%d", request.Generation)] = request.Status
			return request.Response, nil
		}),
		BeforeTool: agent.BeforeToolFunc(func(ctx context.Context, request agent.BeforeToolRequest) (agent.BeforeToolDecision, error) {
			if err := checkContext(ctx); err != nil {
				return agent.BeforeToolDecision{}, err
			}
			statuses["before-tool"] = request.Status
			return agent.BeforeToolDecision{}, nil
		}),
		AfterTool: agent.AfterToolFunc(func(ctx context.Context, request agent.AfterToolRequest) (agent.AfterToolDecision, error) {
			if err := checkContext(ctx); err != nil {
				return agent.AfterToolDecision{}, err
			}
			statuses["after-tool"] = request.Status
			return agent.AfterToolDecision{Result: request.Result}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.WithValue(context.Background(), key, value)
	observer := agent.ObserverFunc(func(observed context.Context, _ agent.Event) error {
		return checkContext(observed)
	})

	if _, err := a.Run(ctx, agent.RunRequest{Input: userMessage("go")}, observer); err != nil {
		t.Fatal(err)
	}
	assertStatusCounts(t, statuses["prepare-0"], 0, 0)
	assertStatusCounts(t, statuses["before-generation-0"], 0, 0)
	assertStatusCounts(t, statuses["after-generation-0"], 0, 0)
	assertStatusCounts(t, statuses["before-tool"], 1, 0)
	assertStatusCounts(t, statuses["after-tool"], 1, 0)
	assertStatusCounts(t, statuses["prepare-1"], 1, 1)
	assertStatusCounts(t, statuses["before-generation-1"], 1, 1)
	assertStatusCounts(t, statuses["after-generation-1"], 1, 1)
	if got, _ := gai.InputTokens(statuses["before-tool"].Usage); got != 3 {
		t.Fatalf("before-tool input tokens = %d", got)
	}
}

func TestHookErrorsReportTheirOwningPhase(t *testing.T) {
	marker := errors.New("hook failed")
	tests := []struct {
		name  string
		phase agent.RunPhase
		build func(*agenttest.ScriptedGenerator) agent.Config
		steps []agenttest.GenerateStep
	}{
		{
			name:  "prepare dialog",
			phase: agent.RunPhasePrepareDialogHook,
			build: func(generator *agenttest.ScriptedGenerator) agent.Config {
				return agent.Config{Generator: generator, Model: "model", PrepareDialog: agent.PrepareDialogFunc(func(context.Context, agent.PrepareDialogRequest) (agent.PrepareDialogDecision, error) {
					return agent.PrepareDialogDecision{}, marker
				})}
			},
		},
		{
			name:  "before generation",
			phase: agent.RunPhaseBeforeGenerationHook,
			build: func(generator *agenttest.ScriptedGenerator) agent.Config {
				return agent.Config{Generator: generator, Model: "model", BeforeGeneration: agent.BeforeGenerationFunc(func(context.Context, agent.BeforeGenerationRequest) (agent.BeforeGenerationDecision, error) {
					return agent.BeforeGenerationDecision{}, marker
				})}
			},
		},
		{
			name:  "after generation",
			phase: agent.RunPhaseAfterGenerationHook,
			steps: []agenttest.GenerateStep{{Response: textResponse("done", gai.EndTurn, nil)}},
			build: func(generator *agenttest.ScriptedGenerator) agent.Config {
				return agent.Config{Generator: generator, Model: "model", AfterGeneration: agent.AfterGenerationFunc(func(context.Context, agent.AfterGenerationRequest) (gai.Response, error) {
					return gai.Response{}, marker
				})}
			},
		},
		{
			name:  "before tool",
			phase: agent.RunPhaseBeforeToolHook,
			steps: []agenttest.GenerateStep{{Response: toolResponse(mustToolCallBlock(t, "before", "tool", map[string]any{}))}},
			build: func(generator *agenttest.ScriptedGenerator) agent.Config {
				return hookErrorToolConfig(generator, marker, true)
			},
		},
		{
			name:  "after tool",
			phase: agent.RunPhaseAfterToolHook,
			steps: []agenttest.GenerateStep{{Response: toolResponse(mustToolCallBlock(t, "after", "tool", map[string]any{}))}},
			build: func(generator *agenttest.ScriptedGenerator) agent.Config {
				return hookErrorToolConfig(generator, marker, false)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator := agenttest.NewScriptedGenerator(test.steps...)
			a, err := agent.New(test.build(generator))
			if err != nil {
				t.Fatal(err)
			}
			observer := agenttest.NewRecordingObserver()

			_, err = a.Run(context.Background(), agent.RunRequest{Input: userMessage("go")}, observer)
			if !errors.Is(err, marker) {
				t.Fatalf("error = %v", err)
			}
			if phase := lastFailure(t, observer.Events()).Phase; phase != test.phase {
				t.Fatalf("phase = %q, want %q", phase, test.phase)
			}
		})
	}
}

func TestFixedToolSetRejectsGenerationHookChanges(t *testing.T) {
	generator := agenttest.NewScriptedGenerator()
	a, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "test-model",
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "fixed"},
			Handler:    countingHandler(new(int)),
		}},
		BeforeGeneration: agent.BeforeGenerationFunc(func(_ context.Context, request agent.BeforeGenerationRequest) (agent.BeforeGenerationDecision, error) {
			request.Request.Tools = nil
			return agent.BeforeGenerationDecision{Request: request.Request}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := agenttest.NewRecordingObserver()

	result, err := a.Run(context.Background(), agent.RunRequest{Input: userMessage("go")}, observer)
	if err == nil {
		t.Fatal("expected fixed-tool validation error")
	}
	if len(result.Dialog) != 1 || len(generator.Requests()) != 0 {
		t.Fatalf("result = %#v, calls = %d", result, len(generator.Requests()))
	}
	if phase := lastFailure(t, observer.Events()).Phase; phase != agent.RunPhaseBeforeGenerationHook {
		t.Fatalf("phase = %q", phase)
	}
}

func hookErrorToolConfig(generator gai.Generator, marker error, before bool) agent.Config {
	config := agent.Config{
		Generator: generator,
		Model:     "model",
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "tool"},
			Handler: agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
				return gai.ToolResultMessage("", gai.TextBlock("ok")), nil
			}),
		}},
	}
	if before {
		config.BeforeTool = agent.BeforeToolFunc(func(context.Context, agent.BeforeToolRequest) (agent.BeforeToolDecision, error) {
			return agent.BeforeToolDecision{}, marker
		})
	} else {
		config.AfterTool = agent.AfterToolFunc(func(context.Context, agent.AfterToolRequest) (agent.AfterToolDecision, error) {
			return agent.AfterToolDecision{}, marker
		})
	}
	return config
}

type contextKey struct{}

type contextCheckingGenerator struct {
	Generator gai.Generator
	Key       contextKey
	Value     any
}

func (g contextCheckingGenerator) Generate(ctx context.Context, request gai.GenerationRequest) (gai.Response, error) {
	if ctx.Value(g.Key) != g.Value {
		return gai.Response{}, errors.New("context value did not reach generator")
	}
	return g.Generator.Generate(ctx, request)
}

func assertStatusCounts(t *testing.T, status agent.RunStatus, generations, tools uint) {
	t.Helper()
	if status.GenerationCalls != generations || status.ToolExecutions != tools {
		t.Fatalf("status = %#v, want generations=%d tools=%d", status, generations, tools)
	}
}

func userMessage(text string) gai.Message {
	return gai.Message{Role: gai.User, Blocks: []gai.Block{gai.TextBlock(text)}}
}

func assistantMessage(text string) gai.Message {
	return gai.Message{Role: gai.Assistant, Blocks: []gai.Block{gai.TextBlock(text)}}
}

func textResponse(text string, reason gai.FinishReason, usage gai.Metadata) gai.Response {
	return gai.Response{
		Candidates:    []gai.Message{assistantMessage(text)},
		FinishReason:  reason,
		UsageMetadata: usage,
	}
}

func toolResponse(blocks ...gai.Block) gai.Response {
	return gai.Response{
		Candidates:   []gai.Message{{Role: gai.Assistant, Blocks: blocks}},
		FinishReason: gai.ToolUse,
	}
}

func mustToolCallBlock(t *testing.T, id, name string, parameters map[string]any) gai.Block {
	t.Helper()
	block, err := gai.ToolCallBlock(id, name, parameters)
	if err != nil {
		t.Fatal(err)
	}
	return block
}

func textOf(message gai.Message) string {
	if len(message.Blocks) == 0 || message.Blocks[0].Content == nil {
		return ""
	}
	return message.Blocks[0].Content.String()
}

func dialogTexts(dialog gai.Dialog) []string {
	texts := make([]string, len(dialog))
	for i := range dialog {
		if len(dialog[i].Blocks) > 0 && dialog[i].Blocks[0].BlockType == gai.ToolCall {
			var call gai.ToolCallInput
			if err := json.Unmarshal([]byte(dialog[i].Blocks[0].Content.String()), &call); err == nil {
				texts[i] = call.Name
				continue
			}
		}
		texts[i] = textOf(dialog[i])
	}
	return texts
}

func countingHandler(calls *int) agent.ToolHandler {
	return agent.ToolHandlerFunc(func(context.Context, agent.ToolRequest) (gai.Message, error) {
		(*calls)++
		return gai.ToolResultMessage("", gai.TextBlock("unexpected")), nil
	})
}

func assertEventKinds(t *testing.T, events []agent.Event, want ...agent.EventKind) {
	t.Helper()
	if len(events) != len(want) {
		t.Fatalf("event count = %d, want %d: %#v", len(events), len(want), events)
	}
	for i := range events {
		if events[i].Sequence != uint64(i) {
			t.Fatalf("event %d sequence = %d", i, events[i].Sequence)
		}
		if events[i].Kind() != want[i] {
			t.Fatalf("event %d kind = %q, want %q", i, events[i].Kind(), want[i])
		}
	}
}

func lastFailure(t *testing.T, events []agent.Event) agent.RunFailedEvent {
	t.Helper()
	if len(events) == 0 {
		t.Fatal("no events")
	}
	failed, ok := events[len(events)-1].Payload.(agent.RunFailedEvent)
	if !ok {
		t.Fatalf("last event = %T", events[len(events)-1].Payload)
	}
	return failed
}
