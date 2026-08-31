package agent_test

import (
	"context"
	"fmt"

	"github.com/spachava753/gai"
	"github.com/spachava753/gai/agent"
	"github.com/spachava753/gai/agent/agenttest"
)

type weatherArguments struct {
	City string `json:"city"`
}

func ExampleAgent_Run() {
	generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{
		Response: gai.Response{
			Candidates: []gai.Message{{
				Role:   gai.Assistant,
				Blocks: []gai.Block{gai.TextBlock("Hello from the model.")},
			}},
			FinishReason: gai.EndTurn,
		},
	})
	chatAgent, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "provider-model",
	})
	if err != nil {
		panic(err)
	}

	result, err := chatAgent.Run(context.Background(), agent.RunRequest{
		Input: gai.Message{
			Role:   gai.User,
			Blocks: []gai.Block{gai.TextBlock("Hello")},
		},
	}, nil)
	if err != nil {
		panic(err)
	}

	fmt.Println(result.Dialog[len(result.Dialog)-1].Blocks[0].Content.String())
	fmt.Println(result.StopReason)
	// Output:
	// Hello from the model.
	// model
}

func ExampleBeforeGenerationFunc() {
	usedModel := ""
	generator := agenttest.NewScriptedGenerator(agenttest.GenerateStep{
		Check: func(request gai.GenerationRequest) error {
			usedModel = request.Model
			return nil
		},
		Response: gai.Response{
			Candidates:   []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{gai.TextBlock("done")}}},
			FinishReason: gai.EndTurn,
		},
	})
	runner, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "default-model",
		BeforeGeneration: agent.BeforeGenerationFunc(func(_ context.Context, request agent.BeforeGenerationRequest) (agent.BeforeGenerationDecision, error) {
			request.Request.Model = "model-for-this-call"
			return agent.BeforeGenerationDecision{Request: request.Request}, nil
		}),
	})
	if err != nil {
		panic(err)
	}

	_, err = runner.Run(context.Background(), agent.RunRequest{
		Input: gai.Message{Role: gai.User, Blocks: []gai.Block{gai.TextBlock("run")}},
	}, nil)
	if err != nil {
		panic(err)
	}

	fmt.Println(usedModel)
	// Output:
	// model-for-this-call
}

func ExampleToolHandlerFunc() {
	schema, err := gai.GenerateSchema[weatherArguments]()
	if err != nil {
		panic(err)
	}
	call, err := gai.ToolCallBlock("call-1", "weather", map[string]any{"city": "Paris"})
	if err != nil {
		panic(err)
	}
	generator := agenttest.NewScriptedGenerator(
		agenttest.GenerateStep{Response: gai.Response{
			Candidates:   []gai.Message{{Role: gai.Assistant, Blocks: []gai.Block{call}}},
			FinishReason: gai.ToolUse,
		}},
		agenttest.GenerateStep{Response: gai.Response{
			Candidates: []gai.Message{{
				Role:   gai.Assistant,
				Blocks: []gai.Block{gai.TextBlock("It is 18 C in Paris.")},
			}},
			FinishReason: gai.EndTurn,
		}},
	)
	weatherAgent, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "provider-model",
		Tools: []agent.Tool{{
			Definition: gai.Tool{Name: "weather", Description: "Get current weather", InputSchema: schema},
			Handler: agent.ToolHandlerFunc(func(_ context.Context, request agent.ToolRequest) (gai.Message, error) {
				city := request.Call.Parameters["city"].(string)
				return gai.ToolResultMessage(request.Block.ID, gai.TextBlock("18 C in "+city)), nil
			}),
		}},
	})
	if err != nil {
		panic(err)
	}

	result, err := weatherAgent.Run(context.Background(), agent.RunRequest{
		Input: gai.Message{Role: gai.User, Blocks: []gai.Block{gai.TextBlock("What is the weather?")}},
	}, nil)
	if err != nil {
		panic(err)
	}

	fmt.Println(result.Dialog[len(result.Dialog)-1].Blocks[0].Content.String())
	// Output:
	// It is 18 C in Paris.
}
