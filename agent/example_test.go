package agent_test

import (
	"context"
	"errors"
	"fmt"
	"maps"

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

	result, err := chatAgent.Run(context.Background(), agent.RunRequest{Dialog: gai.Dialog{{
		Role:   gai.User,
		Blocks: []gai.Block{gai.TextBlock("Hello")},
	}},
	})
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
			next := request.Request // Scalar changes affect only this struct copy.
			next.Model = "model-for-this-call"
			next.Options = maps.Clone(request.Request.Options) // Nested values still share references.
			if next.Options == nil {
				next.Options = make(gai.GenerationOptions)
			}
			for key, value := range gai.NewGenerationOptions(gai.WithTemperature(0.2)) {
				next.Options[key] = value
			}
			return agent.BeforeGenerationDecision{Request: &next}, nil
		}),
	})
	if err != nil {
		panic(err)
	}

	_, err = runner.Run(context.Background(), agent.RunRequest{Dialog: gai.Dialog{{Role: gai.User, Blocks: []gai.Block{gai.TextBlock("run")}}}})
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

	result, err := weatherAgent.Run(context.Background(), agent.RunRequest{Dialog: gai.Dialog{{Role: gai.User, Blocks: []gai.Block{gai.TextBlock("What is the weather?")}}}})
	if err != nil {
		panic(err)
	}

	fmt.Println(result.Dialog[len(result.Dialog)-1].Blocks[0].Content.String())
	// Output:
	// It is 18 C in Paris.
}

func ExampleAgent_Run_continue() {
	call, err := gai.ToolCallBlock("payment-1", "pay", map[string]any{"invoice": "42"})
	if err != nil {
		panic(err)
	}
	dialog := gai.Dialog{
		{Role: gai.User, Blocks: []gai.Block{gai.TextBlock("Pay invoice 42")}},
		{Role: gai.Assistant, Blocks: []gai.Block{call}},
	}
	generator := agenttest.NewScriptedStreamingGenerator(agenttest.StreamStep{
		Chunks: []gai.StreamChunk{{Block: gai.TextBlock("Payment confirmed.")}},
	})
	runner, err := agent.New(agent.Config{
		Generator: generator,
		Model:     "provider-model",
		GenerationChunk: agent.GenerationChunkFunc(func(_ context.Context, event agent.GenerationChunkEvent) error {
			if event.Chunk.Block.Content != nil {
				fmt.Println(event.Chunk.Block.Content.String())
			}
			return nil
		}),
	})
	if err != nil {
		panic(err)
	}
	_, err = runner.Run(context.Background(), agent.RunRequest{Dialog: dialog})
	var failure *agent.RunError
	if errors.As(err, &failure) {
		fmt.Println("unfinished dialog:", failure.Phase)
	}

	// Application reconciliation has confirmed the payment with its service.
	// Supplying that receipt permits generation without executing pay again.
	dialog = append(dialog, gai.ToolResultMessage("payment-1", gai.TextBlock("receipt-42")))
	result, err := runner.Run(context.Background(), agent.RunRequest{Dialog: dialog})
	if err != nil {
		panic(err)
	}
	fmt.Println("messages:", len(result.Dialog))
	// Output:
	// unfinished dialog: validation
	// Payment confirmed.
	// messages: 4
}
