package gai_test

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/spachava753/gai"
)

type exampleGenerator struct{}

func (exampleGenerator) Generate(_ context.Context, request gai.GenerationRequest) (gai.Response, error) {
	return gai.Response{
		Candidates: []gai.Message{{
			Role:   gai.Assistant,
			Blocks: []gai.Block{gai.TextBlock("Echo: " + request.Dialog[0].Blocks[0].Content.String())},
		}},
		FinishReason: gai.EndTurn,
	}, nil
}

func ExampleGenerator() {
	var generator gai.Generator = exampleGenerator{}
	response, err := generator.Generate(context.Background(), gai.GenerationRequest{
		Model:        "example-model",
		Instructions: gai.SystemMessage(gai.TextBlock("Answer briefly.")),
		Dialog: gai.Dialog{{
			Role:   gai.User,
			Blocks: []gai.Block{gai.TextBlock("Hello")},
		}},
		Options: gai.NewGenerationOptions(gai.WithTemperature(0.2)),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(response.Candidates[0].Blocks[0].Content)
	// Output: Echo: Hello
}

func ExampleNewGenerationOptions() {
	options := gai.NewGenerationOptions(
		gai.WithTemperature(0.3),
		gai.WithToolChoice(gai.ToolChoiceAuto),
		gai.WithStopSequences("END"),
	)

	fmt.Println(options[gai.GenerationOptionTemperature])
	fmt.Println(options[gai.GenerationOptionToolChoice])
	fmt.Println(options[gai.GenerationOptionStopSequences])
	// Output:
	// 0.3
	// auto
	// [END]
}

func ExampleTool() {
	type weatherInput struct {
		City string `json:"city"`
	}

	schema, err := gai.GenerateSchema[weatherInput]()
	if err != nil {
		panic(err)
	}
	tool := gai.Tool{
		Name:        "get_weather",
		Description: "Get the weather for a city.",
		InputSchema: schema,
	}

	fmt.Println(tool.Name)
	fmt.Println(tool.InputSchema.Type)
	fmt.Println(tool.InputSchema.Required)
	// Output:
	// get_weather
	// object
	// [city]
}

func ExampleToolCallBlock() {
	block, err := gai.ToolCallBlock("call_123", "get_weather", map[string]any{
		"city": "Paris",
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(block.ID)
	fmt.Println(block.BlockType)
	fmt.Println(block.Content)
	// Output:
	// call_123
	// tool_call
	// {"name":"get_weather","parameters":{"city":"Paris"}}
}

func ExamplePDFBlock() {
	block := gai.PDFBlock([]byte("doc"), "paper.pdf")

	fmt.Println(block.MimeType)
	fmt.Println(block.Content)
	fmt.Println(block.ExtraFields[gai.BlockFieldFilenameKey])
	// Output:
	// application/pdf
	// ZG9j
	// paper.pdf
}

type exampleStreamingGenerator struct{}

func (exampleStreamingGenerator) Stream(_ context.Context, _ gai.GenerationRequest) iter.Seq[gai.StreamChunk] {
	return func(yield func(gai.StreamChunk) bool) {
		if !yield(gai.StreamChunk{Block: gai.TextBlock("Hello, ")}) {
			return
		}
		yield(gai.StreamChunk{Block: gai.TextBlock("world!")})
	}
}

func ExampleStreamingAdapter() {
	adapter := gai.StreamingAdapter{S: exampleStreamingGenerator{}}
	response, err := adapter.Generate(context.Background(), gai.GenerationRequest{})
	if err != nil {
		panic(err)
	}

	fmt.Println(response.Candidates[0].Blocks[0].Content)
	// Output: Hello, world!
}

func ExampleApiErr() {
	err := error(&gai.ApiErr{
		Provider:   gai.ProviderOpenRouter,
		Kind:       gai.APIErrorKindRateLimit,
		StatusCode: 429,
		Message:    "too many requests",
	})

	var apiErr *gai.ApiErr
	fmt.Println(errors.As(err, &apiErr))
	fmt.Println(apiErr.Retryable())
	// Output:
	// true
	// true
}
