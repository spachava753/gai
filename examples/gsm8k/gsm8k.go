package main

import (
	"context"
	"flag"
	"fmt"
	"net/url"
	"os"

	"github.com/parquet-go/parquet-go"
	"github.com/sashabaranov/go-openai"

	"github.com/spachava753/gai"
)

const PoolSize = 80
const DefaultModel = "mistralai/Mixtral-8x7B-Instruct-v0.1"

type RowType struct {
	Question string `parquet:"question"`
	Answer   string `parquet:"answer"`
}

func solveProblem(client gai.Client, messages []openai.ChatCompletionMessage, modelName string, row RowType) (
	string, error,
) {
	completion := make([]openai.ChatCompletionMessage, len(messages), len(messages)+2)
	copy(completion, messages)

	completion = append(
		completion, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: row.Question,
		},
	)

	req := openai.ChatCompletionRequest{
		Model:       modelName,
		Messages:    completion,
		MaxTokens:   500,
		Temperature: 0,
	}

	resp, respErr := client.ChatCompletion(
		context.TODO(), req, new(gai.TogetherRateLimiterBackOff),
	)
	if respErr != nil {
		return "", fmt.Errorf("could not get response for request %+v: %w", req, respErr)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("not choices present in response: %+v", resp)
	}

	return resp.Choices[0].Message.Content, nil
}

func createClient() (gai.Client, error) {
	endpoint := os.Getenv("GAI_ENDPOINT")
	if endpoint == "" {
		return gai.Client{}, fmt.Errorf("endpoint is blank")
	}
	if _, parseErr := url.Parse(endpoint); parseErr != nil {
		return gai.Client{}, fmt.Errorf("endpoint is not a valid url: %w", parseErr)
	}
	apiKey := os.Getenv("GAI_API_KEY")
	if apiKey == "" {
		return gai.Client{}, fmt.Errorf("apiKey is blank")
	}
	return gai.NewClient(endpoint, apiKey), nil
}

func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) != 1 {
		fmt.Println("please specify a gsm8k benchmark to run")
		os.Exit(1)
	}
	bench := args[0]

	client, clientErr := createClient()
	if clientErr != nil {
		fmt.Println("could not create client")
		os.Exit(1)
	}

	testRows, readErr := parquet.ReadFile[RowType]("./test.parquet")
	if readErr != nil {
		fmt.Printf("could not read test.parquet file: %s\n", readErr)
		os.Exit(1)
	}

	var err error
	switch bench {
	case "simple":
		err = runSimple(client, testRows)
	case "pool":
		err = runWithPool(client, testRows)
	case "tool_call":
		err = runWithToolsAndPool(client, testRows)
	default:
		fmt.Println("please select one program to run: simple, pool or tool_call")
		os.Exit(1)
	}

	if err != nil {
		fmt.Println("error while executing:", err)
		os.Exit(1)
	}
}
