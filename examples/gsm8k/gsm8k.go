package main

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/sashabaranov/go-openai"

	"github.com/spachava753/gai"
)

const PoolSize = 80

type RowType struct {
	Question string `parquet:"question"`
	Answer   string `parquet:"answer"`
}

func solveProblem(client gai.Client, messages []openai.ChatCompletionMessage, row RowType) (string, error) {
	completion := make([]openai.ChatCompletionMessage, len(messages), len(messages)+2)
	copy(completion, messages)

	completion = append(
		completion, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: row.Question,
		},
	)

	req := openai.ChatCompletionRequest{
		// Model: "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
		Model:       "mistralai/Mixtral-8x7B-Instruct-v0.1",
		Messages:    completion,
		MaxTokens:   500,
		Temperature: 0,
	}

	resp, respErr := client.ChatCompletion(
		context.TODO(), req, new(gai.TogetherRateLimiter),
	)
	if respErr != nil {
		return "", fmt.Errorf("could not get response for request %+v: %w", req, respErr)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("not choices present in response: %+v", resp)
	}

	return resp.Choices[0].Message.Content, nil
}

func run() error {
	endpoint := os.Getenv("GAI_ENDPOINT")
	if endpoint == "" {
		return fmt.Errorf("endpoint is blank")
	}
	if _, parseErr := url.Parse(endpoint); parseErr != nil {
		return fmt.Errorf("endpoint is not a valid url: %w", parseErr)
	}
	apiKey := os.Getenv("GAI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("apiKey is blank")
	}
	client := gai.NewClient(endpoint, apiKey)

	testRows, readErr := parquet.ReadFile[RowType]("./test.parquet")
	if readErr != nil {
		return fmt.Errorf("could not read test.parquet file: %w", readErr)
	}

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are an expert assistant at solving math word problems.",
		},
	}

	for _, row := range testRows[:8] {
		messages = append(
			messages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: row.Question,
			},
		)
		messages = append(
			messages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: row.Answer,
			},
		)
	}

	evalRows := testRows[8:]

	log.Printf("total number of problems: %d", len(evalRows))

	// evalRows = evalRows[:100]

	startTime := time.Now()

	rowChan := make(chan RowType)
	resultChan := make(chan bool)
	wg := new(sync.WaitGroup)
	wg.Add(PoolSize)
	for i := 0; i < PoolSize; i++ {
		go func() {
			defer wg.Done()
			for row := range rowChan {
				respAnswer, err := solveProblem(client, messages, row)
				if err != nil {
					log.Println(err)
					continue
				}
				splitResp := strings.Split(respAnswer, "####")
				if len(splitResp) != 2 {
					log.Printf("unexpected response format: %s", respAnswer)
					continue
				}
				answerStr := strings.TrimSpace(splitResp[1])
				answer, parseErr := strconv.ParseFloat(answerStr, 64)
				if parseErr != nil {
					log.Printf("could not parse response: %s", answerStr)
					continue
				}

				splitAnswer := strings.Split(row.Answer, "####")
				if len(splitAnswer) != 2 {
					log.Printf("unexpected format for row: %+v", row)
					continue
				}
				groundTruthStr := strings.TrimSpace(splitAnswer[1])
				groundTruth, parseErr := strconv.ParseFloat(groundTruthStr, 64)
				if parseErr != nil {
					log.Printf("could not parse ground truth: %s", groundTruthStr)
					continue
				}

				correct := answer == groundTruth

				log.Printf("ANSWER: %f; REAL: %f; CORRECT: %t", answer, groundTruth, correct)
				resultChan <- correct
			}
		}()
	}

	correctResp := 0
	resultWg := new(sync.WaitGroup)
	resultWg.Add(1)
	go func() {
		defer resultWg.Done()
		for r := range resultChan {
			if r {
				correctResp += 1
			}
		}
	}()

	go func() {
		for _, row := range evalRows {
			rowChan <- row
		}
		close(rowChan)
	}()

	wg.Wait()

	close(resultChan)

	resultWg.Wait()

	log.Printf(
		"%d / %d correct, or %f correct; took %f seconds",
		correctResp,
		len(evalRows),
		float64(correctResp)/float64(len(evalRows)),
		time.Since(startTime).Seconds(),
	)

	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Println("error while executing:", err)
		os.Exit(1)
	}
}
