package main

import (
	"log"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"

	"github.com/spachava753/gai"
)

func runWithPool(client gai.Client, testRows []RowType) error {
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

	startTime := time.Now()

	rowChan := make(chan RowType)
	resultChan := make(chan bool)
	wg := new(sync.WaitGroup)
	wg.Add(PoolSize)
	for i := 0; i < PoolSize; i++ {
		go func() {
			defer wg.Done()
			for row := range rowChan {
				respAnswer, err := solveProblem(client, messages, DefaultModel, row)
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
