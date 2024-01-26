package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"

	"github.com/spachava753/gai"
)

type Calculator struct {
	Operation   string  `json:"operation,omitempty"`
	LeftNumber  float64 `json:"left_number,omitempty"`
	RightNumber float64 `json:"right_number,omitempty"`
}

type ToolCall struct {
	Calculator *Calculator `json:"calculator"`
}

func runWithToolsAndPool(client gai.Client, testRows []RowType) error {
	modelName := "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"

	toolCallSchema := jsonschema.Definition{
		Type:        jsonschema.Object,
		Description: "Tools available to call",
		Properties: map[string]jsonschema.Definition{
			"calculator": {
				Type:        jsonschema.Object,
				Description: "A simple arithmetic calculator",
				Properties: map[string]jsonschema.Definition{
					"operation": {
						Type:        jsonschema.String,
						Description: "The operation for the simple arithmetic",
						Enum: []string{
							"add",
							"subtract",
							"multiply",
							"divide",
						},
					},
					"left_number": {
						Type:        jsonschema.Number,
						Description: "The left hand side number for the operation",
					},
					"right_number": {
						Type:        jsonschema.Number,
						Description: "The right hand side number for the operation",
					},
				},
				Required: []string{
					"operation",
					"left_number",
					"right_number",
				},
			},
		},
	}

	toolCallSchemaJson, marshallErr := json.Marshal(toolCallSchema)
	if marshallErr != nil {
		return marshallErr
	}

	messages := []openai.ChatCompletionMessage{
		{
			Role: openai.ChatMessageRoleSystem,
			Content: fmt.Sprintf(
				`You are an expert assistant at solving math word problems. You are very deliberate and write out all of your steps. You have tools available to you to call in your steps as specified in this JSON schema:
%s

When calling a tool, always put the tool in a mardown code blocked, and the fenced label should be "tool_call". Make sure to call only one tool at a time. Always follow the above JSON schema when calling a tool. The user will provide the output of the tool call, ALWAYS wait for the output to be given to you. Do not generate mulitple tool calls and wait for the generated tool call result. When you have the final answer to the word problem, always write your final answer as a number in the format of "#### <number>"`,
				toolCallSchemaJson,
			),
		},
	}

	messages = append(
		messages,
		// Example 1
		openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: `Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?`,
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Let's first calculate how many duck eggs Janet uses every day. Janet eats three for breakfast and uses four to bake muffins for her friends.
%stool_call
{
  "calculator": {
    "operation": "add",
	"left_number": 3,
    "right_number": 4
  }
}
%s
`, "```", "```",
			),
		}, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf(`7`),
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Janet uses 7 duck eggs every day. Next, lets calculate how many duck eggs Janet sells every day. She starts with 16 duck eggs every day, and uses 7.
%stool_call
{
  "calculator": {
    "operation": "subtract",
	"left_number": 16,
    "right_number": 7
  }
}
%s
`, "```", "```",
			),
		}, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(`9`),
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Janet has 9 duck eggs every day to sell. Next, lets calculate how much Janet earns at the farmer's market. We need to multiply the price per duck egg, which is $2, with how many she has left over every day, which is 9.
%stool_call
{
  "calculator": {
    "operation": "multiply",
	"left_number": 9,
    "right_number": 2
  }
}
%s
`, "```", "```",
			),
		}, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(`18`),
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Janet makes $18 every day.
#### 18`,
			),
		},
		// Example 2
		openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: `A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?`,
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Let's first calculate how many bolts of white fiber is needed while making a robe. There are 2 bolts of blue fiber and there is half as many bolts of white fiber. We need to divide 2 bolts of blue fiber by 2.
%stool_call
{
  "calculator": {
    "operation": "divide",
	"left_number": 2,
    "right_number": 2
  }
}
%s
`, "```", "```",
			),
		}, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf(`1`),
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Making a robe needs 1 bolt of white fiber. Next let's calculate how many total bolts of fiber is needed to make robe. We have 2 bolts of blue fiber and 1 bolt of white fiber, so we need to add the bolts of blue and white fiber together to get the total number of bolts of fiber needed to make a robe.
%stool_call
{
  "calculator": {
    "operation": "subtract",
	"left_number": 2,
    "right_number": 1
  }
}
%s
`, "```", "```",
			),
		}, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(`3`),
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`It takes a total of 3 bolts of fiber to make a robe.
#### 3`,
			),
		},
		// Example 3
		openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: `Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?`,
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`In order to figure out how much profit Josh made, we need to find out he total cost of the house. Josh bought the house $80,000 and put in $50,000 in repairs, so we need add 80000 and 50000 to get the cost of the house.
%stool_call
{
  "calculator": {
    "operation": "add",
	"left_number": 80000,
    "right_number": 50000
  }
}
%s
`, "```", "```",
			),
		}, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf(`130000`),
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`The house cost Josh a total of $130,000. Next, in order to calculate the profit, we need to find out the house value after repairs. The house value before repairs is $80,000, and after repairs, it increased in value 150%s. This means we need to multiply 80000 by 1.5.
%stool_call
{
  "calculator": {
    "operation": "multiply",
	"left_number": 80000,
    "right_number": 1.5
  }
}
%s
`, "%", "```", "```",
			),
		}, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(`120000`),
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`The house increased in value from $80,000 to $120,000. The total cost of the house for Josh is $130,000. We need to subtract the cost of the house from the increased value, so we need to subtract 130000 from 120000. 
%stool_call
{
  "calculator": {
    "operation": "subtract",
	"left_number": 120000,
    "right_number": 130000
  }
}
%s
`, "```", "```",
			),
		}, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(`−10000`),
		}, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Josh made a profit of negative $10,000. It is impossible to make negative profit, so Josh actually lost $10,000. 
#### -10000`,
			),
		},
	)

	evalRows := testRows[3:]

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
					log.Printf("could not get response for request %+v: %s", req, respErr)
					continue
				}

				if len(resp.Choices) == 0 {
					log.Printf("not choices present in response: %+v", resp)
					continue
				}

				respAnswer := resp.Choices[0].Message.Content

				completion = append(
					completion, openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: respAnswer,
					},
				)

				for {
					if strings.Contains(respAnswer, "```tool_call") {
						toolCallIdxStart := strings.Index(respAnswer, "```tool_call")
						toolCallJson := respAnswer[toolCallIdxStart+len("```tool_call"):]
						toolCallIdxEnd := strings.LastIndex(toolCallJson, "```")
						if toolCallIdxEnd == -1 {
							log.Printf("imporoperly formatted toolcall json: %s", respAnswer)
							break
						}
						toolCallJson = toolCallJson[:toolCallIdxEnd]

						var toolCall ToolCall
						if unmarshallErr := json.Unmarshal([]byte(toolCallJson), &toolCall); unmarshallErr != nil {
							log.Printf("cannot unmarshall json: %s", respAnswer)
							break
						}

						if toolCall.Calculator == nil {
							log.Printf("unknown tool: %s", respAnswer)
							break
						}

						answer, calcErr := calc(
							toolCall.Calculator.Operation, toolCall.Calculator.LeftNumber,
							toolCall.Calculator.RightNumber,
						)

						if calcErr != nil {
							log.Printf("error while calculating: %s, response: %s", calcErr, respAnswer)
							break
						}

						completion = append(
							completion, openai.ChatCompletionMessage{
								Role:    openai.ChatMessageRoleUser,
								Content: fmt.Sprintf("%f", answer),
							},
						)

						req.Messages = completion

						resp, respErr = client.ChatCompletion(
							context.TODO(), req, new(gai.TogetherRateLimiterBackOff),
						)

						if respErr != nil {
							log.Printf("could not get response for request %+v: %s", req, respErr)
							break
						}

						if len(resp.Choices) == 0 {
							log.Printf("not choices present in response: %+v", resp)
							break
						}

						respAnswer = resp.Choices[0].Message.Content

						completion = append(
							completion, openai.ChatCompletionMessage{
								Role:    openai.ChatMessageRoleAssistant,
								Content: respAnswer,
							},
						)

						continue
					}
					if !strings.Contains(respAnswer, "####") {
						log.Printf("Unknown response: %s", respAnswer)
						break
					}

					splitResp := strings.Split(respAnswer, "####")
					if len(splitResp) != 2 {
						log.Printf("unexpected response format: %s", respAnswer)
						break
					}
					answerStr := strings.TrimSpace(splitResp[1])
					answer, parseErr := strconv.ParseFloat(answerStr, 64)
					if parseErr != nil {
						log.Printf("could not parse response: %s", answerStr)
						break
					}

					splitAnswer := strings.Split(row.Answer, "####")
					if len(splitAnswer) != 2 {
						log.Printf("unexpected format for row: %+v", row)
						break
					}
					groundTruthStr := strings.TrimSpace(splitAnswer[1])
					groundTruth, parseErr := strconv.ParseFloat(groundTruthStr, 64)
					if parseErr != nil {
						log.Printf("could not parse ground truth: %s", groundTruthStr)
						break
					}

					correct := answer == groundTruth

					log.Printf("ANSWER: %f; REAL: %f; CORRECT: %t", answer, groundTruth, correct)
					resultChan <- correct
					break
				}

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

func calc(operation string, number float64, number2 float64) (float64, error) {
	switch operation {
	case "add":
		return number + number2, nil
	case "subtract":
		return number - number2, nil
	case "multiply":
		return number * number2, nil
	case "divide":
		return number / number2, nil
	default:
		return 0, fmt.Errorf("unknown operation")
	}
}
