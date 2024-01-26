package gai

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/invopop/jsonschema"
	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type QuestionAnswer struct {
	Data struct {
		Reasoning string `json:"reasoning,omitempty" jsonschema_description:"Reasoning that supports the answer in a single line" jsonschema:"required"`
		Answer    string `json:"answer,omitempty" jsonschema_description:"The answer to the question provided" jsonschema:"required"`
	} `json:"data,omitempty" jsonschema_description:"The object that contains the answer"`
	Error string `json:"error,omitempty" jsonschema_description:"Any errors encountered while trying to answer the question"`
}

func TestFinance_ChatCompletion(t *testing.T) {
	require.NotZero(t, *apiKey, "Please provide a valid api key")
	require.NotZero(t, *endpoint, "Please provide a valid endpoint")
	csvFile, readCsvFileErr := os.ReadFile("financebench_sample_150.csv")
	require.NoError(t, readCsvFileErr)
	bench := csv.NewReader(bytes.NewReader(csvFile))
	records, recordsErr := bench.ReadAll()
	require.NoError(t, recordsErr)

	schema := jsonschema.Reflect(&QuestionAnswer{})
	schemaJson, schemaMarshalErr := json.Marshal(schema)
	assert.NoError(t, schemaMarshalErr)
	t.Log(string(schemaJson))
	require.NoError(t, schemaMarshalErr)

	client := NewClient(*endpoint, *apiKey)

	for _, record := range records[1:] {
		t.Run(
			record[0], func(t *testing.T) {
				text := record[7]
				question := record[5]
				require.NotZero(t, text)
				require.NotZero(t, question)

				qaPrompt := fmt.Sprintf(
					`Try to answer the following finance question by using the text provided. Make sure to answer only if the text contains information that can answer the question. Think step by step and provide your reasoning in a single line. Finally, make absolutely sure to provide a final answer:

### Text:
%s

### Question:
%s`, text, question,
				)

				qaResp, err := client.ChatCompletion(
					context.Background(),
					openai.ChatCompletionRequest{
						Model: "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
						Messages: []openai.ChatCompletionMessage{
							{
								Role: openai.ChatMessageRoleSystem,
								Content: fmt.Sprintf(
									`You are a helpful assistant that extracts information and outputs only JSON. Do not output anything else besides JSON. Strictly follow the below JSON schema:

%s

You must always output JSON that follows the above schema. `,
									string(schemaJson),
								),
							},
							{
								Role:    openai.ChatMessageRoleUser,
								Content: qaPrompt,
							},
						},
						MaxTokens: 2048,
					},
					new(TogetherRateLimiterBackOff),
				)
				assert.NoError(t, err)
				if assert.NotZero(t, qaResp.Choices, "Prompt: %s", qaPrompt) {
					t.Log(qaResp.Choices[0].Message.Content)
					var result QuestionAnswer
					if unmarshall := json.Unmarshal(
						[]byte(qaResp.Choices[0].Message.Content), &result,
					); unmarshall != nil {
						t.Logf(
							"could not unmarshal intial response\n%s\n, correcting...",
							qaResp.Choices[0].Message.Content,
						)
						jsonFixPrompt := fmt.Sprintf(
							`The following JSON is not correct. Fix it by escaping newlines in the JSON strings and make sure that the JSON conforms to the schema provided. Make absolutely sure to only output the fixed JSON, do not output anything else besides JSON:
### Schema
%s

### JSON to fix
%s`,
							schemaJson,
							qaResp.Choices[0].Message.Content,
						)
						jsonFixResp, jsonFixRespErr := client.ChatCompletion(
							context.Background(),
							openai.ChatCompletionRequest{
								Model: "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
								Messages: []openai.ChatCompletionMessage{
									{
										Role: openai.ChatMessageRoleSystem,
										Content: fmt.Sprintf(
											`You are a helpful assistant that extracts information and outputs only JSON. Do not output anything else besides JSON. Strictly follow the below JSON schema:

%s

You must always output JSON that follows the above schema. `,
											string(schemaJson),
										),
									},
									{
										Role:    openai.ChatMessageRoleUser,
										Content: jsonFixPrompt,
									},
								},
								MaxTokens: 2048,
							},
							new(TogetherRateLimiterBackOff),
						)
						assert.NoError(t, jsonFixRespErr)
						assert.NoError(
							t, json.Unmarshal([]byte(jsonFixResp.Choices[0].Message.Content), &result), "Response: %s",
							jsonFixResp.Choices[0].Message.Content,
						)

						t.Logf("%+v", result)
					}
				}
			},
		)
	}
}

func TestJsonFix_ChatCompletion(t *testing.T) {
	require.NotZero(t, *apiKey, "Please provide a valid api key")
	require.NotZero(t, *endpoint, "Please provide a valid endpoint")

	schema := jsonschema.Reflect(&QuestionAnswer{})
	schemaJson, schemaMarshalErr := json.Marshal(schema)
	assert.NoError(t, schemaMarshalErr)
	t.Log(string(schemaJson))
	require.NoError(t, schemaMarshalErr)

	client := NewClient(*endpoint, *apiKey)

	jsonFixPrompt := fmt.Sprintf(
		`The following JSON is not correct. Fix it by escaping newlines in the JSON strings and make sure that the JSON conforms to the schema provided. Make absolutely sure to only output the fixed JSON, do not output anything else besides JSON:
### Schema
%s

### JSON to fix
%s`, schemaJson,
		`{"data":{"reasoning":"To find the net PP&E, we need to look at the balance sheet. Net PP&E is calculated by subtracting accumulated depreciation from Property, Plant and Equipment - net. In the balance sheet, we can see that Property, Plant and Equipment - net for 2018 is $8,738 million and Accumulated depreciation is $(16,135) million. So, net PP&E = $8,738 million - $(16,135) million = $24,873 million. To convert this to billions, we divide by 1,000: $24,873 million / 1,000 = $24.87 billion.","answer":"$24.87 billion"}`,
	)

	jsonFixResp, err := client.ChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
			Messages: []openai.ChatCompletionMessage{
				{
					Role: openai.ChatMessageRoleSystem,
					Content: fmt.Sprintf(
						`You are a helpful assistant that extracts information and outputs only JSON. Do not output anything else besides JSON. Strictly follow the below JSON schema:

%s

You must always output JSON that follows the above schema. `,
						string(schemaJson),
					),
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: jsonFixPrompt,
				},
			},
			MaxTokens: 2048,
		},
		new(TogetherRateLimiterBackOff),
	)
	assert.NoError(t, err)
	var result QuestionAnswer
	assert.NoError(
		t, json.Unmarshal([]byte(jsonFixResp.Choices[0].Message.Content), &result), "Response: %s",
		jsonFixResp.Choices[0].Message.Content,
	)

	t.Logf("%+v", result)

}
