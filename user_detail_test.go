package gai

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"testing"

	"github.com/invopop/jsonschema"
	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var apiKey = flag.String("api-key", "", "Your endpoints api key")
var endpoint = flag.String("endpoint", "", "The endpoint you want to call")

type Job struct {
	Company  string `json:"company,omitempty" jsonschema_description:"The company that the user works at"`
	Position string `json:"position,omitempty" jsonschema_description:"The position name that the user has at the company"`
}

type UserDetail struct {
	Name string `json:"name,omitempty" jsonschema_description:"The name of the user" jsonschema:"required"`
	Age  int    `json:"age,omitempty" jsonschema_description:"The age of the user" jsonschema:"required"`
	Job  *Job   `json:"job,omitempty" jsonschema_description:"The job of the user"`
}

type DataExtraction[T any] struct {
	Data  *T      `json:"data,omitempty" jsonschema_description:"The data to be extracted"`
	Error *string `json:"error,omitempty" jsonschema_description:"Any errors encountered while extracting data"`
}

func TestUserDetail_ChatCompletion(t *testing.T) {
	require.NotZero(t, *apiKey, "Please provide a valid api key")
	require.NotZero(t, *endpoint, "Please provide a valid endpoint")
	schema := jsonschema.Reflect(&DataExtraction[UserDetail]{})
	schemaJson, schemaMarshalErr := json.Marshal(schema)
	assert.NoError(t, schemaMarshalErr)
	t.Log(string(schemaJson))
	require.NoError(t, schemaMarshalErr)
	config := openai.DefaultConfig(*apiKey)
	config.BaseURL = *endpoint
	client := openai.NewClientWithConfig(config)
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
			// Model: "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
			Messages: []openai.ChatCompletionMessage{
				// 				{
				// 					Role:    openai.ChatMessageRoleSystem,
				// 					Content: fmt.Sprintf(`You are a helpful assistant that extracts information and outputs only JSON.`),
				// 				},
				// 				{
				// 					Role: openai.ChatMessageRoleUser,
				// 					Content: fmt.Sprintf(
				// 						`Do not output anything else besides JSON. Strictly follow the below JSON schema:
				//
				// %s
				//
				// You must always output JSON that follows the above schema.`,
				// 						string(schemaJson),
				// 					),
				// 				},
				{
					Role: openai.ChatMessageRoleSystem,
					Content: fmt.Sprintf(
						`You are a helpful assistant that extracts information and outputs only JSON. Do not output anything else besides JSON. Strictly follow the below JSON schema:

%s

You must always output JSON that follows the above schema.`,
						string(schemaJson),
					),
				},
				// {
				// 	Role:    openai.ChatMessageRoleUser,
				// 	Content: "Mackenzie is 69 years old woman who is retired",
				// },
				// {
				// 	Role: openai.ChatMessageRoleAssistant,
				// 	Content: func() string {
				// 		contents, marshalErr := json.Marshal(
				// 			UserDetail{
				// 				Name: "Mackenzie",
				// 				Age:  69,
				// 			},
				// 		)
				// 		require.NoError(t, marshalErr)
				// 		return string(contents)
				// 	}(),
				// },
				// {
				// 	Role:    openai.ChatMessageRoleUser,
				// 	Content: "Robert is 18 years old young man who works as a gas station attendant at Circle K. His last name is Tyler",
				// },
				// {
				// 	Role: openai.ChatMessageRoleAssistant,
				// 	Content: func() string {
				// 		contents, marshalErr := json.Marshal(
				// 			UserDetail{
				// 				Name: "Robert Tyler",
				// 				Age:  18,
				// 				Job: &Job{
				// 					Company:  "Circle K",
				// 					Position: "gas station attendant",
				// 				},
				// 			},
				// 		)
				// 		require.NoError(t, marshalErr)
				// 		return string(contents)
				// 	}(),
				// },
				// {
				// 	Role:    openai.ChatMessageRoleUser,
				// 	Content: "Shierra is a PHD candidate and a Google AI researcher. She married Mbappe Williams, and she is 36 years old, married, and have twins.",
				// },
				{
					Role: openai.ChatMessageRoleUser,
					Content: `Try to extract data from the following text. Make sure to extract data only if the text contains information about a person:
				
"Toyata is great company that produces reliable cars. It is a generational 106 year old company that preserves its timeless values"`,
				},
				// 				{
				// 					Role: openai.ChatMessageRoleUser,
				// 					Content: `Try to extract data from the following text. Make sure to extract data only if the text contains information about a person. Remember to only output JSON following the schema provided:
				//
				// "Shierra is a PHD candidate and a Google AI researcher. She married Mbappe Williams, and she is 36 years old, married, and have twins."`,
				// 				},
			},
		},
	)
	assert.NoError(t, err)
	t.Log(resp.Choices[0].Message.Content)
	var result DataExtraction[UserDetail]
	assert.NoError(t, json.Unmarshal([]byte(resp.Choices[0].Message.Content), &result))
	t.Logf("%+v", result)
}
