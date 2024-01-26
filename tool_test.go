package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
	ojs "github.com/sashabaranov/go-openai/jsonschema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var FunctionCall = ojs.Definition{
	Type:        ojs.Object,
	Description: "The function to call",
	Properties: map[string]ojs.Definition{
		"name": {
			Type:        ojs.String,
			Description: "The name of the function to call",
		},
		"arguments": {
			Type:        ojs.Array,
			Description: "Arguments to call the function with",
			Items: &ojs.Definition{
				Type: ojs.Object,
				Properties: map[string]ojs.Definition{
					"name": {
						Type:        ojs.String,
						Description: "The name of the argument",
					},
					"value": {
						Type:        ojs.String,
						Description: "The value of the argument",
					},
				},
				Required: []string{
					"name",
					"value",
				},
			},
		},
	},
	Required: []string{
		"name",
	},
}

func TestToolOutput_ChatCompletion(t *testing.T) {
	var tools []openai.FunctionDefinition
	tools = append(
		tools, openai.FunctionDefinition{
			Name:        "calculator",
			Description: "A simple calculator to do basic arithmetic",
			Parameters: ojs.Definition{
				Type:        ojs.Object,
				Description: "",
				Enum:        nil,
				Properties: map[string]ojs.Definition{
					"operation": {
						Type:        ojs.String,
						Description: "The type of simple arithmetic you would like to do",
						Enum: []string{
							"add",
							"subtract",
							"divide",
							"multiply",
						},
					},
					"left_number": {
						Type:        ojs.Number,
						Description: "A single number that is the left hand side of the operation",
					},
					"right_number": {
						Type:        ojs.Number,
						Description: "A single number that is the right hand side of the operation",
					},
				},
				Required: []string{
					"operation",
					"left_operand",
					"right_operand",
				},
			},
		},
	)
	require.NotZero(t, *apiKey, "Please provide a valid api key")
	require.NotZero(t, *endpoint, "Please provide a valid endpoint")

	schemaJson, schemaMarshalErr := json.Marshal(tools)
	require.NoError(t, schemaMarshalErr)
	t.Log(string(schemaJson))

	functionCallSchemaJson, functionCallSchemaJsonMarshalErr := json.Marshal(&FunctionCall)
	require.NoError(t, functionCallSchemaJsonMarshalErr)
	t.Log(string(functionCallSchemaJson))

	client := NewClient(*endpoint, *apiKey)

	messages := []openai.ChatCompletionMessage{
		{
			Role: openai.ChatMessageRoleSystem,
			Content: fmt.Sprintf(
				`You are a helpful assistant that answers user's question by using the functions given to you as defined below. When you call a function, make sure to ALWAYS surround the call with a markdown code block labeled as "function", such as %s. If you do a function call, ALWAYS make sure the call to the function is the last thing you say and the function call is in JSON. The user will pass the output of the function to you. DO NOT generate the output of function yourself. Make sure to only do ONE function call at a time so the output of the function call can be passed to you. Here are the functions available to you:

%s

Use the following JSON schema to call functions:

%s

Make sure to always adhere to the JSON schema when doing function calls`, "```function```",
				string(schemaJson),
				string(functionCallSchemaJson),
			),
		},
	}

	fewShot := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: `Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?`,
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`To calculate Weng's earnings, we need to convert the 50 minutes of babysitting into hours and then multiply it by her hourly rate.
First, let's convert the minutes to hours:
%sfunction
{
  "name": "calculator",
  "arguments": [
    {
      "name": "left_number",
      "value": "50"
    },
    {
      "name": "operation",
      "value": "divide"
    },
    {
      "name": "right_number",
      "value": "60"
    }
  ]
}
%s`, "````", "```",
			),
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: `0.833333333`,
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Now, let's multiply the hours by her hourly rate:
%sfunction
{
  "name": "calculator",
  "arguments": [
    {
      "name": "left_number",
      "value": "0.833333333"
    },
    {
      "name": "operation",
      "value": "multiply"
    },
    {
      "name": "right_number",
      "value": "12"
    }
  ]
}
%s`, "````", "```",
			),
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: `10`,
		},
		{
			Role:    openai.ChatMessageRoleAssistant,
			Content: `Weng made $10 babysitting`,
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: `Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?`,
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`First, let's calculate the amount of money Betty's grandparents gave her:
%sfunction
{
  "name": "calculator",
  "arguments": [
    {
      "name": "left_number",
      "value": "15"
    },
    {
      "name": "operation",
      "value": "multiply"
    },
    {
      "name": "right_number",
      "value": "2"
    }
  ]
}
%s`, "````", "```",
			),
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: `30`,
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Now, let's add the money Betty received from her parents and grandparents:
%sfunction
{
  "name": "calculator",
  "arguments": [
    {
      "name": "left_number",
      "value": "15"
    },
    {
      "name": "operation",
      "value": "add"
    },
    {
      "name": "right_number",
      "value": "30"
    }
  ]
}
%s`, "````", "```",
			),
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: `45`,
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Now, let's add the money Betty received from her parents and grandparents to what she has. First, lets calculate how much she has saved:
%sfunction
{
  "name": "calculator",
  "arguments": [
    {
      "name": "left_number",
      "value": "100"
    },
    {
      "name": "operation",
      "value": "divide"
    },
    {
      "name": "right_number",
      "value": "2"
    }
  ]
}
%s`, "````", "```",
			),
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: `50`,
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Next, let's add the money Betty received from her parents and grandparents to what she has:
%sfunction
{
  "name": "calculator",
  "arguments": [
    {
      "name": "left_number",
      "value": "45"
    },
    {
      "name": "operation",
      "value": "add"
    },
    {
      "name": "right_number",
      "value": "50"
    }
  ]
}
%s`, "````", "```",
			),
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: `95`,
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			Content: fmt.Sprintf(
				`Next, let's calculate how much more Betty needs to buy her wallet:
%sfunction
{
  "name": "calculator",
  "arguments": [
    {
      "name": "left_number",
      "value": "100"
    },
    {
      "name": "operation",
      "value": "subtract"
    },
    {
      "name": "right_number",
      "value": "95"
    }
  ]
}
%s`, "````", "```",
			),
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: `5`,
		},
	}

	// _ = fewShot
	messages = append(messages, fewShot...)

	qaPrompt := fmt.Sprintf(
		`Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?`,
	)

	messages = append(
		messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: qaPrompt,
		},
	)

	qaResp, err := client.ChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model:     "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
			Messages:  messages,
			MaxTokens: 2048,
		},
		new(TogetherRateLimiterBackOff),
	)
	assert.NoError(t, err)
	t.Log(qaResp.Choices[0].Message.Content)

	messages = append(
		messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: qaResp.Choices[0].Message.Content,
		},
	)

	// check if tool is called
	if strings.Contains(qaResp.Choices[0].Message.Content, "```function") {
		functionCallIdxStart := strings.Index(qaResp.Choices[0].Message.Content, "```function")
		functionCallJson := qaResp.Choices[0].Message.Content[functionCallIdxStart+len("```function"):]
		functionCallIdxEnd := strings.LastIndex(functionCallJson, "```")
		functionCallJson = functionCallJson[:functionCallIdxEnd]
		t.Log(functionCallJson)
		var fc openai.FunctionCall
		if unmarshallErr := json.Unmarshal([]byte(functionCallJson), &fc); unmarshallErr != nil {
			messages = append(
				messages, openai.ChatCompletionMessage{
					Role: openai.ChatMessageRoleUser,
					Content: fmt.Sprintf(
						`The following function call is incorrect. Please call the function call again with the proper arguments:
### Incorrect function call:
%s

### JSON schema for function call
%s`, functionCallJson, tools[0],
					),
				},
			)

			qaResp, err = client.ChatCompletion(
				context.Background(),
				openai.ChatCompletionRequest{
					Model:     "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
					Messages:  messages,
					MaxTokens: 2048,
				},
				new(TogetherRateLimiterBackOff),
			)

			assert.NoError(t, err)
			t.Log(qaResp.Choices[0].Message.Content)
		}
		t.Log(fc)
	}
}

func TestToolCallParse(t *testing.T) {
	toolCall := "To answer this question, I will use the \"Calculator\" tool. Here is the tool call:\n```tool_call\n{\n  \"tool\": {\n    \"calculator\": {\n      \"operation\": \"add\",\n      \"left_operand\": 645,\n      \"right_operand\": 2423\n    }\n  }\n}\n```\n\nPlease wait while the tool processes the request."
	codeBlockIdxStart := strings.Index(toolCall, "```tool_call")
	t.Log(toolCall[codeBlockIdxStart+len("```tool_call"):])
	toolCall = toolCall[codeBlockIdxStart+len("```tool_call"):]
	codeBlockIdxEnd := strings.LastIndex(toolCall, "```")
	t.Log(toolCall[:codeBlockIdxEnd])
}
