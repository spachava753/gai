package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/google/generative-ai-go/genai"
)

// MarshalJSONToolUseInput marshals a ToolUseInput, never panics.
func MarshalJSONToolUseInput(t ToolUseInput) ([]byte, error) {
	data, err := json.Marshal(t)
	if err != nil {
		return []byte("{}"), err
	}
	return data, nil
}

// Register implements gai.ToolRegister for GeminiGenerator
func (g *GeminiGenerator) Register(tool Tool) error {
	// Validate tool name
	if tool.Name == "" {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool name cannot be empty"),
		}
	}
	if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool name cannot be %s", tool.Name),
		}
	}

	if g.tools == nil {
		g.tools = make(map[string]*genai.FunctionDeclaration)
	}
	if _, exists := g.tools[tool.Name]; exists {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool already registered"),
		}
	}

	geminiTool, err := convertToolToGemini(tool)
	if err != nil {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: err,
		}
	}
	g.tools[tool.Name] = geminiTool
	return nil
}

// convertToolToGemini converts gai.Tool to genai.Tool
func convertToolToGemini(tool Tool) (*genai.FunctionDeclaration, error) {
	// Only object input schema supported for now.
	if tool.InputSchema.Type != Object && tool.InputSchema.Type != Null {
		return nil, fmt.Errorf("gemini only supports object/null as root input schema")
	}
	decl := &genai.FunctionDeclaration{
		Name:        tool.Name,
		Description: tool.Description,
	}
	if tool.InputSchema.Type == Object {
		// Recursively map gai schema to Gemini parameter schema
		jschema, err := convertPropertyToGeminiSchema(tool.InputSchema.Properties, tool.InputSchema.Required)
		if err != nil {
			return nil, err
		}
		decl.Parameters = jschema
	}

	return decl, nil
}

// Helper to convert map of properties to *genai.Schema
func convertPropertyToGeminiSchema(props map[string]Property, required []string) (*genai.Schema, error) {
	// If no properties, this can just mean an empty object
	if props == nil {
		return &genai.Schema{Type: genai.TypeObject}, nil
	}
	properties := map[string]*genai.Schema{}
	for name, prop := range props {
		schema, err := convertSinglePropertyToGeminiSchema(prop)
		if err != nil {
			return nil, fmt.Errorf("property %s: %w", name, err)
		}
		properties[name] = schema
	}
	return &genai.Schema{
		Type:       genai.TypeObject,
		Properties: properties,
		Required:   required,
	}, nil
}

func convertSinglePropertyToGeminiSchema(prop Property) (*genai.Schema, error) {
	switch prop.Type {
	case String:
		s := &genai.Schema{Type: genai.TypeString, Description: prop.Description, Enum: prop.Enum}
		return s, nil
	case Integer:
		return &genai.Schema{Type: genai.TypeInteger, Description: prop.Description}, nil
	case Number:
		return &genai.Schema{Type: genai.TypeNumber, Description: prop.Description}, nil
	case Boolean:
		return &genai.Schema{Type: genai.TypeBoolean, Description: prop.Description}, nil
	case Array:
		var itemsSchema *genai.Schema
		if prop.Items != nil {
			var err error
			itemsSchema, err = convertSinglePropertyToGeminiSchema(*prop.Items)
			if err != nil {
				return nil, err
			}
		}
		return &genai.Schema{
			Type:        genai.TypeArray,
			Description: prop.Description,
			Items:       itemsSchema,
		}, nil
	case Object:
		props := map[string]*genai.Schema{}
		for name, subprop := range prop.Properties {
			s, err := convertSinglePropertyToGeminiSchema(subprop)
			if err != nil {
				return nil, err
			}
			props[name] = s
		}
		return &genai.Schema{
			Type:        genai.TypeObject,
			Description: prop.Description,
			Properties:  props,
			Required:    prop.Required,
		}, nil
	case Null:
		return nil, nil
	default:
		return nil, fmt.Errorf("unsupported property type: %v", prop.Type)
	}
}

var _ ToolRegister = (*GeminiGenerator)(nil)

type GeminiGenerator struct {
	client             *genai.Client
	modelName          string
	systemInstructions string
	tools              map[string]*genai.FunctionDeclaration
}

// NewGeminiGenerator creates a new Gemini generator with the specified API key, model name, and system instructions.
// Returns a ToolCapableGenerator that preprocesses dialog for parallel tool use compatibility.
// Example model names: "gemini-1.5-pro", "gemini-1.5-flash"
// The API key should be a valid Google Gemini API Key.
func NewGeminiGenerator(client *genai.Client, modelName, systemInstructions string) (ToolCapableGenerator, error) {
	inner := &GeminiGenerator{
		client:             client,
		modelName:          modelName,
		systemInstructions: systemInstructions,
	}
	return &PreprocessingGenerator{Inner: inner}, nil
}

// Generate implements gai.Generator
func (g *GeminiGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("gemini: client not initialized")
	}
	if len(dialog) == 0 {
		return Response{}, EmptyDialogErr
	}

	// We'll keep a mapping of toolCallID -> functionName for this call.
	toolCallIDToFunc := make(map[string]string)
	toolCallCount := 0

	model := g.client.GenerativeModel(g.modelName)
	// Set system prompt if provided
	if g.systemInstructions != "" {
		model.SystemInstruction = &genai.Content{
			Parts: []genai.Part{genai.Text(g.systemInstructions)},
		}
	}
	// If tools are registered, attach them
	if len(g.tools) > 0 {
		toolList := make([]*genai.FunctionDeclaration, 0, len(g.tools))
		for _, t := range g.tools {
			toolList = append(toolList, t)
		}
		model.Tools = []*genai.Tool{
			{
				FunctionDeclarations: toolList,
			},
		}
	}

	// generation parameters
	if options != nil {
		switch {
		case options.ToolChoice == ToolChoiceAuto:
			model.ToolConfig = &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingAuto,
				},
			}
		case options.ToolChoice == ToolChoiceToolsRequired:
			model.ToolConfig = &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingAny,
				},
			}
		case options.ToolChoice != "":
			model.ToolConfig = &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingAny,
					AllowedFunctionNames: []string{options.ToolChoice},
				},
			}
		}

		model.Temperature = genai.Ptr(float32(options.Temperature))

		if options.MaxGenerationTokens > 0 {
			model.MaxOutputTokens = genai.Ptr(int32(options.MaxGenerationTokens))
		}

		if options.N > 1 {
			model.CandidateCount = genai.Ptr(int32(options.N))
		}

		if options.StopSequences != nil {
			model.StopSequences = options.StopSequences
		}

		if options.TopP > 0 {
			model.TopP = genai.Ptr(float32(options.TopP))
		}

		if options.TopK > 0 {
			model.TopK = genai.Ptr(int32(options.TopK))
		}
	}

	// Split dialog into history and current user input, with tool result references resolved
	history, userInput, err := dialogToGeminiHistoryWithTools(dialog, toolCallIDToFunc)
	if err != nil {
		return Response{}, err
	}

	chat := model.StartChat()
	if len(history) > 0 {
		chat.History = history
	}

	resp, err := chat.SendMessage(ctx, userInput...)
	if err != nil {
		return Response{}, fmt.Errorf("gemini: generation failed: %w", err)
	}

	result := Response{
		UsageMetrics: make(Metrics),
	}

	// Usage metadata if available
	if resp.UsageMetadata != nil {
		if resp.UsageMetadata.PromptTokenCount > 0 {
			result.UsageMetrics[UsageMetricInputTokens] = int(resp.UsageMetadata.PromptTokenCount)
		}
		if resp.UsageMetadata.CandidatesTokenCount > 0 {
			result.UsageMetrics[UsageMetricGenerationTokens] = int(resp.UsageMetadata.CandidatesTokenCount)
		}
	}

	toolCallCount = len(toolCallIDToFunc)

	// Map candidates to gai.Messages
	var hasToolCalls bool
	for _, cand := range resp.Candidates {
		if cand.Content == nil {
			continue
		}
		var blocks []Block
		for _, part := range cand.Content.Parts {
			if txt, ok := part.(genai.Text); ok {
				blocks = append(blocks, Block{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(txt),
				})
			}
			// Map function call/tool use to ToolCall blocks
			if fc, ok := part.(genai.FunctionCall); ok {
				hasToolCalls = true
				toolCallCount++
				id := fmt.Sprintf("toolcall-%d", toolCallCount)
				toolCallIDToFunc[id] = fc.Name
				data, _ := MarshalJSONToolUseInput(ToolUseInput{
					Name:       fc.Name,
					Parameters: fc.Args,
				})
				blocks = append(blocks, Block{
					ID:           id,
					BlockType:    ToolCall,
					ModalityType: Text,
					MimeType:     "application/json",
					Content:      Str(data),
					ExtraFields:  map[string]interface{}{"function_name": fc.Name},
				})
			}
		}
		msg := Message{
			Role:   Assistant,
			Blocks: blocks,
		}
		result.Candidates = append(result.Candidates, msg)
	}

	// Set finish reason if available
	switch resp.Candidates[0].FinishReason {
	case genai.FinishReasonStop:
		result.FinishReason = EndTurn
	case genai.FinishReasonMaxTokens:
		result.FinishReason = MaxGenerationLimit
		return result, MaxGenerationLimitErr
	}

	if hasToolCalls {
		result.FinishReason = ToolUse
	}

	return result, nil
}

// msgToGeminiContent is a helper to map a Message to a Gemini Content, with support for tool calls/results
func msgToGeminiContent(msg Message, toolCallIDToFuncName map[string]string) *genai.Content {
	var parts []genai.Part
	var role string
	switch msg.Role {
	case User:
		role = "user"
		for _, block := range msg.Blocks {
			if block.BlockType != Content {
				continue
			}
			switch block.ModalityType {
			case Text:
				parts = append(parts, genai.Text(block.Content.String()))
			case Image:
				fileContent, decodeErr := base64.StdEncoding.DecodeString(block.Content.String())
				if decodeErr != nil {
					panic("unexpected error decoding image content: " + decodeErr.Error())
				}
				parts = append(parts, genai.Blob{
					MIMEType: block.MimeType,
					Data:     fileContent,
				})
			}
		}
	case Assistant:
		role = "model"
		for _, block := range msg.Blocks {
			if block.BlockType == Content && block.ModalityType == Text {
				parts = append(parts, genai.Text(block.Content.String()))
			}
			if block.BlockType == ToolCall && block.ModalityType == Text {
				// Unmarshal to get function name, params
				var toolUse ToolUseInput
				_ = json.Unmarshal([]byte(block.Content.String()), &toolUse)
				id := block.ID
				if toolUse.Name == "" {
					toolUse.Name, _ = block.ExtraFields["function_name"].(string)
				}
				toolCallIDToFuncName[id] = toolUse.Name
				parts = append(parts, genai.FunctionCall{
					Name: toolUse.Name, Args: toolUse.Parameters,
				})
			}
		}
	case ToolResult:
		role = "user"
		for _, block := range msg.Blocks {
			if block.ModalityType == Text {
				id := block.ID
				fn, ok := toolCallIDToFuncName[id]
				if !ok || fn == "" {
					err := fmt.Errorf("tool result references unknown tool call id: %q (history)", id)
					if err != nil {
						panic("unexpected error: " + err.Error())
					}
					return nil
				}
				var maybeObj map[string]any
				e := json.Unmarshal([]byte(block.Content.String()), &maybeObj)
				respObj := make(map[string]any)
				if e == nil {
					respObj = maybeObj
				} else {
					respObj["result"] = block.Content.String()
				}
				parts = append(parts, genai.FunctionResponse{
					Name: fn, Response: respObj,
				})
			}
		}
	}
	return &genai.Content{Parts: parts, Role: role}
}

// dialogToGeminiHistoryWithTools splits dialog into chat history and user input, with tool result mapping support
// This will populate toolCallIDToFunc with tool call id -> function name when populating ToolCall blocks in history
func dialogToGeminiHistoryWithTools(dialog Dialog, toolCallIDToFuncName map[string]string) (history []*genai.Content, userInput []genai.Part, err error) {
	// Early validation: last message must be User or ToolResult
	if len(dialog) == 0 {
		return nil, nil, nil
	}
	lastMsg := dialog[len(dialog)-1]
	if lastMsg.Role != User && lastMsg.Role != ToolResult {
		return nil, nil, fmt.Errorf("invalid dialog: last message must be user or tool result, got %v", lastMsg.Role)
	}

	// All previous messages go to history EXCEPT the last, which we just validated
	for i := 0; i < len(dialog)-1; i++ {
		history = append(history, msgToGeminiContent(dialog[i], toolCallIDToFuncName))
	}

	// The last message is the input for this call. Handle based on its role:
	if lastMsg.Role == User {
		for _, block := range lastMsg.Blocks {
			if block.BlockType == Content && block.ModalityType == Text {
				userInput = append(userInput, genai.Text(block.Content.String()))
			}
		}
	} else if lastMsg.Role == ToolResult {
		for _, block := range lastMsg.Blocks {
			if block.ModalityType == Text {
				id := block.ID
				fn, ok := toolCallIDToFuncName[id]
				if !ok || fn == "" {
					return nil, nil, fmt.Errorf("tool result references unknown tool call id: %q (input)", id)
				}
				var maybeObj map[string]any
				err := json.Unmarshal([]byte(block.Content.String()), &maybeObj)
				respObj := make(map[string]any)
				if err == nil {
					respObj = maybeObj
				} else {
					respObj["result"] = block.Content.String()
				}
				userInput = append(userInput, genai.FunctionResponse{
					Name: fn, Response: respObj,
				})
			}
		}
	}
	return history, userInput, nil
}

var _ Generator = (*GeminiGenerator)(nil)
