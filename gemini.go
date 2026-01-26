package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"

	"google.golang.org/genai"

	"github.com/google/jsonschema-go/jsonschema"
)

const (
	// GeminiExtraFieldThoughtSignature stores the thought signature for thinking blocks.
	// Present in Block.ExtraFields for Thinking blocks from Gemini responses.
	// This signature is required when sending thinking blocks back to the API.
	GeminiExtraFieldThoughtSignature = "gemini_thought_signature"

	// GeminiExtraFieldFunctionName stores the function name for tool call blocks.
	// Present in Block.ExtraFields for ToolCall blocks from Gemini responses.
	GeminiExtraFieldFunctionName = "function_name"
)

// MarshalJSONToolUseInput marshals a ToolCallInput, never panics.
func MarshalJSONToolUseInput(t ToolCallInput) ([]byte, error) {
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

// convertToolToGemini converts gai.Tool to *[genai.FunctionDeclaration]
func convertToolToGemini(tool Tool) (*genai.FunctionDeclaration, error) {
	if tool.InputSchema != nil && tool.InputSchema.Type != "object" && tool.InputSchema.Type != "" {
		return nil, fmt.Errorf("gemini only supports object/null as root input schema")
	}
	decl := &genai.FunctionDeclaration{
		Name:        tool.Name,
		Description: tool.Description,
	}
	if tool.InputSchema != nil && tool.InputSchema.Type == "object" {
		jschema, err := convertJSONSchemaToGemini(tool.InputSchema)
		if err != nil {
			return nil, err
		}
		decl.Parameters = jschema
	}
	return decl, nil
}

// convertJSONSchemaToGemini converts a jsonschema.Schema to a genai.Schema
func convertJSONSchemaToGemini(schema *jsonschema.Schema) (*genai.Schema, error) {
	if schema == nil {
		return &genai.Schema{Type: genai.TypeObject}, nil
	}

	// Serialize the schema to JSON then unmarshal into genai.Schema
	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}

	var genSchema genai.Schema
	if err := json.Unmarshal(schemaJSON, &genSchema); err != nil {
		return nil, err
	}

	return &genSchema, nil
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
// The returned generator also implements the TokenCounter interface for token counting.
//
// Parameters:
//   - client: A properly initialized genai.Client instance with API key configured
//   - modelName: The Gemini model to use (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
//   - systemInstructions: Optional system instructions that set the model's behavior
//
// Supported modalities:
//   - Text: Both input and output
//   - Image: Input only (base64 encoded, including PDFs with MIME type "application/pdf")
//   - Audio: Input only (base64 encoded)
//
// PDF documents are supported as part of the Image modality. The PDF content is sent
// with the appropriate MIME type and processed by Gemini's multimodal capabilities.
// Use the PDFBlock helper function to create PDF content blocks.
//
// Note on JSON Schema support limitations:
//   - The anyOf property has limited support in Gemini. It only supports the pattern [Type, null] to
//     indicate nullable fields, which is implemented using Schema.Nullable=true.
//   - If you use anyOf with multiple non-null types or with only the null type, this generator will
//     return errors, as the Gemini SDK doesn't support these patterns.
//   - For maximum compatibility across all generators, restrict usage of anyOf to the nullable pattern:
//     e.g., "anyOf": [{"type": "string"}, {"type": "null"}]
//
// Returns a ToolCapableGenerator that also implements TokenCounter, or an error if initialization fails.
func NewGeminiGenerator(client *genai.Client, modelName, systemInstructions string) (interface {
	ToolCapableGenerator
	StreamingGenerator
	TokenCounter
}, error) {
	inner := &GeminiGenerator{
		client:             client,
		modelName:          modelName,
		systemInstructions: systemInstructions,
	}

	return &PreprocessingGenerator{GeneratorWrapper: GeneratorWrapper{Inner: inner}}, nil
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

	genContentConfig := &genai.GenerateContentConfig{
		ThinkingConfig: &genai.ThinkingConfig{
			IncludeThoughts: true,
		},
	}
	// Set system prompt if provided
	if g.systemInstructions != "" {
		genContentConfig.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{genai.NewPartFromText(g.systemInstructions)},
		}
	}
	// If tools are registered, attach them
	if len(g.tools) > 0 {
		toolList := make([]*genai.FunctionDeclaration, 0, len(g.tools))
		for _, t := range g.tools {
			toolList = append(toolList, t)
		}
		genContentConfig.Tools = []*genai.Tool{
			{
				FunctionDeclarations: toolList,
			},
		}
	}

	// generation parameters
	if options != nil {
		if options.ToolChoice != "" {
			tc := &genai.ToolConfig{}
			mode := genai.FunctionCallingConfigModeAuto
			var allowedFuncNames []string
			switch {
			case options.ToolChoice == ToolChoiceToolsRequired:
				mode = genai.FunctionCallingConfigModeAny
			case options.ToolChoice != ToolChoiceAuto:
				mode = genai.FunctionCallingConfigModeAny
				allowedFuncNames = []string{options.ToolChoice}
			}
			tc.FunctionCallingConfig = &genai.FunctionCallingConfig{
				Mode:                 mode,
				AllowedFunctionNames: allowedFuncNames,
			}
			genContentConfig.ToolConfig = tc
		}

		if options.Temperature > 0 {
			genContentConfig.Temperature = genai.Ptr(float32(options.Temperature))
		}
		if options.MaxGenerationTokens > 0 {
			genContentConfig.MaxOutputTokens = int32(options.MaxGenerationTokens)
		}
		if options.N > 1 {
			genContentConfig.CandidateCount = int32(options.N)
		}
		if options.StopSequences != nil {
			genContentConfig.StopSequences = options.StopSequences
		}
		if options.TopP > 0 {
			genContentConfig.TopP = genai.Ptr(float32(options.TopP))
		}
		if options.TopK > 0 {
			genContentConfig.TopK = genai.Ptr(float32(options.TopK))
		}
		if options.ThinkingBudget != "" {
			switch options.ThinkingBudget {
			case "low", "medium", "high":
				genContentConfig.ThinkingConfig.ThinkingLevel = genai.ThinkingLevel(options.ThinkingBudget)
			default:
				return Response{}, InvalidParameterErr{Parameter: "thinking budget", Reason: fmt.Sprintf("invalid thinking budget: %s", options.ThinkingBudget)}
			}
		}
	}

	allContents, err := prepareGeminiChatHistory(dialog, toolCallIDToFunc)
	if err != nil {
		return Response{}, err
	}

	resp, err := g.client.Models.GenerateContent(ctx, g.modelName, allContents, genContentConfig)
	if err != nil {
		var apierr genai.APIError
		if errors.As(err, &apierr) {
			// Map HTTP status codes to our error types
			switch apierr.Code {
			case 401:
				return Response{}, AuthenticationErr(apierr.Error())
			case 403:
				return Response{}, ApiErr{
					StatusCode: apierr.Code,
					Type:       "permission_error",
					Message:    apierr.Error(),
				}
			case 404:
				return Response{}, ApiErr{
					StatusCode: apierr.Code,
					Type:       "not_found_error",
					Message:    apierr.Error(),
				}
			case 429:
				return Response{}, RateLimitErr(apierr.Error())
			case 500:
				return Response{}, ApiErr{
					StatusCode: apierr.Code,
					Type:       "api_error",
					Message:    apierr.Error(),
				}
			case 503:
				return Response{}, ApiErr{
					StatusCode: apierr.Code,
					Type:       "service_unavailable",
					Message:    apierr.Error(),
				}
			default:
				// Default to invalid_request_error for 400 and other status codes
				return Response{}, ApiErr{
					StatusCode: apierr.Code,
					Type:       "invalid_request_error",
					Message:    apierr.Error(),
				}
			}
		}
		return Response{}, fmt.Errorf("gemini: generation failed: %w", err)
	}

	result := Response{
		UsageMetadata: make(Metadata),
	}

	// Usage metadata if available
	if resp.UsageMetadata != nil {
		if resp.UsageMetadata.PromptTokenCount > 0 {
			result.UsageMetadata[UsageMetricInputTokens] = int(resp.UsageMetadata.PromptTokenCount)
		}
		if resp.UsageMetadata.TotalTokenCount > 0 {
			result.UsageMetadata[UsageMetricGenerationTokens] = int(resp.UsageMetadata.TotalTokenCount - resp.UsageMetadata.PromptTokenCount)
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
			if part.Text != "" {
				blkType := Content
				if part.Thought {
					blkType = Thinking
				}

				block := Block{
					BlockType:    blkType,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(part.Text),
				}

				if part.ThoughtSignature != nil {
					block.ExtraFields = map[string]interface{}{
						GeminiExtraFieldThoughtSignature: base64.StdEncoding.EncodeToString(part.ThoughtSignature),
					}
				}

				blocks = append(blocks, block)
			} else if part.InlineData != nil {
				// Handle inline data (could be image, audio, video)
				mimeType := part.InlineData.MIMEType
				data := base64.StdEncoding.EncodeToString(part.InlineData.Data)

				// Determine modality based on MIME type
				var modality Modality
				if strings.HasPrefix(mimeType, "image/") {
					modality = Image
				} else if strings.HasPrefix(mimeType, "audio/") {
					modality = Audio
				} else if strings.HasPrefix(mimeType, "video/") {
					modality = Video
				} else {
					// Default to text for unknown types
					modality = Text
				}

				block := Block{
					BlockType:    Content,
					ModalityType: modality,
					MimeType:     mimeType,
					Content:      Str(data),
				}

				if part.ThoughtSignature != nil {
					block.ExtraFields = map[string]interface{}{
						GeminiExtraFieldThoughtSignature: base64.StdEncoding.EncodeToString(part.ThoughtSignature),
					}
				}

				blocks = append(blocks, block)
			} else if part.FunctionCall != nil {
				fc := part.FunctionCall
				hasToolCalls = true
				toolCallCount++
				id := fmt.Sprintf("toolcall-%d", toolCallCount)
				toolCallIDToFunc[id] = fc.Name

				jsonData, _ := MarshalJSONToolUseInput(ToolCallInput{
					Name:       fc.Name,
					Parameters: fc.Args,
				})

				extraFields := map[string]interface{}{
					GeminiExtraFieldFunctionName: fc.Name,
				}

				if part.ThoughtSignature != nil {
					extraFields[GeminiExtraFieldThoughtSignature] = base64.StdEncoding.EncodeToString(part.ThoughtSignature)
				}

				blocks = append(blocks, Block{
					ID:           id,
					BlockType:    ToolCall,
					ModalityType: Text,
					MimeType:     "application/json",
					Content:      Str(jsonData),
					ExtraFields:  extraFields,
				})
			}
		}
		msg := Message{
			Role:   Assistant,
			Blocks: blocks,
		}
		result.Candidates = append(result.Candidates, msg)
	}

	if len(resp.Candidates) > 0 && resp.Candidates[0] != nil {
		switch resp.Candidates[0].FinishReason {
		case genai.FinishReasonStop:
			result.FinishReason = EndTurn
		case genai.FinishReasonMaxTokens:
			result.FinishReason = MaxGenerationLimit
		default:
			result.FinishReason = EndTurn
		}
	}

	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}

	return result, nil
}

func (g *GeminiGenerator) Stream(ctx context.Context, dialog Dialog, options *GenOpts) iter.Seq2[StreamChunk, error] {
	return func(yield func(StreamChunk, error) bool) {
		if g.client == nil {
			yield(StreamChunk{}, fmt.Errorf("gemini: client not initialized"))
			return
		}

		if len(dialog) == 0 {
			yield(StreamChunk{}, EmptyDialogErr)
			return
		}

		// We'll keep a mapping of toolCallID -> functionName for this call.
		toolCallIDToFunc := make(map[string]string)
		toolCallCount := 0

		genContentConfig := &genai.GenerateContentConfig{}
		// Set system prompt if provided
		if g.systemInstructions != "" {
			genContentConfig.SystemInstruction = &genai.Content{
				Parts: []*genai.Part{genai.NewPartFromText(g.systemInstructions)},
			}
		}
		// If tools are registered, attach them
		if len(g.tools) > 0 {
			toolList := make([]*genai.FunctionDeclaration, 0, len(g.tools))
			for _, t := range g.tools {
				toolList = append(toolList, t)
			}
			genContentConfig.Tools = []*genai.Tool{
				{
					FunctionDeclarations: toolList,
				},
			}
		}

		// generation parameters
		if options != nil {
			if options.ToolChoice != "" {
				tc := &genai.ToolConfig{}
				mode := genai.FunctionCallingConfigModeAuto
				var allowedFuncNames []string
				switch {
				case options.ToolChoice == ToolChoiceToolsRequired:
					mode = genai.FunctionCallingConfigModeAny
				case options.ToolChoice != ToolChoiceAuto:
					mode = genai.FunctionCallingConfigModeAny
					allowedFuncNames = []string{options.ToolChoice}
				}
				tc.FunctionCallingConfig = &genai.FunctionCallingConfig{
					Mode:                 mode,
					AllowedFunctionNames: allowedFuncNames,
				}
				genContentConfig.ToolConfig = tc
			}

			if options.Temperature > 0 {
				genContentConfig.Temperature = genai.Ptr(float32(options.Temperature))
			}
			if options.MaxGenerationTokens > 0 {
				genContentConfig.MaxOutputTokens = int32(options.MaxGenerationTokens)
			}
			if options.N > 1 {
				genContentConfig.CandidateCount = int32(options.N)
			}
			if options.StopSequences != nil {
				genContentConfig.StopSequences = options.StopSequences
			}
			if options.TopP > 0 {
				genContentConfig.TopP = genai.Ptr(float32(options.TopP))
			}
			if options.TopK > 0 {
				genContentConfig.TopK = genai.Ptr(float32(options.TopK))
			}
		}

		allContents, err := prepareGeminiChatHistory(dialog, toolCallIDToFunc)
		if err != nil {
			yield(StreamChunk{}, err)
			return
		}

		// Track cumulative usage
		var totalInputTokens, totalOutputTokens int32

		for resp, err := range g.client.Models.GenerateContentStream(ctx, g.modelName, allContents, genContentConfig) {
			if err != nil {
				var apierr genai.APIError
				if errors.As(err, &apierr) {
					// Map HTTP status codes to our error types
					switch apierr.Code {
					case 401:
						yield(StreamChunk{}, AuthenticationErr(apierr.Error()))
						return
					case 403:
						yield(StreamChunk{}, ApiErr{
							StatusCode: apierr.Code,
							Type:       "permission_error",
							Message:    apierr.Error(),
						})
						return
					case 404:
						yield(StreamChunk{}, ApiErr{
							StatusCode: apierr.Code,
							Type:       "not_found_error",
							Message:    apierr.Error(),
						})
						return
					case 429:
						yield(StreamChunk{}, RateLimitErr(apierr.Error()))
						return
					case 500:
						yield(StreamChunk{}, ApiErr{
							StatusCode: apierr.Code,
							Type:       "api_error",
							Message:    apierr.Error(),
						})
						return
					case 503:
						yield(StreamChunk{}, ApiErr{
							StatusCode: apierr.Code,
							Type:       "service_unavailable",
							Message:    apierr.Error(),
						})
						return
					default:
						// Default to invalid_request_error for 400 and other status codes
						yield(StreamChunk{}, ApiErr{
							StatusCode: apierr.Code,
							Type:       "invalid_request_error",
							Message:    apierr.Error(),
						})
						return
					}
				}
				yield(StreamChunk{}, fmt.Errorf("gemini: generation failed: %w", err))
				return
			}

			// Update cumulative usage if available
			if resp.UsageMetadata != nil {
				if resp.UsageMetadata.PromptTokenCount > 0 {
					totalInputTokens = resp.UsageMetadata.PromptTokenCount
				}
				// CandidatesTokenCount is the output tokens in each response
				if resp.UsageMetadata.CandidatesTokenCount > 0 {
					totalOutputTokens += resp.UsageMetadata.CandidatesTokenCount
				}
			}

			if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil || len(resp.Candidates[0].Content.Parts) == 0 {
				if !yield(StreamChunk{
					Block:           TextBlock(""),
					CandidatesIndex: 0,
				}, nil) {
					return
				}
			}
			if len(resp.Candidates) != 1 {
				panic("cannot handle multiple candidates at this time")
			}

			for _, part := range resp.Candidates[0].Content.Parts {
				if part.Text != "" {
					if part.Thought {
						if !yield(StreamChunk{
							Block: Block{
								BlockType:    Thinking,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(part.Text),
								ExtraFields:  nil,
							},
							CandidatesIndex: 0,
						}, nil) {
							return
						}
					} else {
						if !yield(StreamChunk{
							Block: TextBlock(part.Text),
						}, nil) {
							return
						}
					}
				} else {
					if part.InlineData != nil {
						panic("unknown block type")
					}
					if part.CodeExecutionResult != nil {
						panic("unknown block type")
					}
					if part.ExecutableCode != nil {
						panic("unknown block type")
					}
					if part.FileData != nil {
						panic("unknown block type")
					}
					if part.FunctionCall != nil {
						if part.FunctionCall.Name != "" {
							toolCallCount++
							id := fmt.Sprintf("toolcall-%d", toolCallCount)
							toolCallIDToFunc[id] = part.FunctionCall.Name
							if !yield(StreamChunk{
								Block: Block{
									ID:           id,
									BlockType:    ToolCall,
									ModalityType: Text,
									MimeType:     "text/plain",
									Content:      Str(part.FunctionCall.Name),
								},
								CandidatesIndex: 0,
							}, nil) {
								return
							}
						}
						if part.FunctionCall.Args != nil {
							contentJson, err := json.Marshal(part.FunctionCall.Args)
							if err != nil {
								panic(err)
							}
							if !yield(StreamChunk{
								Block: Block{
									BlockType:    ToolCall,
									ModalityType: Text,
									MimeType:     "text/plain",
									Content:      Str(contentJson),
								},
								CandidatesIndex: 0,
							}, nil) {
								return
							}
						}
					}
					if part.FunctionResponse != nil {
						panic("unexpected block type")
					}
				}
			}
		}

		// Emit metadata block as final block
		if totalInputTokens > 0 || totalOutputTokens > 0 {
			metadata := make(Metadata)

			if totalInputTokens > 0 {
				metadata[UsageMetricInputTokens] = int(totalInputTokens)
			}
			if totalOutputTokens > 0 {
				metadata[UsageMetricGenerationTokens] = int(totalOutputTokens)
			}

			yield(StreamChunk{
				Block:           MetadataBlock(metadata),
				CandidatesIndex: 0,
			}, nil)
		}
	}
}

// msgToGeminiContent is a helper to map a Message to a Gemini Content, with support for tool calls/results
func msgToGeminiContent(msg Message, toolCallIDToFuncName map[string]string) (*genai.Content, error) {
	var parts []*genai.Part
	var role genai.Role

	switch msg.Role {
	case User:
		role = genai.RoleUser
		for _, block := range msg.Blocks {
			if block.BlockType != Content {
				return nil, fmt.Errorf("user message block type %v is not Content", block.BlockType)
			}
			switch block.ModalityType {
			case Text:
				parts = append(parts, genai.NewPartFromText(block.Content.String()))
			case Image:
				fileContent, decodeErr := base64.StdEncoding.DecodeString(block.Content.String())
				if decodeErr != nil {
					return nil, fmt.Errorf("decoding image content failed: %w", decodeErr)
				}
				parts = append(parts, genai.NewPartFromBytes(fileContent, block.MimeType))
			case Audio:
				fileContent, decodeErr := base64.StdEncoding.DecodeString(block.Content.String())
				if decodeErr != nil {
					return nil, fmt.Errorf("decoding audio content failed: %w", decodeErr)
				}
				parts = append(parts, genai.NewPartFromBytes(fileContent, block.MimeType))
			default:
				return nil, fmt.Errorf("unsupported modality type in user message: %v", block.ModalityType)
			}
		}
	case Assistant:
		role = genai.RoleModel
		for _, block := range msg.Blocks {
			if block.BlockType == Content {
				switch block.ModalityType {
				case Text:
					part := genai.NewPartFromText(block.Content.String())
					if sigVal, ok := block.ExtraFields[GeminiExtraFieldThoughtSignature]; ok && sigVal != nil {
						sig, err := base64.StdEncoding.DecodeString(sigVal.(string))
						if err != nil {
							return nil, fmt.Errorf("could not decode base64 thought signature: %w", err)
						}
						part.ThoughtSignature = sig
					}
					parts = append(parts, part)
				case Audio:
					fileContent, decodeErr := base64.StdEncoding.DecodeString(block.Content.String())
					if decodeErr != nil {
						return nil, fmt.Errorf("decoding audio content failed: %w", decodeErr)
					}
					part := genai.NewPartFromBytes(fileContent, block.MimeType)
					if sigVal, ok := block.ExtraFields[GeminiExtraFieldThoughtSignature]; ok && sigVal != nil {
						sig, err := base64.StdEncoding.DecodeString(sigVal.(string))
						if err != nil {
							return nil, fmt.Errorf("could not decode base64 thought signature: %w", err)
						}
						part.ThoughtSignature = sig
					}
					parts = append(parts, part)
				default:
					// Skip unsupported modalities for assistant messages
				}
			}
			if block.BlockType == ToolCall && block.ModalityType == Text {
				// Unmarshal to get function name, params
				var toolUse ToolCallInput
				if err := json.Unmarshal([]byte(block.Content.String()), &toolUse); err != nil {
					return nil, fmt.Errorf("unmarshalling tool call content failed: %w", err)
				}
				id := block.ID
				if toolUse.Name == "" {
					name, ok := block.ExtraFields[GeminiExtraFieldFunctionName].(string)
					if !ok {
						return nil, fmt.Errorf("missing function_name in tool call block extra fields for ID %s", id)
					}
					toolUse.Name = name
				}
				toolCallIDToFuncName[id] = toolUse.Name
				part := genai.NewPartFromFunctionCall(toolUse.Name, toolUse.Parameters)
				if sigVal, ok := block.ExtraFields[GeminiExtraFieldThoughtSignature]; ok && sigVal != nil {
					sig, err := base64.StdEncoding.DecodeString(sigVal.(string))
					if err != nil {
						return nil, fmt.Errorf("could not decode base64 thought signature: %w", err)
					}
					part.ThoughtSignature = sig
				}
				parts = append(parts, part)
			}
		}
	case ToolResult:
		role = genai.RoleUser
		for _, block := range msg.Blocks {
			id := block.ID
			fn, ok := toolCallIDToFuncName[id]
			if !ok || fn == "" {
				return nil, fmt.Errorf("tool result references unknown tool call id: %q", id)
			}
			
			switch block.ModalityType {
			case Text:
				var respObj map[string]any
				if err := json.Unmarshal([]byte(block.Content.String()), &respObj); err != nil {
					respObj = make(map[string]any)
					respObj["output"] = block.Content.String()
				}
				parts = append(parts, genai.NewPartFromFunctionResponse(fn, respObj))
			case Image, Audio:
				fileContent, decodeErr := base64.StdEncoding.DecodeString(block.Content.String())
				if decodeErr != nil {
					return nil, fmt.Errorf("decoding %s content failed: %w", block.ModalityType, decodeErr)
				}
				funcRespPart := genai.NewFunctionResponsePartFromBytes(fileContent, block.MimeType)
				parts = append(parts, genai.NewPartFromFunctionResponseWithParts(fn, nil, []*genai.FunctionResponsePart{funcRespPart}))
			default:
				return nil, fmt.Errorf("unsupported modality type in tool result: %v", block.ModalityType)
			}
		}
	default:
		return nil, fmt.Errorf("unsupported message role: %v", msg.Role)
	}

	if len(parts) == 0 {
		if role == genai.RoleUser {
			return nil, fmt.Errorf("user message resulted in no parts")
		}
	}
	return genai.NewContentFromParts(parts, role), nil
}

func prepareGeminiChatHistory(dialog Dialog, toolCallIDToFuncName map[string]string) ([]*genai.Content, error) {
	if len(dialog) == 0 {
		return nil, fmt.Errorf("empty dialog")
	}
	var history []*genai.Content
	for i, msg := range dialog {
		content, err := msgToGeminiContent(msg, toolCallIDToFuncName)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message at index %d (role: %s) to gemini content: %w", i, msg.Role, err)
		}
		if content == nil {
			return nil, fmt.Errorf("message at index %d (role: %s) converted to nil content", i, msg.Role)
		}
		history = append(history, content)
	}
	return history, nil
}

// Count implements the TokenCounter interface for GeminiGenerator.
// It converts the dialog to Gemini's format and uses Google's official CountTokens API.
//
// Like the Anthropic implementation, this method makes an API call to obtain accurate
// token counts directly from Google's tokenizer. This ensures the count matches exactly
// what would be used in actual generation.
//
// The method accounts for:
//   - System instructions (if set during generator initialization)
//   - All messages in the dialog with their respective blocks
//   - Multi-modal content including text and images
//   - Tool definitions registered with the generator
//
// Special considerations:
//   - For multi-turn conversations, all dialog turns are included in the count
//   - The system instructions are prepended to the dialog for accurate counting
//   - Image tokens are counted based on Google's own token calculation
//
// The context parameter allows for cancellation of the API call.
//
// Returns:
//   - The total token count as uint, representing the combined input tokens
//   - An error if the API call fails or if dialog conversion fails
//
// Note: Gemini's CountTokens API returns the total tokens for the entire dialog,
// including system instructions, unlike some other providers that break this down
// into more detailed metrics.
func (g *GeminiGenerator) Count(ctx context.Context, dialog Dialog) (uint, error) {
	if g.client == nil {
		return 0, fmt.Errorf("gemini: client not initialized")
	}
	if len(dialog) == 0 {
		return 0, EmptyDialogErr
	}

	// We'll need a map to track tool call IDs to function names, even though we are not executing tools.
	// This is because the prepareGeminiChatHistory function requires it.
	toolCallIDToFunc := make(map[string]string)

	allContents, err := prepareGeminiChatHistory(dialog, toolCallIDToFunc)
	if err != nil {
		return 0, fmt.Errorf("failed to prepare gemini chat history for token counting: %w", err)
	}

	var countTokenConfig genai.CountTokensConfig

	// Add system prompt if provided, as it would be part of the context for the model
	if g.systemInstructions != "" {
		// When using the below config, get error "Error counting tokens: gemini: token counting failed: systemInstruction parameter is not supported in Gemini API"
		//countTokenConfig.SystemInstruction = &genai.Content{
		//	Parts: []*genai.Part{genai.NewPartFromText(g.systemInstructions)},
		//}
		allContents = append([]*genai.Content{
			{
				Parts: []*genai.Part{genai.NewPartFromText(g.systemInstructions)},
				Role:  "model",
			},
		}, allContents...)
	}

	// If tools are registered, attach them
	if len(g.tools) > 0 {
		countTokenConfig.Tools = []*genai.Tool{
			{
				FunctionDeclarations: slices.Collect(maps.Values(g.tools)),
			},
		}
	}

	resp, err := g.client.Models.CountTokens(ctx, g.modelName, allContents, &countTokenConfig)
	if err != nil {
		var apierr *genai.APIError
		if errors.As(err, &apierr) {
			// Map HTTP status codes to our error types
			switch apierr.Code {
			case 401:
				return 0, AuthenticationErr(apierr.Error())
			case 403:
				return 0, ApiErr{
					StatusCode: apierr.Code,
					Type:       "permission_error",
					Message:    apierr.Error(),
				}
			case 404:
				return 0, ApiErr{
					StatusCode: apierr.Code,
					Type:       "not_found_error",
					Message:    apierr.Error(),
				}
			case 429:
				return 0, RateLimitErr(apierr.Error())
			case 500:
				return 0, ApiErr{
					StatusCode: apierr.Code,
					Type:       "api_error",
					Message:    apierr.Error(),
				}
			case 503:
				return 0, ApiErr{
					StatusCode: apierr.Code,
					Type:       "service_unavailable",
					Message:    apierr.Error(),
				}
			default:
				// Default to invalid_request_error for 400 and other status codes
				return 0, ApiErr{
					StatusCode: apierr.Code,
					Type:       "invalid_request_error",
					Message:    apierr.Error(),
				}
			}
		}
		return 0, fmt.Errorf("gemini: token counting failed: %w", err)
	}

	return uint(resp.TotalTokens), nil
}

var _ Generator = (*GeminiGenerator)(nil)
var _ TokenCounter = (*GeminiGenerator)(nil)
