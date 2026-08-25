package gai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"time"

	"google.golang.org/genai"

	"github.com/google/jsonschema-go/jsonschema"
)

func geminiResponseError(response *genai.GenerateContentResponse) error {
	if response == nil {
		return errors.New("gemini: empty generation response")
	}
	if feedback := response.PromptFeedback; feedback != nil && feedback.BlockReason != genai.BlockedReasonUnspecified && feedback.BlockReason != "" {
		message := feedback.BlockReasonMessage
		if message == "" {
			message = fmt.Sprintf("prompt blocked: %s", feedback.BlockReason)
		}
		return ContentPolicyErr(message)
	}
	if len(response.Candidates) == 0 || response.Candidates[0] == nil {
		return nil
	}

	candidate := response.Candidates[0]
	reason := candidate.FinishReason
	message := candidate.FinishMessage
	if message == "" {
		message = string(reason)
	}
	switch reason {
	case "", genai.FinishReasonUnspecified, genai.FinishReasonStop:
		return nil
	case genai.FinishReasonMaxTokens:
		return ErrMaxGenerationLimit
	case genai.FinishReasonSafety,
		genai.FinishReasonRecitation,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent,
		genai.FinishReasonImageRecitation:
		return ContentPolicyErr(message)
	case genai.FinishReasonMalformedFunctionCall, genai.FinishReasonUnexpectedToolCall:
		return fmt.Errorf("gemini: generation failed: %s", message)
	default:
		return fmt.Errorf("gemini: generation stopped: %s", message)
	}
}

func classifyGeminiError(apiErr genai.APIError) APIErrorKind {
	for _, detail := range apiErr.Details {
		reason, _ := detail["reason"].(string)
		switch reason {
		case "API_KEY_INVALID":
			return APIErrorKindAuthentication
		case "ACCESS_TOKEN_SCOPE_INSUFFICIENT":
			return APIErrorKindPermission
		}
	}

	switch apiErr.Status {
	case "UNAUTHENTICATED":
		return APIErrorKindAuthentication
	case "PERMISSION_DENIED":
		return APIErrorKindPermission
	case "NOT_FOUND":
		return APIErrorKindNotFound
	case "RESOURCE_EXHAUSTED":
		return APIErrorKindRateLimit
	case "DEADLINE_EXCEEDED":
		return APIErrorKindTimeout
	case "UNAVAILABLE":
		return APIErrorKindServiceUnavailable
	case "INTERNAL", "DATA_LOSS":
		return APIErrorKindServer
	case "INVALID_ARGUMENT", "FAILED_PRECONDITION", "OUT_OF_RANGE":
		return APIErrorKindInvalidRequest
	default:
		return classifyHTTPStatus(apiErr.Code)
	}
}

func retryAfterFromGeminiError(apiErr genai.APIError) *time.Duration {
	for _, detail := range apiErr.Details {
		if detail["@type"] != "type.googleapis.com/google.rpc.RetryInfo" {
			continue
		}
		value, ok := detail["retryDelay"].(string)
		if !ok {
			return nil
		}
		delay, err := time.ParseDuration(value)
		if err != nil || delay < 0 {
			return nil
		}
		return &delay
	}
	return nil
}

func mapGeminiError(err error) *ApiErr {
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		var apiErrPointer *genai.APIError
		if !errors.As(err, &apiErrPointer) || apiErrPointer == nil {
			return nil
		}
		apiErr = *apiErrPointer
	}

	return &ApiErr{
		Provider:           ProviderGemini,
		Kind:               classifyGeminiError(apiErr),
		StatusCode:         apiErr.Code,
		Message:            apiErr.Message,
		RetryAfterDuration: retryAfterFromGeminiError(apiErr),
		Cause:              err,
	}
}

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

func convertToolsToGemini(tools []Tool) ([]*genai.FunctionDeclaration, error) {
	converted := make([]*genai.FunctionDeclaration, 0, len(tools))
	seen := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		if tool.Name == "" {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be empty")}
		}
		if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be %s", tool.Name)}
		}
		if _, exists := seen[tool.Name]; exists {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool already provided")}
		}
		seen[tool.Name] = struct{}{}

		providerTool, err := convertToolToGemini(tool)
		if err != nil {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
		}
		converted = append(converted, providerTool)
	}
	return converted, nil
}

type geminiGenerationOptions struct {
	Temperature         *float64
	TopP                *float64
	TopK                *uint
	CandidateCount      *uint
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	ThinkingBudget      string
}

func parseGeminiGenerationOptions(values GenerationOptions) (*geminiGenerationOptions, error) {
	options := &geminiGenerationOptions{}

	temperature, ok, err := generationOption[float64](values, GenerationOptionTemperature)
	if err != nil {
		return nil, err
	}
	if ok {
		options.Temperature = &temperature
	}
	topP, ok, err := generationOption[float64](values, GenerationOptionTopP)
	if err != nil {
		return nil, err
	}
	if ok {
		options.TopP = &topP
	}
	topK, ok, err := generationOption[uint](values, GenerationOptionTopK)
	if err != nil {
		return nil, err
	}
	if ok {
		options.TopK = &topK
	}
	candidateCount, ok, err := generationOption[uint](values, GenerationOptionCandidateCount)
	if err != nil {
		return nil, err
	}
	if ok {
		options.CandidateCount = &candidateCount
	}
	maxTokens, ok, err := generationOption[int](values, GenerationOptionMaxGenerationTokens)
	if err != nil {
		return nil, err
	}
	if ok {
		options.MaxGenerationTokens = &maxTokens
	}
	if options.ToolChoice, _, err = generationOption[string](values, GenerationOptionToolChoice); err != nil {
		return nil, err
	}
	if options.StopSequences, _, err = generationOption[[]string](values, GenerationOptionStopSequences); err != nil {
		return nil, err
	}
	if options.ThinkingBudget, _, err = generationOption[string](values, GenerationOptionThinkingBudget); err != nil {
		return nil, err
	}
	return options, nil
}

type GeminiGenerator struct {
	client *genai.Client
}

// NewGeminiGenerator creates a stateless Gemini generator.
// It preprocesses parallel tool results for Gemini compatibility.
//
// Parameters:
//   - client: A properly initialized genai.Client instance with API key configured
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
// Returns a generator that also implements StreamingGenerator and TokenCounter.
func NewGeminiGenerator(client *genai.Client) interface {
	Generator
	StreamingGenerator
	TokenCounter
} {
	inner := &GeminiGenerator{client: client}
	return &PreprocessingGenerator{GeneratorWrapper: GeneratorWrapper{Inner: inner}}
}

// Generate implements gai.Generator
func (g *GeminiGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("gemini: client not initialized")
	}
	dialog := request.Dialog
	if len(dialog) == 0 {
		return Response{}, ErrEmptyDialog
	}
	options, err := parseGeminiGenerationOptions(request.Options)
	if err != nil {
		return Response{}, err
	}
	tools, err := convertToolsToGemini(request.Tools)
	if err != nil {
		return Response{}, err
	}
	instructions, err := textInstructions(request.Instructions)
	if err != nil {
		return Response{}, err
	}

	// We'll keep a mapping of toolCallID -> functionName for this call.
	toolCallIDToFunc := make(map[string]string)
	toolCallCount := 0

	genContentConfig := &genai.GenerateContentConfig{
		ThinkingConfig: &genai.ThinkingConfig{
			IncludeThoughts: true,
		},
	}
	if len(instructions) > 0 {
		parts := make([]*genai.Part, 0, len(instructions))
		for _, instruction := range instructions {
			parts = append(parts, genai.NewPartFromText(instruction))
		}
		genContentConfig.SystemInstruction = &genai.Content{Parts: parts}
	}
	if len(tools) > 0 {
		genContentConfig.Tools = []*genai.Tool{{FunctionDeclarations: tools}}
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

		if options.Temperature != nil {
			genContentConfig.Temperature = genai.Ptr(float32(*options.Temperature))
		}
		if options.MaxGenerationTokens != nil {
			genContentConfig.MaxOutputTokens = int32(*options.MaxGenerationTokens)
		}
		if options.CandidateCount != nil && *options.CandidateCount > 1 {
			genContentConfig.CandidateCount = int32(*options.CandidateCount)
		}
		if options.StopSequences != nil {
			genContentConfig.StopSequences = options.StopSequences
		}
		if options.TopP != nil {
			genContentConfig.TopP = genai.Ptr(float32(*options.TopP))
		}
		if options.TopK != nil {
			genContentConfig.TopK = genai.Ptr(float32(*options.TopK))
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

	resp, err := g.client.Models.GenerateContent(ctx, request.Model, allContents, genContentConfig)
	if err != nil {
		if mapped := mapGeminiError(err); mapped != nil {
			return Response{}, mapped
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
		// CachedContentTokenCount represents tokens read from cached content
		if resp.UsageMetadata.CachedContentTokenCount > 0 {
			result.UsageMetadata[UsageMetricCacheReadTokens] = int(resp.UsageMetadata.CachedContentTokenCount)
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

	if err := geminiResponseError(resp); err != nil {
		if errors.Is(err, ErrMaxGenerationLimit) {
			result.FinishReason = MaxGenerationLimit
		}
		var policyErr ContentPolicyErr
		if errors.As(err, &policyErr) {
			result.FinishReason = ContentPolicyViolation
		}
		return result, err
	}
	if len(resp.Candidates) > 0 && resp.Candidates[0] != nil {
		result.FinishReason = EndTurn
	}

	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}

	return result, nil
}

func (g *GeminiGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("gemini: client not initialized")})
			return
		}

		dialog := request.Dialog
		if len(dialog) == 0 {
			yield(StreamChunk{Err: ErrEmptyDialog})
			return
		}
		options, err := parseGeminiGenerationOptions(request.Options)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		tools, err := convertToolsToGemini(request.Tools)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		instructions, err := textInstructions(request.Instructions)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}

		// We'll keep a mapping of toolCallID -> functionName for this call.
		toolCallIDToFunc := make(map[string]string)
		toolCallCount := 0

		genContentConfig := &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true},
		}
		if len(instructions) > 0 {
			parts := make([]*genai.Part, 0, len(instructions))
			for _, instruction := range instructions {
				parts = append(parts, genai.NewPartFromText(instruction))
			}
			genContentConfig.SystemInstruction = &genai.Content{Parts: parts}
		}
		if len(tools) > 0 {
			genContentConfig.Tools = []*genai.Tool{{FunctionDeclarations: tools}}
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

			if options.Temperature != nil {
				genContentConfig.Temperature = genai.Ptr(float32(*options.Temperature))
			}
			if options.MaxGenerationTokens != nil {
				genContentConfig.MaxOutputTokens = int32(*options.MaxGenerationTokens)
			}
			if options.CandidateCount != nil && *options.CandidateCount > 1 {
				genContentConfig.CandidateCount = int32(*options.CandidateCount)
			}
			if options.StopSequences != nil {
				genContentConfig.StopSequences = options.StopSequences
			}
			if options.TopP != nil {
				genContentConfig.TopP = genai.Ptr(float32(*options.TopP))
			}
			if options.TopK != nil {
				genContentConfig.TopK = genai.Ptr(float32(*options.TopK))
			}
			if options.ThinkingBudget != "" {
				switch options.ThinkingBudget {
				case "low", "medium", "high":
					genContentConfig.ThinkingConfig.ThinkingLevel = genai.ThinkingLevel(options.ThinkingBudget)
				default:
					yield(StreamChunk{Err: InvalidParameterErr{Parameter: "thinking budget", Reason: fmt.Sprintf("invalid thinking budget: %s", options.ThinkingBudget)}})
					return
				}
			}
		}

		allContents, err := prepareGeminiChatHistory(dialog, toolCallIDToFunc)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}

		// Track cumulative usage
		var totalInputTokens, totalOutputTokens, totalCacheReadTokens int32

		for resp, err := range g.client.Models.GenerateContentStream(ctx, request.Model, allContents, genContentConfig) {
			if err != nil {
				if mapped := mapGeminiError(err); mapped != nil {
					yield(StreamChunk{Err: mapped})
				} else {
					yield(StreamChunk{Err: fmt.Errorf("gemini: generation failed: %w", err)})
				}
				return
			}

			if err := geminiResponseError(resp); err != nil {
				yield(StreamChunk{Err: err})
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
				// CachedContentTokenCount represents tokens read from cached content
				if resp.UsageMetadata.CachedContentTokenCount > 0 {
					totalCacheReadTokens = resp.UsageMetadata.CachedContentTokenCount
				}
			}

			if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil || len(resp.Candidates[0].Content.Parts) == 0 {
				if !yield(StreamChunk{
					Block:           TextBlock(""),
					CandidatesIndex: 0,
				}) {
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
								ExtraFields: map[string]interface{}{
									ThinkingExtraFieldGeneratorKey: ThinkingGeneratorGemini,
								},
							},
							CandidatesIndex: 0,
						}) {
							return
						}
					} else {
						if !yield(StreamChunk{
							Block: TextBlock(part.Text),
						}) {
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
							}) {
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
							}) {
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
		if totalInputTokens > 0 || totalOutputTokens > 0 || totalCacheReadTokens > 0 {
			metadata := make(Metadata)

			if totalInputTokens > 0 {
				metadata[UsageMetricInputTokens] = int(totalInputTokens)
			}
			if totalOutputTokens > 0 {
				metadata[UsageMetricGenerationTokens] = int(totalOutputTokens)
			}
			// CachedContentTokenCount represents tokens read from cached content
			if totalCacheReadTokens > 0 {
				metadata[UsageMetricCacheReadTokens] = int(totalCacheReadTokens)
			}

			yield(StreamChunk{
				Block:           MetadataBlock(metadata),
				CandidatesIndex: 0,
			})
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

// Count sends the request's model, instructions, dialog, and tools to Gemini's
// CountTokens API. It includes multimodal input and all conversation turns. The
// context can cancel the remote call.
func (g *GeminiGenerator) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	if g.client == nil {
		return 0, fmt.Errorf("gemini: client not initialized")
	}
	dialog := request.Dialog
	if len(dialog) == 0 {
		return 0, ErrEmptyDialog
	}
	tools, err := convertToolsToGemini(request.Tools)
	if err != nil {
		return 0, err
	}
	instructions, err := textInstructions(request.Instructions)
	if err != nil {
		return 0, err
	}

	// We'll need a map to track tool call IDs to function names, even though we are not executing tools.
	// This is because the prepareGeminiChatHistory function requires it.
	toolCallIDToFunc := make(map[string]string)

	allContents, err := prepareGeminiChatHistory(dialog, toolCallIDToFunc)
	if err != nil {
		return 0, fmt.Errorf("failed to prepare gemini chat history for token counting: %w", err)
	}

	var countTokenConfig genai.CountTokensConfig

	if len(instructions) > 0 {
		parts := make([]*genai.Part, 0, len(instructions))
		for _, instruction := range instructions {
			parts = append(parts, genai.NewPartFromText(instruction))
		}
		// CountTokens does not accept SystemInstruction, so include the same text in
		// the counted contents.
		allContents = append([]*genai.Content{{Parts: parts, Role: "model"}}, allContents...)
	}

	if len(tools) > 0 {
		countTokenConfig.Tools = []*genai.Tool{{FunctionDeclarations: tools}}
	}

	resp, err := g.client.Models.CountTokens(ctx, request.Model, allContents, &countTokenConfig)
	if err != nil {
		if mapped := mapGeminiError(err); mapped != nil {
			return 0, mapped
		}
		return 0, fmt.Errorf("gemini: token counting failed: %w", err)
	}

	return uint(resp.TotalTokens), nil
}

var _ Generator = (*GeminiGenerator)(nil)
var _ StreamingGenerator = (*GeminiGenerator)(nil)
var _ TokenCounter = (*GeminiGenerator)(nil)
