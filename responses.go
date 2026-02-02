package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
)

// ResponsesThoughtSummaryDetailParam is a key used for storing the thought summary detail level
// in GenOpts.ExtraArgs. Setting parameter will set the level of detail of thought summaries that
// are returned from the OpenAI Responses API. One of `auto`, `concise`, or `detailed`.
const ResponsesThoughtSummaryDetailParam = "responses_thought_summary_detail"

// ResponsesPrevRespId is a key used for storing the previous response id in GenOpts.ExtraArgs.
// When set, the previous response id will set in the param request. This allows the model to
// refer back to earlier reasoning traces stored in the OpenAI Responses API, boosting performance.
const ResponsesPrevRespId = "responses_prev_resp_id"

type ResponsesService interface {
	New(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (res *responses.Response, err error)
	NewStreaming(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (stream *ssestream.Stream[responses.ResponseStreamEventUnion])
}

// ResponsesGenerator is a generator that calls OpenAI models via the Responses API
type ResponsesGenerator struct {
	client             ResponsesService
	model              string
	tools              map[string]responses.ToolUnionParam
	systemInstructions string
}

// NewResponsesGenerator creates a new OpenAI Responses API generator with the specified model.
// The returned generator implements the Generator and ToolRegister interfaces.
//
// Parameters:
//   - client: An OpenAI completion service (typically &client.Responses)
//   - model: The OpenAI model to use (e.g., "gpt-5")
//   - systemInstructions: Optional system instructions that set the model's behavior
//
// Supported modalities:
//   - Text: Both input and output
//   - Image: Input only (base64 encoded, including PDFs with MIME type "application/pdf")
//   - Audio: Input only (base64 encoded, WAV and MP3 formats)
//
// For audio input, use models with audio support like:
//   - openai.ChatModelGPT4oAudioPreview
//   - openai.ChatModelGPT4oMiniAudioPreview
//
// PDF documents are supported as a special case of the Image modality. Use the PDFBlock
// helper function to create PDF content blocks.
//
// This generator fully supports the anyOf JSON Schema feature.
func NewResponsesGenerator(client ResponsesService, model, systemInstructions string) ResponsesGenerator {
	return ResponsesGenerator{
		client:             client,
		systemInstructions: systemInstructions,
		model:              model,
		tools:              make(map[string]responses.ToolUnionParam),
	}
}

var _ ToolRegister = (*ResponsesGenerator)(nil)
var _ Generator = (*ResponsesGenerator)(nil)
var _ StreamingGenerator = (*ResponsesGenerator)(nil)

func (r *ResponsesGenerator) Register(tool Tool) error {
	if tool.Name == "" {
		return &ToolRegistrationErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be empty")}
	}
	if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
		return &ToolRegistrationErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be %s", tool.Name)}
	}
	if r.tools == nil {
		r.tools = make(map[string]responses.ToolUnionParam)
	}
	if _, exists := r.tools[tool.Name]; exists {
		return &ToolRegistrationErr{Tool: tool.Name, Cause: fmt.Errorf("tool already registered")}
	}

	params := make(map[string]any)
	if tool.InputSchema != nil {
		b, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return &ToolRegistrationErr{Tool: tool.Name, Cause: err}
		}
		if err := json.Unmarshal(b, &params); err != nil {
			return &ToolRegistrationErr{Tool: tool.Name, Cause: err}
		}
	}
	if len(params) == 1 {
		if t, ok := params["type"].(string); ok && t == "object" {
			params = map[string]any{}
		}
	}

	fn := responses.ToolParamOfFunction(tool.Name, params, false)
	fn.OfFunction.Description = openai.String(tool.Description)
	r.tools[tool.Name] = fn
	return nil
}

func (r *ResponsesGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if r.client == nil {
		return Response{}, fmt.Errorf("responses: client not initialized")
	}
	if len(dialog) == 0 {
		return Response{}, EmptyDialogErr
	}

	var inputItems []responses.ResponseInputItemUnionParam
	for _, msg := range dialog {
		switch msg.Role {
		case User:
			var contents responses.ResponseInputMessageContentListParam

			for _, blk := range msg.Blocks {
				switch blk.BlockType {
				case Content:
					switch blk.ModalityType {
					case Text:
						contents = append(contents, responses.ResponseInputContentParamOfInputText(blk.Content.String()))
					case Image:
						if blk.MimeType == "" {
							return Response{}, fmt.Errorf("image block missing mimetype")
						}
						dataURL := fmt.Sprintf("data:%s;base64,%s", blk.MimeType, blk.Content.String())
						if blk.MimeType == "application/pdf" {
							val, ok := blk.ExtraFields[BlockFieldFilenameKey]
							if !ok {
								return Response{}, fmt.Errorf("filename field missing in extra fields")
							}
							filename, ok := val.(string)
							if !ok {
								return Response{}, fmt.Errorf("filename field is not a string")
							}
							file := responses.ResponseInputContentUnionParam{OfInputFile: &responses.ResponseInputFileParam{}}
							file.OfInputFile.FileData = openai.Opt(dataURL)
							file.OfInputFile.Filename = openai.Opt(filename)
							contents = append(contents, file)
						} else {
							img := responses.ResponseInputContentParamOfInputImage(responses.ResponseInputImageDetailAuto)
							img.OfInputImage.ImageURL = openai.Opt(dataURL)
							contents = append(contents, img)
						}
					default:
						return Response{}, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
				default:
					return Response{}, fmt.Errorf("unsupported block type in message: %s", blk.BlockType)
				}
			}

			if len(contents) > 0 {
				role := responses.EasyInputMessageRoleUser
				inputItems = append(inputItems, responses.ResponseInputItemParamOfMessage(contents, role))
			}

		case Assistant:
			var contents []responses.ResponseOutputMessageContentUnionParam
			var toolCalls []responses.ResponseInputItemUnionParam

			for _, blk := range msg.Blocks {
				switch blk.BlockType {
				case Content:
					switch blk.ModalityType {
					case Text:
						contents = append(contents, responses.ResponseOutputMessageContentUnionParam{
							OfOutputText: &responses.ResponseOutputTextParam{
								Text: blk.Content.String(),
							},
						})
					default:
						return Response{}, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
				case Thinking:
					// ignore thinking type blocks, responses API does not support injected thinking traces
				case ToolCall:
					if blk.ID == "" {
						return Response{}, fmt.Errorf("tool call block missing ID")
					}
					var tci ToolCallInput
					if err := json.Unmarshal([]byte(blk.Content.String()), &tci); err != nil {
						return Response{}, fmt.Errorf("invalid tool call content: %w", err)
					}
					argsJSON, err := json.Marshal(tci.Parameters)
					if err != nil {
						return Response{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
					}
					toolCalls = append(toolCalls, responses.ResponseInputItemParamOfFunctionCall(string(argsJSON), blk.ID, tci.Name))
				default:
					return Response{}, fmt.Errorf("unsupported block type in message: %s", blk.BlockType)
				}
			}

			if len(contents) > 0 {
				inputItems = append(inputItems, responses.ResponseInputItemUnionParam{
					OfOutputMessage: &responses.ResponseOutputMessageParam{
						ID:      "",
						Content: contents,
					},
				})
			}
			if len(toolCalls) > 0 {
				inputItems = append(inputItems, toolCalls...)
			}

		case ToolResult:
			if len(msg.Blocks) == 0 {
				return Response{}, fmt.Errorf("tool result message must have at least one block")
			}
			toolID := msg.Blocks[0].ID
			if toolID == "" {
				return Response{}, fmt.Errorf("tool result message block must have an ID")
			}

			// Check if all blocks are text-only (simple case)
			allText := true
			for _, blk := range msg.Blocks {
				if blk.ModalityType != Text {
					allText = false
					break
				}
			}

			if allText {
				// Simple text-only case: concatenate all text blocks
				var sb strings.Builder
				for _, blk := range msg.Blocks {
					if blk.ID != toolID {
						return Response{}, fmt.Errorf("all blocks in tool result must share the same ID")
					}
					sb.WriteString(blk.Content.String())
				}
				inputItems = append(inputItems, responses.ResponseInputItemParamOfFunctionCallOutput(toolID, sb.String()))
			} else {
				// Complex case: blocks with images/PDFs - use array output format
				var outputItems responses.ResponseFunctionCallOutputItemListParam
				for _, blk := range msg.Blocks {
					if blk.ID != toolID {
						return Response{}, fmt.Errorf("all blocks in tool result must share the same ID")
					}
					switch blk.ModalityType {
					case Text:
						outputItems = append(outputItems, responses.ResponseFunctionCallOutputItemParamOfInputText(blk.Content.String()))
					case Image:
						if blk.MimeType == "application/pdf" {
							// PDF files use input_file type
							fileParam := responses.ResponseInputFileContentParam{
								FileData: openai.String("data:" + blk.MimeType + ";base64," + blk.Content.String()),
							}
							if filename, ok := blk.ExtraFields[BlockFieldFilenameKey].(string); ok && filename != "" {
								fileParam.Filename = openai.String(filename)
							}
							outputItems = append(outputItems, responses.ResponseFunctionCallOutputItemUnionParam{
								OfInputFile: &fileParam,
							})
						} else {
							// Regular images use input_image type with data URL
							outputItems = append(outputItems, responses.ResponseFunctionCallOutputItemUnionParam{
								OfInputImage: &responses.ResponseInputImageContentParam{
									ImageURL: openai.String("data:" + blk.MimeType + ";base64," + blk.Content.String()),
								},
							})
						}
					default:
						return Response{}, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
				}
				inputItems = append(inputItems, responses.ResponseInputItemParamOfFunctionCallOutput(toolID, outputItems))
			}
		default:
			return Response{}, fmt.Errorf("unsupported message role: %s", msg.Role.String())
		}
	}

	params := responses.ResponseNewParams{
		Model: r.model,
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: inputItems},
	}

	if r.systemInstructions != "" {
		params.Instructions = openai.Opt(r.systemInstructions)
	}

	if len(r.tools) > 0 {
		var tools []responses.ToolUnionParam
		for _, t := range r.tools {
			tools = append(tools, t)
		}
		params.Tools = tools
	}

	if options != nil {
		if options.Temperature != 0 {
			params.Temperature = openai.Opt(options.Temperature)
		}
		if options.TopP != 0 {
			params.TopP = openai.Opt(options.TopP)
		}
		if options.MaxGenerationTokens > 0 {
			params.MaxOutputTokens = openai.Opt(int64(options.MaxGenerationTokens))
		}
		if options.ToolChoice != "" {
			switch options.ToolChoice {
			case ToolChoiceAuto:
				params.ToolChoice.OfToolChoiceMode = openai.Opt(responses.ToolChoiceOptionsAuto)
			case ToolChoiceToolsRequired:
				params.ToolChoice.OfToolChoiceMode = openai.Opt(responses.ToolChoiceOptionsRequired)
			default:
				params.ToolChoice.OfFunctionTool = &responses.ToolChoiceFunctionParam{Name: options.ToolChoice}
			}
		}
		if options.ThinkingBudget != "" {
			switch options.ThinkingBudget {
			case "minimal", "low", "medium", "high":
				params.Reasoning = responses.ReasoningParam{Effort: responses.ReasoningEffort(options.ThinkingBudget)}
			default:
				return Response{}, InvalidParameterErr{Parameter: "thinking budget", Reason: fmt.Sprintf("invalid thinking budget: %s", options.ThinkingBudget)}
			}
			if options.ExtraArgs != nil {
				if val, ok := options.ExtraArgs[ResponsesThoughtSummaryDetailParam]; ok {
					params.Reasoning.Summary = val.(responses.ReasoningSummary)
				}
			}
		}

		if len(options.OutputModalities) > 0 {
			for _, m := range options.OutputModalities {
				if m != Text {
					return Response{}, UnsupportedOutputModalityErr(m.String())
				}
			}
		}
		if val, ok := options.ExtraArgs[ResponsesPrevRespId]; ok {
			prevId := val.(string)
			if prevId != "" {
				params.PreviousResponseID = openai.Opt[string](prevId)
			}
		}
	}

	res, err := r.client.New(ctx, params)
	if err != nil {
		var apierr *responses.Error
		if errors.As(err, &apierr) {
			switch apierr.StatusCode {
			case 401:
				return Response{}, AuthenticationErr(apierr.Error())
			case 403:
				return Response{}, ApiErr{StatusCode: apierr.StatusCode, Type: "permission_error", Message: apierr.Error()}
			case 404:
				return Response{}, ApiErr{StatusCode: apierr.StatusCode, Type: "not_found_error", Message: apierr.Error()}
			case 429:
				return Response{}, RateLimitErr(apierr.Error())
			case 500:
				return Response{}, ApiErr{StatusCode: apierr.StatusCode, Type: "api_error", Message: apierr.Error()}
			case 503:
				return Response{}, ApiErr{StatusCode: apierr.StatusCode, Type: "service_unavailable", Message: apierr.Error()}
			default:
				return Response{}, ApiErr{StatusCode: apierr.StatusCode, Type: "invalid_request_error", Message: apierr.Error()}
			}
		}
		return Response{}, fmt.Errorf("responses: generation failed: %w", err)
	}

	result := Response{UsageMetadata: make(Metadata)}
	if usage := res.Usage; usage.InputTokens > 0 || usage.OutputTokens > 0 {
		if usage.InputTokens > 0 {
			result.UsageMetadata[UsageMetricInputTokens] = int(usage.InputTokens)
		}
		if usage.OutputTokens > 0 {
			result.UsageMetadata[UsageMetricGenerationTokens] = int(usage.OutputTokens)
		}
		// Report cached tokens if available (OpenAI Responses API prompt caching)
		if usage.InputTokensDetails.CachedTokens > 0 {
			result.UsageMetadata[UsageMetricCacheReadTokens] = int(usage.InputTokensDetails.CachedTokens)
		}
	}
	result.UsageMetadata[ResponsesPrevRespId] = res.ID

	var blocks []Block
	var hasToolCalls bool
	var refusal string
	for _, item := range res.Output {
		switch item.Type {
		case "message":
			msg := item.AsMessage()
			for _, c := range msg.Content {
				switch c.Type {
				case "output_text":
					blocks = append(blocks, Block{BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str(c.Text)})
				case "refusal":
					if c.Refusal != "" {
						refusal = c.Refusal
					}
				}
			}
		case "function_call":
			fc := item.AsFunctionCall()
			var paramsMap map[string]any
			if fc.Arguments != "" {
				_ = json.Unmarshal([]byte(fc.Arguments), &paramsMap)
			}
			tci := ToolCallInput{Name: fc.Name, Parameters: paramsMap}
			b, _ := json.Marshal(tci)
			id := fc.CallID
			if id == "" {
				id = fc.ID
			}
			blocks = append(blocks, Block{ID: id, BlockType: ToolCall, ModalityType: Text, MimeType: "application/json", Content: Str(b)})
			hasToolCalls = true
		case "reasoning":
			reas := item.AsReasoning()
			// Process actual reasoning content as Thinking blocks
			for _, rc := range reas.Content {
				blocks = append(blocks, Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str(rc.Text)})
			}
			// Also process summaries as Thinking blocks (as close as we get when full traces unavailable)
			for _, rc := range reas.Summary {
				blocks = append(blocks, Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str(rc.Text)})
			}
		}
	}

	result.Candidates = append(result.Candidates, Message{Role: Assistant, Blocks: blocks})

	if res.IncompleteDetails.Reason == "max_output_tokens" {
		result.FinishReason = MaxGenerationLimit
		return result, MaxGenerationLimitErr
	}
	if res.IncompleteDetails.Reason == "content_filter" || refusal != "" {
		result.FinishReason = Unknown
		if refusal == "" {
			refusal = "content policy violation detected"
		}
		return result, ContentPolicyErr(refusal)
	}
	if hasToolCalls {
		result.FinishReason = ToolUse
	} else {
		result.FinishReason = EndTurn
	}
	return result, nil
}

func (r *ResponsesGenerator) Stream(ctx context.Context, dialog Dialog, options *GenOpts) iter.Seq2[StreamChunk, error] {
	return func(yield func(StreamChunk, error) bool) {
		if r.client == nil {
			yield(StreamChunk{}, fmt.Errorf("responses: client not initialized"))
			return
		}
		if len(dialog) == 0 {
			yield(StreamChunk{}, EmptyDialogErr)
			return
		}

		var inputItems []responses.ResponseInputItemUnionParam
		for _, msg := range dialog {
			switch msg.Role {
			case User:
				var contents responses.ResponseInputMessageContentListParam

				for _, blk := range msg.Blocks {
					switch blk.BlockType {
					case Content:
						switch blk.ModalityType {
						case Text:
							contents = append(contents, responses.ResponseInputContentParamOfInputText(blk.Content.String()))
						case Image:
							if blk.MimeType == "" {
								yield(StreamChunk{}, fmt.Errorf("image block missing mimetype"))
								return
							}
							dataURL := fmt.Sprintf("data:%s;base64,%s", blk.MimeType, blk.Content.String())
							if blk.MimeType == "application/pdf" {
								val, ok := blk.ExtraFields[BlockFieldFilenameKey]
								if !ok {
									yield(StreamChunk{}, fmt.Errorf("filename field missing in extra fields"))
									return
								}
								filename, ok := val.(string)
								if !ok {
									yield(StreamChunk{}, fmt.Errorf("filename field is not a string"))
									return
								}
								file := responses.ResponseInputContentUnionParam{OfInputFile: &responses.ResponseInputFileParam{}}
								file.OfInputFile.FileData = openai.Opt(dataURL)
								file.OfInputFile.Filename = openai.Opt(filename)
								contents = append(contents, file)
							} else {
								img := responses.ResponseInputContentParamOfInputImage(responses.ResponseInputImageDetailAuto)
								img.OfInputImage.ImageURL = openai.Opt(dataURL)
								contents = append(contents, img)
							}
						default:
							yield(StreamChunk{}, UnsupportedInputModalityErr(blk.ModalityType.String()))
							return
						}
					default:
						yield(StreamChunk{}, fmt.Errorf("unsupported block type in message: %s", blk.BlockType))
						return
					}
				}

				if len(contents) > 0 {
					role := responses.EasyInputMessageRoleUser
					inputItems = append(inputItems, responses.ResponseInputItemParamOfMessage(contents, role))
				}

			case Assistant:
				var contents []responses.ResponseOutputMessageContentUnionParam
				var toolCalls []responses.ResponseInputItemUnionParam

				for _, blk := range msg.Blocks {
					switch blk.BlockType {
					case Content:
						switch blk.ModalityType {
						case Text:
							contents = append(contents, responses.ResponseOutputMessageContentUnionParam{
								OfOutputText: &responses.ResponseOutputTextParam{
									Text: blk.Content.String(),
								},
							})
						default:
							yield(StreamChunk{}, UnsupportedInputModalityErr(blk.ModalityType.String()))
							return
						}
					case Thinking:
						// ignore thinking type blocks, responses API does not support injected thinking traces
					case ToolCall:
						if blk.ID == "" {
							yield(StreamChunk{}, fmt.Errorf("tool call block missing ID"))
							return
						}
						var tci ToolCallInput
						if err := json.Unmarshal([]byte(blk.Content.String()), &tci); err != nil {
							yield(StreamChunk{}, fmt.Errorf("invalid tool call content: %w", err))
							return
						}
						argsJSON, err := json.Marshal(tci.Parameters)
						if err != nil {
							yield(StreamChunk{}, fmt.Errorf("failed to marshal tool parameters: %w", err))
							return
						}
						toolCalls = append(toolCalls, responses.ResponseInputItemParamOfFunctionCall(string(argsJSON), blk.ID, tci.Name))
					default:
						yield(StreamChunk{}, fmt.Errorf("unsupported block type in message: %s", blk.BlockType))
						return
					}
				}

				if len(contents) > 0 {
					inputItems = append(inputItems, responses.ResponseInputItemUnionParam{
						OfOutputMessage: &responses.ResponseOutputMessageParam{
							ID:      "",
							Content: contents,
						},
					})
				}
				if len(toolCalls) > 0 {
					inputItems = append(inputItems, toolCalls...)
				}

			case ToolResult:
				if len(msg.Blocks) == 0 {
					yield(StreamChunk{}, fmt.Errorf("tool result message must have at least one block"))
					return
				}
				toolID := msg.Blocks[0].ID
				if toolID == "" {
					yield(StreamChunk{}, fmt.Errorf("tool result message block must have an ID"))
					return
				}

				// Check if all blocks are text-only (simple case)
				allText := true
				for _, blk := range msg.Blocks {
					if blk.ModalityType != Text {
						allText = false
						break
					}
				}

				if allText {
					// Simple text-only case: concatenate all text blocks
					var sb strings.Builder
					for _, blk := range msg.Blocks {
						if blk.ID != toolID {
							yield(StreamChunk{}, fmt.Errorf("all blocks in tool result must share the same ID"))
							return
						}
						sb.WriteString(blk.Content.String())
					}
					inputItems = append(inputItems, responses.ResponseInputItemParamOfFunctionCallOutput(toolID, sb.String()))
				} else {
					// Complex case: blocks with images/PDFs - use array output format
					var outputItems responses.ResponseFunctionCallOutputItemListParam
					for _, blk := range msg.Blocks {
						if blk.ID != toolID {
							yield(StreamChunk{}, fmt.Errorf("all blocks in tool result must share the same ID"))
							return
						}
						switch blk.ModalityType {
						case Text:
							outputItems = append(outputItems, responses.ResponseFunctionCallOutputItemParamOfInputText(blk.Content.String()))
						case Image:
							if blk.MimeType == "application/pdf" {
								// PDF files use input_file type
								fileParam := responses.ResponseInputFileContentParam{
									FileData: openai.String("data:" + blk.MimeType + ";base64," + blk.Content.String()),
								}
								if filename, ok := blk.ExtraFields[BlockFieldFilenameKey].(string); ok && filename != "" {
									fileParam.Filename = openai.String(filename)
								}
								outputItems = append(outputItems, responses.ResponseFunctionCallOutputItemUnionParam{
									OfInputFile: &fileParam,
								})
							} else {
								// Regular images use input_image type with data URL
								outputItems = append(outputItems, responses.ResponseFunctionCallOutputItemUnionParam{
									OfInputImage: &responses.ResponseInputImageContentParam{
										ImageURL: openai.String("data:" + blk.MimeType + ";base64," + blk.Content.String()),
									},
								})
							}
						default:
							yield(StreamChunk{}, UnsupportedInputModalityErr(blk.ModalityType.String()))
							return
						}
					}
					inputItems = append(inputItems, responses.ResponseInputItemParamOfFunctionCallOutput(toolID, outputItems))
				}
			default:
				yield(StreamChunk{}, fmt.Errorf("unsupported message role: %s", msg.Role.String()))
				return
			}
		}

		params := responses.ResponseNewParams{
			Model: r.model,
			Input: responses.ResponseNewParamsInputUnion{OfInputItemList: inputItems},
		}

		if r.systemInstructions != "" {
			params.Instructions = openai.Opt(r.systemInstructions)
		}

		if len(r.tools) > 0 {
			var tools []responses.ToolUnionParam
			for _, t := range r.tools {
				tools = append(tools, t)
			}
			params.Tools = tools
		}

		if options != nil {
			if options.Temperature != 0 {
				params.Temperature = openai.Opt(options.Temperature)
			}
			if options.TopP != 0 {
				params.TopP = openai.Opt(options.TopP)
			}
			if options.MaxGenerationTokens > 0 {
				params.MaxOutputTokens = openai.Opt(int64(options.MaxGenerationTokens))
			}
			if options.ToolChoice != "" {
				switch options.ToolChoice {
				case ToolChoiceAuto:
					params.ToolChoice.OfToolChoiceMode = openai.Opt(responses.ToolChoiceOptionsAuto)
				case ToolChoiceToolsRequired:
					params.ToolChoice.OfToolChoiceMode = openai.Opt(responses.ToolChoiceOptionsRequired)
				default:
					params.ToolChoice.OfFunctionTool = &responses.ToolChoiceFunctionParam{Name: options.ToolChoice}
				}
			}
			if options.ThinkingBudget != "" {
				switch options.ThinkingBudget {
				case "minimal", "low", "medium", "high":
					params.Reasoning = responses.ReasoningParam{Effort: responses.ReasoningEffort(options.ThinkingBudget)}
				default:
					yield(StreamChunk{}, InvalidParameterErr{Parameter: "thinking budget", Reason: fmt.Sprintf("invalid thinking budget: %s", options.ThinkingBudget)})
					return
				}
				if options.ExtraArgs != nil {
					if val, ok := options.ExtraArgs[ResponsesThoughtSummaryDetailParam]; ok {
						params.Reasoning.Summary = val.(responses.ReasoningSummary)
					}
				}
			}

			if len(options.OutputModalities) > 0 {
				for _, m := range options.OutputModalities {
					if m != Text {
						yield(StreamChunk{}, UnsupportedOutputModalityErr(m.String()))
						return
					}
				}
			}
			if val, ok := options.ExtraArgs[ResponsesPrevRespId]; ok {
				prevId := val.(string)
				if prevId != "" {
					params.PreviousResponseID = openai.Opt[string](prevId)
				}
			}
		}

		// Start the stream
		stream := r.client.NewStreaming(ctx, params)
		defer stream.Close()

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_item.added":
				item := event.AsResponseOutputItemAdded().Item
				switch item.Type {
				case "message":
					msg := item.AsMessage()
					for _, c := range msg.Content {
						switch c.Type {
						case "output_text":
							if !yield(StreamChunk{
								Block: Block{
									BlockType:    Content,
									ModalityType: Text,
									MimeType:     "text/plain",
									Content:      Str(c.Text),
								},
								CandidatesIndex: 0,
							}, nil) {
								return
							}
						case "refusal":
							if c.Refusal != "" {
								yield(StreamChunk{}, ContentPolicyErr(c.Refusal))
								return
							}
						}
					}
				case "function_call":
					fc := item.AsFunctionCall()
					if !yield(StreamChunk{
						Block: Block{
							ID:           fc.CallID,
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "application/json",
							Content:      Str(fc.Name),
						},
						CandidatesIndex: 0,
					}, nil) {
						return
					}
				case "reasoning":
					reas := item.AsReasoning()
					// Process actual reasoning content
					for _, rc := range reas.Content {
						if !yield(StreamChunk{
							Block: Block{
								BlockType:    Thinking,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(rc.Text),
							},
							CandidatesIndex: 0,
						}, nil) {
							return
						}
					}
					// Also process summaries as Thinking blocks
					for _, rc := range reas.Summary {
						if !yield(StreamChunk{
							Block: Block{
								BlockType:    Thinking,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(rc.Text),
							},
							CandidatesIndex: 0,
						}, nil) {
							return
						}
					}
				}
			case "response.output_text.delta":
				textDelta := event.AsResponseOutputTextDelta()
				if textDelta.Delta != "" {
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    Content,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(textDelta.Delta),
						},
						CandidatesIndex: 0,
					}, nil) {
						return
					}
				}
			case "response.function_call_arguments.delta":
				fcDelta := event.AsResponseFunctionCallArgumentsDelta()
				if fcDelta.Delta != "" {
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(fcDelta.Delta),
						},
						CandidatesIndex: 0,
					}, nil) {
						return
					}
				}
			case "response.reasoning_text.delta":
				reasoningDelta := event.AsResponseReasoningTextDelta()
				if reasoningDelta.Delta != "" {
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    Thinking,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(reasoningDelta.Delta),
						},
						CandidatesIndex: 0,
					}, nil) {
						return
					}
				}
			case "response.reasoning_summary_text.delta":
				summaryDelta := event.AsResponseReasoningSummaryTextDelta()
				if summaryDelta.Delta != "" {
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    Thinking,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(summaryDelta.Delta),
						},
						CandidatesIndex: 0,
					}, nil) {
						return
					}
				}
			case "response.refusal.delta":
				refusalDelta := event.AsResponseRefusalDelta()
				if refusalDelta.Delta != "" {
					yield(StreamChunk{}, ContentPolicyErr(refusalDelta.Delta))
					return
				}
			case "response.completed":
				completed := event.AsResponseCompleted()

				// Emit usage metadata as final block with response ID included
				metadata := make(Metadata)

				if usage := completed.Response.Usage; usage.InputTokens > 0 {
					metadata[UsageMetricInputTokens] = int(usage.InputTokens)
				}
				if usage := completed.Response.Usage; usage.OutputTokens > 0 {
					metadata[UsageMetricGenerationTokens] = int(usage.OutputTokens)
				}
				// Report cached tokens if available (OpenAI Responses API prompt caching)
				if cachedTokens := completed.Response.Usage.InputTokensDetails.CachedTokens; cachedTokens > 0 {
					metadata[UsageMetricCacheReadTokens] = int(cachedTokens)
				}
				// Store the response ID in metadata for multi-turn support
				metadata[ResponsesPrevRespId] = completed.Response.ID

				yield(StreamChunk{
					Block:           MetadataBlock(metadata),
					CandidatesIndex: 0,
				}, nil)
				return
			case "response.failed":
				yield(StreamChunk{}, fmt.Errorf("response failed"))
				return
			case "response.incomplete":
				yield(StreamChunk{}, MaxGenerationLimitErr)
				return
			case "error":
				errorEvent := event.AsError()
				yield(StreamChunk{}, fmt.Errorf("stream error: %s", errorEvent.Message))
				return
			}
		}

		if stream.Err() != nil {
			yield(StreamChunk{}, stream.Err())
		}
	}
}
