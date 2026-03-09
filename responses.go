package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
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

// ResponsesPromptCacheKeyParam is a key used in GenOpts.ExtraArgs to set the OpenAI
// Responses API prompt_cache_key request field. Reuse the same key across requests that
// share a long static prefix to improve cache routing and hit rates.
const ResponsesPromptCacheKeyParam = "responses_prompt_cache_key"

// ResponsesExtraFieldReasoningID is the key used in Block.ExtraFields for Thinking blocks
// to store the reasoning item's unique ID from the Responses API. This is needed to reconstruct
// reasoning input items when passing back in multi-turn conversations.
const ResponsesExtraFieldReasoningID = "responses_reasoning_id"

// ResponsesExtraFieldEncryptedContent is the key used in Block.ExtraFields for Thinking blocks
// to store the encrypted reasoning content from the Responses API. When using the API in
// stateless mode (store=false), encrypted reasoning items must be passed back to the API
// during multi-turn function-calling conversations so the model can continue its reasoning.
//
// Per the OpenAI docs: reasoning items should be included in the input for subsequent turns
// during ongoing function call chains (i.e., between the last user message and the current
// request). Once the assistant produces a non-tool-call response and a new user message begins
// a new turn, previous encrypted reasoning items are no longer needed. The API will
// automatically ignore reasoning items that aren't relevant to the current context.
const ResponsesExtraFieldEncryptedContent = "responses_encrypted_content"

// ResponsesMessageExtraFieldPhase is the key used in Message.ExtraFields for assistant messages
// returned by the OpenAI Responses API. The API documents that assistant message phase should be
// preserved and resent in follow-up requests when present.
const ResponsesMessageExtraFieldPhase = "responses_phase"

const (
	ResponsesMessagePhaseCommentary  = "commentary"
	ResponsesMessagePhaseFinalAnswer = "final_answer"
)

type ResponsesService interface {
	New(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (res *responses.Response, err error)
	NewStreaming(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (stream *ssestream.Stream[responses.ResponseStreamEventUnion])
}

// ResponsesGenerator is a stateless generator that calls OpenAI models via the Responses API.
//
// This generator operates in fully stateless mode: it sets store=false on every request
// and includes "reasoning.encrypted_content" so that encrypted reasoning tokens are
// returned in API responses. These encrypted tokens are stored in Thinking block ExtraFields
// and automatically reconstructed as reasoning input items when the dialog is passed back
// for subsequent turns (e.g., during multi-step function calling).
type ResponsesGenerator struct {
	client             ResponsesService
	model              string
	tools              map[string]responses.ToolUnionParam
	systemInstructions string
}

// NewResponsesGenerator creates a new OpenAI Responses API generator with the specified model.
// The returned generator implements the Generator, StreamingGenerator, and ToolRegister interfaces.
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

func responsesMessagePhase(extraFields map[string]interface{}) (responses.ResponseOutputMessagePhase, error) {
	if extraFields == nil {
		return "", nil
	}
	raw, ok := extraFields[ResponsesMessageExtraFieldPhase]
	if !ok || raw == nil {
		return "", nil
	}
	phase, ok := raw.(string)
	if !ok {
		return "", fmt.Errorf("responses message phase must be a string")
	}
	switch phase {
	case "":
		return "", nil
	case ResponsesMessagePhaseCommentary:
		return responses.ResponseOutputMessagePhaseCommentary, nil
	case ResponsesMessagePhaseFinalAnswer:
		return responses.ResponseOutputMessagePhaseFinalAnswer, nil
	default:
		return "", fmt.Errorf("invalid responses message phase %q", phase)
	}
}

func mergeResponsesMessagePhase(extraFields map[string]interface{}, phase string) (map[string]interface{}, error) {
	if phase == "" {
		return extraFields, nil
	}
	if extraFields == nil {
		return map[string]interface{}{ResponsesMessageExtraFieldPhase: phase}, nil
	}
	if existing, ok := extraFields[ResponsesMessageExtraFieldPhase]; ok && existing != nil {
		existingPhase, ok := existing.(string)
		if !ok {
			return nil, fmt.Errorf("responses message phase must be a string")
		}
		if existingPhase != phase {
			return nil, fmt.Errorf("responses output contained multiple assistant message phases (%q, %q)", existingPhase, phase)
		}
		return extraFields, nil
	}
	merged := maps.Clone(extraFields)
	merged[ResponsesMessageExtraFieldPhase] = phase
	return merged, nil
}

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

// buildInputItems converts a Dialog into the input item list format expected by the
// Responses API. For Assistant messages, it reconstructs reasoning items from Thinking
// blocks that carry encrypted content in their ExtraFields, preserving the model's
// chain-of-thought across function-calling turns.
func (r *ResponsesGenerator) buildInputItems(dialog Dialog) ([]responses.ResponseInputItemUnionParam, error) {
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
							return nil, fmt.Errorf("image block missing mimetype")
						}
						dataURL := fmt.Sprintf("data:%s;base64,%s", blk.MimeType, blk.Content.String())
						if blk.MimeType == "application/pdf" {
							val, ok := blk.ExtraFields[BlockFieldFilenameKey]
							if !ok {
								return nil, fmt.Errorf("filename field missing in extra fields")
							}
							filename, ok := val.(string)
							if !ok {
								return nil, fmt.Errorf("filename field is not a string")
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
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
				default:
					return nil, fmt.Errorf("unsupported block type in message: %s", blk.BlockType)
				}
			}

			if len(contents) > 0 {
				role := responses.EasyInputMessageRoleUser
				inputItems = append(inputItems, responses.ResponseInputItemParamOfMessage(contents, role))
			}

		case Assistant:
			var contents []responses.ResponseOutputMessageContentUnionParam
			var toolCalls []responses.ResponseInputItemUnionParam
			phase, err := responsesMessagePhase(msg.ExtraFields)
			if err != nil {
				return nil, err
			}
			// Collect reasoning items from Thinking blocks. In stateless mode, we must
			// pass back reasoning items (with encrypted content) to the API so the model
			// can continue its chain of thought across function-calling turns. The API
			// will ignore reasoning items that aren't relevant to the current context.
			var reasoningItems []responses.ResponseInputItemUnionParam
			// Track seen reasoning IDs to deduplicate. A single reasoning item from the
			// API may produce multiple Thinking blocks (content + summaries), but we
			// must only send one reasoning input item per unique ID back to the API.
			seenReasoningIDs := make(map[string]bool)

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
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
				case Thinking:
					// Reconstruct reasoning items from Thinking blocks that have encrypted
					// content stored in ExtraFields. These are needed for multi-turn
					// function-calling conversations in stateless mode. Blocks without
					// encrypted content (e.g., summary-only blocks) are skipped since the
					// API cannot use them as input.
					if blk.ExtraFields != nil {
						if encContent, ok := blk.ExtraFields[ResponsesExtraFieldEncryptedContent].(string); ok && encContent != "" {
							reasoningID, _ := blk.ExtraFields[ResponsesExtraFieldReasoningID].(string)
							// Deduplicate: a single reasoning item from the API can produce
							// multiple Thinking blocks (content entries + summaries), all
							// sharing the same reasoning ID and encrypted content. We only
							// need to send one reasoning input item per unique ID.
							if seenReasoningIDs[reasoningID] {
								continue
							}
							seenReasoningIDs[reasoningID] = true
							reasoningParam := responses.ResponseReasoningItemParam{
								ID:               reasoningID,
								EncryptedContent: openai.Opt(encContent),
								// Summary is required by the SDK but the API only needs the
								// encrypted content to restore reasoning state, so we pass
								// an empty summary slice.
								Summary: []responses.ResponseReasoningItemSummaryParam{},
							}
							reasoningItems = append(reasoningItems, responses.ResponseInputItemUnionParam{
								OfReasoning: &reasoningParam,
							})
						}
					}
				case ToolCall:
					if blk.ID == "" {
						return nil, fmt.Errorf("tool call block missing ID")
					}
					var tci ToolCallInput
					if err := json.Unmarshal([]byte(blk.Content.String()), &tci); err != nil {
						return nil, fmt.Errorf("invalid tool call content: %w", err)
					}
					argsJSON, err := json.Marshal(tci.Parameters)
					if err != nil {
						return nil, fmt.Errorf("failed to marshal tool parameters: %w", err)
					}
					toolCalls = append(toolCalls, responses.ResponseInputItemParamOfFunctionCall(string(argsJSON), blk.ID, tci.Name))
				default:
					return nil, fmt.Errorf("unsupported block type in message: %s", blk.BlockType)
				}
			}

			// Reasoning items must appear before the message/tool-call items they
			// accompany, mirroring the order the API originally returned them in.
			if len(reasoningItems) > 0 {
				inputItems = append(inputItems, reasoningItems...)
			}
			if len(contents) > 0 || phase != "" {
				messageContents := contents
				if messageContents == nil {
					messageContents = []responses.ResponseOutputMessageContentUnionParam{}
				}
				inputItems = append(inputItems, responses.ResponseInputItemUnionParam{
					OfOutputMessage: &responses.ResponseOutputMessageParam{
						ID:      "",
						Content: messageContents,
						Phase:   phase,
					},
				})
			}
			if len(toolCalls) > 0 {
				inputItems = append(inputItems, toolCalls...)
			}

		case ToolResult:
			if len(msg.Blocks) == 0 {
				return nil, fmt.Errorf("tool result message must have at least one block")
			}
			toolID := msg.Blocks[0].ID
			if toolID == "" {
				return nil, fmt.Errorf("tool result message block must have an ID")
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
						return nil, fmt.Errorf("all blocks in tool result must share the same ID")
					}
					sb.WriteString(blk.Content.String())
				}
				inputItems = append(inputItems, responses.ResponseInputItemParamOfFunctionCallOutput(toolID, sb.String()))
			} else {
				// Complex case: blocks with images/PDFs - use array output format
				var outputItems responses.ResponseFunctionCallOutputItemListParam
				for _, blk := range msg.Blocks {
					if blk.ID != toolID {
						return nil, fmt.Errorf("all blocks in tool result must share the same ID")
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
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
				}
				inputItems = append(inputItems, responses.ResponseInputItemParamOfFunctionCallOutput(toolID, outputItems))
			}
		default:
			return nil, fmt.Errorf("unsupported message role: %s", msg.Role.String())
		}
	}
	return inputItems, nil
}

// buildParams constructs the ResponseNewParams from the input items, generator config,
// and user-provided options. It always sets store=false and includes
// "reasoning.encrypted_content" for stateless operation.
func (r *ResponsesGenerator) buildParams(inputItems []responses.ResponseInputItemUnionParam, options *GenOpts) (responses.ResponseNewParams, error) {
	params := responses.ResponseNewParams{
		Model: r.model,
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: inputItems},
		// Stateless mode: do not store responses on OpenAI servers. This means
		// we cannot use previous_response_id and must manually manage conversation
		// state by passing all input items explicitly each turn.
		Store: openai.Opt(false),
		// Request encrypted reasoning content so we can pass reasoning items back
		// in subsequent turns. Without this, reasoning items won't have the
		// encrypted_content field needed for stateless multi-turn function calling.
		Include: []responses.ResponseIncludable{
			responses.ResponseIncludableReasoningEncryptedContent,
		},
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
		if options.Temperature != nil {
			params.Temperature = openai.Opt(*options.Temperature)
		}
		if options.TopP != nil {
			params.TopP = openai.Opt(*options.TopP)
		}
		if options.MaxGenerationTokens != nil {
			params.MaxOutputTokens = openai.Opt(int64(*options.MaxGenerationTokens))
		}
		if options.ExtraArgs != nil {
			if val, ok := options.ExtraArgs[ResponsesPromptCacheKeyParam]; ok {
				key, ok := val.(string)
				if !ok {
					return params, fmt.Errorf("responses prompt cache key must be a string")
				}
				if key != "" {
					params.PromptCacheKey = openai.Opt(key)
				}
			}
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
			params.Reasoning = responses.ReasoningParam{Effort: responses.ReasoningEffort(options.ThinkingBudget)}
			if options.ExtraArgs != nil {
				if val, ok := options.ExtraArgs[ResponsesThoughtSummaryDetailParam]; ok {
					params.Reasoning.Summary = val.(responses.ReasoningSummary)
				}
			}
		}
		if len(options.OutputModalities) > 0 {
			for _, m := range options.OutputModalities {
				if m != Text {
					return params, UnsupportedOutputModalityErr(m.String())
				}
			}
		}
	}

	return params, nil
}

// processResponseOutput converts the API response output items into an assistant Message.
// For reasoning items, it stores the encrypted content and reasoning ID in ExtraFields
// so that subsequent calls can reconstruct the reasoning input items.
func processResponseOutput(output []responses.ResponseOutputItemUnion) (message Message, hasToolCalls bool, refusal string, err error) {
	message.Role = Assistant
	for _, item := range output {
		switch item.Type {
		case "message":
			msg := item.AsMessage()
			message.ExtraFields, err = mergeResponsesMessagePhase(message.ExtraFields, string(msg.Phase))
			if err != nil {
				return Message{}, false, "", err
			}
			for _, c := range msg.Content {
				switch c.Type {
				case "output_text":
					message.Blocks = append(message.Blocks, Block{BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str(c.Text)})
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
				if unmarshalErr := json.Unmarshal([]byte(fc.Arguments), &paramsMap); unmarshalErr != nil {
					err = fmt.Errorf("malformed function call arguments for %q: %w", fc.Name, unmarshalErr)
					return
				}
			}
			tci := ToolCallInput{Name: fc.Name, Parameters: paramsMap}
			b, _ := json.Marshal(tci)
			id := fc.CallID
			if id == "" {
				id = fc.ID
			}
			message.Blocks = append(message.Blocks, Block{ID: id, BlockType: ToolCall, ModalityType: Text, MimeType: "application/json", Content: Str(b)})
			hasToolCalls = true
		case "reasoning":
			reas := item.AsReasoning()

			// Build extra fields for this reasoning item. We store the reasoning ID
			// and encrypted content so that when this block is passed back in a
			// subsequent Assistant message, we can reconstruct the reasoning input item.
			extraFields := map[string]interface{}{
				ThinkingExtraFieldGeneratorKey: ThinkingGeneratorResponses,
				ResponsesExtraFieldReasoningID: reas.ID,
			}
			if reas.EncryptedContent != "" {
				extraFields[ResponsesExtraFieldEncryptedContent] = reas.EncryptedContent
			}

			// Process actual reasoning content as Thinking blocks.
			// Each reasoning item may contain multiple content entries and/or summary entries.
			// We emit them all as separate Thinking blocks but share the same reasoning ID
			// and encrypted content across all of them (the encrypted content is per
			// reasoning item, not per content entry).
			for _, rc := range reas.Content {
				message.Blocks = append(message.Blocks, Block{
					BlockType:    Thinking,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(rc.Text),
					ExtraFields:  maps.Clone(extraFields),
				})
			}
			// Also process summaries as Thinking blocks. Summaries are the closest
			// we get to the model's reasoning when full traces are unavailable.
			for _, rc := range reas.Summary {
				message.Blocks = append(message.Blocks, Block{
					BlockType:    Thinking,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(rc.Text),
					ExtraFields:  maps.Clone(extraFields),
				})
			}
			// If the reasoning item has encrypted content but no visible content/summaries,
			// emit a placeholder block so the encrypted content is preserved in the dialog
			// for the next turn.
			if len(reas.Content) == 0 && len(reas.Summary) == 0 && reas.EncryptedContent != "" {
				message.Blocks = append(message.Blocks, Block{
					BlockType:    Thinking,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(""),
					ExtraFields:  maps.Clone(extraFields),
				})
			}
		}
	}
	return message, hasToolCalls, refusal, nil
}

func (r *ResponsesGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if r.client == nil {
		return Response{}, fmt.Errorf("responses: client not initialized")
	}
	if len(dialog) == 0 {
		return Response{}, EmptyDialogErr
	}

	inputItems, err := r.buildInputItems(dialog)
	if err != nil {
		return Response{}, err
	}

	params, err := r.buildParams(inputItems, options)
	if err != nil {
		return Response{}, err
	}

	res, err := r.client.New(ctx, params)
	if err != nil {
		if mapped := mapResponsesRequestError(err); mapped != nil {
			return Response{}, mapped
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
		// Report reasoning tokens if available (models with reasoning/thinking enabled)
		if usage.OutputTokensDetails.ReasoningTokens > 0 {
			result.UsageMetadata[UsageMetricReasoningTokens] = int(usage.OutputTokensDetails.ReasoningTokens)
		}
	}

	assistantMsg, hasToolCalls, refusal, parseErr := processResponseOutput(res.Output)
	if parseErr != nil {
		return Response{}, parseErr
	}
	result.Candidates = append(result.Candidates, assistantMsg)

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

		inputItems, err := r.buildInputItems(dialog)
		if err != nil {
			yield(StreamChunk{}, err)
			return
		}

		params, err := r.buildParams(inputItems, options)
		if err != nil {
			yield(StreamChunk{}, err)
			return
		}

		var assistantMessageExtraFields map[string]interface{}

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
					assistantMessageExtraFields, err = mergeResponsesMessagePhase(assistantMessageExtraFields, string(msg.Phase))
					if err != nil {
						yield(StreamChunk{}, err)
						return
					}
				case "function_call":
					fc := item.AsFunctionCall()
					id := fc.CallID
					if id == "" {
						id = fc.ID
					}
					if !yield(StreamChunk{
						Block: Block{
							ID:           id,
							BlockType:    ToolCall,
							ModalityType: Text,
							MimeType:     "application/json",
							Content:      Str(fc.Name),
						},
						MessageExtraFields: maps.Clone(assistantMessageExtraFields),
						CandidatesIndex:    0,
					}, nil) {
						return
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
						MessageExtraFields: maps.Clone(assistantMessageExtraFields),
						CandidatesIndex:    0,
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
						MessageExtraFields: maps.Clone(assistantMessageExtraFields),
						CandidatesIndex:    0,
					}, nil) {
						return
					}
				}
			case "response.reasoning_text.delta":
				// Empirical note for future maintainers: on 2026-03-09 we probed the raw
				// Responses streaming API directly (gpt-5.4, high reasoning, repeated
				// difficult HLE-style prompts) to verify whether one streamed response can
				// produce multiple distinct reasoning items. Across the observed runs, every
				// reasoning-related delta belonged to a single stable reasoning item ID, and
				// encrypted_content only appeared on the final response.output_item.done
				// event for that item. Some runs emitted no reasoning deltas at all but still
				// ended with one completed reasoning item carrying encrypted_content.
				//
				// Because of that observed API behavior, we stream reasoning deltas as plain
				// Thinking chunks here and attach the replay-critical reasoning ID and
				// encrypted content when the reasoning item completes below. If OpenAI ever
				// starts emitting multiple reasoning items within one streamed response,
				// revisit both this logic and StreamingAdapter thinking-block compression.
				reasoningDelta := event.AsResponseReasoningTextDelta()
				if reasoningDelta.Delta != "" {
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    Thinking,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(reasoningDelta.Delta),
							ExtraFields: map[string]interface{}{
								ThinkingExtraFieldGeneratorKey: ThinkingGeneratorResponses,
							},
						},
						MessageExtraFields: maps.Clone(assistantMessageExtraFields),
						CandidatesIndex:    0,
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
							ExtraFields: map[string]interface{}{
								ThinkingExtraFieldGeneratorKey: ThinkingGeneratorResponses,
							},
						},
						MessageExtraFields: maps.Clone(assistantMessageExtraFields),
						CandidatesIndex:    0,
					}, nil) {
						return
					}
				}
			case "response.output_item.done":
				// When a reasoning item completes, emit a zero-content thinking chunk
				// carrying the encrypted content and reasoning ID in ExtraFields.
				// compressStreamingBlocks merges ExtraFields across consecutive
				// thinking chunks via maps.Copy, so the final compressed block will
				// carry the encrypted content needed for stateless multi-turn
				// function calling.
				//
				// Raw SDK probes run on 2026-03-09 against gpt-5.4 with high reasoning
				// observed that one streamed response yielded at most one completed
				// reasoning item, while many reasoning summary deltas all shared that
				// same item_id. That is the behavior this merge strategy relies on.
				// If OpenAI ever returns multiple completed reasoning items in one
				// streamed response, consecutive thinking chunks would collapse into a
				// single block and only the last item's replay metadata would survive.
				// The non-streaming Generate path does not have this limitation.
				item := event.AsResponseOutputItemDone().Item
				switch item.Type {
				case "message":
					msg := item.AsMessage()
					assistantMessageExtraFields, err = mergeResponsesMessagePhase(assistantMessageExtraFields, string(msg.Phase))
					if err != nil {
						yield(StreamChunk{}, err)
						return
					}
				case "reasoning":
					reas := item.AsReasoning()
					if reas.EncryptedContent != "" || reas.ID != "" {
						extra := map[string]interface{}{
							ThinkingExtraFieldGeneratorKey: ThinkingGeneratorResponses,
							ResponsesExtraFieldReasoningID: reas.ID,
						}
						if reas.EncryptedContent != "" {
							extra[ResponsesExtraFieldEncryptedContent] = reas.EncryptedContent
						}
						if !yield(StreamChunk{
							Block: Block{
								BlockType:    Thinking,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(""),
								ExtraFields:  extra,
							},
							MessageExtraFields: maps.Clone(assistantMessageExtraFields),
							CandidatesIndex:    0,
						}, nil) {
							return
						}
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

				metadata := make(Metadata)
				usage := completed.Response.Usage
				if usage.InputTokens > 0 {
					metadata[UsageMetricInputTokens] = int(usage.InputTokens)
				}
				if usage.OutputTokens > 0 {
					metadata[UsageMetricGenerationTokens] = int(usage.OutputTokens)
				}
				if usage.InputTokensDetails.CachedTokens > 0 {
					metadata[UsageMetricCacheReadTokens] = int(usage.InputTokensDetails.CachedTokens)
				}
				if usage.OutputTokensDetails.ReasoningTokens > 0 {
					metadata[UsageMetricReasoningTokens] = int(usage.OutputTokensDetails.ReasoningTokens)
				}

				yield(StreamChunk{
					Block:              MetadataBlock(metadata),
					MessageExtraFields: maps.Clone(assistantMessageExtraFields),
					CandidatesIndex:    0,
				}, nil)
				return
			case "response.failed":
				failed := event.AsResponseFailed()
				yield(StreamChunk{}, newResponsesStreamAPIError(string(failed.Response.Error.Code), failed.Response.Error.Message, failed.RawJSON()))
				return
			case "response.incomplete":
				yield(StreamChunk{}, MaxGenerationLimitErr)
				return
			case "error":
				errorEvent := event.AsError()
				yield(StreamChunk{}, newResponsesStreamAPIError(errorEvent.Code, errorEvent.Message, errorEvent.RawJSON()))
				return
			}
		}

		if stream.Err() != nil {
			if mapped := mapResponsesRequestError(stream.Err()); mapped != nil {
				yield(StreamChunk{}, mapped)
			} else {
				yield(StreamChunk{}, stream.Err())
			}
		}
	}
}
