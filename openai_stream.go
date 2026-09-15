package gai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"sort"
	"strings"

	"github.com/oapi-codegen/nullable"

	wire "github.com/spachava753/gai/internal/openai"
)

// openAISSE reads data fields without reconnecting. The generous event limit
// bounds untrusted input while allowing large audio and tool argument events.
// A final event without a blank line is dispatched at EOF.
func openAISSE(reader io.Reader) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		scanner := bufio.NewScanner(reader)
		scanner.Buffer(make([]byte, 4096), 16<<20)
		scanner.Split(openAISSELine)
		var data []byte
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				if len(data) > 0 {
					if !yield(bytes.TrimSuffix(data, []byte{'\n'}), nil) {
						return
					}
					data = nil
				}
				continue
			}
			if line[0] == ':' {
				continue
			}
			field, value, found := bytes.Cut(line, []byte{':'})
			if string(field) != "data" {
				continue
			}
			if !found {
				value = nil
			}
			value = bytes.TrimPrefix(value, []byte{' '})
			if len(data)+len(value) > 16<<20 {
				yield(nil, fmt.Errorf("openai SSE event exceeds 16 MiB"))
				return
			}
			data = append(data, value...)
			data = append(data, '\n')
		}
		if err := scanner.Err(); err != nil {
			yield(nil, err)
			return
		}
		if len(data) > 0 {
			yield(bytes.TrimSuffix(data, []byte{'\n'}), nil)
		}
	}
}

// openAIMergeStreamFields assembles the provider-defined delta shapes. Unknown
// fields must be stable; conflicting opaque values are errors, not silent loss.
func openAIMergeStreamFields(target map[string]json.RawMessage, update map[string]json.RawMessage) error {
	for key, value := range update {
		previous, exists := target[key]
		if !exists {
			target[key] = bytes.Clone(value)
			continue
		}
		switch key {
		case "reasoning", "reasoning_content", "refusal", "data", "transcript":
			var a, b string
			if json.Unmarshal(previous, &a) != nil || json.Unmarshal(value, &b) != nil {
				return fmt.Errorf("invalid string delta for %s", key)
			}
			target[key], _ = json.Marshal(a + b)
		case "audio", "function":
			var a, b map[string]json.RawMessage
			if json.Unmarshal(previous, &a) != nil || json.Unmarshal(value, &b) != nil || a == nil {
				return fmt.Errorf("invalid object delta for %s", key)
			}
			if err := openAIMergeStreamFields(a, b); err != nil {
				return err
			}
			target[key], _ = json.Marshal(a)
		case "reasoning_details":
			var a, b []map[string]json.RawMessage
			if json.Unmarshal(previous, &a) != nil || json.Unmarshal(value, &b) != nil {
				return fmt.Errorf("invalid reasoning details delta")
			}
			for _, detail := range b {
				found := false
				for _, existing := range a {
					if detail["index"] != nil && bytes.Equal(existing["index"], detail["index"]) {
						for field, fragment := range detail {
							if field == "text" || field == "data" || field == "signature" {
								var before, after string
								if json.Unmarshal(existing[field], &before) == nil && json.Unmarshal(fragment, &after) == nil {
									existing[field], _ = json.Marshal(before + after)
									continue
								}
							}
							if old, ok := existing[field]; ok && !bytes.Equal(old, fragment) {
								return fmt.Errorf("conflicting reasoning detail field %s", field)
							}
							existing[field] = fragment
						}
						found = true
						break
					}
				}
				if !found {
					a = append(a, detail)
				}
			}
			target[key], _ = json.Marshal(a)
		case "annotations":
			var a, b []json.RawMessage
			if json.Unmarshal(previous, &a) != nil || json.Unmarshal(value, &b) != nil {
				return fmt.Errorf("invalid annotation delta")
			}
			target[key], _ = json.Marshal(append(a, b...))
		default:
			if !bytes.Equal(previous, value) {
				return fmt.Errorf("conflicting openai stream field %s", key)
			}
		}
	}
	return nil
}

func openAISSELine(data []byte, atEOF bool) (int, []byte, error) {
	for i, b := range data {
		if b == '\n' {
			return i + 1, data[:i], nil
		}
		if b == '\r' {
			if i+1 == len(data) && !atEOF {
				return 0, nil, nil
			}
			advance := i + 1
			if advance < len(data) && data[advance] == '\n' {
				advance++
			}
			return advance, data[:i], nil
		}
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}

type openAIStreamTool struct {
	fields    map[string]json.RawMessage
	id, name  string
	arguments strings.Builder
}
type openAIStreamCandidate struct {
	tools   map[int64]*openAIStreamTool
	fields  map[string]json.RawMessage
	reason  string
	refused string
}

// Stream emits incremental text and reasoning. Indexed tool deltas are buffered
// per candidate and emitted as complete header/argument pairs at EOF, preventing
// interleaved parallel tool calls from corrupting StreamingAdapter assembly.
// The stream owns its HTTP body and closes it on every exit, including an early
// consumer stop. EOF without [DONE] or a finish reason is an incomplete stream.
func (g *OpenAiGenerator) Stream(ctx context.Context, request GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("openai: client not initialized")})
			return
		}
		params, options, err := openAIRequest(request, true)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}
		response, err := g.client.CreateChatCompletion(ctx, nil, params)
		if err != nil {
			yield(StreamChunk{Err: fmt.Errorf("openai stream: %w", err)})
			return
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			yield(StreamChunk{Err: mapHTTPAPIError(ProviderOpenAI, response)})
			return
		}
		defer response.Body.Close()
		if !strings.HasPrefix(strings.ToLower(response.Header.Get("Content-Type")), "text/event-stream") {
			body, err := io.ReadAll(response.Body)
			if err != nil {
				yield(StreamChunk{Err: err})
				return
			}
			yield(StreamChunk{Err: &ApiErr{Provider: ProviderOpenAI, Kind: APIErrorKindUnknown, StatusCode: response.StatusCode, Message: parseAPIErrorMessage(string(body)), RawBody: string(body)}})
			return
		}
		candidates := map[int64]*openAIStreamCandidate{}
		fields := map[string]json.RawMessage{}
		var usage wire.Usage
		done := false
		for data, err := range openAISSE(response.Body) {
			if err != nil {
				yield(StreamChunk{Err: fmt.Errorf("read openai stream: %w", err)})
				return
			}
			if string(bytes.TrimSpace(data)) == "[DONE]" {
				done = true
				continue
			}
			var chunk wire.ChatCompletionChunk
			if err := json.Unmarshal(data, &chunk); err != nil {
				yield(StreamChunk{Err: fmt.Errorf("decode openai event: %w", err)})
				return
			}
			if (chunk.Error.IsSpecified() && !chunk.Error.IsNull()) || (chunk.Type != nil && *chunk.Type == "error") {
				yield(StreamChunk{Err: &ApiErr{Provider: ProviderOpenAI, Kind: APIErrorKindUnknown, StatusCode: response.StatusCode, RawBody: string(data), Message: parseAPIErrorMessage(string(data))}})
				return
			}
			for key, value := range openAIRawFields(chunk, "choices", "usage") {
				fields[key] = value
			}
			if value, err := chunk.Usage.Get(); err == nil {
				usage = value
			}
			if chunk.Choices == nil {
				continue
			}
			for _, choice := range *chunk.Choices {
				index := int64(0)
				if choice.Index != nil {
					index = *choice.Index
				}
				if index < 0 {
					yield(StreamChunk{Err: fmt.Errorf("negative candidate index")})
					return
				}
				candidate := candidates[index]
				if candidate == nil {
					candidate = &openAIStreamCandidate{tools: map[int64]*openAIStreamTool{}, fields: map[string]json.RawMessage{"content": json.RawMessage(`""`)}}
					candidates[index] = candidate
				}
				if value, err := choice.Usage.Get(); err == nil {
					usage = value
				}
				if reason, err := choice.FinishReason.Get(); err == nil && reason != "" {
					candidate.reason = reason
				}
				delta, deltaErr := choice.Delta.Get()
				if deltaErr != nil {
					continue
				}
				if refusal, err := delta.Refusal.Get(); err == nil {
					candidate.refused += refusal
				}
				if text, err := delta.Content.Get(); err == nil && text != "" {
					delete(candidate.fields, "content")
					if !yield(StreamChunk{Block: TextBlock(text), CandidatesIndex: int(index)}) {
						return
					}
				}
				for _, value := range []nullable.Nullable[string]{delta.ReasoningContent, delta.Reasoning} {
					if text, err := value.Get(); err == nil && text != "" {
						if !yield(StreamChunk{Block: Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str(text), ExtraFields: map[string]interface{}{ThinkingExtraFieldGeneratorKey: "openai"}}, CandidatesIndex: int(index)}) {
							return
						}
					}
				}
				// Delta fields can repeat or grow. Publish only their assembled value at
				// completion rather than conflicting snapshots on successive chunks.
				if err := openAIMergeStreamFields(candidate.fields, openAIRawFields(delta, "role", "content", "tool_calls")); err != nil {
					yield(StreamChunk{Err: err})
					return
				}
				calls, _ := delta.ToolCalls.Get()
				for _, call := range calls {
					if call.Index == nil || *call.Index < 0 {
						yield(StreamChunk{Err: fmt.Errorf("stream tool call missing valid index")})
						return
					}
					tool := candidate.tools[*call.Index]
					if tool == nil {
						tool = &openAIStreamTool{fields: map[string]json.RawMessage{}}
						candidate.tools[*call.Index] = tool
					}
					if id, err := call.Id.Get(); err == nil && id != "" {
						tool.id = id
					}
					function, _ := call.Function.Get()
					if call.Custom != nil {
						yield(StreamChunk{Err: fmt.Errorf("custom tool streaming is not supported")})
						return
					}
					extra := openAIRawFields(call, "id", "index", "type", "function")
					if functionFields := openAIRawFields(function, "name", "arguments"); len(functionFields) > 0 {
						extra["function"], _ = json.Marshal(functionFields)
					}
					if err := openAIMergeStreamFields(tool.fields, extra); err != nil {
						yield(StreamChunk{Err: err})
						return
					}
					if name, err := function.Name.Get(); err == nil {
						tool.name += name
					}
					if args, err := function.Arguments.Get(); err == nil {
						tool.arguments.WriteString(args)
					}
				}
			}
		}
		indices := make([]int64, 0, len(candidates))
		for index := range candidates {
			indices = append(indices, index)
		}
		sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
		for _, index := range indices {
			candidate := candidates[index]
			if !done && candidate.reason == "" {
				yield(StreamChunk{Err: io.ErrUnexpectedEOF})
				return
			}
			tools := make([]int64, 0, len(candidate.tools))
			for index := range candidate.tools {
				tools = append(tools, index)
			}
			sort.Slice(tools, func(i, j int) bool { return tools[i] < tools[j] })
			for _, toolIndex := range tools {
				tool := candidate.tools[toolIndex]
				if tool.id == "" || tool.name == "" {
					yield(StreamChunk{Err: fmt.Errorf("incomplete streamed tool call")})
					return
				}
				var arguments map[string]json.RawMessage
				if err := json.Unmarshal([]byte(tool.arguments.String()), &arguments); err != nil {
					yield(StreamChunk{Err: fmt.Errorf("invalid streamed tool arguments: %w", err)})
					return
				}
				if !yield(StreamChunk{Block: Block{ID: tool.id, BlockType: ToolCall, ModalityType: Text, MimeType: "text/plain", Content: Str(tool.name), ExtraFields: map[string]interface{}{OpenAIExtraFieldWireFields: tool.fields}}, CandidatesIndex: int(index)}) {
					return
				}
				if !yield(StreamChunk{Block: Block{BlockType: ToolCall, ModalityType: Text, MimeType: "text/plain", Content: Str(tool.arguments.String())}, CandidatesIndex: int(index)}) {
					return
				}
				if !yield(StreamChunk{Block: SeparatorBlock(), CandidatesIndex: int(index)}) {
					return
				}
			}
			if audioJSON, ok := candidate.fields["audio"]; ok {
				var audio wire.OutputAudio
				if err := json.Unmarshal(audioJSON, &audio); err != nil {
					yield(StreamChunk{Err: err})
					return
				}
				message, err := openAIResponseMessage(wire.ResponseMessage{Audio: nullable.NewNullableWithValue(audio)}, options.AudioConfig.Format)
				if err != nil {
					yield(StreamChunk{Err: err})
					return
				}
				if !yield(StreamChunk{Block: SeparatorBlock(), CandidatesIndex: int(index)}) {
					return
				}
				for _, block := range message.Blocks {
					if !yield(StreamChunk{Block: block, CandidatesIndex: int(index)}) {
						return
					}
				}
				delete(candidate.fields, "audio")
			}
			if len(candidate.fields) > 0 {
				if !yield(StreamChunk{Block: SeparatorBlock(), MessageExtraFields: map[string]interface{}{OpenAIExtraFieldWireFields: candidate.fields}, CandidatesIndex: int(index)}) {
					return
				}
			}
			if _, err := openAIFinish(candidate.reason, candidate.refused, len(tools) > 0); err != nil {
				yield(StreamChunk{Err: err})
				return
			}
		}
		if !done && len(candidates) == 0 {
			yield(StreamChunk{Err: io.ErrUnexpectedEOF})
			return
		}
		yield(StreamChunk{Block: MetadataBlock(openAIUsage(usage)), ResponseExtraFields: map[string]interface{}{OpenAIResponseExtraFieldHeaders: response.Header.Clone(), OpenAIResponseExtraFieldUsage: openAIRawFields(usage), OpenAIResponseExtraFieldWireFields: fields}})
	}
}
