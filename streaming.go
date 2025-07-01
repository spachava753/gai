package gai

import (
	"context"
	"iter"
)

// A StreamingGenerator takes a Dialog and optional GenOpts and returns an iterator used for streaming generation.
// The iterator yields a Block and an error, which may be nil.
//
// A [context.Context] is provided to the Generator as to provide not only cancellation and request specific values,
// but also to pass Generator implementation specific parameters if needed.
//
// An example would be an implementation of Generator offering a beta feature not yet offered as common functionality,
// or utilizing a special feature specific to an implementation of Generator.
//
// A Generator implementation may return several types of errors:
//   - [MaxGenerationLimitErr] when the maximum token generation limit is exceeded
//   - [UnsupportedInputModalityErr] when encountering an unsupported input modality
//   - [UnsupportedOutputModalityErr] when requested to generate an unsupported output modality
//   - [InvalidToolChoiceErr] when an invalid tool choice is specified
//   - [InvalidParameterErr] when generation parameters are invalid or out of range
//   - [ContextLengthExceededErr] when input dialog is too long
//   - [ContentPolicyErr] when content violates usage policies
//   - [EmptyDialogErr] when no messages are provided in the dialog
//   - [AuthenticationErr] when there are authentication or authorization issues
type StreamingGenerator interface {
	Stream(ctx context.Context, dialog Dialog, options *GenOpts) iter.Seq2[Block, error]
}

// StreamingAdapter converts a StreamingGenerator to a Generator
type StreamingAdapter struct {
	S StreamingGenerator
}

func (s StreamingAdapter) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	// TODO: accumulate all blocks, compress blocks into a single block (bunch of text blocks into a single block, bunch of audio blocks into a single block) and return Response
	panic("implement me")
}
