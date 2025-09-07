package gai

import (
	"context"
	"fmt"
)

type ResponsesToolGeneratorAdapter struct {
	r          ResponsesGenerator
	prevRespID string
}

func NewResponsesToolGeneratorAdapter(r ResponsesGenerator, initialPrevRespID string) *ResponsesToolGeneratorAdapter {
	return &ResponsesToolGeneratorAdapter{r: r, prevRespID: initialPrevRespID}
}

func (a *ResponsesToolGeneratorAdapter) Register(tool Tool) error {
	return a.r.Register(tool)
}

func (a *ResponsesToolGeneratorAdapter) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if a == nil {
		return Response{}, fmt.Errorf("responses adapter: inner generator is nil")
	}
	prev := a.prevRespID

	opts := GenOpts{}
	if options != nil {
		cpy := *options
		opts = cpy
	}
	if opts.ExtraArgs == nil {
		opts.ExtraArgs = make(map[string]any)
	}
	if _, exists := opts.ExtraArgs[ResponsesPrevRespId]; !exists && prev != "" {
		opts.ExtraArgs[ResponsesPrevRespId] = prev
	}

	resp, err := a.r.Generate(ctx, dialog, &opts)
	if err != nil {
		return resp, err
	}
	if resp.UsageMetadata != nil {
		if v, ok := resp.UsageMetadata[ResponsesPrevRespId]; ok {
			if id, ok2 := v.(string); ok2 && id != "" {
				a.prevRespID = id
			}
		}
	}
	return resp, nil
}

var _ ToolCapableGenerator = (*ResponsesToolGeneratorAdapter)(nil)
