// Package perplexity contains the generated Perplexity Decisions wire client.
// It backs x/classifier. Supply authentication and per-call headers through
// request editors. Raw response methods retain status, headers, and body;
// callers must close the body. The client does not add retries.
package perplexity

//go:generate go tool oapi-codegen -config config.yaml -o client.gen.go api.yaml
