// Package cloudflare contains the generated Workers AI Clef wire client.
// It backs x/classifier. Supply authentication and per-call headers through
// request editors. Raw response methods retain the Cloudflare envelope and
// HTTP errors for the adapter to interpret; callers must close the body.
package cloudflare

//go:generate go tool oapi-codegen -config config.yaml -o client.gen.go api.yaml
