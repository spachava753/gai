// Package typesafe contains the generated TypeSafe System One wire client.
// It backs the Jev classifier in x/classifier. Authentication and per-call
// headers are supplied through request editors. Use the raw response methods
// to preserve error bodies and numeric tokens; callers must close the body.
package typesafe

//go:generate go tool oapi-codegen -config config.yaml -o client.gen.go api.yaml
