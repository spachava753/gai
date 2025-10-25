# GAI (Go AI generation utilities)

GAI is a Go library for interacting with LLM providers (OpenAI, Anthropic, Google Gemini) with consistent APIs, composable generators, streaming support, metrics, and robust testing. The core library is in the repository root as a standard Go module. Additional MCP (Model Context Protocol) helpers live under `x/mcp/`.

@README.md provides a general overview and quick start. See @ROADMAP.md for planned work.

## Project structure and organization

- Root Go module (go.mod) with all library code colocated for ease of import
  - Provider clients: `openai.go`, `anthropic.go`, `gemini.go`
  - Generation pipeline and composition: `generate.go`, `retry_generator.go`, `fallback_generator.go`, `preprocessing_generator.go`
  - Streaming primitives: `streaming.go`
  - Shared domain types and helpers: `message.go`, `tool.go`, `errors.go`, `metrics.go`, `callback.go`
  - Examples and usage via `*_example_test.go` files
- Tests: colocated `*_test.go` for each area, plus provider-specific tests
- Samples: `sample.jpg`, `sample.pdf`, `sample.wav` for multimodal tests/examples
- Scripts: `scripts/` includes docs-generation helper
- Experimental/extended: `x/mcp/` includes MCP helpers and example tests

Conventions
- Single module, no internal/ submodules yet
- Public APIs live at repository root package `gai`
- Experimental/stability-in-flux code goes under `x/`

## Build, test, and development commands

Requirements: Go 1.22+.

Common commands
- Install deps: `go mod download`
- Lint (if golangci-lint installed): `golangci-lint run` (optional)
- Run tests (all): `go test ./...`
- Run tests with race: `go test -race ./...`
- Run a single test file: `go test -run TestName ./...`
- Examples as docs: `go test ./...` executes `*_example_test.go`
- Generate README from script (if desired): `go run ./scripts/generate-readme.go` or `bash ./scripts/generate-readme.sh`

Dev tips
- Use `RG_COLOR=never` if your environment requires plain output
- Prefer ripgrep (rg) to locate call sites quickly, e.g. `rg 'Generate\('`

## Code style and conventions

Go style
- Follow standard Go formatting: `gofmt`/`go fmt` and `go vet` clean
- Keep package surface minimal and clear; prefer small, composable types
- Error handling: return `error` values, wrap with `%w` using `fmt.Errorf`
- Context-first: public methods that block should accept `context.Context`
- Naming: exported identifiers use full words; keep acronyms consistent (ID, URL, API)
- Avoid panics in library code; prefer errors
- Keep provider-specific types in their files to prevent cross-coupling
- Use table-driven tests for variations

Documentation
- Package-level overview in @README.md and `doc.go`
- Example-driven documentation via `*_example_test.go`

Imports
- Standard -> third-party -> local groupings
- Avoid unnecessary type aliases; prefer direct types

## Architecture and design patterns

Key concepts
- Generator abstraction: pluggable components that transform inputs to model calls and outputs
- Composition patterns: retry, fallback, and preprocessing generators compose functionality without duplication
- Provider adapters: thin wrappers that expose a unified interface over OpenAI, Anthropic, and Gemini SDKs/HTTP APIs
- Streaming: unified stream interface emitting chunks, with helpers for incremental assembly
- Tools: typed function/tool-call support with validation and safe dispatch
- Metrics and callbacks: hooks for observability, tracing, and policy checks

Design principles
- Interface-first, implementation-behind adapters
- Small, testable units with clear contracts
- Provider-agnostic core, provider-specific edges
- Opt into features (metrics, callbacks) without forcing dependencies

## Testing guidelines

- Use `go test ./...` locally and in CI
- Structure tests as table-driven where relevant
- Keep external calls mocked or recorded; tests should be deterministic and offline by default
- Example tests (`*_example_test.go`) should compile and run as documentation
- For streaming, assert on ordered chunk assembly and termination conditions
- Include negative tests: timeouts, API errors, invalid tool payloads
- Use sample media files for multimodal inputs to avoid external fetches

Naming and location
- Unit tests live alongside implementation: `file_test.go`
- Example files: `*_example_test.go` demonstrate canonical usage

## Security considerations

Secrets
- Never commit API keys; use environment variables for provider credentials:
  - OpenAI: `OPENAI_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`
  - Google: `GOOGLE_API_KEY` or ADC where applicable
- Provide `.env` locally but do not commit secrets; prefer `.env.example` when introduced

Data handling
- Avoid logging sensitive prompts or completions by default
- Redact potentially sensitive tool inputs in logs/metrics callbacks
- Validate tool-call payloads before execution
- Enforce timeouts and context cancellation for remote calls
- Follow provider usage policies and rate limits; implement exponential backoff in retry generators

Dependencies
- Keep dependencies minimal; regularly `go get -u` and review changelogs

## Testing frameworks and execution

- Standard library `testing` is used; no external test framework required
- Use `-run`, `-bench`, `-count=1` flags to target tests and disable caching when needed
- For race detection and leak checks: `go test -race ./...`

## Configuration

Environment variables
- Provider keys as above
- Tuning flags when present should be wired through options structs; document defaults in README

Configuration management
- Prefer functional options to long parameter lists for public constructors
- Validate config at construction time; return descriptive errors

When adding new configuration
1) Add fields to the appropriate options or config struct
2) Validate in constructor
3) Document in @README.md and add an example test

## Git and contribution workflow

- Default branch: `main`; use feature branches for changes
- Keep commits small, focused, and with descriptive messages
- Ensure `go fmt`, `go vet`, and `go test ./...` pass before pushing
- Avoid force pushes on `main`; use `--force-with-lease` only on feature branches

### Commit message conventions

Use Conventional Commits format:

```text
type(scope)!: short summary

Commit body. Write a detailed breakdown and use full sentences in short paragraphs over lists

...

BREAKING CHANGE: footer describing breaking change if necessary  
```

- Common types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
- Imperative, present tense; no trailing period or whitespace; scope optional; add ! for breaking changes
- Describe what changed and why, not how. Avoid describing surface level code changes; can just view the code diff. Should instead detail the reason for this commit and feature wise what changed.
- Include body/footer when helpful; use BREAKING CHANGE: and issue refs (e.g., Closes #123)
- Always include the body of the commit, never skip it

## Development environment

- Go 1.22+
- Optional tools: `golangci-lint`, `rg` (ripgrep)
- IDE hints stored under `.idea/` are not required for builds

## Parsing notes for agentic tools

- Section headers are stable and use H2 for primary topics
- File references use @-mentions (e.g., @README.md, @ROADMAP.md)
- Commands appear in inline code blocks with backticks

## Documentation for Go Symbols

When gathering context about symbols like types, global variables, constants, functions and methods, prefer to use `go doc` command. You may use `go doc -all github.com/example/pkg` to get a full overview of a package, but use sparingly, as it may overwhelm the context window. Alternatively, you may use `go doc github.com/example/pkg.Type` to get documentation about a specific symbol. Try to avoid usage of `rg` cli to search over source code of dependencies, as it can easily overwhelm your context window.