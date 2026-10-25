# GAI

Keep this file limited to non-obvious workflows and repository constraints. Read the code and package documentation for architecture and API details.

## Checks

- Run `go test ./...` and `go vet ./...` for Go changes.
- Run the pinned linter with:
  ```sh
  go tool laas -exclude-packages='^github\.com/spachava753/gai/internal/(openai|cloudflare|perplexity|typesafe)$' ./...
  ```
- After cloning, enable the tracked hooks with `git config --local core.hooksPath .githooks`.

## Generated provider clients

- Do not edit `internal/**/client.gen.go` directly. Change the schema or generator configuration, then run `go generate ./internal/...`.
- Use the generated clients or existing provider SDKs for provider HTTP operations. Do not add handwritten HTTP clients; existing legacy calls are not a pattern to copy.
- Keep generated clients under `internal/`. Public APIs should accept standard HTTP clients, base URLs, and explicit credentials rather than generated types or environment lookup.

## Request validation

- Let provider APIs validate provider-specific ranges, enums, cache lifetimes, and model capabilities. Do not duplicate those checks with local allowlists or model-name inference.
- Keep local validation to the library's own contracts and safe, lossless serialization, such as option types and incompatible option combinations.
- Shared OpenAI-compatible serializers serve multiple providers and custom endpoints. Do not impose OpenAI's current accepted values or capabilities on all of them.

## Tests and live APIs

- Never run live provider tests unless the user explicitly requests them, even if credentials and `LIVE_TESTS=1` are already set.
- Gate every live test with `requireLiveAPIKey`. Live tests require both `LIVE_TESTS=1` and the relevant provider credentials; they must not retrieve credentials themselves.
- The SDK request corpus under `testdata/openai_sdk/` is replayed offline. Tests must never refresh captures. Read `testdata/openai_sdk/README.md` before recording or replacing captures, and obtain explicit approval before provider calls.
- LAAS enforces test structure: keep each scenario's setup, execution, and assertions in its inline `t.Run` callback. Shared fixtures and focused assertion helpers are fine, but do not move whole scenarios into helper functions.

## Workflow

- Work directly on `main`; feature branches and pull requests are optional.
- The module is pre-`v1.0.0`, so intentional breaking API changes are acceptable.
- Do not force-push `main`.
