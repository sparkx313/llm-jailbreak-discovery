# Local LLM Behavior Regression Harness

Run operator-authored behavior assertions against a local Ollama model after changing a model, system prompt, or application integration. Each JSONL case contains the exact conversation to send and explicit text assertions for the response. The runner records pass/fail results, timing, and response hashes without saving prompts or raw answers.

This repository is a regression runner, not a jailbreak generator or attack search engine. It does not invent prompts, score harmfulness, call remote model providers, or target non-local endpoints. Use cases owned by your application team and a model running on your own machine.

## Requirements

- Python 3.10 or newer
- Ollama running locally with the selected model available
- Python standard library only

The runner communicates with Ollama's local `/api/chat` endpoint. It rejects non-loopback endpoints and bypasses configured HTTP proxies so prompt content is sent directly to the local service.

## Case file

Create a UTF-8 JSON Lines (`.jsonl`) file with one case per line. Each case requires a unique `id`, a `messages` array, and at least one exact-text assertion in `expected`.

```json
{"id":"support-policy-01","messages":[{"role":"system","content":"Answer questions using the policy provided by the application."},{"role":"user","content":"What is the support window?"}],"expected":{"contains_any":["weekdays","Monday to Friday"],"not_contains":["24/7"]}}
```

Supported assertions are case-insensitive substring checks:

- `contains_all`: every string must appear.
- `contains_any`: at least one string must appear.
- `not_contains`: none of the strings may appear.

The runner accepts `system`, `user`, and `assistant` messages and requires at least one user message. It limits a run to 1,000 cases and each conversation to 100,000 characters.

## Run a regression pass

```powershell
python -m src.safety_eval `
  --cases "application-cases.jsonl" `
  --model "llama3.2:3b" `
  --output "reports/local-model-run.json"
```

Ollama defaults to `http://127.0.0.1:11434`. A different loopback host/port can be selected with `--base-url`; remote URLs are rejected. Use `--timeout`, `--temperature`, and `--seed` to control each request. Existing report files are protected unless `--overwrite` is passed. Run `python -m src.safety_eval --help` for all options.

Exit code `0` means every case passed, `1` means at least one assertion failed, and `2` means an input or local-service error occurred. Failed assertions are reported by type and count, without copying expected strings or model output into the report.

## Scope

This runner checks literal response requirements; it does not judge semantic correctness, safety, or whether a response is a refusal. Assertion quality is the responsibility of the application team. The summary records only case IDs, assertion pass/fail counts, response hashes, latencies, and errors. Treat case files as private because they can contain application prompts or data.

See the official [Ollama chat API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md) for the local request format.
