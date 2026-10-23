"""Run user-authored behavior assertions against a local Ollama model.

This harness evaluates fixed cases supplied by the operator. It does not create
jailbreak prompts, search for bypasses, or send prompts to remote services.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_CASES = 1000
MAX_JSONL_LINE_BYTES = 1_000_000
MAX_CASE_MESSAGES = 20
MAX_CASE_CHARS = 100_000
MAX_ASSERTIONS_PER_TYPE = 20
MAX_RESPONSE_BYTES = 10_000_000
ALLOWED_ROLES = {"system", "user", "assistant"}
ASSERTION_TYPES = {"contains_all", "contains_any", "not_contains"}


def local_chat_url(base_url: str) -> str:
    """Accept only a direct HTTP loopback endpoint, never a remote service."""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL must be a plain HTTP loopback URL without credentials or query parameters")
    host = (parsed.hostname or "").lower()
    is_loopback_name = host == "localhost"
    try:
        is_loopback_ip = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback_ip = False
    if not (is_loopback_name or is_loopback_ip):
        raise ValueError("only a local loopback Ollama endpoint is allowed")
    if parsed.path not in ("", "/"):
        raise ValueError("base URL must be the Ollama server root, such as http://127.0.0.1:11434")
    try:
        port = parsed.port or 11434
    except ValueError as exc:
        raise ValueError("base URL contains an invalid port") from exc
    if not 1 <= port <= 65535:
        raise ValueError("base URL port must be between 1 and 65535")
    host_text = f"[{host}]" if ":" in host else host
    return f"http://{host_text}:{port}/api/chat"


def validate_case(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("case must be a JSON object")
    case_id = value.get("id")
    if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 100 or not case_id.isprintable():
        raise ValueError("id must be a non-empty string of at most 100 characters")

    raw_messages = value.get("messages")
    if not isinstance(raw_messages, list) or not 1 <= len(raw_messages) <= MAX_CASE_MESSAGES:
        raise ValueError(f"messages must contain between 1 and {MAX_CASE_MESSAGES} entries")
    messages = []
    total_chars = 0
    has_user_message = False
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be a JSON object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in ALLOWED_ROLES or not isinstance(content, str):
            raise ValueError("message role must be system, user, or assistant, and content must be text")
        has_user_message = has_user_message or role == "user"
        total_chars += len(content)
        messages.append({"role": role, "content": content})
    if not has_user_message:
        raise ValueError("messages must include at least one user message")
    if total_chars > MAX_CASE_CHARS:
        raise ValueError(f"combined message content exceeds {MAX_CASE_CHARS} characters")

    expected = value.get("expected")
    if not isinstance(expected, dict) or set(expected) - ASSERTION_TYPES:
        raise ValueError("expected must use only contains_all, contains_any, and not_contains")
    checks: dict[str, list[str]] = {}
    for assertion_type, terms in expected.items():
        if not isinstance(terms, list) or not 1 <= len(terms) <= MAX_ASSERTIONS_PER_TYPE:
            raise ValueError(f"{assertion_type} must be a list of 1 to {MAX_ASSERTIONS_PER_TYPE} strings")
        if any(not isinstance(term, str) or not term or len(term) > 500 for term in terms):
            raise ValueError(f"{assertion_type} entries must be non-empty strings of at most 500 characters")
        checks[assertion_type] = terms
    if not checks:
        raise ValueError("expected must contain at least one assertion")
    return {"id": case_id.strip(), "messages": messages, "expected": checks}


def load_cases(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
                errors.append({"line": line_number, "error": "case exceeds the one-megabyte line limit"})
                continue
            try:
                case = validate_case(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append({"line": line_number, "error": str(exc)})
                continue
            if case["id"] in seen_ids:
                errors.append({"line": line_number, "error": "duplicate case id"})
                continue
            seen_ids.add(case["id"])
            cases.append(case)
            if len(cases) > MAX_CASES:
                raise ValueError(f"case file exceeds the {MAX_CASES}-case per-run limit")
    if not cases:
        raise ValueError("no valid test cases were found")
    return cases, errors


def call_ollama(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    seed: int,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("local model response exceeded the ten-megabyte limit")
    result = json.loads(data)
    content = result.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("local model response did not contain message.content text")
    return content


def evaluate_response(response: str, expected: dict[str, list[str]]) -> tuple[bool, dict[str, dict[str, Any]]]:
    normalized = response.casefold()
    results: dict[str, dict[str, Any]] = {}
    for assertion_type, terms in expected.items():
        if assertion_type == "contains_all":
            failed = sum(term.casefold() not in normalized for term in terms)
        elif assertion_type == "contains_any":
            failed = 0 if any(term.casefold() in normalized for term in terms) else 1
        else:
            failed = sum(term.casefold() in normalized for term in terms)
        results[assertion_type] = {"passed": failed == 0, "failed_count": failed}
    return all(item["passed"] for item in results.values()), results


def run_case(
    endpoint: str,
    model: str,
    case: dict[str, Any],
    temperature: float,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = call_ollama(endpoint, model, case["messages"], temperature, seed, timeout)
        passed, checks = evaluate_response(response, case["expected"])
        return {
            "id": case["id"],
            "status": "passed" if passed else "failed",
            "checks": checks,
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except urllib.error.HTTPError as exc:
        error = f"local Ollama server returned HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError):
        error = "could not reach the local Ollama server or the request timed out"
    except json.JSONDecodeError:
        error = "local Ollama response was not valid JSON chat output"
    except ValueError:
        error = "local Ollama response was invalid or exceeded the response-size limit"
    except Exception as exc:
        error = f"request failed ({type(exc).__name__})"
    return {
        "id": case["id"],
        "status": "error",
        "error": error,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed behavior assertions against a local Ollama model.")
    parser.add_argument("--cases", required=True, type=Path, help="JSONL file of operator-authored regression cases")
    parser.add_argument("--model", required=True, help="model name already available in the local Ollama server")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="local Ollama root URL (loopback only)")
    parser.add_argument("--timeout", type=float, default=120.0, help="timeout for each local model response")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature from 0 to 2")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be a positive finite number")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        parser.error("temperature must be between 0 and 2")
    try:
        endpoint = local_chat_url(args.base_url)
        cases_path = args.cases.expanduser()
        if not cases_path.is_file():
            parser.error(f"case file not found: {cases_path}")
        cases, input_errors = load_cases(cases_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    output_path = args.output.expanduser() if args.output else None
    if output_path is not None:
        if output_path.resolve() == cases_path.resolve():
            parser.error("report path must not overwrite the case file")
        if output_path.exists() and not args.overwrite:
            parser.error(f"report already exists (use --overwrite): {output_path}")

    started_at = datetime.now(timezone.utc)
    results = [run_case(endpoint, args.model, case, args.temperature, args.seed, args.timeout) for case in cases]
    report = {
        "tool": "local-llm-safety-regression-harness",
        "model": args.model,
        "endpoint": args.base_url,
        "cases_file": cases_path.name,
        "started_at": started_at.isoformat(),
        "results": results,
        "input_errors": input_errors,
        "summary": {
            "passed": sum(item["status"] == "passed" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "errors": sum(item["status"] == "error" for item in results) + len(input_errors),
        },
        "privacy": "Prompts and raw model responses are not included in the report; only IDs, assertion counts, response hashes, and timing are recorded.",
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            parser.error(f"could not write report: {exc}")

    if args.json:
        print(serialized, end="")
    else:
        summary = report["summary"]
        print(f"Model: {args.model}")
        print(f"Cases: {len(results)} | passed: {summary['passed']} | failed: {summary['failed']} | errors: {summary['errors']}")
        for result in results:
            print(f"[{result['status'].upper()}] {result['id']}")
        if output_path is not None:
            print(f"Report: {output_path}")

    if report["summary"]["errors"]:
        return 2
    if report["summary"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
