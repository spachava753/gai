# /// script
# requires-python = ">=3.11"
# dependencies = ["openai==3.13.0"]
# ///
"""Run one SDK conversation through a local capture proxy. Requires live approval."""

import argparse
import base64
import importlib.metadata
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx2
from openai import APIStatusError, OpenAI


def dump(value):
    return value.model_dump(mode="json", exclude_unset=True)


def content(step):
    if not step.get("images") and not step.get("pdfs"):
        return step["text"]
    parts = [{"type": "text", "text": step["text"]}] if step.get("text") else []
    for name in step.get("images", []):
        data = (Path(__file__).resolve().parents[2] / name).read_bytes()
        mime, _ = mimetypes.guess_type(name)
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
                },
            }
        )
    for name in step.get("pdfs", []):
        data = (Path(__file__).resolve().parents[2] / name).read_bytes()
        parts.append(
            {
                "type": "file",
                "file": {
                    "filename": Path(name).name,
                    "file_data": "data:application/pdf;base64,"
                    + base64.b64encode(data).decode("ascii"),
                },
            }
        )
    return parts


def synthetic_results(message, scenario, step):
    prices = scenario.get("prices", {})
    results = []
    for call in message.get("tool_calls") or []:
        if (
            call.get("type") == "function"
            and call["function"]["name"] == "get_image"
            and json.loads(call["function"]["arguments"]) == {}
            and step.get("images")
        ):
            results.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content(step)}
            )
            continue
        if call.get("type") != "function":
            raise ValueError("unexpected tool type")
        arguments = json.loads(call["function"]["arguments"])
        name = call["function"]["name"]
        if name == "quote_order":
            value = {"quoted": True, "item_count": len(arguments["items"])}
        elif name == "lookup_shipping" and arguments.get("region") in ("EU", "US"):
            value = {
                "region": arguments["region"],
                "fee_cents": {"EU": 250, "US": 500}[arguments["region"]],
            }
        elif name == "lookup_price" and set(arguments) == {"product_code"}:
            code = arguments["product_code"]
            if code in prices:
                value = {"product_code": code, "price_cents": prices[code]}
            else:
                value = {
                    "error": "unknown_product",
                    "suggested_product_code": min(prices),
                }
        else:
            raise ValueError("unexpected synthetic tool; no arbitrary dispatch")
        results.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(
                    value,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    sort_keys=name != "lookup_price",
                ),
            }
        )
    return results


def assemble(chunks):
    """SDK-side assembly, independent of the Go adapter and its wire types."""
    message = {"role": "assistant", "content": ""}
    tools = {}
    finish = None
    for chunk in chunks:
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                raise ValueError("this recording scenario requests one candidate")
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            for key, value in delta.items():
                if key == "tool_calls":
                    for part in value or []:
                        index = part["index"]
                        tool = tools.setdefault(
                            index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        for field in ("id", "type"):
                            if part.get(field):
                                tool[field] = part[field]
                        for field, fragment in (part.get("function") or {}).items():
                            if field in ("name", "arguments") and fragment is not None:
                                tool["function"][field] += fragment
                            elif field not in ("name", "arguments"):
                                tool["function"][field] = fragment
                        for field, fragment in part.items():
                            if field not in ("id", "type", "index", "function"):
                                tool[field] = fragment
                elif key in ("content", "reasoning_content", "reasoning", "refusal"):
                    if value is not None:
                        message[key] = (message.get(key) or "") + value
                    elif key not in message:
                        message[key] = None
                elif key == "role":
                    if value:
                        message[key] = value
                elif key not in message or message[key] == value:
                    message[key] = value
                else:
                    raise ValueError(
                        f"unimplemented SDK-side delta assembly for {key}; do not silently overwrite"
                    )
    if tools:
        message["tool_calls"] = [tools[index] for index in sorted(tools)]
    return message, finish


def check_response(message, finish, scenario, step, options):
    calls = message.get("tool_calls") or []
    if (
        finish != ("tool_calls" if calls else "stop")
        or len(calls) != step["expected_tool_calls"]
    ):
        raise ValueError(f"unexpected scenario outcome: {step['name']}")
    ids = [call["id"] for call in calls]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate tool call IDs")
    declared = {tool["function"]["name"] for tool in scenario["tools"]}
    arguments = {}
    for call in calls:
        name = call["function"]["name"]
        value = json.loads(call["function"]["arguments"])
        if (
            call.get("type") != "function"
            or name not in declared
            or not isinstance(value, dict)
        ):
            raise ValueError("invalid function call")
        arguments[name] = value
    if "expected_arguments" in step and arguments != step["expected_arguments"]:
        raise ValueError(f"wrong function arguments: {step['name']}")
    if not calls and not (message.get("content") or "").strip():
        raise ValueError("empty final answer")
    if (
        "expected_json" in step
        and json.loads(message["content"]) != step["expected_json"]
    ):
        raise ValueError(f"wrong JSON result: {step['name']}")
    if any(
        term not in (message.get("content") or "")
        for term in step.get("expected_text", [])
    ):
        raise ValueError(f"missing answer fact: {step['name']}")
    extra = options.get("extra_body", {})
    if (
        extra.get("thinking", {}).get("type") == "disabled"
        or extra.get("reasoning_effort", options.get("reasoning_effort")) == "none"
    ) and (message.get("reasoning_content") or message.get("reasoning")):
        raise ValueError("reasoning returned while disabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument(
        "--matrix", type=Path, default=Path(__file__).with_name("matrix.json")
    )
    parser.add_argument("--scenario")
    parser.add_argument(
        "--model", help="Override the case model, e.g. a vision-capable variant"
    )
    parser.add_argument("--proxy", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    cases = {case["name"]: case for case in matrix["cases"]}
    if args.case not in cases or (
        args.scenario and args.scenario not in matrix["scenarios"]
    ):
        parser.error("unknown case or scenario in selected matrix")
    proxy = urlsplit(args.proxy)
    if proxy.scheme != "http" or proxy.hostname not in (
        "127.0.0.1",
        "localhost",
        "::1",
    ):
        parser.error("--proxy must be a local HTTP reverse proxy")
    case = cases[args.case]
    scenario = matrix["scenarios"][args.scenario or case["scenario"]]
    if (
        case["sdk"] != "openai"
        or importlib.metadata.version("openai") != case["sdk_version"]
    ):
        parser.error("SDK version does not match matrix")
    # Reverse mode forwards the path unchanged; retain the provider's API prefix.
    base_url = args.proxy.rstrip("/") + urlsplit(case["base_url"]).path
    messages = [{"role": "system", "content": scenario["instructions"]}]
    with OpenAI(
        api_key=os.environ[case["key_env"]],
        base_url=base_url,
        max_retries=0,
        http_client=httpx2.Client(timeout=180, trust_env=False, follow_redirects=False),
    ) as client:
        for step in scenario["steps"]:
            if step["input"] == "user":
                messages.append({"role": "user", "content": content(step)})
            else:
                messages.extend(synthetic_results(messages[-1], scenario, step))
            options = {**case["sdk_options"], **step.get("sdk_options", {})}
            result = client.chat.completions.create(
                model=args.model or case["model"],
                messages=messages,
                **({"tools": scenario["tools"]} if scenario["tools"] else {}),
                **options,
            )
            if case["stream"]:
                try:
                    message, finish = assemble(dump(chunk) for chunk in result)
                finally:
                    close = getattr(result, "close", None)
                    if close is not None:
                        close()
            else:
                choice = dump(result)["choices"][0]
                message, finish = choice["message"], choice["finish_reason"]
            check_response(message, finish, scenario, step, options)
            messages.append(message)
            print(step["name"])


if __name__ == "__main__":
    try:
        main()
    except APIStatusError as error:
        # Keep proxy/HTML pages out of CLI output, retaining structured API
        # errors so rejected parameters can be diagnosed without another call.
        detail = error.body if isinstance(error.body, dict) else {}
        detail = detail.get("error", detail)
        message = detail.get("message", "") if isinstance(detail, dict) else ""
        raise SystemExit(
            f"{type(error).__name__}: HTTP {error.status_code} {str(message)[:300]}"
        ) from None
