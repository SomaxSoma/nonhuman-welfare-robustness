#!/usr/bin/env python
"""Build the mixed SFT dataset: ~40% recovery / ~60% APIGen-MT.

Downloads both source datasets from HuggingFace, normalizes each into one
unified chat schema, mixes and shuffles them together (NOT sequentially),
and writes a single combined JSONL plus a stats JSON consumed by train.py.

Unified row schema (what this script emits, one JSON object per line):
  {
    "messages": [
      {"role": "system"|"user"|"assistant"|"tool", "content": str,
       # assistant messages that call tools additionally carry:
       "tool_calls": [{"type": "function",
                       "function": {"name": str, "arguments": dict}}]}
    ],
    "tools":  [{"type": "function", "function": {name, description?, parameters}}],
    "source": "apigen" | "recovery"
  }

Source formats being normalized:
  - Salesforce/APIGen-MT-5k (GATED — accept terms on HF while logged in):
    ShareGPT `conversations` with from/value keys (human, gpt, function_call,
    observation) plus separate `system` and `tools` (JSON string) fields.
  - CompassioninMachineLearning/agentic-tool-recovery-sft:
    `messages` format; tool_calls[].function.arguments is already a dict.

The `arguments` field is normalized to a dict in BOTH sources so every row is
identical in shape (Qwen2.5's chat template tojson-dumps dicts itself).

Usage (on the pod, HF_HOME pointed at the large volume):
  python build_dataset.py --output-dir /workspace/data --show-samples 3 --render
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

APIGEN_REPO = "Salesforce/APIGen-MT-5k"
RECOVERY_REPO = "CompassioninMachineLearning/agentic-tool-recovery-sft"

VALID_ROLES = {"system", "user", "assistant", "tool"}


class NormError(Exception):
    """Row cannot be normalized; carries a short reason for the drop report."""


def parse_json(value, what):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as e:
        raise NormError(f"unparseable {what}") from e


def normalize_tool(tool):
    """Wrap a tool definition into OpenAI style {type, function:{...}}."""
    if not isinstance(tool, dict):
        raise NormError("tool is not a dict")
    if "function" in tool and isinstance(tool["function"], dict):
        fn = tool["function"]
    else:
        fn = tool
    if "name" not in fn:
        raise NormError("tool missing name")
    out_fn = {"name": fn["name"]}
    if fn.get("description") is not None:
        out_fn["description"] = fn["description"]
    params = fn.get("parameters")
    if params is not None:
        out_fn["parameters"] = parse_json(params, "tool parameters")
    return {"type": "function", "function": out_fn}


def normalize_tool_call(call):
    """Normalize one tool call into {type, function:{name, arguments: dict}}."""
    if not isinstance(call, dict):
        raise NormError("tool call is not a dict")
    fn = call.get("function", call)
    name = fn.get("name")
    if not name:
        raise NormError("tool call missing name")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        args = parse_json(args, "tool call arguments")
    if not isinstance(args, dict):
        raise NormError("tool call arguments not a dict")
    return {"type": "function", "function": {"name": name, "arguments": args}}


def convert_apigen(row):
    """ShareGPT (from/value) -> unified messages. Mapping:
    human->user, gpt->assistant, function_call->assistant w/ tool_calls,
    observation->tool message(s)."""
    messages = []
    system = row.get("system")
    if system:
        messages.append({"role": "system", "content": system})

    for turn in row["conversations"]:
        frm, value = turn.get("from"), turn.get("value")
        if frm == "human":
            messages.append({"role": "user", "content": value or ""})
        elif frm == "gpt":
            messages.append({"role": "assistant", "content": value or ""})
        elif frm == "function_call":
            parsed = parse_json(value, "function_call value")
            calls = parsed if isinstance(parsed, list) else [parsed]
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [normalize_tool_call(c) for c in calls],
            })
        elif frm == "observation":
            try:
                parsed = parse_json(value, "observation value")
            except NormError:
                parsed = value  # keep raw string observation as-is
            results = parsed if isinstance(parsed, list) else [parsed]
            for r in results:
                content = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
                messages.append({"role": "tool", "content": content})
        else:
            raise NormError(f"unknown ShareGPT role {frm!r}")

    tools = parse_json(row.get("tools") or "[]", "tools field")
    return finish_row(messages, tools, "apigen")


def convert_recovery(row):
    """Already messages-format; verify roles and normalize tool_calls args to dicts."""
    messages = []
    for msg in row["messages"]:
        role = msg.get("role")
        if role not in VALID_ROLES:
            raise NormError(f"unknown role {role!r}")
        out = {"role": role, "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            out["tool_calls"] = [normalize_tool_call(c) for c in msg["tool_calls"]]
        messages.append(out)

    tools = row.get("tools") or []
    if isinstance(tools, str):
        tools = parse_json(tools, "tools field")
    return finish_row(messages, tools, "recovery")


def finish_row(messages, tools, source):
    if not any(m["role"] == "assistant" for m in messages):
        raise NormError("no assistant message")
    if not isinstance(tools, list) or not tools:
        raise NormError("empty tools list")
    return {
        "messages": messages,
        "tools": [normalize_tool(t) for t in tools],
        "source": source,
    }


def convert_all(dataset, converter, label):
    rows, drops = [], Counter()
    for raw in dataset:
        try:
            rows.append(converter(raw))
        except NormError as e:
            drops[str(e)] += 1
    if drops:
        print(f"[{label}] dropped {sum(drops.values())} rows: {dict(drops)}")
    return rows, drops


def check_schema_compat(apigen_rows, recovery_rows):
    """Both sources' tools must have the same normalized function shape."""
    def shapes(rows):
        s = set()
        for r in rows[:200]:
            for t in r["tools"]:
                s.add(tuple(sorted(t["function"].keys())))
        return s

    a, b = shapes(apigen_rows), shapes(recovery_rows)
    print(f"tool-function key shapes — apigen: {sorted(a)}  recovery: {sorted(b)}")
    for rows in (apigen_rows, recovery_rows):
        for r in rows:
            for t in r["tools"]:
                assert t["type"] == "function" and "name" in t["function"]
    print("schema compatibility check passed (all tools are OpenAI-style function defs)")


def show_samples(rows, n, label):
    with_calls = [r for r in rows if any(m.get("tool_calls") for m in r["messages"])]
    print(f"\n{'=' * 20} {label}: {n} fully-normalized sample rows {'=' * 20}")
    for r in with_calls[:n]:
        print(json.dumps(r, indent=2, ensure_ascii=False)[:4000])
        print("-" * 70)


def render_comparison(apigen_rows, recovery_rows, tokenizer_id):
    """Render one tool-calling row per source through the Qwen chat template so
    the tool-call formatting can be eyeballed for cross-source consistency."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    for label, rows in (("APIGEN", apigen_rows), ("RECOVERY", recovery_rows)):
        row = next(r for r in rows if any(m.get("tool_calls") for m in r["messages"]))
        text = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
        print(f"\n{'=' * 20} rendered chat template — {label} {'=' * 20}")
        print(text[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/workspace/data")
    ap.add_argument("--recovery-frac", type=float, default=0.40,
                    help="target recovery share of the final mix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-samples", type=int, default=0,
                    help="print N normalized sample rows per source")
    ap.add_argument("--render", action="store_true",
                    help="render one tool-calling row per source via the chat template")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    args = ap.parse_args()

    print(f"loading {APIGEN_REPO} (gated — if this 401/403s, accept the terms "
          f"on its HF page while logged in; the token itself is likely fine)")
    apigen_raw = load_dataset(APIGEN_REPO, split="train")
    print(f"loading {RECOVERY_REPO}")
    recovery_raw = load_dataset(RECOVERY_REPO, split="train")
    print(f"raw counts — apigen: {len(apigen_raw)}  recovery: {len(recovery_raw)}")

    apigen, apigen_drops = convert_all(apigen_raw, convert_apigen, "apigen")
    recovery, recovery_drops = convert_all(recovery_raw, convert_recovery, "recovery")

    check_schema_compat(apigen, recovery)

    # Use ALL recovery rows; size the APIGen side toward the target ratio.
    # (3,750 recovery at 40% would need 5,625 APIGen rows; the 5k set can't
    # supply that, so with all of both the achieved mix lands near 43/57.)
    rng = random.Random(args.seed)
    target_apigen = round(len(recovery) * (1 - args.recovery_frac) / args.recovery_frac)
    if len(apigen) > target_apigen:
        apigen = rng.sample(apigen, target_apigen)

    mixed = apigen + recovery
    rng.shuffle(mixed)  # interleaved from the start — never sequential

    n_total, n_rec = len(mixed), len(recovery)
    achieved = n_rec / n_total
    stats = {
        "apigen_repo": APIGEN_REPO,
        "recovery_repo": RECOVERY_REPO,
        "total_examples": n_total,
        "apigen_examples": len(apigen),
        "recovery_examples": n_rec,
        "achieved_recovery_frac": round(achieved, 4),
        "target_recovery_frac": args.recovery_frac,
        "dropped": {"apigen": dict(apigen_drops), "recovery": dict(recovery_drops)},
        "seed": args.seed,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "combined.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nwrote {data_path}")
    print(f"TOTAL examples:     {n_total}")
    print(f"  from APIGen-MT:   {len(apigen)}  ({1 - achieved:.1%})")
    print(f"  from recovery:    {n_rec}  ({achieved:.1%})")
    print(f"ACHIEVED ratio:     {achieved:.1%} recovery / {1 - achieved:.1%} APIGen-MT "
          f"(target was {args.recovery_frac:.0%}/{1 - args.recovery_frac:.0%})")

    if args.show_samples:
        show_samples([r for r in mixed if r["source"] == "apigen"], args.show_samples, "APIGEN")
        show_samples([r for r in mixed if r["source"] == "recovery"], args.show_samples, "RECOVERY")
    if args.render:
        render_comparison(apigen, recovery, args.tokenizer)


if __name__ == "__main__":
    sys.exit(main())
