#!/usr/bin/env python
"""Post-training sanity check: does the adapter emit <tool_call> tags?

Loads the trained adapter on the 4-bit base, generates on a few TAC-style
prompts, and asserts well-formed <tool_call>{JSON}</tool_call> output naming a
tool from the provided schema. This catches the modules_to_save/embedding bug
immediately: if embed_tokens/lm_head didn't train, the model cannot emit the
tool-call control tokens and this fails on every prompt.

Usage (GPU must be free of vLLM first — see README):
  python check_tags.py --adapter /workspace/runs/anchor-v2/final
"""

import argparse
import json
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-7B"

TOOLS = [
    {"type": "function", "function": {
        "name": "search_experiences",
        "description": "Search bookable travel experiences by keyword.",
        "parameters": {"type": "object", "properties": {
            "keywords": {"type": "string", "description": "Search keywords, e.g. a city name"},
        }, "required": ["keywords"]}}},
    {"type": "function", "function": {
        "name": "check_availability",
        "description": "Check availability of an experience for a party size.",
        "parameters": {"type": "object", "properties": {
            "experience_id": {"type": "string"},
            "num_tickets": {"type": "integer"},
        }, "required": ["experience_id", "num_tickets"]}}},
]

PROMPTS = [
    "Hi! I'd like to book a food tour in Barcelona for 4 people this Saturday.",
    "Can you find snorkeling experiences in Oahu? There are two of us.",
    "I want tickets to a flamenco show in Seville for tonight, party of 3.",
]

SYSTEM = ("You are a travel booking assistant. Use the provided tools to search "
          "for and book experiences for the user.")

TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    tool_names = {t["function"]["name"] for t in TOOLS}
    failures = []
    for prompt in PROMPTS:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        n_prompt = inputs["input_ids"].shape[1]
        completion = tokenizer.decode(out[0][n_prompt:], skip_special_tokens=False)
        print(f"\n=== PROMPT: {prompt}\n--- COMPLETION:\n{completion}\n")

        m = TAG_RE.search(completion)
        if not m:
            failures.append(f"no <tool_call>...</tool_call> tags: {prompt!r}")
            continue
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError:
            failures.append(f"tool_call body is not valid JSON: {prompt!r}")
            continue
        if call.get("name") not in tool_names:
            failures.append(f"called unknown tool {call.get('name')!r}: {prompt!r}")
            continue
        print(f"OK — well-formed call to {call['name']}({call.get('arguments')})")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        print("\nIf every prompt failed with missing tags, suspect the "
              "modules_to_save/embedding bug: verify the adapter dir contains "
              "the saved embed_tokens/lm_head weights.")
        sys.exit(1)
    print(f"\nALL {len(PROMPTS)} PROMPTS PASSED — <tool_call> emission works.")


if __name__ == "__main__":
    main()
