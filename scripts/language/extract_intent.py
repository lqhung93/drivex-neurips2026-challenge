"""Structured intent of every cache instruction with Qwen2.5-3B-Instruct, run once on one GPU (fp16).

  python scripts/language/extract_intent.py --splits train val --out-dir results/runs/intent-3b
  python scripts/language/extract_intent.py --splits test --purpose submission --out-dir results/runs/intent-3b

Unique instructions are run once (greedy, so identical text gives identical output); outputs that
fail to parse get one retry with a longer budget, then stay null. Writes <out-dir>/intent_<split>.json:
one entry per cache row, in cache order, plus the sha256 of the instruction column, which
scripts/language/encode_intent.py turns into the L3 input. The test split needs --purpose submission.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from drivex26.data import assert_split_allowed, load_cache
from drivex26.text import texts_sha256

ROOT = Path(__file__).resolve().parents[2]
MODEL, BATCH = "Qwen/Qwen2.5-3B-Instruct", 32
# same lists as encode_intent.py
MANEUVERS = ["go_straight", "turn_left", "turn_right", "lane_change_left", "lane_change_right", "stop",
             "slow_down", "follow", "u_turn", "other"]
SPEEDS = ["stop", "slow", "keep", "faster"]
SYSTEM = (
    "You convert a driving instruction for the ego car into JSON. Keys: "
    f'"maneuver" (one of {MANEUVERS}), "then" (the next maneuver from the same list, or null), '
    f'"speed" (one of {SPEEDS}), "wait_first" (true if the car must wait before acting), '
    '"reference" (short noun phrase of the object it refers to, or null). Reply with JSON only.'
)


def parse(text: str):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        out = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    ok = (out.get("maneuver") in MANEUVERS and out.get("speed") in SPEEDS
          and (out.get("then") is None or out.get("then") in MANEUVERS))
    return out if ok else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--purpose", default="eval")
    args = p.parse_args()
    for split in args.splits:
        assert_split_allowed(split, args.purpose)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda:0").eval()

    def generate(texts, max_new_tokens):
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
                                           tokenize=False, add_generation_prompt=True) for t in texts]
        out = []
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
            out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        texts = [str(t) for t in load_cache(args.cache_dir / f"{split}.npz")["instruction"]]
        unique = sorted(set(texts))
        start = time.perf_counter()
        raw = dict(zip(unique, generate(unique, 80)))
        parsed = {t: parse(r) for t, r in raw.items()}
        failed = [t for t, v in parsed.items() if v is None]
        for t, r in zip(failed, generate(failed, 160)):
            raw[t], parsed[t] = r, parse(r)
        (args.out_dir / f"intent_{split}.json").write_text(json.dumps(
            {"model": MODEL, "split": split, "texts_sha256": texts_sha256(texts), "rows": [parsed[t] for t in texts],
             "raw": [raw[t] for t in texts]}, indent=1))
        print(json.dumps({split: {"rows": len(texts), "unique": len(unique), "retried": len(failed),
                                  "seconds": round(time.perf_counter() - start, 1),
                                  "valid_unique": float(np.mean([v is not None for v in parsed.values()]))}}),
              flush=True)


if __name__ == "__main__":
    main()
