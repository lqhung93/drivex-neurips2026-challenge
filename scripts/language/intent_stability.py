"""How often the LLM intent survives a paraphrase: compare intents of the original and of the
paraphrased instructions, row by row, and write metrics.json into the paraphrase run directory.

  python scripts/language/intent_stability.py results/runs/<intent run> results/runs/<paraphrase run>
"""

import json
import sys
from pathlib import Path

FIELDS = ("maneuver", "then", "speed", "wait_first")


def main() -> None:
    orig_run, para_run = map(Path, sys.argv[1:3])
    out = {"intent_run": orig_run.name, "paraphrase_run": para_run.name, "splits": {}}
    for split in ("train", "val"):
        a = json.loads((orig_run / f"intent_{split}.json").read_text())
        b = json.loads((para_run / f"intent_{split}.json").read_text())
        if a["texts_sha256"] != b["texts_sha256"]:
            raise ValueError(f"{split}: the two runs cover different instructions")
        both = [(x, y) for x, y in zip(a["rows"], b["rows"]) if x and y]
        same = {f: sum((x.get(f) or False) == (y.get(f) or False) for x, y in both) / len(both) for f in FIELDS}
        same["all_fields"] = sum(all((x.get(f) or False) == (y.get(f) or False) for f in FIELDS) for x, y in both) / len(both)
        out["splits"][split] = {"rows": len(a["rows"]), "both_parsed": len(both), "same": same}
    (para_run / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["splits"], indent=1))


if __name__ == "__main__":
    main()
