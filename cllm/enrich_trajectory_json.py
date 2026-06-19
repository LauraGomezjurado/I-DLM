"""
Add a compact `tok_display` map ({token_id: display_string}) to a trajectory_data.json
so the Streamlit dashboard can label every convergence-grid cell with its token text
WITHOUT loading a tokenizer at runtime (keeps the deployed app dependency-light).

The display string is computed with the exact same `tok_cell` rule the app/HTML use, so
the precomputed labels match what `visualize_trajectory.py` renders locally. Only the
unique token ids that actually appear in the trajectories are stored, so the map is tiny.

    python cllm/enrich_trajectory_json.py results/trajectory_viz/trajectory_data.json
    python cllm/enrich_trajectory_json.py <json> [<json> ...]   # enrich several in place
"""
import argparse
import json
import sys

from transformers import AutoTokenizer


def tok_cell(tok, tid):
    """One token id -> short, whitespace-visible display string (matches the app)."""
    s = tok.decode([tid])
    if s == "":                                            # control / byte piece
        s = tok.convert_ids_to_tokens(tid) or "∅"
    s = s.replace("\n", "⏎").replace("\t", "⇥")
    if s != "" and s.strip() == "":                        # pure whitespace
        s = "·" * len(s)
    elif s.startswith(" "):                                # show a leading space
        s = "·" + s[1:]
    return s


def unique_ids(d):
    ids = set()
    for p in d["prompts"]:
        for side in ("pre", "post"):
            s = p.get(side)
            if not s:
                continue
            for b in s["blocks"]:
                for row in b["states"]:
                    ids.update(row)
    return ids


def enrich(path, tok_cache):
    with open(path) as f:
        d = json.load(f)
    model = d.get("model")
    if model not in tok_cache:
        print(f"[load] tokenizer {model}")
        tok_cache[model] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tok = tok_cache[model]
    ids = unique_ids(d)
    d["tok_display"] = {str(tid): tok_cell(tok, tid) for tid in sorted(ids)}
    with open(path, "w") as f:
        json.dump(d, f)
    print(f"[enriched] {path}  ({len(ids)} unique tokens labeled)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+", help="trajectory_data.json file(s) to enrich in place")
    args = ap.parse_args()
    tok_cache = {}
    for p in args.json:
        enrich(p, tok_cache)


if __name__ == "__main__":
    sys.exit(main())
