"""
Collect Jacobi trajectories from Qwen3-8B over GSM8K (train split) for consistency
distillation. Port of CLLM's data/generate_trajectory.py, adapted to Qwen + HF.

Each saved record is ONE Jacobi block:
    {
      "trajectory": [[ids...], ...],   # K iterates, each length L = context_len + n,
                                       #   trajectory[-1] is the fixed point.
      "labels":     [ids...],          # length L; -100 on context, fixed-point tokens
                                       #   on the block region (self-distillation AR target).
      "context_len": int,
    }

Usage:
    python cllm/generate_trajectory_qwen.py \
        --model Qwen/Qwen3-8B --n 16 --max-new-tokens 256 \
        --num-questions 1000 --out data/gsm8k_qwen_traj.jsonl --verify
"""
import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from jacobi import jacobi_block, verify_fixed_point_matches_greedy

IGNORE = -100


def build_prompt(tokenizer, question):
    """GSM8K prompt, Qwen chat template with thinking OFF (keeps answers short)."""
    user = (f"{question}\nPlease reason step by step, and put your final answer "
            f"within \\boxed{{}}.")
    msgs = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # tokenizers without the enable_thinking kwarg
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=16, help="Jacobi block size (n_token_seq_size)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-questions", type=int, default=1000)
    ap.add_argument("--max-blocks-per-q", type=int, default=8)
    ap.add_argument("--out", default="data/gsm8k_qwen_traj.jsonl")
    ap.add_argument("--device", default="cuda",
                    help='"cuda" (single GPU), "auto" (shard across GPUs -- needed for 8B '
                         'on 16GB cards), or e.g. "cuda:0"')
    ap.add_argument("--dtype", default="float16",
                    help="float16 on V100 (bf16 is emulated/slow pre-Ampere)")
    ap.add_argument("--verify", action="store_true",
                    help="check fixed point == greedy on the first question, then continue")
    args = ap.parse_args()

    # On 16GB cards an 8B model must be sharded; inputs then live on the first shard.
    device_map = "auto" if args.device == "auto" else args.device
    dev = "cuda:0" if args.device == "auto" else args.device

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True, device_map=device_map).eval()

    ds = load_dataset("openai/gsm8k", "main", split="train").select(range(args.num_questions))
    pad = tok.pad_token_id
    eos = tok.eos_token_id

    if args.verify:
        ids = tok(build_prompt(tok, ds[0]["question"]), return_tensors="pt").input_ids
        ok, jb, gd = verify_fixed_point_matches_greedy(model, tok, ids, args.n, dev)
        print(f"[verify] fixed_point == greedy : {ok}")
        if not ok:
            print(f"  jacobi: {jb}\n  greedy: {gd}")
            raise SystemExit("Fixed-point invariant FAILED -- fix jacobi.py before collecting.")

    n_records = 0
    with open(args.out, "w") as f:
        for qi, ex in enumerate(ds):
            prompt = build_prompt(tok, ex["question"])
            ctx = tok(prompt, return_tensors="pt").input_ids.to(dev)
            produced = 0
            for _ in range(args.max_blocks_per_q):
                if produced >= args.max_new_tokens:
                    break
                traj, _ = jacobi_block(model, ctx, args.n, pad, device=dev, record=True)
                L = ctx.shape[1] + args.n
                fixed = traj[-1]                                  # [1, L]

                labels = torch.full((L,), IGNORE, dtype=torch.long)
                labels[ctx.shape[1]:] = fixed[0, ctx.shape[1]:].cpu()

                rec = {
                    "trajectory": [t[0].cpu().tolist() for t in traj],
                    "labels": labels.tolist(),
                    "context_len": ctx.shape[1],
                }
                f.write(json.dumps(rec) + "\n")
                n_records += 1

                block = fixed[0, ctx.shape[1]:]
                produced += args.n
                if eos in block.tolist():
                    break
                ctx = torch.cat([ctx, block.view(1, -1)], dim=1)

            if (qi + 1) % 50 == 0:
                print(f"  {qi + 1}/{len(ds)} questions -> {n_records} block records")

    print(f"Done. Wrote {n_records} trajectory records to {args.out}")


if __name__ == "__main__":
    main()
