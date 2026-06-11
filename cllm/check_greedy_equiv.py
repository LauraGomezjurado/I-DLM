"""
Direct test of the Jacobi greedy-equivalence guarantee.

For each prompt, generate with (1) greedy AR and (2) the Jacobi decoder(s), to the
SAME token budget, and compare the output token sequences directly -- not just
accuracy. Reports the fraction of generations that are token-identical and, when
they differ, the first divergence position.

This disambiguates the two candidate causes of Jacobi-pre != greedy:
  - run with --dtype float32: if equivalence becomes (near) exact, the fp16
    parallel-vs-incremental arithmetic path was the cause.
  - --decoder cached vs full: the cached decoder shares greedy's KV-cache
    arithmetic and should match far more closely than the full-recompute one.

Usage:
  python cllm/check_greedy_equiv.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --n 16 --num 50 --max-new-tokens 256 --decoder cached --dtype float16
  # then repeat with --dtype float32 and/or --decoder full to compare.
"""
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from jacobi import jacobi_generate, jacobi_generate_cached


def build_prompt(tokenizer, question):
    user = (f"{question}\nPlease reason step by step, and put your final answer "
            f"within \\boxed{{}}.")
    msgs = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def first_divergence(a, b):
    """Index of first differing token, or -1 if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--decoder", choices=["cached", "full"], default="cached")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float16", help="float16 or float32")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # cuda can shard via device_map; cpu/mps load then move.
    if args.device.startswith("cuda") or args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype),
            trust_remote_code=True, device_map=args.device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype),
            trust_remote_code=True).to(args.device).eval()
    gen = jacobi_generate_cached if args.decoder == "cached" else jacobi_generate

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.num))

    identical, div_positions, tok_agree, total_tok = 0, [], 0, 0
    for i, ex in enumerate(ds):
        ids = tok(build_prompt(tok, ex["question"]), return_tensors="pt").input_ids.to(args.device)
        with torch.inference_mode():
            # PURE argmax: override Qwen's generation_config (repetition_penalty=1.1,
            # do_sample=True), else "greedy" != Jacobi's pure argmax.
            g = model.generate(ids, max_new_tokens=args.max_new_tokens,
                               do_sample=False, repetition_penalty=1.0,
                               temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.pad_token_id)
        greedy_ids = g[0, ids.shape[1]:].tolist()
        jac_ids, _ = gen(model, tok, ids, args.n, args.max_new_tokens, device=args.device)
        jac_ids = jac_ids[:args.max_new_tokens]

        L = min(len(greedy_ids), len(jac_ids))
        same = sum(1 for a, b in zip(greedy_ids[:L], jac_ids[:L]) if a == b)
        tok_agree += same; total_tok += L
        d = first_divergence(greedy_ids, jac_ids)
        if d == -1 and len(greedy_ids) == len(jac_ids):
            identical += 1
        else:
            div_positions.append(d if d >= 0 else L)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.num}  identical={identical}  "
                  f"tok_agree={tok_agree/max(total_tok,1):.4f}")

    print("\n==== GREEDY-EQUIVALENCE ====")
    print(f"model/decoder : {args.model}  |  jacobi-{args.decoder}  |  {args.dtype}")
    print(f"token-identical generations : {identical}/{args.num} "
          f"({100*identical/args.num:.1f}%)")
    print(f"mean token agreement        : {tok_agree/max(total_tok,1):.4f}")
    if div_positions:
        mp = sum(div_positions) / len(div_positions)
        print(f"mean first-divergence pos   : {mp:.1f} tokens "
              f"(of {args.max_new_tokens})  [{len(div_positions)} differing]")


if __name__ == "__main__":
    main()
