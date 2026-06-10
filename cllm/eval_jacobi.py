"""
Three-way GSM8K eval: greedy baseline vs Jacobi (pre-distillation) vs Jacobi
(post-distillation). Reuses the prompt + answer-extraction logic from the repo's
inference/eval/eval_gsm8k.py, but runs decoding locally (HF) so we can measure
forward-passes-per-token for Jacobi.

Reports per run:
    accuracy            -- exact-match on the boxed/final number
    fwd_per_token       -- speed metric; greedy AR == 1.0, lower is faster
                           (post-distillation Jacobi should drop well below pre)

Usage:
    # 1) greedy baseline
    python cllm/eval_jacobi.py --model Qwen/Qwen3-8B --mode greedy --num 200
    # 2) Jacobi on the base model (pre-distillation)
    python cllm/eval_jacobi.py --model Qwen/Qwen3-8B --mode jacobi --n 16 --num 200
    # 3) Jacobi on the distilled model (LoRA adapter on top of the base)
    python cllm/eval_jacobi.py --model Qwen/Qwen3-8B --mode jacobi --n 16 --num 200 \
        --lora-adapter ckpts/cllm-qwen3-8b-gsm8k
"""
import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from jacobi import jacobi_generate


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


def extract_pred(text):
    """Same extraction policy as inference/eval/eval_gsm8k.py."""
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].replace(",", "").replace("$", "").replace("\\", "").strip()
    after = text.split("</think>")[-1] if "</think>" in text else text
    nums = re.findall(r'[\d,]+', after)
    return nums[-1].replace(",", "") if nums else "?"


def extract_gold(answer):
    return answer.split("####")[-1].replace(",", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--lora-adapter", default=None,
                    help="path to distilled LoRA adapter (post-distillation eval)")
    ap.add_argument("--mode", choices=["greedy", "jacobi"], default="jacobi")
    ap.add_argument("--n", type=int, default=16, help="Jacobi block size")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="float16",
                    help="float16 on V100 (bf16 is emulated/slow pre-Ampere)")
    ap.add_argument("--device", default="cuda",
                    help='"cuda", "auto" (shard 8B across GPUs), or e.g. "cuda:0"')
    args = ap.parse_args()

    device_map = "auto" if args.device == "auto" else args.device
    dev = "cuda:0" if args.device == "auto" else args.device

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True, device_map=device_map).eval()
    if args.lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
        print(f"Loaded LoRA adapter: {args.lora_adapter}")

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.num))

    correct, total_fwd, total_tok = 0, 0, 0
    for i, ex in enumerate(ds):
        prompt = build_prompt(tok, ex["question"])
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)

        if args.mode == "greedy":
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            new_ids = out[0, ids.shape[1]:].tolist()
            total_fwd += len(new_ids); total_tok += len(new_ids)   # AR: 1 fwd/token
        else:
            new_ids, stats = jacobi_generate(
                model, tok, ids, args.n, args.max_new_tokens, device=dev)
            total_fwd += stats["forward_passes"]; total_tok += stats["tokens"]

        text = tok.decode(new_ids, skip_special_tokens=True)
        if extract_pred(text) == extract_gold(ex["answer"]):
            correct += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ds)}  acc={correct / (i + 1):.3f}  "
                  f"fwd/tok={total_fwd / max(total_tok, 1):.3f}")

    print("\n==== RESULT ====")
    print(f"mode          : {args.mode}"
          + (f" + LoRA({args.lora_adapter})" if args.lora_adapter else ""))
    print(f"n (block)     : {args.n if args.mode == 'jacobi' else '-'}")
    print(f"accuracy      : {correct / len(ds):.4f}  ({correct}/{len(ds)})")
    print(f"fwd_per_token : {total_fwd / max(total_tok, 1):.4f}  "
          f"(greedy=1.0; lower=faster)")


if __name__ == "__main__":
    main()
