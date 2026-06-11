"""
Consistency distillation on collected Jacobi trajectories (port of CLLM's
cllm_trainer_global.py loss, simplified to a readable single-file loop).

Loss per step (batch size 1; trajectories are variable length):
    consistency (global): push student logits at a RANDOM intermediate trajectory
        point toward the (stop-grad) teacher logits at the fixed point.
    AR: cross-entropy predicting the fixed-point block tokens (self-distillation).
    total = w_ar * loss_ar + loss_global          # CLLM uses w_ar ~ 10

Defaults to LoRA so a single 80GB GPU can run it on Qwen3-8B. Use --full for a
full fine-tune (needs multi-GPU / DeepSpeed -- see training/ for the heavier setup).

Usage:
    python cllm/train_cllm.py \
        --model Qwen/Qwen3-8B --data data/gsm8k_qwen_traj.jsonl \
        --out ckpts/cllm-qwen3-8b-gsm8k --epochs 1 --lr 1e-5
"""
import argparse
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE = -100


class TrajectoryDataset(Dataset):
    def __init__(self, path):
        self.records = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return (
            [torch.tensor(t, dtype=torch.long) for t in r["trajectory"]],
            torch.tensor(r["labels"], dtype=torch.long),
        )


def soft_cross_entropy(pred_logits, target_logits, pad_mask):
    """Push student dist (pred) toward teacher dist (target); pad_mask True = ignore."""
    logp = F.log_softmax(pred_logits, dim=-1)
    q = F.softmax(target_logits, dim=-1)
    ent = -(q * logp)                                       # [B, T, V]
    ent = ent.masked_fill(pad_mask.unsqueeze(-1), 0.0)
    denom = (~pad_mask).sum().clamp(min=1)
    return ent.sum() / denom


def cllm_loss(model, trajectory, labels, w_ar, device):
    fixed_pt = trajectory[-1].unsqueeze(0).to(device)          # [1, L]
    labels = labels.unsqueeze(0).to(device)                    # [1, L]

    # --- global consistency: random intermediate point -> fixed point ---
    i = random.randrange(len(trajectory) - 1) if len(trajectory) > 1 else 0
    point = trajectory[i].unsqueeze(0).to(device)
    logits_i = model(point).logits                            # student at intermediate point (grad)
    # One fixed-point forward (with grad), reused for BOTH terms: the AR loss uses
    # it directly; the consistency target is the same logits stop-grad'd (teacher).
    logits_fp = model(fixed_pt).logits

    pad_mask = (labels == IGNORE)                            # mask context + padding
    loss_global = soft_cross_entropy(
        logits_i[:, :-1], logits_fp[:, :-1].detach(), pad_mask[:, :-1])

    # --- AR loss on the fixed-point block (self-distillation anchor) ---
    V = logits_fp.size(-1)
    loss_ar = F.cross_entropy(
        logits_fp[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
        ignore_index=IGNORE)

    return w_ar * loss_ar + loss_global, loss_ar.item(), loss_global.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default="data/gsm8k_qwen_traj.jsonl")
    ap.add_argument("--out", default="ckpts/cllm-qwen3-8b-gsm8k")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--w-ar", type=float, default=10.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--dtype", default="float16",
                    help="float16 on V100 (bf16 is emulated/slow pre-Ampere)")
    ap.add_argument("--device", default="cuda",
                    help='"cuda" (single GPU), or "auto" to shard 8B across GPUs')
    ap.add_argument("--full", action="store_true", help="full finetune instead of LoRA")
    ap.add_argument("--lora-r", type=int, default=128)
    args = ap.parse_args()

    device_map = "auto" if args.device == "auto" else args.device
    dev = "cuda:0" if args.device == "auto" else args.device

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True, device_map=device_map)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # needed for grad-ckpt + frozen base (LoRA)
    model.config.use_cache = False

    if not args.full:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    model.train()
    ds = TrajectoryDataset(args.data)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        for traj, labels in dl:
            loss, lar, lg = cllm_loss(model, traj, labels, args.w_ar, dev)
            (loss / args.grad_accum).backward()
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} "
                      f"loss {loss.item():.4f} (ar {lar:.4f}, global {lg:.4f})")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved {'LoRA adapter' if not args.full else 'model'} to {args.out}")


if __name__ == "__main__":
    main()
