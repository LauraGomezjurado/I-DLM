"""
Core Jacobi decoding primitives (first-principles port of CLLM's fixed-point loop).

A Jacobi block guesses `n` tokens at once and refines ALL of them in parallel
each forward pass until the block stops changing (a fixed point). The fixed point
is provably identical to greedy AR decoding for that block -- that invariant is the
unit test for this file (see `verify_fixed_point_matches_greedy`).

Used by both generate_trajectory_qwen.py (collect trajectories for distillation)
and eval_jacobi.py (decode + measure forward-passes-per-token).
"""
import random
import torch


@torch.inference_mode()
def jacobi_block(model, context_ids, n, pad_id, device="cuda",
                 max_iters=512, record=True, seed_from_context=True):
    """
    Run Jacobi fixed-point iteration on a block of `n` new tokens appended to context.

    Args:
        model:        HF causal LM (any architecture).
        context_ids:  LongTensor [1, plen] -- fixed prefix (prompt + already-accepted tokens).
        n:            block size (number of new tokens refined in parallel).
        pad_id:       token id used to pad the tensor before seeding.
        record:       if True, return every intermediate iterate (needed for distillation).
        seed_from_context: seed the n new positions with random tokens drawn from the
                      context (CLLM's init); if False, seed with pad_id.

    Returns:
        (trajectory, n_iters)
          trajectory: list of LongTensor [1, plen+n]; trajectory[-1] is the fixed point.
                      If record=False, only the fixed point is returned (list of length 1).
          n_iters:    number of forward passes used to converge.
    """
    plen = context_ids.shape[1]
    total = plen + n
    tokens = torch.full((1, total), pad_id, dtype=torch.long, device=device)
    tokens[0, :plen] = context_ids[0]
    if seed_from_context:
        tokens[0, plen:] = torch.tensor(
            random.choices(context_ids[0].tolist(), k=n), device=device)

    attn = torch.ones_like(tokens)
    traj = [tokens.clone()] if record else None
    cur = tokens
    n_iters = 0
    while True:
        logits = model(cur, attention_mask=attn).logits          # [1, total, V]
        n_iters += 1
        nxt = logits.argmax(dim=-1)                              # greedy at ALL positions
        new = cur.clone()
        # AR left-shift: token at pos t comes from logits at t-1; prompt stays fixed.
        new[0, plen:] = nxt[0, plen - 1:total - 1]
        if torch.equal(new, cur):                               # converged
            break
        if record:
            traj.append(new.clone())
        cur = new
        if n_iters >= max_iters:                                # safety valve
            break

    if record:
        # traj[-1] is now the fixed point (no trailing duplicate).
        return traj, n_iters
    return [new], n_iters


@torch.inference_mode()
def jacobi_generate(model, tokenizer, prompt_ids, n, max_new_tokens,
                    device="cuda", max_iters_per_block=512):
    """
    Full blockwise Jacobi generation: converge a block, accept it, slide forward,
    repeat until EOS or max_new_tokens.

    Returns:
        (new_token_ids, stats)
          new_token_ids: list[int] of generated tokens (excludes the prompt).
          stats: {"forward_passes", "tokens", "fwd_per_token"}.
                 fwd_per_token is the speed metric: greedy AR == 1.0, lower is faster.
    """
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    ctx = prompt_ids.to(device)
    out, total_iters, produced = [], 0, 0

    while produced < max_new_tokens:
        traj, it = jacobi_block(model, ctx, n, pad, device=device,
                                record=False, max_iters=max_iters_per_block)
        total_iters += it
        block = traj[-1][0, ctx.shape[1]:].tolist()             # the n converged tokens
        if eos in block:
            block = block[:block.index(eos) + 1]
            out.extend(block); produced += len(block)
            break
        out.extend(block); produced += len(block)
        ctx = torch.cat([ctx, torch.tensor(block, device=device).view(1, -1)], dim=1)

    return out, {
        "forward_passes": total_iters,
        "tokens": produced,
        "fwd_per_token": total_iters / max(produced, 1),
    }


@torch.inference_mode()
def jacobi_generate_cached(model, tokenizer, prompt_ids, n, max_new_tokens,
                           device="cuda", max_iters_per_block=512):
    """
    KV-cache Jacobi decoding -- the wall-clock-optimized decoder.

    Unlike jacobi_generate (which re-runs the whole [prompt, block] every iteration),
    this prefills the prompt ONCE into a DynamicCache and then forwards only the
    n-token block each Jacobi iteration, attending to the cached prompt+accepted KV.
    The provisional block KV is dropped (cache.crop) before each iteration and kept
    permanently only once the block converges. Per-iteration cost drops from
    O(prompt+generated) to O(n) -> real tokens/sec speedup, and because it uses the
    exact AR cache path the output matches greedy without the fp16 drift of the
    full-recompute path.

    Returns (new_token_ids, stats) with the same keys as jacobi_generate plus
    "seconds" and "tokens_per_sec".
    """
    import time
    from transformers import DynamicCache

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    prompt_ids = prompt_ids.to(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # --- prefill prompt once ---
    cache = DynamicCache()
    out = model(prompt_ids, past_key_values=cache, use_cache=True)
    first_tok_logit = out.logits[:, -1, :]              # predicts the first generated token
    cache_len = cache.get_seq_length()                 # == prompt length

    generated, total_iters = [], 0
    while len(generated) < max_new_tokens:
        # seed block: position 0 is the greedy next token (fixed); rest repeat it
        first_tok = first_tok_logit.argmax(-1)          # [1]
        block = first_tok.view(1, 1).expand(1, n).clone()

        it = 0
        while True:
            cache.crop(cache_len)                       # drop provisional block KV
            pos = torch.arange(cache_len, cache_len + n, device=device)
            out = model(block, past_key_values=cache, use_cache=True, cache_position=pos)
            it += 1
            logits = out.logits[0]                      # [n, V]
            new_block = block.clone()
            new_block[0, 1:] = logits[:-1].argmax(-1)   # pos i+1 predicted by logits[i]
            # new_block[0,0] stays = greedy first token
            if torch.equal(new_block, block) or it >= max_iters_per_block:
                block = new_block
                break
            block = new_block

        total_iters += it
        cache_len += n                                  # keep converged block KV
        first_tok_logit = logits[-1:].clone()           # predicts next block's first token

        toks = block[0].tolist()
        if eos in toks:
            toks = toks[:toks.index(eos) + 1]
            generated.extend(toks)
            break
        generated.extend(toks)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    produced = len(generated)
    return generated, {
        "forward_passes": total_iters,
        "tokens": produced,
        "fwd_per_token": total_iters / max(produced, 1),
        "seconds": secs,
        "tokens_per_sec": produced / max(secs, 1e-9),
    }


@torch.inference_mode()
def verify_fixed_point_matches_greedy(model, tokenizer, prompt_ids, n, device="cuda"):
    """Sanity check: one Jacobi block's fixed point must equal greedy decoding."""
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    traj, _ = jacobi_block(model, prompt_ids.to(device), n, pad, device=device)
    jacobi_block_tokens = traj[-1][0, prompt_ids.shape[1]:].tolist()

    greedy = model.generate(prompt_ids.to(device), max_new_tokens=n, do_sample=False,
                            pad_token_id=pad)[0, prompt_ids.shape[1]:].tolist()
    return jacobi_block_tokens == greedy, jacobi_block_tokens, greedy
