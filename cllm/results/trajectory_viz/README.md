# Jacobi trajectory visualizations

Artifacts produced by [`cllm/visualize_trajectory.py`](../../visualize_trajectory.py).
Each run compares **PRE** (base Qwen) vs **POST** (base + CLLM-distilled LoRA) Jacobi
decoding on the same OpenThoughts prompts.

## Interactive dashboard (`streamlit_app.py`)

An interactive, **shareable** twin of `trajectory.html`. Same story — Overview +
per-prompt convergence staircases + decoded text per iteration — but as a live app you
can host. It reads `trajectory_data.json` directly, so it needs **no model, no torch,
no tokenizer** (decoded text is baked into the JSON).

Run it locally:

```bash
pip install -r cllm/results/trajectory_viz/requirements.txt
streamlit run cllm/results/trajectory_viz/streamlit_app.py
```

The sidebar picks the run (the top-level file or any `archive/<name>/`), switches
between the Overview and individual prompts, and can **upload** any other
`trajectory_data.json`.

Grid cells are labeled with their token text from a compact `tok_display` map
(`{token_id: text}`) baked into the JSON, so **no tokenizer is needed at runtime**.
`visualize_trajectory.py` now writes that map automatically; for JSONs produced before
this change, retro-fit it once:

```bash
python cllm/enrich_trajectory_json.py cllm/results/trajectory_viz/trajectory_data.json
```

Only a JSON missing the map falls back to an optional "per-token cell labels" toggle
(loads a tokenizer via `transformers`).

### Share it (Streamlit Community Cloud — free)

1. Commit `streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`, and at
   least one `trajectory_data.json` to GitHub (this repo already has the remote).
2. Go to <https://share.streamlit.io> → **Create app** → **Deploy a public app from
   GitHub**, and sign in with GitHub.
3. Pick repo `LauraGomezjurado/I-DLM`, the branch, and set **Main file path** to
   `cllm/results/trajectory_viz/streamlit_app.py`.
4. (Optional) **Advanced settings** → choose a Python version. Deploy. Community Cloud
   installs from the co-located `requirements.txt` and gives you a public
   `*.streamlit.app` URL to share.

> The deployed app only sees files committed to the repo, so the `trajectory_data.json`
> (and any `archive/` runs) you want visible must be pushed. Each is ~0.4–0.7 MB.

## Files (latest run, top level)

| file | what it is |
|------|-----------|
| `trajectory.html` | tabbed report — Overview tab + one tab per prompt (flip with tabs or ←/→). Drift drill-downs inline. |
| `trajectory_data.json` | **the data** — full per-iteration trajectories (token ids + decoded text), stats, drift, for every prompt × {PRE, POST} × block. |
| `convergence_grid.png` | prompt-0 block-0 staircase, PRE vs POST. |
| `summary_per_prompt.png` | per-prompt fwd/token, PRE vs POST. |
| `position_convergence.png` | mean settle-iteration per block position (where the forwards go). |
| `drift_vs_depth.png` | how greedy-equivalence breaks compound with block depth. |

Re-render any saved JSON without the model:
`python cllm/visualize_trajectory.py --from-json <path/to/trajectory_data.json>`

## `archive/` — preserved runs (top level gets overwritten each run)

- `8b-ot_20p_3blk/` — Qwen3-8B + `cllm-8b-ot`, 20 prompts, 3 blocks each.

## Inspecting `trajectory_data.json`

Stdlib browser (no torch/tokenizer needed):

```bash
python cllm/inspect_traj.py results/trajectory_viz/trajectory_data.json            # list all prompts
python cllm/inspect_traj.py results/trajectory_viz/trajectory_data.json --idx 4    # dump prompt 4
```

### JSON schema

```jsonc
{
  "model": "Qwen/Qwen3-8B", "lora_adapter": "ckpts/cllm-8b-ot",
  "dataset": "openthoughts", "n": 16, "max_blocks": 3, "seed": 0,
  "prompts": [
    {
      "idx": 0,
      "raw": "<the OpenThoughts prompt text>",
      "prompt_text": "<full chat-templated prompt fed to the model>",
      "drift": [ ...block indices where POST fixed point != PRE fixed point... ],
      "pre":  { "stats": {"blocks","total_iters","tokens","fwd_per_token"},
                "blocks": [ <block>, ... ] },
      "post": { ...same shape... }   // null if no adapter
    }
  ]
}
```

Each `<block>`:

```jsonc
{
  "block_index": 0,
  "context_len": 696,           // prompt+accepted tokens before this 16-token block
  "n_iters": 15,                // Jacobi forward passes to converge (the speed metric)
  "n_real": 16,                 // tokens counted toward fwd/token (trimmed at EOS)
  "has_eos": false,
  "states":      [[id,...], ...],   // every iterate, each length n; states[0]=random seed,
                                    //   states[-1]=fixed point. Token IDS.
  "fixed_text":  "To solve this problem, ...",   // decoded fixed point
  "state_texts": ["<seed text>", "<iter1 text>", ..., "<fixed text>"]  // decoded per iterate
}
```

**Only the first drift block is a same-context comparison** — after PRE and POST
disagree once, every later block trivially differs (the two models are continuing
different text). `inspect_traj.py` and the HTML both report that **first divergence**.
