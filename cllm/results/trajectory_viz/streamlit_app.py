"""
Shareable Streamlit dashboard for Jacobi decoding trajectories (PRE vs POST CLLM).

This is the interactive, deployable twin of the static `trajectory.html` produced by
`cllm/visualize_trajectory.py`. It reads a saved `trajectory_data.json` and lets you:

  * skim an Overview: aggregate speedup, a sortable per-prompt table, and interactive
    fwd/token / drift / settle-position charts;
  * open any single prompt and read its PRE (base Qwen) vs POST (base + CLLM LoRA)
    Jacobi trajectories block-by-block — the convergence "staircase" colored
    settled-vs-unsettled, plus the decoded text at every refinement iteration.

It is intentionally **self-contained from the JSON**: the decoded text of every iterate
is stored in the file, so the app needs no model, no torch, and (by default) no
tokenizer. That keeps the Streamlit Community Cloud deploy tiny and fast. If
`transformers` happens to be installed, the convergence grid is additionally labeled
with the per-token text; otherwise the grid still shows the colored wavefront and the
full per-iteration text below it.

Run locally:
    streamlit run cllm/results/trajectory_viz/streamlit_app.py

Deploy: push the repo to GitHub, then point https://share.streamlit.io at this file
(see README.md → "Share it (Streamlit Community Cloud)").
"""
import glob
import html as _html
import json
import os

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Jacobi trajectories — CLLM", layout="wide", page_icon="🪜")


# --------------------------------------------------------------------------- #
#  Data discovery + loading                                                     #
# --------------------------------------------------------------------------- #
def discover_runs():
    """Map a friendly label -> path for every trajectory_data.json we can find.

    Top-level file is the latest run; archive/<name>/ holds preserved runs.
    """
    runs = {}
    top = os.path.join(APP_DIR, "trajectory_data.json")
    if os.path.exists(top):
        runs["latest (top level)"] = top
    for p in sorted(glob.glob(os.path.join(APP_DIR, "archive", "*", "trajectory_data.json"))):
        runs[os.path.basename(os.path.dirname(p))] = p
    return runs


@st.cache_data(show_spinner=False)
def load_run(path):
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_run_bytes(data: bytes):
    return json.loads(data)


# --------------------------------------------------------------------------- #
#  Optional tokenizer (labels grid cells with per-token text when available)    #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading tokenizer for per-token labels…")
def get_tokenizer(model_name):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None


def tok_cell(tok, tid):
    """One token id -> short, whitespace-visible display string (or '' w/o tokenizer)."""
    if tok is None:
        return ""
    s = tok.decode([tid])
    if s == "":
        s = tok.convert_ids_to_tokens(tid) or "∅"
    s = s.replace("\n", "⏎").replace("\t", "⇥")
    if s != "" and s.strip() == "":
        s = "·" * len(s)
    elif s.startswith(" "):
        s = "·" + s[1:]
    return s


# --------------------------------------------------------------------------- #
#  Trajectory helpers (operate on the JSON block dicts; ported 1:1)             #
# --------------------------------------------------------------------------- #
def settle_iters(block):
    """Per position, first iteration index from which it stays == fixed point."""
    states, fixed = block["states"], block["states"][-1]
    T = len(states)
    out = []
    for p in range(len(fixed)):
        i = T - 1
        while i - 1 >= 0 and states[i - 1][p] == fixed[p]:
            i -= 1
        out.append(i)
    return out


def first_divergence(pre_blocks, post_blocks, drift):
    """First block+position where PRE and POST disagree on IDENTICAL context.

    Up to (and including) min(drift) every earlier block matched, so this is a genuine
    greedy-equivalence break, not a downstream artifact. None if no drift.
    """
    if not drift or post_blocks is None:
        return None
    j = min(drift)
    pf, qf = pre_blocks[j]["states"][-1], post_blocks[j]["states"][-1]
    p = next((i for i in range(min(len(pf), len(qf))) if pf[i] != qf[i]), 0)
    return {"block": j, "pos": p, "pre_id": pf[p], "post_id": qf[p]}


def speedup(p):
    if not p.get("post"):
        return None
    return p["pre"]["stats"]["fwd_per_token"] / max(p["post"]["stats"]["fwd_per_token"], 1e-9)


def snippet(raw, k=80):
    raw = " ".join(raw.split())
    return raw[:k] + ("…" if len(raw) > k else "")


# --------------------------------------------------------------------------- #
#  Rendering: the convergence grid + the per-iteration text                     #
# --------------------------------------------------------------------------- #
GRID_CSS = """
<style>
.gridwrap{overflow-x:auto;max-width:100%;border:1px solid #e8e8e8;border-radius:6px;
  background:#fff;margin:2px 0 6px}
table.jgrid{border-collapse:collapse;margin:0;font-family:ui-monospace,Menlo,monospace;
  font-size:11px;table-layout:fixed}
table.jgrid td{border:1px solid #eee;padding:2px 4px;text-align:center;width:46px;
  max-width:46px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#222}
table.jgrid td.tag{color:#999;text-align:right;border:none;padding-right:8px;font-size:10px;
  width:60px;max-width:60px;position:sticky;left:0;background:#fafafa;z-index:1}
table.jgrid td.settled{background:#d6f3d6}
table.jgrid td.unsettled{background:#ffdede}
table.jgrid td.changed{outline:2px solid #2a7;outline-offset:-2px;font-weight:700}
.jlegend{font-size:12px;color:#555;margin:4px 0 8px;line-height:1.7}
.jlegend span{padding:2px 6px;border-radius:4px;margin-right:4px}
.itext{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap;
  background:#fff;border:1px solid #eee;border-radius:6px;padding:6px 8px;margin:3px 0;color:#333}
.itext b{color:#111}
</style>
"""

LEGEND = (
    '<div class="jlegend">'
    '<span style="background:#ffdede">pink</span> token will still change &nbsp;&nbsp;'
    '<span style="background:#d6f3d6">green</span> token settled to its final value '
    '(matches the fixed point)&nbsp;&nbsp;'
    '<span style="outline:2px solid #2a7">green outline</span> changed on this iteration.'
    '<br>The green region grows left→right — that staircase is the autoregressive '
    'convergence wavefront. Fewer rows (forward passes) = faster.</div>'
)


def _cell(tok, tid, cls):
    disp = tok_cell(tok, tid)
    short = disp if len(disp) <= 8 else disp[:7] + "…"
    return f'<td class="{cls}" title="{_html.escape(disp) or tid}">{_html.escape(short)}</td>'


def block_grid_html(tok, block):
    states, fixed = block["states"], block["states"][-1]
    rows = []
    for i, row in enumerate(states):
        tag = "init" if i == 0 else ("fixed pt" if i == len(states) - 1 else f"iter {i}")
        cells = [f'<td class="tag">{tag}</td>']
        for p, tid in enumerate(row):
            cls = "settled" if tid == fixed[p] else "unsettled"
            if i > 0 and tid != states[i - 1][p]:
                cls += " changed"
            cells.append(_cell(tok, tid, cls))
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="gridwrap"><table class="jgrid">{"".join(rows)}</table></div>'


def block_text_html(block):
    """The decoded block text at every refinement iteration (the readable story)."""
    texts = block.get("state_texts")
    if not texts:
        return ""
    lines = []
    for i, t in enumerate(texts):
        tag = "init" if i == 0 else ("fixed" if i == len(texts) - 1 else f"it{i}")
        lines.append(f"<b>{tag:>6}</b> │ {_html.escape(t)}")
    return '<div class="itext">' + "<br>".join(lines) + "</div>"


def render_side(label, side, tok):
    """One model's blocks: stats line + per-block grid and per-iteration text."""
    s = side["stats"]
    st.markdown(f"#### {label}")
    st.caption(
        f"blocks **{s['blocks']}** · forward passes **{s['total_iters']}** · "
        f"tokens **{s['tokens']}** · fwd/token **{s['fwd_per_token']:.3f}**"
    )
    for b in side["blocks"]:
        eos = " · reached EOS (generation stops)" if b.get("has_eos") else ""
        st.markdown(
            f"**Block {b['block_index']}** — converged in **{b['n_iters']}** "
            f"forward passes{eos}"
        )
        st.markdown(block_grid_html(tok, b), unsafe_allow_html=True)
        with st.expander("decoded text at each iteration"):
            st.markdown(block_text_html(b), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Pages                                                                         #
# --------------------------------------------------------------------------- #
def page_overview(d):
    P = d["prompts"]
    have_post = any(p.get("post") for p in P)
    st.subheader("Overview")

    if have_post:
        rr = [speedup(p) for p in P if p.get("post")]
        pre_m = sum(p["pre"]["stats"]["fwd_per_token"] for p in P) / len(P)
        post_m = sum(p["post"]["stats"]["fwd_per_token"] for p in P if p.get("post")) / max(len(rr), 1)
        ndrift = sum(1 for p in P if p.get("drift"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mean base fwd/token", f"{pre_m:.3f}")
        c2.metric("mean distilled fwd/token", f"{post_m:.3f}", delta=f"-{pre_m - post_m:.3f}",
                  delta_color="inverse")
        c3.metric("mean speedup", f"{sum(rr) / len(rr):.2f}×", help=f"range {min(rr):.2f}–{max(rr):.2f}×")
        c4.metric("prompts with fixed-point drift", f"{ndrift}/{len(P)}")
    else:
        pre_m = sum(p["pre"]["stats"]["fwd_per_token"] for p in P) / len(P)
        st.metric("mean base fwd/token", f"{pre_m:.3f}")

    # ---- sortable summary table ----
    rows = []
    for p in P:
        ps = p["pre"]["stats"]
        row = {"prompt": p["idx"], "blocks": ps["blocks"],
               "PRE fwd/tok": round(ps["fwd_per_token"], 3),
               "snippet": snippet(p["raw"], 70)}
        if have_post and p.get("post"):
            r = speedup(p)
            row["POST fwd/tok"] = round(p["post"]["stats"]["fwd_per_token"], 3)
            row["speedup"] = round(r, 2)
            row["drift@block"] = (min(p["drift"]) if p.get("drift") else None)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("prompt")
    st.dataframe(df, width="stretch", height=min(560, 56 + 35 * len(df)))

    st.markdown(GRID_CSS + LEGEND, unsafe_allow_html=True)

    # ---- charts ----
    st.markdown("##### Speedup varies by prompt — fwd/token, PRE vs POST")
    chart_df = pd.DataFrame({
        "prompt": [f"#{p['idx']}" for p in P],
        "PRE (base)": [p["pre"]["stats"]["fwd_per_token"] for p in P],
    })
    if have_post:
        chart_df["POST (distilled)"] = [
            p["post"]["stats"]["fwd_per_token"] if p.get("post") else None for p in P
        ]
    st.bar_chart(chart_df.set_index("prompt"))

    if have_post:
        col_a, col_b = st.columns(2)
        # mean settle-iteration per block position (where the forward passes go)
        n = len(P[0]["pre"]["blocks"][0]["states"][-1])
        pre_si = [0.0] * n; pre_c = [0] * n
        post_si = [0.0] * n; post_c = [0] * n
        for p in P:
            for b in p["pre"]["blocks"]:
                for pos, v in enumerate(settle_iters(b)):
                    pre_si[pos] += v; pre_c[pos] += 1
            for b in (p["post"]["blocks"] if p.get("post") else []):
                for pos, v in enumerate(settle_iters(b)):
                    post_si[pos] += v; post_c[pos] += 1
        pos_df = pd.DataFrame({
            "position in block": list(range(n)),
            "PRE (base)": [pre_si[i] / max(pre_c[i], 1) for i in range(n)],
            "POST (distilled)": [post_si[i] / max(post_c[i], 1) for i in range(n)],
        }).set_index("position in block")
        with col_a:
            st.markdown("##### Where forward passes go (mean settle-iteration per position)")
            st.line_chart(pos_df)

        # drift vs block depth (cumulative fraction diverged)
        maxb = max(len(p["pre"]["blocks"]) for p in P)
        firsts = [min(p["drift"]) if p.get("drift") else None for p in P]
        cum = [sum(1 for f in firsts if f is not None and f <= k) / len(P) for k in range(maxb)]
        first_at = [sum(1 for f in firsts if f == k) for k in range(maxb)]
        drift_df = pd.DataFrame({
            "block depth": list(range(maxb)),
            "first break here (# prompts)": first_at,
            "cumulative fraction diverged": cum,
        }).set_index("block depth")
        with col_b:
            st.markdown("##### Greedy-equivalence breaks compound with depth")
            st.bar_chart(drift_df[["first break here (# prompts)"]])
            st.line_chart(drift_df[["cumulative fraction diverged"]])


def page_prompt(d, p, tok):
    have_post = bool(p.get("post"))
    st.subheader(f"Prompt #{p['idx']}")

    if have_post:
        r = speedup(p)
        c1, c2, c3 = st.columns(3)
        c1.metric("base fwd/token", f"{p['pre']['stats']['fwd_per_token']:.3f}")
        c2.metric("distilled fwd/token", f"{p['post']['stats']['fwd_per_token']:.3f}")
        c3.metric("speedup", f"{r:.2f}×", help="base ÷ distilled forward passes")

    with st.expander("full prompt fed to the model", expanded=False):
        st.code(p["prompt_text"], language=None)

    # ---- first greedy-equivalence break (same-context divergence) ----
    fd = first_divergence(
        p["pre"]["blocks"], p["post"]["blocks"] if have_post else None, p.get("drift")
    )
    if fd:
        j, pos = fd["block"], fd["pos"]
        pre_tok, post_tok = tok_cell(tok, fd["pre_id"]), tok_cell(tok, fd["post_id"])
        tok_note = (f": base → `{pre_tok}` vs distilled → `{post_tok}`"
                    if tok is not None else "")
        st.warning(
            f"**First greedy-equivalence break:** block {j}, position {pos} — identical "
            f"context up to here, so this is a real divergence, not a downstream "
            f"artifact{tok_note}."
        )
        cc1, cc2 = st.columns(2)
        cc1.markdown(f"**base** block {j}:")
        cc1.markdown(f'<div class="itext">{_html.escape(p["pre"]["blocks"][j]["fixed_text"])}</div>',
                     unsafe_allow_html=True)
        cc2.markdown(f"**distilled** block {j}:")
        cc2.markdown(f'<div class="itext">{_html.escape(p["post"]["blocks"][j]["fixed_text"])}</div>',
                     unsafe_allow_html=True)
    elif have_post:
        st.success("Same fixed-point output as base (no drift on the visualized blocks).")

    st.markdown(GRID_CSS + LEGEND, unsafe_allow_html=True)

    if have_post:
        col_pre, col_post = st.columns(2)
        with col_pre:
            render_side("PRE — base model", p["pre"], tok)
        with col_post:
            render_side("POST — CLLM-distilled", p["post"], tok)
    else:
        render_side("PRE — base model", p["pre"], tok)


# --------------------------------------------------------------------------- #
#  App                                                                           #
# --------------------------------------------------------------------------- #
def main():
    st.title("🪜 Jacobi decoding trajectories — Consistency-LLM")
    st.caption(
        "Each Jacobi block guesses 16 tokens at once and refines them in parallel until "
        "the block stops changing (the fixed point == greedy autoregressive). Distillation "
        "makes that converge in fewer forward passes — the staircases below, made legible."
    )

    runs = discover_runs()
    with st.sidebar:
        st.header("Run")
        choices = list(runs.keys()) + ["upload a trajectory_data.json…"]
        pick = st.selectbox("dataset", choices, index=0 if choices else None)
        if pick == "upload a trajectory_data.json…":
            up = st.file_uploader("trajectory_data.json", type="json")
            if up is None:
                st.info("Upload a `trajectory_data.json` produced by visualize_trajectory.py.")
                st.stop()
            d = load_run_bytes(up.getvalue())
        elif not runs:
            st.error("No trajectory_data.json found next to the app or in archive/.")
            st.stop()
        else:
            d = load_run(runs[pick])

        st.divider()
        st.caption(
            f"**model** {d.get('model', '?')}  \n"
            f"**adapter** {d.get('lora_adapter', '—')}  \n"
            f"**dataset** {d.get('dataset', '?')} · n={d.get('n', '?')} · "
            f"max_blocks={d.get('max_blocks', '?')} · prompts={len(d['prompts'])}"
        )

        st.divider()
        labels = ["📊 Overview"] + [
            f"#{p['idx']}"
            + (f"  ·  {speedup(p):.2f}×" if p.get("post") else "")
            + ("  ⚠" if p.get("drift") else "")
            for p in d["prompts"]
        ]
        view = st.radio("view", labels, index=0, label_visibility="collapsed")

        st.divider()
        want_labels = st.toggle(
            "per-token cell labels", value=False,
            help="Loads the tokenizer (needs `transformers`) to print the token text in "
                 "each grid cell. Off = colored grid only (lighter, no extra deps).",
        )

    tok = get_tokenizer(d.get("model", "")) if want_labels else None
    if want_labels and tok is None:
        st.sidebar.warning("Couldn't load a tokenizer (is `transformers` installed?). "
                           "Showing the colored grid without per-token labels.")

    if view == "📊 Overview":
        page_overview(d)
    else:
        idx = int(labels.index(view)) - 1
        page_prompt(d, d["prompts"][idx], tok)


if __name__ == "__main__":
    main()
