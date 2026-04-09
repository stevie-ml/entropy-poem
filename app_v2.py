import streamlit as st
import torch
import math
import pandas as pd
import os
import json
import anthropic
import plotly.graph_objects as go
from transformers import GPT2LMHeadModel, GPT2Tokenizer

claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

st.set_page_config(page_title="Entropic Writing Workshop", layout="wide", page_icon="favicon.png")

# ── Typography & style ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'EB Garamond', Georgia, serif;
    color: #1e1b3a;
    background: #fdf8f0;
}

/* Title */
h1 { font-size: 2rem; font-weight: 400; letter-spacing: -0.02em; margin-bottom: 0; color: #1e1b3a; }
h2, h3 { font-weight: 400; letter-spacing: -0.01em; color: #1e1b3a; }

/* Subheader */
.stSubheader { font-size: 1.1rem; font-weight: 400; color: #5a5580; border-bottom: 1px solid #d4cfe8; padding-bottom: 0.4rem; margin-bottom: 1rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #d4cfe8; gap: 0; background: transparent; }
.stTabs [data-baseweb="tab"] {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1rem;
    color: #7c7a9e;
    padding: 0.4rem 1.2rem;
    border-bottom: 2px solid transparent;
    background: none;
}
.stTabs [aria-selected="true"] {
    color: #1e1b3a;
    font-weight: 500;
}
/* Override Streamlit's default red/orange tab indicator with purple */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #7c5cbf !important;
}

/* Buttons */
.stButton > button {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 0.95rem;
    background: #7c5cbf;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 0.4rem 1.6rem;
    letter-spacing: 0.02em;
    transition: background 0.15s;
}
.stButton > button:hover { background: #6244a8; color: #fff; }

/* Inputs */
.stTextArea textarea, .stTextInput input {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1rem;
    border: 1px solid #d4cfe8;
    border-radius: 4px;
    background: #faf6ef;
    color: #1e1b3a;
}

/* Radio */
.stRadio label { font-family: 'EB Garamond', Georgia, serif; font-size: 0.95rem; color: #1e1b3a; }
[data-testid="stRadio"] [role="radio"][aria-checked="true"] { background-color: #7c5cbf !important; border-color: #7c5cbf !important; }
[data-testid="stRadio"] [role="radio"] { border-color: #c4bfe0 !important; }
[data-testid="stRadio"] [role="radio"]:focus { box-shadow: 0 0 0 2px rgba(124,92,191,0.3) !important; }

/* Metrics — strip Streamlit's default styling */
[data-testid="metric-container"] {
    background: #e8e5f5;
    border: 1px solid #d4cfe8;
    border-radius: 6px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #7c7a9e; text-transform: uppercase; letter-spacing: 0.06em; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 400; color: #1e1b3a; font-family: 'EB Garamond', Georgia, serif; }

/* Expander */
.streamlit-expanderHeader { font-family: 'EB Garamond', Georgia, serif; font-size: 0.9rem; color: #5a5580; }

/* Captions */
.stCaption { font-size: 0.8rem; color: #7c7a9e; font-family: 'JetBrains Mono', monospace; }

/* Code blocks */
.stCodeBlock { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Reduce top padding */
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
header[data-testid="stHeader"] { height: 0; min-height: 0; }
h1 a.anchor-link { display: none; }

/* Horizontal rule */
hr { border: none; border-top: 1px solid #d4cfe8; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
h1 a.anchor-link { display: none; }
.title-wrap { display: inline-flex; align-items: center; gap: 0.4rem; }
.title-wrap h1 { margin: 0; }
.copy-btn {
    opacity: 0; font-size: 0.9rem; cursor: pointer;
    color: #9b8ec4; transition: opacity 0.15s;
    background: none; border: none; padding: 0; line-height: 1;
}
.title-wrap:hover .copy-btn { opacity: 1; }
.copy-btn:hover { color: #7c5cbf; }
</style>
<div class="title-wrap">
  <h1>Entropic Writing Workshop</h1>
  <button class="copy-btn" title="Copy link" onclick="navigator.clipboard.writeText('https://s2lab.stevie.art/').then(() => { this.textContent='✓'; setTimeout(() => this.textContent='⇗', 1000); })">⇗</button>
</div>
""", unsafe_allow_html=True)
st.caption("token-level surprisal, entropy and S₂ using DistilGPT-2 and Claude")

# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_resources():
    import nltk
    from nltk.corpus import words as nltk_words
    nltk.download("words", quiet=True)
    word_list = set(w.lower() for w in nltk_words.words())
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()
    return model, tokenizer, word_list

# ── Core analysis ──────────────────────────────────────────────────────────────

def analyze_text(text):
    model, tokenizer, _ = load_resources()
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids
    if input_ids.shape[1] < 2:
        return []
    with torch.no_grad():
        logits = model(input_ids).logits
    results = []
    for i in range(1, input_ids.shape[1]):
        probs = torch.softmax(logits[0, i - 1], dim=-1)
        token_id = input_ids[0, i].item()
        p = probs[token_id].item()
        surprisal = -math.log2(p) if p > 1e-12 else 50.0
        log_probs = torch.log2(probs.clamp(min=1e-12))
        entropy = -(probs * log_probs).sum().item()
        s2 = surprisal - entropy
        results.append({
            "token": tokenizer.decode([token_id]),
            "surprisal": round(surprisal, 3),
            "entropy": round(entropy, 3),
            "s2": round(s2, 3),
            "prob_%": round(p * 100, 5),
        })
    return results

def render_colored_tokens(tokens, metric):
    vals = [t[metric] for t in tokens]
    vmin, vmax = min(vals), max(vals)
    rng = vmax - vmin + 1e-8

    # Color: low = lavender purple, high = amber orange. Opacity scales with value.
    LO = (124, 92, 191)   # lavender purple
    HI  = (232, 126, 18)  # amber orange

    html = (
        "<div style='font-size:1.15rem;line-height:3.2;font-family:\"JetBrains Mono\",monospace;"
        "padding:1.2rem 0.5rem;margin:1rem 0'>"
    )
    for t in tokens:
        norm = (t[metric] - vmin) / rng
        r = int(HI[0] * norm + LO[0] * (1 - norm))
        g = int(HI[1] * norm + LO[1] * (1 - norm))
        b = int(HI[2] * norm + LO[2] * (1 - norm))
        alpha = 0.08 + 0.78 * norm
        text_color = "#1e1b3a"  # dark navy always readable on purple→orange
        tok = t["token"].replace("<", "&lt;").replace(">", "&gt;")
        html += (
            f"<span style='background:rgba({r},{g},{b},{alpha:.2f});"
            f"color:{text_color};border-radius:3px;"
            f"padding:2px 4px;margin:0 1px'"
            f" title='{metric}: {t[metric]:.3f}'>{tok}</span>"
        )

    # Gradient key
    lo_label = f"{vmin:.1f} bits"
    hi_label = f"{vmax:.1f} bits"
    html += (
        "<div style='margin-top:1.4rem;display:flex;align-items:center;gap:0.7rem;"
        "font-size:0.72rem;font-family:\"JetBrains Mono\",monospace;color:#7c7a9e'>"
        f"<span>{lo_label}</span>"
        "<div style='width:140px;height:8px;border-radius:4px;"
        "background:linear-gradient(to right,rgba(124,92,191,0.12),rgba(232,126,18,0.85))'></div>"
        f"<span>{hi_label}</span>"
        f"<span style='margin-left:1rem;color:#aaa'>↑ {metric}</span>"
        "</div>"
    )
    html += "</div>"
    return html

def metric_chart(tokens):
    labels = [t["token"].strip() or "·" for t in tokens]
    x = list(range(len(tokens)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=[t["surprisal"] for t in tokens], name="Surprisal",
        line=dict(color="#b44a7c", width=1.5), mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=[t["entropy"] for t in tokens], name="Entropy",
        line=dict(color="#7c5cbf", width=1.5, dash="dash"), mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=[t["s2"] for t in tokens], name="S₂",
        line=dict(color="#9b8ec4", width=1, dash="dot"), mode="lines"
    ))
    fig.update_layout(
        xaxis=dict(
            tickmode="array", tickvals=x, ticktext=labels,
            tickangle=45, tickfont=dict(size=10, family="JetBrains Mono"),
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title="bits", tickfont=dict(size=10, family="JetBrains Mono"),
            gridcolor="#d4cfe8", zeroline=True, zerolinecolor="#c4bfe0", zerolinewidth=1,
        ),
        height=280,
        margin=dict(t=20, b=90, l=45, r=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(size=11, family="EB Garamond, Georgia, serif"),
        ),
        plot_bgcolor="#fdf8f0",
        paper_bgcolor="#fdf8f0",
        font=dict(color="#1e1b3a", family="EB Garamond, Georgia, serif"),
    )
    return fig

# ── Next token ─────────────────────────────────────────────────────────────────

def get_next_token_candidates(context):
    model, tokenizer, word_list = load_resources()
    inputs = tokenizer(context, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(inputs.input_ids).logits[0, -1], dim=-1).cpu().numpy()
    results = []
    for idx in range(len(probs)):
        tok = tokenizer.decode([idx])
        word = tok.strip().lower()
        if tok.startswith(" ") and word.isalpha() and len(word) > 2 and word in word_list:
            p = probs[idx]
            if p > 1e-12:
                results.append({"word": word, "surprisal": round(-math.log2(p), 2), "prob_%": round(p * 100, 5)})
    results.sort(key=lambda x: x["surprisal"])
    return results

# ── GPT-2 oracle ───────────────────────────────────────────────────────────────

GPT2_TOOL = {
    "name": "get_token_surprisals",
    "description": (
        "Given a text context and a list of candidate words, returns GPT-2 surprisal (bits) "
        "for each word as the next token. Higher surprisal = more unexpected. "
        "Use this to compare candidates and pick the most surprising grammatically valid one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "context": {"type": "string"},
            "words": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["context", "words"],
    },
}

def score_words_gpt2(context, words):
    model, tokenizer, _ = load_resources()
    inputs = tokenizer(context, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(inputs.input_ids).logits[0, -1], dim=-1).cpu()
    results = []
    for word in words:
        scored = False
        for prefix in [" " + word, word]:
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            if len(ids) == 1:
                p = probs[ids[0]].item()
                results.append({"word": word, "surprisal_bits": round(-math.log2(p), 3) if p > 1e-12 else 50.0})
                scored = True
                break
        if not scored:
            results.append({"word": word, "surprisal_bits": None, "note": "multi-token"})
    return results

def claude_generate(system, prompt, status_fn=None):
    oracle_system = system + (
        "\n\nIMPORTANT: At every word position use get_token_surprisals. "
        "Propose 6-8 diverse candidates, get their surprisal scores, then pick the "
        "highest-surprisal option that is still grammatically correct."
    )
    messages = [{"role": "user", "content": prompt}]
    tool_calls = 0
    while True:
        resp = claude.messages.create(
            model="claude-opus-4-6",
            max_tokens=8000,
            system=oracle_system,
            tools=[GPT2_TOOL],
            messages=messages,
        )
        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use" and block.name == "get_token_surprisals":
                    tool_calls += 1
                    result = score_words_gpt2(block.input["context"], block.input["words"])
                    if status_fn:
                        status_fn(f"call {tool_calls}: scored {len(result)} candidates · …{block.input['context'][-35:]!r}")
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": tool_results})
        elif resp.stop_reason == "end_turn":
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text.strip(), tool_calls
            return "", tool_calls

# ── UI ─────────────────────────────────────────────────────────────────────────

DEFAULT_TEXT = "Let be be finale of seem.\nThe only emperor is the emperor of ice-cream."

if "shared_text" not in st.session_state:
    st.session_state.shared_text = DEFAULT_TEXT

shared_text = st.text_area("Text", height=200, label_visibility="collapsed", key="shared_text")

tab1, tab2, tab3 = st.tabs(["Analyze", "Next token", "Generate"])

EXPLAINER = """
**Surprisal** = −log₂ P(token | context)

How unexpected was this specific token? DeDeo: *"the number of questions to get to choice x is equal to −log₂ P(x) — the less likely possibilities are buried deeper in the question tree."* High surprisal = the model had to dig deep. ([DeDeo, *Information Theory for Intelligent People*, 2017](http://santafe.edu/~simon/it.pdf))

---

**Entropy** = −Σ P(w) log₂ P(w) over all possible next tokens

How uncertain was the model *before* seeing the token? DeDeo: *"H(X) goes by a number of different names: 'uncertainty', 'information', even 'entropy'... how uncertain we are about the outcome, how much information is in the process."* High entropy = many continuations were plausible. ([DeDeo, 2017](http://santafe.edu/~simon/it.pdf))

---

**S₂** = surprisal − entropy

Was this token more surprising than the situation warranted? Positive S₂: harder to predict than the general uncertainty would suggest. Negative S₂: easier than expected. Follows from the definitions of self-information and entropy in Cover & Thomas, [*Elements of Information Theory*](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959) (2006).
"""

# ── Tab 1: Analyze ─────────────────────────────────────────────────────────────
with tab1:
    metric_sel = st.radio("Color by", ["surprisal", "entropy", "s2"], horizontal=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing…"):
            tokens = analyze_text(shared_text)
        if tokens:
            st.markdown(render_colored_tokens(tokens, metric_sel), unsafe_allow_html=True)
            st.plotly_chart(metric_chart(tokens), use_container_width=True)
            df = pd.DataFrame(tokens)
            c1, c2, c3 = st.columns(3)
            c1.metric("mean surprisal", f"{df['surprisal'].mean():.2f} bits")
            c2.metric("mean entropy", f"{df['entropy'].mean():.2f} bits")
            c3.metric("mean S₂", f"{df['s2'].mean():.2f} bits")
            with st.expander("Raw data"):
                st.dataframe(df, use_container_width=True)

    with st.expander("What are these metrics?"):
        st.markdown(EXPLAINER)

# ── Tab 2: Next token ──────────────────────────────────────────────────────────
with tab2:
    if st.button("Score distribution"):
        with st.spinner("Scoring…"):
            all_candidates = get_next_token_candidates(shared_text)
        df_all = pd.DataFrame(all_candidates)
        st.caption(f"{len(df_all)} dictionary words scored · sorted by surprisal")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**most predictable**")
            st.dataframe(df_all.head(20), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**most surprising**")
            st.dataframe(df_all.tail(20).iloc[::-1], use_container_width=True, hide_index=True)
        with st.expander("Full distribution"):
            st.dataframe(df_all, use_container_width=True, hide_index=True)

# ── Tab 3: Generate ────────────────────────────────────────────────────────────
with tab3:
    sys_prompt = st.text_area(
        "System prompt",
        "Write a 5-line poem, exactly 11 syllables per line, with one metaphor (no like/as).",
        height=80,
    )
    user_prompt = st.text_area("Prompt", height=80)
    notes = st.text_input("Notes", "")

    if st.button("Generate"):
        full_prompt = user_prompt + (f"\n{notes}" if notes else "")
        status_box = st.empty()
        log_lines = []
        def upd(msg):
            log_lines.append(msg)
            status_box.caption("\n".join(log_lines[-4:]))
        with st.spinner(""):
            result, n_calls = claude_generate(sys_prompt, full_prompt, status_fn=upd)
        status_box.empty()
        st.caption(f"{n_calls} GPT-2 oracle calls")
        st.code(result)

        if result:
            st.markdown("---")
            with st.spinner(""):
                tokens = analyze_text(result)
            if tokens:
                st.markdown(render_colored_tokens(tokens, "surprisal"), unsafe_allow_html=True)
                st.plotly_chart(metric_chart(tokens), use_container_width=True)
                df = pd.DataFrame(tokens)
                c1, c2, c3 = st.columns(3)
                c1.metric("mean surprisal", f"{df['surprisal'].mean():.2f} bits")
                c2.metric("mean entropy", f"{df['entropy'].mean():.2f} bits")
                c3.metric("mean S₂", f"{df['s2'].mean():.2f} bits")
