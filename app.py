"""
SIFT // TRIAGE CONSOLE
Neon SOC edition. Same model, same triage brain, considerably more voltage.

    streamlit run app.py
"""

import json
import time
from pathlib import Path

import skops.io as sio

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sift_core import (FEATURES, FEATURE_LABELS, evidence_for,
                       policy_metrics, triage_lane, triage_lanes)
import sift_agent

# ---------------------------------------------------------------- palette
BG = "#070B14"
CARD = "#0E1626"
CYAN = "#00F0FF"
MAGENTA = "#FF2BD6"
GREEN = "#00FF9C"
AMBER = "#FFC24B"
RED = "#FF3B5C"
PURPLE = "#8B5CF6"
MUT = "#7C8CA8"

LANE_STYLE = {
    "ALLOW": (GREEN, "CLEARED", "Package proceeds. No elevated risk signals."),
    "REVIEW": (AMBER, "ESCALATED", "Routed to a human analyst with evidence attached."),
    "BLOCK": (RED, "NEUTRALIZED", "Install halted. Verdict and evidence logged for audit."),
}

st.set_page_config(page_title="SIFT // Triage Console", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- css
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

/* animated aurora background */
[data-testid="stAppViewContainer"] {{
    background:
      radial-gradient(1200px 600px at 85% -10%, {PURPLE}22, transparent 60%),
      radial-gradient(900px 500px at -10% 110%, {CYAN}1e, transparent 60%),
      radial-gradient(700px 400px at 50% 120%, {MAGENTA}14, transparent 60%),
      {BG};
    background-attachment: fixed;
}}
/* scanline overlay */
[data-testid="stAppViewContainer"]::after {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 9999;
    background: repeating-linear-gradient(0deg, transparent 0 2px, rgba(255,255,255,0.012) 2px 4px);
    mix-blend-mode: overlay;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0A1120 0%, #0B0F1E 100%);
    border-right: 1px solid {CYAN}33;
}}
h1, h2, h3 {{ font-family: 'Orbitron', sans-serif !important; letter-spacing: 1px; }}

/* glitch title */
.glitch {{
    font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 3rem;
    color: #EAF6FF; position: relative; letter-spacing: 6px; margin: 0;
    text-shadow: 0 0 18px {CYAN}88;
    animation: flicker 4s infinite;
}}
.glitch::before, .glitch::after {{
    content: attr(data-text); position: absolute; left: 0; top: 0; width: 100%;
}}
.glitch::before {{ color: {MAGENTA}; clip-path: inset(0 0 55% 0); animation: gl1 3.1s infinite linear alternate-reverse; }}
.glitch::after  {{ color: {CYAN};    clip-path: inset(55% 0 0 0); animation: gl2 2.3s infinite linear alternate-reverse; }}
@keyframes gl1 {{ 0%,92% {{transform:none; opacity:.25}} 94% {{transform:translate(-3px,-2px)}} 98% {{transform:translate(3px,1px)}} 100% {{transform:none}} }}
@keyframes gl2 {{ 0%,90% {{transform:none; opacity:.25}} 93% {{transform:translate(3px,2px)}} 97% {{transform:translate(-3px,-1px)}} 100% {{transform:none}} }}
@keyframes flicker {{ 0%,97%,100% {{opacity:1}} 98% {{opacity:.75}} 99% {{opacity:.92}} }}

.subtitle {{
    font-family: 'Share Tech Mono', monospace; color: {MUT};
    letter-spacing: 3px; margin-top: 2px;
}}
.subtitle b {{ color: {CYAN}; }}

/* neon cards for metrics */
div[data-testid="stMetric"] {{
    background: linear-gradient(160deg, {CARD} 0%, #101B31 100%);
    border: 1px solid {CYAN}2e; border-radius: 14px; padding: 14px 18px;
    box-shadow: 0 0 22px {CYAN}12, inset 0 0 30px #00000055;
}}
div[data-testid="stMetric"] label {{ font-family: 'Share Tech Mono', monospace; color: {MUT} !important; }}
div[data-testid="stMetricValue"] {{ font-family: 'Orbitron', sans-serif; text-shadow: 0 0 14px {CYAN}66; }}

/* tabs */
button[data-baseweb="tab"] {{
    font-family: 'Share Tech Mono', monospace !important; letter-spacing: 1px;
    background: transparent !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {CYAN} !important; text-shadow: 0 0 12px {CYAN};
}}

/* verdict banner */
.verdict {{
    border-radius: 16px; padding: 26px 30px; margin: 6px 0 14px 0;
    border: 1px solid; position: relative; overflow: hidden;
    animation: vpulse 2.2s ease-in-out infinite;
}}
.verdict h2 {{ margin: 0; letter-spacing: 6px; font-size: 2.1rem; }}
.verdict .stamp {{ font-family:'Share Tech Mono',monospace; letter-spacing:4px; font-size:.8rem; opacity:.85; }}
.verdict p {{ margin: 6px 0 0 0; color: {MUT}; font-family: 'Share Tech Mono', monospace; }}
@keyframes vpulse {{ 0%,100% {{ box-shadow: 0 0 18px var(--glow); }} 50% {{ box-shadow: 0 0 44px var(--glow); }} }}

/* terminal */
.term {{
    background: #050910; border: 1px solid {GREEN}33; border-radius: 12px;
    font-family: 'Share Tech Mono', monospace; font-size: .85rem;
    padding: 16px 18px; color: {GREEN}; line-height: 1.65;
    box-shadow: inset 0 0 24px #000, 0 0 18px {GREEN}11;
    min-height: 172px;
}}
.term .dim {{ color: {MUT}; }} .term .hot {{ color: {MAGENTA}; }} .term .warn {{ color: {AMBER}; }}

/* scan button */
div.stButton > button {{
    font-family: 'Orbitron', sans-serif; letter-spacing: 3px; font-weight: 700;
    background: linear-gradient(90deg, {CYAN}22, {MAGENTA}22);
    border: 1px solid {CYAN}; color: {CYAN}; border-radius: 12px;
    padding: .6rem 1.4rem; text-shadow: 0 0 10px {CYAN};
    box-shadow: 0 0 18px {CYAN}33;
    transition: all .15s ease;
}}
div.stButton > button:hover {{
    border-color: {MAGENTA}; color: {MAGENTA}; text-shadow: 0 0 12px {MAGENTA};
    box-shadow: 0 0 26px {MAGENTA}55; transform: translateY(-1px);
}}
.smallcap {{ font-family:'Share Tech Mono',monospace; color:{MUT}; letter-spacing:2px; font-size:.75rem; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- artifacts
ART = Path(__file__).parent / "artifacts"


@st.cache_resource
def load_artifacts():
    # model.skops is deliberately gitignored (never ship a serialized model in
    # git). Regenerate it deterministically on first run if absent.
    if not (ART / "model.skops").exists():
        import train
        train.main()
    # skops load: inspect the file's required types and only trust the sklearn
    # / numpy machinery the model legitimately needs. Anything unexpected in the
    # file raises instead of silently executing (unlike pickle.load).
    untrusted = sio.get_untrusted_types(file=ART / "model.skops")
    unexpected = [t for t in untrusted
                  if not t.startswith(("sklearn.", "numpy.", "scipy."))]
    if unexpected:
        raise ValueError(f"Refusing to load model, unexpected types: {unexpected}")
    model = sio.load(ART / "model.skops", trusted=untrusted)
    test = pd.read_csv(ART / "test_scores.csv")
    importance = pd.read_csv(ART / "feature_importance.csv")
    benign_p90 = json.loads((ART / "benign_p90.json").read_text())
    metrics = json.loads((ART / "metrics.json").read_text())
    benign_med = test[test.label == 0][FEATURES].median().to_dict()
    return model, test, importance, benign_p90, metrics, benign_med


try:
    model, TEST, IMPORTANCE, BENIGN_P90, METRICS, BENIGN_MED = load_artifacts()
except FileNotFoundError:
    st.error("Artifacts not found. Run `python train.py` first (or use the Docker image).")
    st.stop()

IMP_ORDER = IMPORTANCE["feature"].tolist()

# ---------------------------------------------------------------- presets
PRESETS = {
    "fast-jsonn 1.0.2 :: typosquat, writes agent configs": dict(
        name_dist_top1k=1, has_install_script=1, install_entropy=0.81,
        obfuscation_score=0.72, maintainer_age_days=12, maintainer_pkg_count=3,
        log_weekly_downloads=2.1, dependency_count=4, readme_length=310,
        repo_linked=0, version_jump_anomaly=0, new_comaintainer_30d=0,
        net_calls_in_install=1, base64_blob_count=4, touches_agent_config=1,
        touches_credential_paths=1, publish_burst_7d=6),
    "lodash-extras 4.2.0 :: healthy utility": dict(
        name_dist_top1k=9, has_install_script=0, install_entropy=0.40,
        obfuscation_score=0.08, maintainer_age_days=2400, maintainer_pkg_count=14,
        log_weekly_downloads=8.9, dependency_count=6, readme_length=5200,
        repo_linked=1, version_jump_anomaly=0, new_comaintainer_30d=0,
        net_calls_in_install=0, base64_blob_count=0, touches_agent_config=0,
        touches_credential_paths=0, publish_burst_7d=0),
    "popular-lib 9.0.0 :: anomalous release of trusted package": dict(
        name_dist_top1k=11, has_install_script=1, install_entropy=0.58,
        obfuscation_score=0.34, maintainer_age_days=1900, maintainer_pkg_count=22,
        log_weekly_downloads=8.1, dependency_count=12, readme_length=7800,
        repo_linked=1, version_jump_anomaly=1, new_comaintainer_30d=1,
        net_calls_in_install=0, base64_blob_count=1, touches_agent_config=0,
        touches_credential_paths=0, publish_burst_7d=1),
}
BINARY = ["has_install_script", "repo_linked", "version_jump_anomaly",
          "new_comaintainer_30d", "net_calls_in_install",
          "touches_agent_config", "touches_credential_paths"]

# ---------------------------------------------------------------- header
st.markdown('<p class="glitch" data-text="SIFT // TRIAGE CONSOLE">SIFT // TRIAGE CONSOLE</p>',
            unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">SUPPLY-CHAIN INTELLIGENCE + FORENSIC TRIAGE '
            f'&nbsp;·&nbsp; <b>MODEL ONLINE</b> &nbsp;·&nbsp; ACC {METRICS["accuracy"]:.3f} '
            f'&nbsp;·&nbsp; AUC {METRICS["roc_auc"]:.2f}</p>', unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"<h2 style='color:{CYAN}; text-shadow:0 0 12px {CYAN};'>🛰️ SIFT</h2>",
                unsafe_allow_html=True)
    st.markdown('<p class="smallcap">POLICY DIALS · LIVE</p>', unsafe_allow_html=True)
    lo = st.slider("Auto-allow below", 0.0, 0.5, 0.30, 0.05)
    hi = st.slider("Auto-block at or above", 0.5, 1.0, 0.70, 0.05)
    st.markdown(f'<p class="smallcap">UNCERTAINTY BAND {lo:.2f} → {hi:.2f} = HUMAN REVIEW</p>',
                unsafe_allow_html=True)
    st.divider()
    st.markdown('<p class="smallcap">TUNED GRADIENT BOOSTING · CORPUS MODELED ON '
                "BACKSTABBER'S KNIFE COLLECTION + OPENSSF · CAPSTONE · STUART WUBBENA · 2026</p>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------- helpers


def dark_fig(fig, h=340):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE9FF", family="Share Tech Mono"),
        margin=dict(l=10, r=10, t=42, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#152238")
    fig.update_yaxes(gridcolor="#152238")
    return fig


def threat_gauge(p, lane):
    color = LANE_STYLE[lane][0]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"family": "Orbitron", "size": 44, "color": color}},
        title={"text": "THREAT PROBABILITY", "font": {"family": "Share Tech Mono", "size": 13, "color": MUT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUT},
            "bar": {"color": color, "thickness": 0.26},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, lo * 100], "color": "rgba(0,255,156,0.10)"},
                {"range": [lo * 100, hi * 100], "color": "rgba(255,194,75,0.10)"},
                {"range": [hi * 100, 100], "color": "rgba(255,59,92,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": p * 100},
        }))
    return dark_fig(fig, 260)


RADAR_FEATS = ["install_entropy", "obfuscation_score", "base64_blob_count",
               "publish_burst_7d", "net_calls_in_install",
               "touches_agent_config", "touches_credential_paths", "name_dist_top1k"]
RADAR_MAX = {"install_entropy": 1, "obfuscation_score": 1, "base64_blob_count": 6,
             "publish_burst_7d": 8, "net_calls_in_install": 1,
             "touches_agent_config": 1, "touches_credential_paths": 1, "name_dist_top1k": 15}


def radar(vals):
    def norm(src):
        out = []
        for f in RADAR_FEATS:
            v = src.get(f, 0) / RADAR_MAX[f]
            if f == "name_dist_top1k":
                v = 1 - v  # closer to a popular name = hotter
            out.append(min(max(v, 0), 1))
        return out
    theta = [FEATURE_LABELS[f].replace(" (days)", "") for f in RADAR_FEATS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm(BENIGN_MED) + [norm(BENIGN_MED)[0]],
                                  theta=theta + [theta[0]], name="Benign median",
                                  line=dict(color=MUT, width=1.5), fill="toself",
                                  fillcolor="rgba(124,140,168,0.10)"))
    fig.add_trace(go.Scatterpolar(r=norm(vals) + [norm(vals)[0]],
                                  theta=theta + [theta[0]], name="This package",
                                  line=dict(color=MAGENTA, width=2.5), fill="toself",
                                  fillcolor="rgba(255,43,214,0.14)"))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#152238"),
        angularaxis=dict(gridcolor="#152238", tickfont=dict(size=9))))
    fig.update_layout(title="ATTACK-SURFACE PROFILE")
    return dark_fig(fig, 330)


def verdict_banner(lane, p):
    color, stamp, desc = LANE_STYLE[lane]
    st.markdown(
        f"""<div class="verdict" style="--glow:{color}44; border-color:{color}66;
             background:linear-gradient(120deg, {color}12, transparent 60%);">
        <span class="stamp" style="color:{color};">VERDICT :: {stamp}</span>
        <h2 style="color:{color}; text-shadow:0 0 22px {color};">{lane}</h2>
        <p>P(malicious) = <b style="color:#EAF6FF;">{p:.4f}</b> · {desc}</p>
        </div>""", unsafe_allow_html=True)


SCAN_LINES = [
    ('<span class="dim">[ingest]</span>  pulling registry metadata + tarball manifest ...', 0.25),
    ('<span class="dim">[sandbox]</span> detonating install in isolated container ...', 0.35),
    ('<span class="warn">[trace]</span>   syscall capture: fs writes, network egress, env reads ...', 0.35),
    ('<span class="dim">[feature]</span> vectorizing 17 behavioral + reputation signals ...', 0.25),
    ('<span class="hot">[model]</span>   tuned gradient boosting :: inference ...', 0.3),
]

# ---------------------------------------------------------------- tabs
tab_ask, tab_assess, tab_queue, tab_policy, tab_fleet = st.tabs(
    ["💬 ASK SIFT", "⚡ SCAN", "🗂 REVIEW QUEUE", "🎚 POLICY LAB", "📡 FLEET RADAR"])

# ================================================================ ASK SIFT
# Pull an API key from Streamlit secrets into the env the SDK reads, if present.
# Never hardcode a key; secrets.toml is gitignored.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        import os as _os
        _os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass  # no secrets.toml is fine; key can come from the environment instead

with tab_ask:
    st.markdown('<p class="smallcap">NATURAL-LANGUAGE INTAKE :: DESCRIBE A PACKAGE, '
                'GET A TRIAGED VERDICT</p>', unsafe_allow_html=True)

    if not sift_agent.has_api_key():
        st.markdown(f"""<div class="term" style="border-color:{AMBER}44; color:{AMBER};">
        <span class="warn">[offline]</span> language layer needs an Anthropic API key.<br><br>
        <span class="dim">local:  export ANTHROPIC_API_KEY=sk-ant-...<br>
        cloud:  add ANTHROPIC_API_KEY in the Streamlit app's Secrets<br><br>
        the SCAN, QUEUE, POLICY and FLEET tabs work fully without it.</span>
        </div>""", unsafe_allow_html=True)
        st.info("Design note: the language model only translates your words into "
                "the 17 features and explains the result. The ALLOW / REVIEW / BLOCK "
                "verdict is always computed by the scikit-learn model, so no phrasing "
                "or injection attempt can change a score.")
    else:
        st.caption("Try: \"a brand new npm package with 3 downloads that runs a "
                   "postinstall script, makes network calls, and writes to mcp.json\" "
                   "or \"is lodash safe\" or \"how does the review lane work\"")
        q = st.text_input("Ask SIFT", placeholder="describe a package in plain English...",
                          label_visibility="collapsed")
        go_ask = st.button("💬 SEND TO SIFT", width="stretch")

        if go_ask and q.strip():
            with st.spinner("SIFT is reasoning + scoring..."):
                try:
                    res = sift_agent.ask(
                        q.strip(), model=model, benign_med=BENIGN_MED,
                        benign_p90=BENIGN_P90, imp_order=IMP_ORDER, lo=lo, hi=hi)
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
                    res = None

            if res:
                a = res.get("assessment")
                if a:
                    verdict_banner(a["lane"], a["p_malicious"])
                    c1, c2 = st.columns([1, 1.1])
                    with c1:
                        st.plotly_chart(threat_gauge(a["p_malicious"], a["lane"]),
                                        width="stretch")
                    with c2:
                        st.markdown('<p class="smallcap">SIFT ANALYST NOTE</p>',
                                    unsafe_allow_html=True)
                        st.write(res["answer"])
                        if a["evidence"]:
                            st.markdown('<p class="smallcap">EVIDENCE</p>',
                                        unsafe_allow_html=True)
                            st.write(" · ".join(a["evidence"]))
                    with st.expander("Features the model inferred from your description"):
                        st.json(res["tool_input"] or {})
                else:
                    st.markdown('<p class="smallcap">SIFT</p>', unsafe_allow_html=True)
                    st.write(res["answer"])

# ================================================================ SCAN
with tab_assess:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown('<p class="smallcap">TARGET ACQUISITION</p>', unsafe_allow_html=True)
        preset_name = st.selectbox("Package under assessment", list(PRESETS.keys()))
        vals = dict(PRESETS[preset_name])

        with st.expander("⚙ Override feature evidence", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                vals["name_dist_top1k"] = st.slider("Name distance to top-1k", 0, 15, int(vals["name_dist_top1k"]))
                vals["install_entropy"] = st.slider("Install-code entropy", 0.0, 1.0, float(vals["install_entropy"]), 0.01)
                vals["obfuscation_score"] = st.slider("Obfuscation ratio", 0.0, 1.0, float(vals["obfuscation_score"]), 0.01)
                vals["maintainer_age_days"] = st.slider("Account age (days)", 0, 4000, int(vals["maintainer_age_days"]))
                vals["log_weekly_downloads"] = st.slider("Weekly downloads (log)", 0.0, 14.0, float(vals["log_weekly_downloads"]), 0.1)
                vals["base64_blob_count"] = st.slider("Base64 blobs", 0, 10, int(vals["base64_blob_count"]))
                vals["publish_burst_7d"] = st.slider("Versions last 7 days", 0, 12, int(vals["publish_burst_7d"]))
            with c2:
                for b in BINARY:
                    vals[b] = int(st.toggle(FEATURE_LABELS[b], bool(vals[b])))
                vals["maintainer_pkg_count"] = st.number_input("Maintainer package count", 0, 500, int(vals["maintainer_pkg_count"]))
                vals["dependency_count"] = st.number_input("Dependency count", 0, 200, int(vals["dependency_count"]))
                vals["readme_length"] = st.number_input("README length", 0, 50000, int(vals["readme_length"]))

        run = st.button("⚡ INITIATE SCAN", width="stretch")

        term = st.empty()
        if run:
            shown = []
            for line, delay in SCAN_LINES:
                shown.append(line)
                term.markdown('<div class="term">' + "<br>".join(shown) +
                              '<br><span class="dim">█</span></div>', unsafe_allow_html=True)
                time.sleep(delay)
            st.session_state["scan"] = {"vals": vals, "name": preset_name}
        if "scan" in st.session_state:
            done = [l for l, _ in SCAN_LINES] + [f'<span style="color:{CYAN}">[triage]</span>  verdict rendered. evidence attached.']
            term.markdown('<div class="term">' + "<br>".join(done) + '</div>', unsafe_allow_html=True)
        else:
            term.markdown('<div class="term"><span class="dim">console idle. acquire a target '
                          'and initiate scan.</span><br><span class="dim">█</span></div>',
                          unsafe_allow_html=True)

    with right:
        if "scan" in st.session_state:
            s_vals = st.session_state["scan"]["vals"]
            row = pd.DataFrame([s_vals])[FEATURES]
            p = float(model.predict_proba(row)[0, 1])
            lane = triage_lane(p, lo, hi)
            verdict_banner(lane, p)
            if lane == "BLOCK":
                st.toast("Threat neutralized. Install halted.", icon="🚫")
            elif lane == "REVIEW":
                st.toast("Escalated to analyst queue.", icon="🕵️")
            else:
                st.toast("Package cleared.", icon="✅")

            g, r = st.columns(2)
            with g:
                st.plotly_chart(threat_gauge(p, lane), width="stretch")
            with r:
                st.plotly_chart(radar(s_vals), width="stretch")

            ev = evidence_for(s_vals, BENIGN_P90, IMP_ORDER, k=6)
            if ev:
                st.markdown('<p class="smallcap">EVIDENCE :: SIGNALS ABOVE BENIGN P90</p>',
                            unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=[s_vals[f] for f in ev][::-1],
                    y=[FEATURE_LABELS[f] for f in ev][::-1],
                    orientation="h", marker=dict(color=MAGENTA),
                    text=[f"p90 {BENIGN_P90[f]:.2f}" for f in ev][::-1],
                    textposition="outside", textfont=dict(color=MUT, size=10)))
                st.plotly_chart(dark_fig(fig, 240), width="stretch")
            else:
                st.success("No signals exceed the benign baseline. This is what healthy looks like.")
            st.markdown('<p class="smallcap">IN PRODUCTION THIS VERDICT IS SERVED OVER MCP :: '
                        'AN AI CODING AGENT CALLS assess_package BEFORE TOUCHING A DEPENDENCY FILE</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="border:1px dashed {CYAN}44; border-radius:16px;
                padding:70px 30px; text-align:center; color:{MUT};
                font-family:'Share Tech Mono',monospace;">
                AWAITING SCAN<br><br>
                <span style="font-size:2.2rem;">🛰️</span><br><br>
                verdict gauge, attack-surface radar, and evidence render here
                </div>""", unsafe_allow_html=True)

# ================================================================ QUEUE
with tab_queue:
    st.markdown('<p class="smallcap">HUMAN REVIEW QUEUE :: PACKAGES THE MODEL REFUSES TO GUESS ON</p>',
                unsafe_allow_html=True)

    q = TEST.copy()
    q["lane"] = triage_lanes(q["p_malicious"].values, lo, hi)
    queue = q[q["lane"] == "REVIEW"].sort_values("p_malicious", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("PACKAGES IN QUEUE", f"{len(queue):,}")
    c2.metric("SHARE OF TRAFFIC", f"{len(queue)/len(q)*100:.1f}%")
    c3.metric("MEDIAN SCORE", f"{queue['p_malicious'].median():.2f}" if len(queue) else "n/a")

    reveal = st.toggle("🔦 Reveal ground truth (demo mode)", value=False)

    if len(queue):
        show = queue.copy()
        show["top evidence"] = show.apply(
            lambda r: ", ".join(evidence_for(r[FEATURES].to_dict(), BENIGN_P90, IMP_ORDER, k=3))
            or "borderline profile", axis=1)
        cols = ["p_malicious", "top evidence", "maintainer_age_days",
                "log_weekly_downloads", "version_jump_anomaly", "new_comaintainer_30d"]
        if reveal:
            show["ground truth"] = np.where(show["label"] == 1, "☠ malicious", "✅ benign")
            cols = ["p_malicious", "ground truth", "archetype", "top evidence"]
        st.dataframe(
            show[cols].rename(columns={"p_malicious": "P(malicious)"}),
            width="stretch", height=420,
            column_config={"P(malicious)": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0)})
    else:
        st.success("Queue is empty at this policy. Widen the uncertainty band to see escalations.")

# ================================================================ POLICY
with tab_policy:
    st.markdown('<p class="smallcap">POLICY LAB :: THE SIDEBAR DIALS RECOMPUTE EVERYTHING '
                'LIVE ON THE HELD-OUT TEST SET</p>', unsafe_allow_html=True)
    m = policy_metrics(TEST["p_malicious"].values, TEST["label"].values, lo, hi)
    single = policy_metrics(TEST["p_malicious"].values, TEST["label"].values, 0.5, 0.5)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DECISIONS AUTOMATED", f"{m['automation_rate']*100:.1f}%")
    c2.metric("AUTOMATED ACCURACY", f"{m['auto_accuracy']*100:.2f}%")
    c3.metric("REVIEW QUEUE", f"{m['review_rate']*100:.1f}%")
    c4.metric("ATTACKS SLIPPING THROUGH", f"{m['missed_attack_rate']*100:.1f}%",
              delta=f"{(m['missed_attack_rate']-single['missed_attack_rate'])*100:+.1f} pts vs single threshold",
              delta_color="inverse")

    bands = [(round(l, 2), round(1 - l, 2)) for l in np.arange(0.5, 0.04, -0.05)]
    sweep = pd.DataFrame([
        {**policy_metrics(TEST["p_malicious"].values, TEST["label"].values, l, h),
         "band": f"{l:.2f}/{h:.2f}"} for l, h in bands])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["band"], y=sweep["auto_accuracy"] * 100,
                             mode="lines+markers", name="Automated accuracy",
                             line=dict(color=GREEN, width=3)))
    fig.add_trace(go.Scatter(x=sweep["band"], y=(1 - sweep["missed_attack_rate"]) * 100,
                             mode="lines+markers", name="Attacks caught or escalated",
                             line=dict(color=RED, width=3)))
    fig.add_trace(go.Bar(x=sweep["band"], y=sweep["review_rate"] * 100,
                         name="Review queue (%)", marker_color=CYAN, opacity=0.25, yaxis="y2"))
    cur = f"{lo:.2f}/{hi:.2f}"
    bands_list = list(sweep["band"])
    if cur in bands_list:
        idx = bands_list.index(cur)
        fig.add_trace(go.Scatter(
            x=[cur], y=[sweep["auto_accuracy"].iloc[idx] * 100],
            mode="markers", name="Your policy",
            marker=dict(color=MAGENTA, size=16, symbol="diamond",
                        line=dict(color="#fff", width=1))))
        fig.add_annotation(x=cur, y=101, yref="y", text="YOUR POLICY",
                           showarrow=False, font=dict(color=MAGENTA, size=11))
    fig.update_layout(
        title="WIDEN THE BAND :: AUTOMATION QUALITY VS HUMAN WORKLOAD",
        yaxis=dict(title="Percent", range=[85, 101]),
        yaxis2=dict(title="Queue %", overlaying="y", side="right", range=[0, 50]),
        xaxis_title="Policy band (allow-below / block-above)")
    st.plotly_chart(dark_fig(fig, 400), width="stretch")

    st.markdown('<p class="smallcap">A REGULATED ENTERPRISE WIDENS THE BAND AND STAFFS THE '
                'QUEUE. A STARTUP NARROWS IT. THE MODEL DOES NOT CHANGE; THE POLICY DOES.</p>',
                unsafe_allow_html=True)

# ================================================================ FLEET
with tab_fleet:
    st.markdown('<p class="smallcap">FLEET RADAR :: DISTRIBUTION SHIFTS HERE ARE THE '
                'DRIFT ALARM THAT TRIGGERS RETRAINING</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for lbl, name, color in [(0, "Benign", CYAN), (1, "Malicious", RED)]:
            fig.add_trace(go.Histogram(
                x=TEST.loc[TEST.label == lbl, "p_malicious"], name=name,
                marker_color=color, opacity=0.6, nbinsx=40))
        fig.add_vrect(x0=lo, x1=hi, fillcolor=AMBER, opacity=0.10,
                      annotation_text="review band", annotation_font_color=AMBER)
        fig.update_layout(barmode="overlay", title="SCORE DISTRIBUTION BY GROUND TRUTH",
                          xaxis_title="P(malicious)", yaxis_type="log",
                          yaxis_title="Packages (log)")
        st.plotly_chart(dark_fig(fig), width="stretch")
    with c2:
        mal = TEST[(TEST.label == 1) & (TEST.archetype != "benign")]
        mix = mal["archetype"].value_counts()
        fig = go.Figure(go.Bar(x=mix.values, y=mix.index, orientation="h",
                               marker_color=[RED, MAGENTA, AMBER, PURPLE][:len(mix)]))
        fig.update_layout(title="MALICIOUS TRAFFIC BY ATTACK ARCHETYPE",
                          xaxis_title="Packages")
        st.plotly_chart(dark_fig(fig), width="stretch")

    imp_sorted = IMPORTANCE.sort_values("importance")
    imp_fig = go.Figure(go.Bar(
        x=imp_sorted["importance"],
        y=[FEATURE_LABELS[f] for f in imp_sorted["feature"]],
        orientation="h",
        marker=dict(color=imp_sorted["importance"], colorscale=[[0, "#153048"], [1, CYAN]])))
    imp_fig.update_layout(title="WHAT THE MODEL RELIES ON :: PERMUTATION IMPORTANCE (F1)")
    st.plotly_chart(dark_fig(imp_fig, 430), width="stretch")
