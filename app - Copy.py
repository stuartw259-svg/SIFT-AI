"""
SIFT — Supply-chain package triage
Professional console. Same model and triage logic; presentation rebuilt around
a restrained product design system.

    streamlit run app.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skops.io as sio
import streamlit as st

import sift_agent
from sift_core import (FEATURES, FEATURE_LABELS, evidence_for,
                       policy_metrics, triage_lane, triage_lanes)

# ---------------------------------------------------------------- tokens
BG = "#0C0E12"
SURFACE = "#12151C"
SURFACE2 = "#171B24"
BORDER = "#232936"
TEXT = "#E6EAF2"
MUT = "#8A93A6"
ACCENT = "#4C8DFF"
ALLOW = "#3DD68C"
REVIEW = "#E5A83B"
BLOCK = "#F2555A"

LANE_STYLE = {
    "ALLOW": (ALLOW, "Package proceeds. No elevated risk signals."),
    "REVIEW": (REVIEW, "Routed to a human analyst with evidence attached."),
    "BLOCK": (BLOCK, "Install halted. Verdict and evidence logged for audit."),
}

st.set_page_config(page_title="SIFT — Package triage", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Inter+Tight:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
    font-family: 'Inter', -apple-system, sans-serif;
}}
[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
h1, h2, h3 {{ font-family: 'Inter Tight', 'Inter', sans-serif !important; letter-spacing: -0.01em; }}

/* header */
.hdr {{ display: flex; align-items: baseline; gap: 14px; margin-bottom: 2px; }}
.hdr .mark {{
    font-family: 'Inter Tight', sans-serif; font-weight: 700; font-size: 1.7rem;
    color: {TEXT}; letter-spacing: -0.02em;
}}
.hdr .sub {{ color: {MUT}; font-size: .95rem; }}
.statusline {{
    font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: {MUT};
    margin-bottom: 1.4rem;
}}
.statusline .dot {{
    display:inline-block; width:7px; height:7px; border-radius:50%;
    background:{ALLOW}; margin-right:6px; vertical-align:1px;
}}

/* cards + metrics */
div[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 14px 16px;
}}
div[data-testid="stMetric"] label p {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .72rem !important; letter-spacing: .06em;
    text-transform: uppercase; color: {MUT} !important;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'Inter Tight', sans-serif; font-weight: 600;
}}

/* tabs: quiet underline */
button[data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif !important; font-weight: 500;
    background: transparent !important; color: {MUT} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEXT} !important; }}
div[data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; height: 2px; }}
div[data-baseweb="tab-border"] {{ background-color: {BORDER} !important; }}

/* section label */
.eyebrow {{
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
    letter-spacing: .08em; text-transform: uppercase; color: {MUT};
    margin: 4px 0 10px 0;
}}

/* verdict card + triage rail (signature element) */
.verdict {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 20px 22px 18px 22px; margin-bottom: 14px;
    border-left: 3px solid var(--lane);
}}
.verdict .lane {{
    font-family: 'Inter Tight', sans-serif; font-weight: 700; font-size: 1.5rem;
    color: var(--lane); letter-spacing: .01em;
}}
.verdict .prob {{
    font-family: 'IBM Plex Mono', monospace; font-size: .85rem; color: {TEXT};
    margin-left: 10px;
}}
.verdict .desc {{ color: {MUT}; font-size: .9rem; margin-top: 4px; }}
.rail {{ position: relative; height: 8px; border-radius: 4px; margin-top: 16px;
         background: {SURFACE2}; border: 1px solid {BORDER}; }}
.rail .band {{ position: absolute; top: 0; bottom: 0; background: {REVIEW}22;
               border-left: 1px solid {REVIEW}55; border-right: 1px solid {REVIEW}55; }}
.rail .marker {{
    position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%;
    background: var(--lane); border: 2px solid {BG};
    transform: translate(-50%, -50%);
}}
.rail-labels {{
    display: flex; justify-content: space-between; margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: .68rem; color: {MUT};
}}

/* pipeline steps */
.steps {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 14px 16px; font-family: 'IBM Plex Mono', monospace;
    font-size: .78rem; color: {MUT}; line-height: 1.9; min-height: 150px;
}}
.steps .done {{ color: {TEXT}; }}
.steps .done::before {{ content: "✓  "; color: {ALLOW}; }}
.steps .pending::before {{ content: "·  "; color: {MUT}; }}

/* buttons */
div.stButton > button {{
    font-family: 'Inter', sans-serif; font-weight: 500;
    background: {ACCENT}; color: #0A0C10; border: none; border-radius: 8px;
    padding: .55rem 1.2rem; transition: background .12s ease;
}}
div.stButton > button:hover {{ background: #6BA1FF; color: #0A0C10; }}

/* inputs quieter */
div[data-baseweb="select"] > div, .stTextInput input {{
    background: {SURFACE} !important; border-color: {BORDER} !important;
}}
.smallnote {{ color: {MUT}; font-size: .8rem; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- artifacts
ART = Path(__file__).parent / "artifacts"


@st.cache_resource
def load_artifacts():
    if not (ART / "model.skops").exists():
        import train
        train.main()
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
    st.error("Artifacts not found. Run `python train.py` first.")
    st.stop()

IMP_ORDER = IMPORTANCE["feature"].tolist()

# ---------------------------------------------------------------- presets
PRESETS = {
    "fast-jsonn 1.0.2 — typosquat that writes agent configs": dict(
        name_dist_top1k=1, has_install_script=1, install_entropy=0.81,
        obfuscation_score=0.72, maintainer_age_days=12, maintainer_pkg_count=3,
        log_weekly_downloads=2.1, dependency_count=4, readme_length=310,
        repo_linked=0, version_jump_anomaly=0, new_comaintainer_30d=0,
        net_calls_in_install=1, base64_blob_count=4, touches_agent_config=1,
        touches_credential_paths=1, publish_burst_7d=6),
    "lodash-extras 4.2.0 — healthy utility": dict(
        name_dist_top1k=9, has_install_script=0, install_entropy=0.40,
        obfuscation_score=0.08, maintainer_age_days=2400, maintainer_pkg_count=14,
        log_weekly_downloads=8.9, dependency_count=6, readme_length=5200,
        repo_linked=1, version_jump_anomaly=0, new_comaintainer_30d=0,
        net_calls_in_install=0, base64_blob_count=0, touches_agent_config=0,
        touches_credential_paths=0, publish_burst_7d=0),
    "popular-lib 9.0.0 — anomalous release of a trusted package": dict(
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
st.markdown(f"""
<div class="hdr">
  <span class="mark">SIFT</span>
  <span class="sub">Supply-chain package triage</span>
</div>
<div class="statusline">
  <span class="dot"></span>model online&nbsp;&nbsp;·&nbsp;&nbsp;tuned gradient boosting
  &nbsp;&nbsp;·&nbsp;&nbsp;accuracy {METRICS['accuracy']:.3f}
  &nbsp;&nbsp;·&nbsp;&nbsp;precision {METRICS['precision']:.2f}
  &nbsp;&nbsp;·&nbsp;&nbsp;recall {METRICS['recall']:.2f}
  &nbsp;&nbsp;·&nbsp;&nbsp;auc {METRICS['roc_auc']:.2f}
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"<div style='font-family:Inter Tight,sans-serif;font-weight:700;"
                f"font-size:1.15rem;color:{TEXT};margin-bottom:2px;'>Triage policy</div>",
                unsafe_allow_html=True)
    st.markdown('<p class="smallnote">Thresholds are configuration, not code. '
                'Every tab recomputes live when these change.</p>', unsafe_allow_html=True)
    lo = st.slider("Allow below", 0.0, 0.5, 0.30, 0.05)
    hi = st.slider("Block at or above", 0.5, 1.0, 0.70, 0.05)
    st.markdown(f'<p class="smallnote">Scores from {lo:.2f} to {hi:.2f} go to human review.</p>',
                unsafe_allow_html=True)
    st.divider()
    st.markdown('<p class="smallnote">Corpus modeled on Backstabber\'s Knife Collection '
                'and the OpenSSF malicious-packages feed.<br><br>Capstone project · '
                'Stuart Wubbena · Institute of Data / UTSA · 2026</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- helpers


def fig_style(fig, h=330):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter", size=12),
        title_font=dict(family="Inter", size=13, color=MUT),
        margin=dict(l=8, r=8, t=38, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def verdict_card(lane, p):
    """Signature element: verdict + triage rail. The rail shows where this
    package landed relative to the current policy band."""
    color, desc = LANE_STYLE[lane]
    st.markdown(f"""
    <div class="verdict" style="--lane:{color};">
      <span class="lane">{lane}</span>
      <span class="prob">P(malicious) = {p:.4f}</span>
      <div class="desc">{desc}</div>
      <div class="rail">
        <div class="band" style="left:{lo*100:.0f}%; width:{(hi-lo)*100:.0f}%;"></div>
        <div class="marker" style="left:{min(max(p,0.005),0.995)*100:.1f}%;"></div>
      </div>
      <div class="rail-labels">
        <span>0 · allow</span><span>review band {lo:.2f}–{hi:.2f}</span><span>block · 1</span>
      </div>
    </div>""", unsafe_allow_html=True)


RADAR_FEATS = ["install_entropy", "obfuscation_score", "base64_blob_count",
               "publish_burst_7d", "net_calls_in_install",
               "touches_agent_config", "touches_credential_paths", "name_dist_top1k"]
RADAR_MAX = {"install_entropy": 1, "obfuscation_score": 1, "base64_blob_count": 6,
             "publish_burst_7d": 8, "net_calls_in_install": 1,
             "touches_agent_config": 1, "touches_credential_paths": 1,
             "name_dist_top1k": 15}


def radar(vals):
    def norm(src):
        out = []
        for f in RADAR_FEATS:
            v = src.get(f, 0) / RADAR_MAX[f]
            if f == "name_dist_top1k":
                v = 1 - v
            out.append(min(max(v, 0), 1))
        return out
    theta = [FEATURE_LABELS[f].replace(" (days)", "") for f in RADAR_FEATS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm(BENIGN_MED) + [norm(BENIGN_MED)[0]], theta=theta + [theta[0]],
        name="Benign median", line=dict(color=MUT, width=1.2),
        fill="toself", fillcolor="rgba(138,147,166,0.08)"))
    fig.add_trace(go.Scatterpolar(
        r=norm(vals) + [norm(vals)[0]], theta=theta + [theta[0]],
        name="This package", line=dict(color=ACCENT, width=2),
        fill="toself", fillcolor="rgba(76,141,255,0.14)"))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor=BORDER),
        angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=9, color=MUT))),
        title="Attack-surface profile")
    return fig_style(fig, 320)


PIPELINE = [
    "Pull registry metadata and tarball manifest",
    "Detonate install in an isolated sandbox",
    "Capture syscalls: file writes, network egress, credential reads",
    "Vectorize 17 behavioral and reputation signals",
    "Score with the tuned gradient boosting model",
]

# ---------------------------------------------------------------- tabs
tab_ask, tab_assess, tab_queue, tab_policy, tab_fleet = st.tabs(
    ["Ask SIFT", "Assess", "Review queue", "Policy", "Monitoring"])

# ================================================================ ASK
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        import os as _os
        _os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

with tab_ask:
    eyebrow("Natural-language intake")

    if not sift_agent.has_api_key():
        st.markdown(f"""
        <div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
                    padding:18px 20px;">
          <div style="font-weight:600; color:{TEXT}; margin-bottom:6px;">
            Language layer is offline</div>
          <div class="smallnote">Add an Anthropic API key to enable plain-English queries.
          Locally: <code>export ANTHROPIC_API_KEY=...</code> · Streamlit Cloud: add it under
          the app's Secrets. Every other tab works without it.</div>
        </div>""", unsafe_allow_html=True)
        st.write("")
        st.info("Design note: the language model only translates a description into the "
                "17 structured features and explains the result. The verdict is always "
                "computed by the scoring model, so phrasing cannot change a score.")
    else:
        st.markdown('<p class="smallnote">Describe a package in plain English. SIFT extracts '
                    'the signals, scores them, and explains the verdict. Try: “a brand new npm '
                    'package that runs a postinstall script, makes network calls, and writes '
                    'to mcp.json”.</p>', unsafe_allow_html=True)
        q = st.text_input("Ask SIFT", placeholder="Describe a package…",
                          label_visibility="collapsed")
        ask_go = st.button("Run assessment", key="ask_btn")

        if ask_go and q.strip():
            with st.spinner("Extracting signals and scoring…"):
                try:
                    res = sift_agent.ask(
                        q.strip(), model=model, benign_med=BENIGN_MED,
                        benign_p90=BENIGN_P90, imp_order=IMP_ORDER, lo=lo, hi=hi)
                except Exception as e:
                    st.error(f"Language layer error: {e}")
                    res = None
            if res:
                a = res.get("assessment")
                if a:
                    verdict_card(a["lane"], a["p_malicious"])
                    c1, c2 = st.columns([1.1, 1])
                    with c1:
                        eyebrow("Analyst note")
                        st.write(res["answer"])
                        if a["evidence"]:
                            eyebrow("Evidence")
                            st.markdown(" · ".join(a["evidence"]))
                    with c2:
                        with st.expander("Signals inferred from your description"):
                            st.json(res["tool_input"] or {})
                else:
                    st.write(res["answer"])

# ================================================================ ASSESS
with tab_assess:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        eyebrow("Package under assessment")
        preset_name = st.selectbox("Scenario", list(PRESETS.keys()),
                                   label_visibility="collapsed")
        vals = dict(PRESETS[preset_name])

        with st.expander("Edit signals"):
            c1, c2 = st.columns(2)
            with c1:
                vals["name_dist_top1k"] = st.slider("Name distance to top-1k", 0, 15, int(vals["name_dist_top1k"]))
                vals["install_entropy"] = st.slider("Install-code entropy", 0.0, 1.0, float(vals["install_entropy"]), 0.01)
                vals["obfuscation_score"] = st.slider("Obfuscation ratio", 0.0, 1.0, float(vals["obfuscation_score"]), 0.01)
                vals["maintainer_age_days"] = st.slider("Account age (days)", 0, 4000, int(vals["maintainer_age_days"]))
                vals["log_weekly_downloads"] = st.slider("Weekly downloads (log)", 0.0, 14.0, float(vals["log_weekly_downloads"]), 0.1)
                vals["base64_blob_count"] = st.slider("Base64 blobs", 0, 10, int(vals["base64_blob_count"]))
                vals["publish_burst_7d"] = st.slider("Versions in last 7 days", 0, 12, int(vals["publish_burst_7d"]))
            with c2:
                for b in BINARY:
                    vals[b] = int(st.toggle(FEATURE_LABELS[b], bool(vals[b])))
                vals["maintainer_pkg_count"] = st.number_input("Maintainer package count", 0, 500, int(vals["maintainer_pkg_count"]))
                vals["dependency_count"] = st.number_input("Dependency count", 0, 200, int(vals["dependency_count"]))
                vals["readme_length"] = st.number_input("README length", 0, 50000, int(vals["readme_length"]))

        run = st.button("Run assessment", key="scan_btn")

        steps_slot = st.empty()

        def render_steps(done_n):
            rows = []
            for i, s in enumerate(PIPELINE):
                cls = "done" if i < done_n else "pending"
                rows.append(f'<div class="{cls}">{s}</div>')
            steps_slot.markdown('<div class="steps">' + "".join(rows) + "</div>",
                                unsafe_allow_html=True)

        if run:
            for i in range(1, len(PIPELINE) + 1):
                render_steps(i)
                time.sleep(0.22)
            st.session_state["scan"] = {"vals": vals, "name": preset_name}
        elif "scan" in st.session_state:
            render_steps(len(PIPELINE))
        else:
            render_steps(0)

    with right:
        if "scan" in st.session_state:
            s_vals = st.session_state["scan"]["vals"]
            row = pd.DataFrame([s_vals])[FEATURES]
            p = float(model.predict_proba(row)[0, 1])
            lane = triage_lane(p, lo, hi)
            verdict_card(lane, p)

            ev = evidence_for(s_vals, BENIGN_P90, IMP_ORDER, k=6)
            g1, g2 = st.columns([1, 1])
            with g1:
                st.plotly_chart(radar(s_vals), width="stretch")
            with g2:
                if ev:
                    fig = go.Figure(go.Bar(
                        x=[s_vals[f] for f in ev][::-1],
                        y=[FEATURE_LABELS[f] for f in ev][::-1],
                        orientation="h", marker=dict(color=ACCENT),
                        text=[f"benign p90 {BENIGN_P90[f]:.2f}" for f in ev][::-1],
                        textposition="outside",
                        textfont=dict(color=MUT, size=10)))
                    fig.update_layout(title="Evidence above the benign baseline")
                    st.plotly_chart(fig_style(fig, 320), width="stretch")
                else:
                    st.markdown(f"""<div style="background:{SURFACE};border:1px solid {BORDER};
                        border-radius:10px;padding:40px 20px;text-align:center;color:{MUT};
                        height:320px;display:flex;align-items:center;justify-content:center;">
                        No signals exceed the benign baseline.<br>This is what healthy looks like.
                        </div>""", unsafe_allow_html=True)
            st.markdown('<p class="smallnote">In production this verdict is served over MCP: '
                        'an AI coding agent calls assess_package before touching a dependency '
                        'file.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="border:1px dashed {BORDER}; border-radius:12px;
                padding:80px 30px; text-align:center; color:{MUT};">
                Select a scenario and run an assessment.<br>
                The verdict, attack-surface profile, and evidence render here.
                </div>""", unsafe_allow_html=True)

# ================================================================ QUEUE
with tab_queue:
    eyebrow("Human review queue · packages the model refuses to guess on")

    q = TEST.copy()
    q["lane"] = triage_lanes(q["p_malicious"].values, lo, hi)
    queue = q[q["lane"] == "REVIEW"].sort_values("p_malicious", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Packages in queue", f"{len(queue):,}")
    c2.metric("Share of traffic", f"{len(queue)/len(q)*100:.1f}%")
    c3.metric("Median score", f"{queue['p_malicious'].median():.2f}" if len(queue) else "n/a")

    reveal = st.toggle("Reveal ground truth (demo)", value=False)

    if len(queue):
        show = queue.copy()
        show["top evidence"] = show.apply(
            lambda r: ", ".join(evidence_for(r[FEATURES].to_dict(), BENIGN_P90, IMP_ORDER, k=3))
            or "borderline profile", axis=1)
        cols = ["p_malicious", "top evidence", "maintainer_age_days",
                "log_weekly_downloads", "version_jump_anomaly", "new_comaintainer_30d"]
        if reveal:
            show["ground truth"] = np.where(show["label"] == 1, "malicious", "benign")
            cols = ["p_malicious", "ground truth", "archetype", "top evidence"]
        st.dataframe(
            show[cols].rename(columns={"p_malicious": "P(malicious)"}),
            width="stretch", height=420,
            column_config={"P(malicious)": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0)})
    else:
        st.success("Queue is empty at this policy. Widen the review band to see escalations.")

# ================================================================ POLICY
with tab_policy:
    eyebrow("Policy impact · recomputed live on the held-out test set")
    m = policy_metrics(TEST["p_malicious"].values, TEST["label"].values, lo, hi)
    single = policy_metrics(TEST["p_malicious"].values, TEST["label"].values, 0.5, 0.5)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisions automated", f"{m['automation_rate']*100:.1f}%")
    c2.metric("Automated accuracy", f"{m['auto_accuracy']*100:.2f}%")
    c3.metric("Review queue", f"{m['review_rate']*100:.1f}%")
    c4.metric("Attacks slipping through", f"{m['missed_attack_rate']*100:.1f}%",
              delta=f"{(m['missed_attack_rate']-single['missed_attack_rate'])*100:+.1f} pts vs single threshold",
              delta_color="inverse")

    bands = [(round(l, 2), round(1 - l, 2)) for l in np.arange(0.5, 0.04, -0.05)]
    sweep = pd.DataFrame([
        {**policy_metrics(TEST["p_malicious"].values, TEST["label"].values, l, h),
         "band": f"{l:.2f}/{h:.2f}"} for l, h in bands])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["band"], y=sweep["auto_accuracy"] * 100,
                             mode="lines+markers", name="Automated accuracy",
                             line=dict(color=ALLOW, width=2.2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=sweep["band"], y=(1 - sweep["missed_attack_rate"]) * 100,
                             mode="lines+markers", name="Attacks caught or escalated",
                             line=dict(color=BLOCK, width=2.2), marker=dict(size=6)))
    fig.add_trace(go.Bar(x=sweep["band"], y=sweep["review_rate"] * 100,
                         name="Review queue (%)", marker_color=ACCENT,
                         opacity=0.22, yaxis="y2"))
    cur = f"{lo:.2f}/{hi:.2f}"
    bands_list = list(sweep["band"])
    if cur in bands_list:
        idx = bands_list.index(cur)
        fig.add_trace(go.Scatter(
            x=[cur], y=[sweep["auto_accuracy"].iloc[idx] * 100],
            mode="markers", name="Current policy",
            marker=dict(color=ACCENT, size=13, symbol="diamond",
                        line=dict(color=TEXT, width=1))))
    fig.update_layout(
        title="Widening the review band: automation quality vs human workload",
        yaxis=dict(title="Percent", range=[85, 101]),
        yaxis2=dict(title="Queue %", overlaying="y", side="right",
                    range=[0, 50], gridcolor="rgba(0,0,0,0)"),
        xaxis_title="Policy band (allow-below / block-above)")
    st.plotly_chart(fig_style(fig, 400), width="stretch")

    st.markdown('<p class="smallnote">A regulated enterprise widens the band and staffs the '
                'queue. A startup narrows it. The model does not change; the policy does.</p>',
                unsafe_allow_html=True)

# ================================================================ MONITORING
with tab_fleet:
    eyebrow("Monitoring · distribution shifts here are the drift alarm that triggers retraining")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for lbl, name, color in [(0, "Benign", ACCENT), (1, "Malicious", BLOCK)]:
            fig.add_trace(go.Histogram(
                x=TEST.loc[TEST.label == lbl, "p_malicious"], name=name,
                marker_color=color, opacity=0.55, nbinsx=40))
        fig.add_vrect(x0=lo, x1=hi, fillcolor=REVIEW, opacity=0.08,
                      line_width=0)
        fig.update_layout(barmode="overlay", title="Score distribution by ground truth",
                          xaxis_title="P(malicious)", yaxis_type="log",
                          yaxis_title="Packages (log)")
        st.plotly_chart(fig_style(fig), width="stretch")
    with c2:
        mal = TEST[(TEST.label == 1) & (TEST.archetype != "benign")]
        mix = mal["archetype"].value_counts()
        fig = go.Figure(go.Bar(x=mix.values, y=mix.index, orientation="h",
                               marker_color=ACCENT))
        fig.update_layout(title="Malicious traffic by attack archetype",
                          xaxis_title="Packages")
        st.plotly_chart(fig_style(fig), width="stretch")

    imp_sorted = IMPORTANCE.sort_values("importance")
    imp_fig = go.Figure(go.Bar(
        x=imp_sorted["importance"],
        y=[FEATURE_LABELS[f] for f in imp_sorted["feature"]],
        orientation="h", marker=dict(color=ACCENT)))
    imp_fig.update_layout(title="What the model relies on (permutation importance, F1)")
    st.plotly_chart(fig_style(imp_fig, 430), width="stretch")
