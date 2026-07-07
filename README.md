# SIFT // Triage Console

Neon SOC-style Streamlit deployment layer for **SIFT** (Supply-chain
Intelligence and Forensic Triage), an AI supply-chain threat intelligence
capstone. Scores open source packages and routes each one to
**ALLOW / REVIEW / BLOCK** using a confidence-based triage policy.

## Tabs

- **💬 Ask SIFT** — describe a package in plain English; an LLM extracts the
  structured features and SIFT scores it. The verdict is always computed by the
  model, never the LLM (see Security).
- **⚡ Scan** — pick or hand-build a package, run a theatrical scan, get a
  verdict with a threat gauge, attack-surface radar, and evidence.
- **🗂 Review Queue** — the analyst queue produced by the current policy.
- **🎚 Policy Lab** — the sidebar thresholds are product policy dials; metrics
  and the automation-vs-workload curve recompute live on the held-out test set.
- **📡 Fleet Radar** — score distributions, archetype mix, feature importance.

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

The model regenerates itself on first run (`train.py`) if `artifacts/model.skops`
is missing, so no serialized model is ever committed or downloaded.

Optional, to enable the Ask SIFT tab:

    export ANTHROPIC_API_KEY=sk-ant-...

## Run with Docker

    docker build -t sift-console .
    docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... sift-console

Open http://localhost:8501

## Deploy to Streamlit Community Cloud

Push to GitHub, point Streamlit Cloud at `app.py`, and add
`ANTHROPIC_API_KEY` under the app's **Secrets** (Settings → Secrets). Never put
the key in code.

## Security

- **No pickle.** The model is serialized with `skops`, and the loader
  allowlists only the sklearn/numpy types the model needs, so a tampered model
  file raises instead of executing code. Fitting for a supply-chain security
  tool.
- **The LLM cannot change a verdict.** In Ask SIFT the language model only
  translates plain English into the 17 features and explains the result. The
  ALLOW / REVIEW / BLOCK decision and probability come from the deterministic
  scikit-learn model, so prompt injection cannot alter a score.
- **No secrets in git.** `secrets.toml` and `.env` are gitignored; the API key
  is read from the environment or Streamlit secrets at runtime.
- **Non-root container.** The Docker image runs as an unprivileged user.
- **No free-text-to-HTML sink.** Every scoring input is numeric or a fixed
  choice; there is no user-controlled string rendered as HTML.

## Layout

    sift_core.py    corpus generator, feature schema, triage logic (shared with the notebook)
    sift_agent.py   natural-language layer: NL -> features -> deterministic scoring
    train.py        trains the tuned Gradient Boosting model, writes artifacts/ (skops)
    app.py          the console
    artifacts/      test_scores.csv, feature_importance.csv, benign_p90.json, metrics.json
                    (model.skops is generated locally, not committed)

Dataset is a synthesized corpus modeled on Backstabber's Knife Collection
(arXiv:2005.09535) and the OpenSSF malicious-packages feed; see the capstone
notebook header for the full sourcing rationale.
