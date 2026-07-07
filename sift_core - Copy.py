"""
sift_core.py
Shared core for the SIFT platform: corpus synthesis, feature schema,
and triage logic. Kept identical to the capstone notebook so the app,
the training script, and the notebook never drift apart.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42

FEATURES = [
    "name_dist_top1k", "has_install_script", "install_entropy",
    "obfuscation_score", "maintainer_age_days", "maintainer_pkg_count",
    "log_weekly_downloads", "dependency_count", "readme_length",
    "repo_linked", "version_jump_anomaly", "new_comaintainer_30d",
    "net_calls_in_install", "base64_blob_count", "touches_agent_config",
    "touches_credential_paths", "publish_burst_7d",
]

FEATURE_LABELS = {
    "name_dist_top1k": "Name distance to top-1k packages",
    "has_install_script": "Declares install lifecycle script",
    "install_entropy": "Install-code entropy",
    "obfuscation_score": "Obfuscation ratio",
    "maintainer_age_days": "Maintainer account age (days)",
    "maintainer_pkg_count": "Maintainer package count",
    "log_weekly_downloads": "Weekly downloads (log)",
    "dependency_count": "Dependency count",
    "readme_length": "README length (chars)",
    "repo_linked": "Linked source repository",
    "version_jump_anomaly": "Version cadence anomaly",
    "new_comaintainer_30d": "Co-maintainer added in last 30 days",
    "net_calls_in_install": "Network calls during install",
    "base64_blob_count": "Base64 payload blobs",
    "touches_agent_config": "Writes AI-agent config paths",
    "touches_credential_paths": "Reads credential paths",
    "publish_burst_7d": "Versions published, last 7 days",
}


def synthesize_corpus(n=6000, malicious_rate=0.08, seed=RANDOM_STATE):
    """Synthesize a package corpus modeled on documented attack archetypes.

    Identical to the capstone notebook generator. See the notebook header
    for the dataset source rationale (Backstabber's Knife Collection,
    OpenSSF malicious-packages, npm registry schema).
    """
    rng = np.random.default_rng(seed)
    n_mal = int(n * malicious_rate)
    n_ben = n - n_mal

    def clip01(x):
        return np.clip(x, 0, 1)

    ben = pd.DataFrame({
        "name_dist_top1k": rng.integers(2, 15, n_ben).astype(float),
        "has_install_script": rng.binomial(1, 0.14, n_ben),
        "install_entropy": clip01(rng.normal(0.40, 0.12, n_ben)),
        "obfuscation_score": clip01(rng.beta(1.4, 9, n_ben)),
        "maintainer_age_days": rng.gamma(3.2, 380, n_ben),
        "maintainer_pkg_count": rng.gamma(2.0, 6.0, n_ben),
        "log_weekly_downloads": rng.normal(6.4, 2.6, n_ben),
        "dependency_count": rng.poisson(9, n_ben).astype(float),
        "readme_length": rng.gamma(2.2, 1400, n_ben),
        "repo_linked": rng.binomial(1, 0.87, n_ben),
        "version_jump_anomaly": rng.binomial(1, 0.04, n_ben),
        "new_comaintainer_30d": rng.binomial(1, 0.05, n_ben),
        "net_calls_in_install": rng.binomial(1, 0.035, n_ben),
        "base64_blob_count": rng.poisson(0.15, n_ben).astype(float),
        "touches_agent_config": rng.binomial(1, 0.008, n_ben),
        "touches_credential_paths": rng.binomial(1, 0.01, n_ben),
        "publish_burst_7d": rng.poisson(0.6, n_ben).astype(float),
    })
    ben["label"] = 0

    def archetype(kind, m):
        if kind == "typosquat":
            d = {
                "name_dist_top1k": rng.integers(1, 3, m).astype(float),
                "has_install_script": rng.binomial(1, 0.72, m),
                "install_entropy": clip01(rng.normal(0.66, 0.14, m)),
                "obfuscation_score": clip01(rng.beta(4, 4, m)),
                "maintainer_age_days": rng.gamma(1.1, 45, m),
                "maintainer_pkg_count": rng.gamma(1.2, 2.2, m),
                "log_weekly_downloads": rng.normal(2.2, 1.4, m),
                "dependency_count": rng.poisson(3, m).astype(float),
                "readme_length": rng.gamma(1.2, 350, m),
                "repo_linked": rng.binomial(1, 0.30, m),
                "version_jump_anomaly": rng.binomial(1, 0.08, m),
                "new_comaintainer_30d": rng.binomial(1, 0.06, m),
                "net_calls_in_install": rng.binomial(1, 0.55, m),
                "base64_blob_count": rng.poisson(1.6, m).astype(float),
                "touches_agent_config": rng.binomial(1, 0.05, m),
                "touches_credential_paths": rng.binomial(1, 0.35, m),
                "publish_burst_7d": rng.poisson(2.5, m).astype(float),
            }
        elif kind == "dropper":
            d = {
                "name_dist_top1k": rng.integers(3, 14, m).astype(float),
                "has_install_script": rng.binomial(1, 0.93, m),
                "install_entropy": clip01(rng.normal(0.78, 0.10, m)),
                "obfuscation_score": clip01(rng.beta(6, 2.5, m)),
                "maintainer_age_days": rng.gamma(1.4, 90, m),
                "maintainer_pkg_count": rng.gamma(1.4, 3.0, m),
                "log_weekly_downloads": rng.normal(3.4, 2.0, m),
                "dependency_count": rng.poisson(4, m).astype(float),
                "readme_length": rng.gamma(1.6, 700, m),
                "repo_linked": rng.binomial(1, 0.45, m),
                "version_jump_anomaly": rng.binomial(1, 0.15, m),
                "new_comaintainer_30d": rng.binomial(1, 0.10, m),
                "net_calls_in_install": rng.binomial(1, 0.85, m),
                "base64_blob_count": rng.poisson(3.2, m).astype(float),
                "touches_agent_config": rng.binomial(1, 0.08, m),
                "touches_credential_paths": rng.binomial(1, 0.55, m),
                "publish_burst_7d": rng.poisson(1.8, m).astype(float),
            }
        elif kind == "compromise":
            d = {
                "name_dist_top1k": rng.integers(4, 15, m).astype(float),
                "has_install_script": rng.binomial(1, 0.60, m),
                "install_entropy": clip01(rng.normal(0.60, 0.15, m)),
                "obfuscation_score": clip01(rng.beta(3.5, 4, m)),
                "maintainer_age_days": rng.gamma(3.5, 400, m),
                "maintainer_pkg_count": rng.gamma(2.2, 7.0, m),
                "log_weekly_downloads": rng.normal(7.5, 2.0, m),
                "dependency_count": rng.poisson(10, m).astype(float),
                "readme_length": rng.gamma(2.4, 1500, m),
                "repo_linked": rng.binomial(1, 0.88, m),
                "version_jump_anomaly": rng.binomial(1, 0.70, m),
                "new_comaintainer_30d": rng.binomial(1, 0.62, m),
                "net_calls_in_install": rng.binomial(1, 0.55, m),
                "base64_blob_count": rng.poisson(1.8, m).astype(float),
                "touches_agent_config": rng.binomial(1, 0.06, m),
                "touches_credential_paths": rng.binomial(1, 0.40, m),
                "publish_burst_7d": rng.poisson(1.2, m).astype(float),
            }
        else:  # agent_worm
            d = {
                "name_dist_top1k": rng.integers(1, 8, m).astype(float),
                "has_install_script": rng.binomial(1, 0.90, m),
                "install_entropy": clip01(rng.normal(0.74, 0.11, m)),
                "obfuscation_score": clip01(rng.beta(5, 3, m)),
                "maintainer_age_days": rng.gamma(1.8, 220, m),
                "maintainer_pkg_count": rng.gamma(1.8, 5.0, m),
                "log_weekly_downloads": rng.normal(4.5, 2.2, m),
                "dependency_count": rng.poisson(6, m).astype(float),
                "readme_length": rng.gamma(1.8, 900, m),
                "repo_linked": rng.binomial(1, 0.60, m),
                "version_jump_anomaly": rng.binomial(1, 0.35, m),
                "new_comaintainer_30d": rng.binomial(1, 0.20, m),
                "net_calls_in_install": rng.binomial(1, 0.75, m),
                "base64_blob_count": rng.poisson(2.2, m).astype(float),
                "touches_agent_config": rng.binomial(1, 0.82, m),
                "touches_credential_paths": rng.binomial(1, 0.70, m),
                "publish_burst_7d": rng.poisson(4.5, m).astype(float),
            }
        df = pd.DataFrame(d)
        df["label"] = 1
        df["archetype"] = kind
        return df

    mix = [("typosquat", 0.34), ("dropper", 0.30),
           ("compromise", 0.16), ("agent_worm", 0.20)]
    mal_frames = [archetype(k, int(n_mal * w)) for k, w in mix]
    ben["archetype"] = "benign"
    df = pd.concat([ben] + mal_frames, ignore_index=True)

    # 1% label noise: imperfect threat-feed ground truth
    flip = rng.random(len(df)) < 0.01
    df.loc[flip, "label"] = 1 - df.loc[flip, "label"]

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def triage_lane(p, lo, hi):
    """Route a probability into ALLOW / REVIEW / BLOCK."""
    if p < lo:
        return "ALLOW"
    if p >= hi:
        return "BLOCK"
    return "REVIEW"


def triage_lanes(p, lo, hi):
    """Vectorized lane assignment."""
    lanes = np.full(len(p), "REVIEW", dtype=object)
    lanes[p < lo] = "ALLOW"
    lanes[p >= hi] = "BLOCK"
    return lanes


def policy_metrics(p, y_true, lo, hi):
    """Compute the operational impact of a threshold policy."""
    p = np.asarray(p)
    y_true = np.asarray(y_true)
    lanes = triage_lanes(p, lo, hi)
    auto = lanes != "REVIEW"
    n_mal = max((y_true == 1).sum(), 1)
    out = {
        "automation_rate": float(auto.mean()),
        "review_rate": float((~auto).mean()),
        "review_count": int((~auto).sum()),
        "missed_attack_rate": float(((lanes == "ALLOW") & (y_true == 1)).sum() / n_mal),
        "auto_accuracy": float("nan"),
    }
    if auto.any():
        auto_pred = (p[auto] >= 0.5).astype(int)
        out["auto_accuracy"] = float((auto_pred == y_true[auto]).mean())
    return out


def evidence_for(features: dict, benign_p90: dict, importance_order: list, k=5):
    """Rank the features of one package that exceed the benign 90th percentile,
    ordered by global model importance. This is what an analyst sees."""
    return [f for f in importance_order
            if features.get(f, 0) > benign_p90.get(f, np.inf)][:k]
