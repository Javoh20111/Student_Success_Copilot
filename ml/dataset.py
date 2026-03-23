"""
ml/dataset.py
=============
Synthetic student dataset generator.

Generation method
-----------------
A latent risk score drawn from N(0,1) drives all feature distributions,
creating realistic inter-column correlations without hand-coding each pair.
Higher latent_risk → lower attendance, lower quiz, higher stress, more tasks.

Target variables
----------------
success    (0=Fail, 1=Pass)  — binary classification target for ML models
risk_level (0=Low, 1=Med, 2=High) — for rules engine only, NOT an ML feature

Leakage guard
-------------
risk_level is derived from the same latent variable as success.
It must never be used as a feature when training ML models.

Gender
------
Generated independently of latent_risk (no correlation by design).
Whether gender predicts success is an empirical question answered
during training — not a design assumption baked into the data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_PATH   = Path(__file__).resolve().parent.parent / "data" / "student_data.csv"
RANDOM_SEED = 42


# =============================================================================
# Generator  (provided implementation, integrated into project API)
# =============================================================================

def generate_dataset(n: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic student dataset with realistic inter-column
    correlations via a shared latent risk variable.

    Parameters
    ----------
    n    : number of rows  (default 1000)
    seed : random seed for reproducibility

    Returns
    -------
    pd.DataFrame  — also saved to data/student_data.csv
    """
    rng = np.random.default_rng(seed)
    N   = n

    # ── STEP 1: Latent risk score ─────────────────────────────────────────────
    # Hidden variable drawn from N(0,1). Higher value = more at risk.
    # All features are shifted toward "worse" values for high-risk students,
    # creating realistic inter-column correlations.
    latent_risk = rng.normal(0, 1, N)

    # ── STEP 2: Feature columns ───────────────────────────────────────────────

    student_id = np.arange(1, N + 1)

    # gender: 0=m, 1=f — no intentional correlation with risk
    gender = rng.integers(0, 2, N)

    # attendance (0–100): high-risk students attend less
    attendance_raw = 75 - 12 * latent_risk + rng.normal(0, 8, N)
    attendance = np.clip(np.round(attendance_raw, 1), 0, 100)

    # confidence_level (1–5 ordinal): low risk → high confidence
    confidence_raw = 3.2 - 0.7 * latent_risk + rng.normal(0, 0.6, N)
    confidence_level = np.clip(np.round(confidence_raw).astype(int), 1, 5)

    # stress_level (1–5 ordinal): high risk → high stress
    stress_raw = 2.8 + 0.8 * latent_risk + rng.normal(0, 0.5, N)
    stress_level = np.clip(np.round(stress_raw).astype(int), 1, 5)

    # deadlines (0–10 int): high risk → more upcoming deadlines
    deadlines_raw = 4.0 + 1.5 * latent_risk + rng.normal(0, 1.2, N)
    deadlines = np.clip(np.round(deadlines_raw).astype(int), 0, 10)

    # workload_tasks (0–12 int): high risk → more active tasks
    tasks_raw = 5.0 + 1.8 * latent_risk + rng.normal(0, 1.5, N)
    workload_tasks = np.clip(np.round(tasks_raw).astype(int), 0, 12)

    # workload_credits (0–80 in multiples of 20): more credits = more load
    base_probs = np.array([0.05, 0.15, 0.40, 0.30, 0.10])
    shift      = np.array([-0.02, -0.03, 0.0, 0.03, 0.02])
    modules = np.array([
        rng.choice(5, p=np.clip(base_probs + shift * latent_risk[i], 0.01, None) /
                   np.clip(base_probs + shift * latent_risk[i], 0.01, None).sum())
        for i in range(N)
    ])
    workload_credits = modules * 20

    # workload_hours (0–60 float): derived from tasks, ~3.5 hrs per task
    hours_raw = workload_tasks * 3.5 + 1.5 * latent_risk + rng.normal(0, 2.5, N)
    workload_hours = np.clip(np.round(hours_raw, 1), 0, 60)

    # availability_constraints (1–3 ordinal): 1=low restriction, 3=high restriction
    avail_raw = 2.0 + 0.6 * latent_risk + rng.normal(0, 0.5, N)
    availability_constraints = np.clip(np.round(avail_raw).astype(int), 1, 3)

    # quiz_score_avg (0–100 float): correlated with attendance and inverse risk
    quiz_raw = 65 - 10 * latent_risk + 0.25 * (attendance - 75) + rng.normal(0, 8, N)
    quiz_score_avg = np.clip(np.round(quiz_raw, 1), 0, 100)

    # ── STEP 3: Labels ────────────────────────────────────────────────────────

    # risk_level (0=Low, 1=Medium, 2=High)
    # Thresholds at ±0.43 sigma give roughly equal thirds
    risk_level = np.where(
        latent_risk < -0.43, 0,
        np.where(latent_risk < 0.43, 1, 2)
    )

    # success (0=fail, 1=pass)
    # Pass score combines latent risk, quiz score, and attendance.
    # Threshold chosen so ~70% of students pass (realistic).
    pass_score = (
        - 1.2 * latent_risk
        + 0.03 * (quiz_score_avg - 65)
        + 0.02 * (attendance - 75)
        + rng.normal(0, 0.4, N)
    )
    success = (pass_score > -0.3).astype(int)

    # ── STEP 4: Assemble DataFrame ────────────────────────────────────────────

    df = pd.DataFrame({
        "id":                       student_id,
        "gender":                   gender,
        "attendance":               attendance,
        "confidence_level":         confidence_level,
        "stress_level":             stress_level,
        "deadlines":                deadlines,
        "workload_tasks":           workload_tasks,
        "workload_credits":         workload_credits,
        "workload_hours":           workload_hours,
        "availability_constraints": availability_constraints,
        "quiz_score_avg":           quiz_score_avg,
        "success":                  success,
        "risk_level":               risk_level,
    })

    # ── STEP 5: Validation ────────────────────────────────────────────────────

    assert df.shape == (N, 13),                                 "Shape mismatch"
    assert df["id"].nunique() == N,                             "id not unique"
    assert set(df["gender"].unique()).issubset({0, 1}),         "gender OOR"
    assert df["attendance"].between(0, 100).all(),              "attendance OOR"
    assert df["confidence_level"].between(1, 5).all(),          "confidence OOR"
    assert df["stress_level"].between(1, 5).all(),              "stress OOR"
    assert df["deadlines"].between(0, 10).all(),                "deadlines OOR"
    assert df["workload_tasks"].between(0, 12).all(),           "tasks OOR"
    assert df["workload_credits"].isin([0,20,40,60,80]).all(),  "credits invalid"
    assert df["workload_hours"].between(0, 60).all(),           "hours OOR"
    assert df["availability_constraints"].between(1, 3).all(),  "availability OOR"
    assert df["quiz_score_avg"].between(0, 100).all(),          "quiz OOR"
    assert set(df["success"].unique()).issubset({0, 1}),        "success OOR"
    assert set(df["risk_level"].unique()).issubset({0, 1, 2}),  "risk_level OOR"
    assert df.isnull().sum().sum() == 0,                        "Unexpected nulls"

    # ── STEP 6: Save ──────────────────────────────────────────────────────────

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    return df


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the dataset CSV, generating it first if absent."""
    p = path or DATA_PATH
    if not p.exists():
        return generate_dataset()
    return pd.read_csv(p)


# =============================================================================
# Standalone runner  — mirrors the original step-by-step output exactly
# =============================================================================

if __name__ == "__main__":
    import os

    rng = np.random.default_rng(RANDOM_SEED)
    N   = 1000

    print("=" * 60)
    print("  Section 4 — Synthetic dataset generation")
    print("=" * 60)

    latent_risk = rng.normal(0, 1, N)
    print("\n[1/6] Latent risk score generated")

    student_id   = np.arange(1, N + 1)
    gender       = rng.integers(0, 2, N)
    attendance   = np.clip(np.round(75 - 12*latent_risk + rng.normal(0,8,N), 1), 0, 100)
    conf_raw     = 3.2 - 0.7*latent_risk + rng.normal(0, 0.6, N)
    confidence_level = np.clip(np.round(conf_raw).astype(int), 1, 5)
    stress_raw   = 2.8 + 0.8*latent_risk + rng.normal(0, 0.5, N)
    stress_level = np.clip(np.round(stress_raw).astype(int), 1, 5)
    deadlines    = np.clip(np.round(4.0 + 1.5*latent_risk + rng.normal(0,1.2,N)).astype(int), 0, 10)
    tasks_raw    = 5.0 + 1.8*latent_risk + rng.normal(0, 1.5, N)
    workload_tasks = np.clip(np.round(tasks_raw).astype(int), 0, 12)
    base_probs   = np.array([0.05, 0.15, 0.40, 0.30, 0.10])
    shift        = np.array([-0.02, -0.03, 0.0, 0.03, 0.02])
    modules      = np.array([
        rng.choice(5, p=np.clip(base_probs + shift*latent_risk[i], 0.01, None) /
                   np.clip(base_probs + shift*latent_risk[i], 0.01, None).sum())
        for i in range(N)
    ])
    workload_credits = modules * 20
    hours_raw    = workload_tasks*3.5 + 1.5*latent_risk + rng.normal(0, 2.5, N)
    workload_hours = np.clip(np.round(hours_raw, 1), 0, 60)
    avail_raw    = 2.0 + 0.6*latent_risk + rng.normal(0, 0.5, N)
    availability_constraints = np.clip(np.round(avail_raw).astype(int), 1, 3)
    quiz_raw     = 65 - 10*latent_risk + 0.25*(attendance - 75) + rng.normal(0, 8, N)
    quiz_score_avg = np.clip(np.round(quiz_raw, 1), 0, 100)
    print("[2/6] All feature columns generated")

    risk_level = np.where(latent_risk < -0.43, 0, np.where(latent_risk < 0.43, 1, 2))
    pass_score = (-1.2*latent_risk + 0.03*(quiz_score_avg-65)
                  + 0.02*(attendance-75) + rng.normal(0, 0.4, N))
    success = (pass_score > -0.3).astype(int)
    print("[3/6] Labels derived")
    print(f"      risk_level  — Low:{(risk_level==0).sum()}  "
          f"Med:{(risk_level==1).sum()}  High:{(risk_level==2).sum()}")
    print(f"      success     — Pass:{success.sum()}  Fail:{(1-success).sum()}")

    df = pd.DataFrame({
        "id": student_id, "gender": gender, "attendance": attendance,
        "confidence_level": confidence_level, "stress_level": stress_level,
        "deadlines": deadlines, "workload_tasks": workload_tasks,
        "workload_credits": workload_credits, "workload_hours": workload_hours,
        "availability_constraints": availability_constraints,
        "quiz_score_avg": quiz_score_avg, "success": success,
        "risk_level": risk_level,
    })
    print(f"\n[4/6] DataFrame assembled — shape: {df.shape}")

    print("\n[5/6] Running validation checks...")
    assert df.shape == (1000, 13)
    assert df["id"].nunique() == 1000
    assert set(df["gender"].unique()).issubset({0, 1})
    assert df["attendance"].between(0, 100).all()
    assert df["confidence_level"].between(1, 5).all()
    assert df["stress_level"].between(1, 5).all()
    assert df["deadlines"].between(0, 10).all()
    assert df["workload_tasks"].between(0, 12).all()
    assert df["workload_credits"].isin([0,20,40,60,80]).all()
    assert df["workload_hours"].between(0, 60).all()
    assert df["availability_constraints"].between(1, 3).all()
    assert df["quiz_score_avg"].between(0, 100).all()
    assert set(df["success"].unique()).issubset({0, 1})
    assert set(df["risk_level"].unique()).issubset({0, 1, 2})
    assert df.isnull().sum().sum() == 0
    print("      Schema checks passed ✓")

    corr = df[["attendance","confidence_level","stress_level","deadlines",
               "workload_tasks","workload_hours","quiz_score_avg","risk_level"]].corr()
    checks = [
        ("attendance","risk_level","-"), ("confidence_level","risk_level","-"),
        ("stress_level","risk_level","+"), ("deadlines","risk_level","+"),
        ("workload_hours","risk_level","+"), ("quiz_score_avg","risk_level","-"),
        ("attendance","quiz_score_avg","+"), ("workload_tasks","workload_hours","+"),
    ]
    corr_errors = []
    for a, b, sign in checks:
        r   = corr.loc[a, b]
        ok  = (r > 0.05) if sign == "+" else (r < -0.05)
        tick = "✓" if ok else "✗"
        print(f"      {tick}  {a:<28} vs {b:<20} r={r:+.3f}  (expected {sign})")
        if not ok:
            corr_errors.append(f"{a} vs {b} = {r:.3f}")
    if corr_errors:
        print(f"\n  WARNING: {len(corr_errors)} correlation issue(s) found.")
    else:
        print("\n      All correlation checks passed ✓")

    print("\n[6/6] Summary statistics")
    print("-" * 60)
    summary = df.describe().T[["count","mean","std","min","max"]]
    summary["mean"] = summary["mean"].round(2)
    summary["std"]  = summary["std"].round(2)
    print(summary.to_string())

    os.makedirs("ml", exist_ok=True)
    csv_path = "ml/student_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path}  ({os.path.getsize(csv_path):,} bytes)")
    print("\n" + "=" * 60)
    print("  Dataset generation complete. No errors.")
    print("=" * 60)
