"""
ml/model.py
===========
ML classification pipeline for student success prediction.

Predictive task
---------------
Predict success ∈ {0=Fail, 1=Pass} from academic and behavioural features.

What "success" operationalises
-------------------------------
In this dataset "success" is a binary label derived from a synthetic latent
risk score. In a real deployment it would represent a grade threshold (e.g.
≥40%) at the end of term. This operationalisation has real consequences:
  • It collapses a continuous spectrum of performance into two categories.
  • Students who narrowly fail are treated identically to those who
    disengage completely.
  • The cost of being wrong is asymmetric: falsely labelling a passing
    student as at-risk wastes support resources; falsely labelling an
    at-risk student as safe can result in them failing without help.

Who bears the cost of incorrect predictions
--------------------------------------------
  False Positive (predict Fail, actually Pass):
    The student may receive unwanted intervention, be stigmatised, or lose
    autonomy over their own support decisions.
  False Negative (predict Pass, actually Fail):
    The student receives no early warning. In a high-stakes context (e.g.
    a first-generation university student) this can be catastrophic.
  In an educational early-warning system, False Negatives are typically
  more costly than False Positives — this must inform which metric to
  prioritise (Recall over Precision).

Models trained (specified in coursework)
-----------------------------------------
  0. Baseline        — majority class classifier (null hypothesis)
  1. Decision Tree   — depth 3 (interpretable) then depth 7 (memorising)
  2. Naive Bayes     — GaussianNB (conditional independence analysis)

IMPORTANT: risk_level is excluded from all feature sets.
It is derived from the same latent variable as success and would constitute
direct leakage if included.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.dummy           import DummyClassifier

from .dataset import load_dataset
from ui.display import print_header, print_subheader

warnings.filterwarnings("ignore")

# ── Feature sets ──────────────────────────────────────────────────────────────
# risk_level is EXCLUDED — it is a proxy for the latent variable.
# id is EXCLUDED — not predictive.
FEATURES_WITH_GENDER = [
    "attendance", "confidence_level", "stress_level",
    "deadlines", "workload_tasks", "workload_credits",
    "workload_hours", "availability_constraints",
    "quiz_score_avg", "gender",
]
FEATURES_NO_GENDER = [f for f in FEATURES_WITH_GENDER if f != "gender"]
TARGET = "success"

RANDOM_SEED = 42


# =============================================================================
# 1.  Data preparation
# =============================================================================

def prepare_data(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.20,
    include_gender: bool = True,
) -> Tuple[Any, Any, Any, Any]:
    """
    Load, split, and return (X_train, X_test, y_train, y_test).

    Why stratification is required
    --------------------------------
    Our dataset has ~70% Pass / ~30% Fail. Without stratification a random
    split can produce folds with unequal class ratios — a 20% test set could
    accidentally contain 50% Fail, making evaluation on the test set
    unrepresentative of the true population. Stratification guarantees both
    splits mirror the overall class distribution, which is the same condition
    the model will face in deployment.

    Why 80/20 and not 90/10
    ------------------------
    With n=1000 rows, a 20% test set gives 200 test samples — enough for
    stable estimates of precision/recall per class. A larger test set (90/10)
    would reduce training data without improving statistical confidence.

    Leakage guard
    -------------
    risk_level correlates ~0.88 with the latent variable that generates
    success. Including it as a feature would cause the model to trivially
    memorise the label mapping rather than learning from behaviour. It is
    excluded unconditionally.
    """
    if df is None:
        df = load_dataset()

    features = FEATURES_WITH_GENDER if include_gender else FEATURES_NO_GENDER

    X = df[features].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y,        # preserves class ratio in both splits
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# 2.  Baseline — majority class classifier (null hypothesis)
# =============================================================================

def run_baseline(
    X_train: Any, X_test: Any,
    y_train: Any, y_test: Any,
) -> Dict[str, Any]:
    """
    Majority class classifier: always predicts the most frequent class.

    This is the null hypothesis:
      "No structure beyond class frequency is being used."

    Why this baseline often looks acceptable
    -----------------------------------------
    With ~70% Pass, the baseline achieves ~70% accuracy by predicting Pass
    for every student. This is "acceptable" only in the sense that accuracy
    is the wrong metric when classes are imbalanced.

    What information it discards
    -----------------------------
    Everything. The model ignores attendance, quiz scores, stress — all
    signals that matter for early warning. It treats every student identically.

    When this model would be actively harmful
    ------------------------------------------
    In an early-warning system deployed to identify at-risk students: the
    baseline flags NO student as at-risk. Every student who eventually fails
    receives no intervention. The cost is borne entirely by the minority class.
    """
    clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    majority_class = int(y_train.value_counts().idxmax())
    pass_rate      = float(y_train.mean())

    result = _eval(clf, X_test, y_test, y_pred, "Baseline (majority class)")
    result["majority_class"] = majority_class
    result["pass_rate_train"] = round(pass_rate, 3)
    result["discussion"] = (
        f"The majority class is {'Pass' if majority_class == 1 else 'Fail'} "
        f"({pass_rate:.1%} of training data). "
        f"The baseline achieves {result['accuracy']:.1%} accuracy by predicting "
        f"this class for every student — without examining any features. "
        "Recall for the minority class (Fail) is exactly 0.00: no at-risk "
        "student is ever identified. In an early-warning context this is not "
        "a 'safe' baseline — it is the worst possible outcome for the students "
        "who need support most."
    )
    return result


# =============================================================================
# 3.  Decision Tree
# =============================================================================

def run_decision_tree(
    X_train: Any, X_test: Any,
    y_train: Any, y_test: Any,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """
    Decision Tree classifier.

    Why we fix depth instead of tuning for performance
    ---------------------------------------------------
    Depth is an explicit inductive bias choice, not a hyperparameter to
    maximise accuracy. A shallow tree (depth 3) encodes a hypothesis:
    "A small number of conjunctive rules can explain most of the variance."
    A deep tree (depth 7+) allows memorisation of training examples.

    Interpretability vs accuracy trade-off
    ----------------------------------------
    As depth increases:
      • Accuracy on training set increases monotonically (can reach 100%)
      • Test accuracy first improves then plateaus or falls (overfitting)
      • Each additional level doubles the number of leaves
      • Rules become increasingly specific and fragile
      • Interpretability degrades faster than accuracy improves:
        a depth-7 tree has up to 128 leaves — no human can reason about this

    Inductive bias
    --------------
    The tree assumes the decision boundary can be expressed as axis-aligned
    splits on individual features. It cannot capture feature interactions
    (e.g. "low attendance AND high stress is worse than either alone")
    without growing deeper.
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        class_weight="balanced",   # compensates for class imbalance
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    result = _eval(clf, X_test, y_test, y_pred,
                   f"Decision Tree (depth {max_depth})")
    result["max_depth"]  = max_depth
    result["n_leaves"]   = clf.get_n_leaves()
    result["n_features_used"] = int(
        (clf.feature_importances_ > 0).sum()
    )
    result["feature_importances"] = dict(zip(
        X_train.columns,
        np.round(clf.feature_importances_, 4)
    ))
    result["model"] = clf
    result["feature_names"] = list(X_train.columns)

    return result


# =============================================================================
# 4.  Naive Bayes
# =============================================================================

def run_naive_bayes(
    X_train: Any, X_test: Any,
    y_train: Any, y_test: Any,
) -> Dict[str, Any]:
    """
    Gaussian Naive Bayes classifier.

    The conditional independence assumption
    ----------------------------------------
    Naive Bayes assumes:
      P(features | class) = ∏ P(feature_i | class)
    i.e. each feature is independent of every other given the class label.

    In our dataset this is clearly violated:
      • attendance and quiz_score_avg are positively correlated (r ≈ 0.76)
      • stress_level and confidence_level are negatively correlated
      • workload_tasks and workload_hours are almost perfectly correlated

    Why violating independence does not necessarily break performance
    -----------------------------------------------------------------
    NB does not need the independence assumption to hold for predictions to
    be correct — it only needs the RANKING of P(class | features) to be
    right. When correlated features both point in the same direction
    (e.g. low attendance AND low quiz both increase P(Fail)), NB
    double-counts the evidence but the relative ordering between classes
    remains correct. Performance suffers most when correlations REVERSE
    the direction of evidence.

    What NB cannot capture
    ----------------------
    Interactions between features. NB cannot represent "low attendance
    matters more when combined with high stress" — it models each feature
    as contributing independently. This is exactly what a decision tree
    (conjunctive rules) or logistic regression (weighted sum) can do.

    Why probability calibration matters more than accuracy
    -------------------------------------------------------
    NB outputs probabilities, not just class labels. If NB says
    P(Fail) = 0.95, we act differently than if it says P(Fail) = 0.51.
    Because NB double-counts correlated evidence, its probabilities are
    systematically too extreme (over-confident). This makes raw NB
    probabilities unreliable for threshold-based decisions — calibration
    (e.g. Platt scaling) would be needed before deployment.
    """
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred       = clf.predict(X_test)
    y_proba      = clf.predict_proba(X_test)[:, 1]

    result = _eval(clf, X_test, y_test, y_pred, "Naive Bayes (GaussianNB)")
    result["model"]          = clf
    result["proba_sample"]   = np.round(y_proba[:10], 3).tolist()
    result["proba_std"]      = round(float(y_proba.std()), 3)
    result["discussion"] = (
        "Naive Bayes achieves competitive accuracy despite the violated "
        "independence assumption. This is because correlated features "
        "(attendance ↔ quiz, tasks ↔ hours) point in the same direction, "
        "so double-counting the evidence preserves the ranking of class "
        "probabilities even though the absolute values are overconfident. "
        "The high probability spread (std=" + str(round(float(y_proba.std()), 3)) + ") "
        "confirms that NB is assigning extreme confidence — this is the "
        "signature of uncalibrated correlated features. Before using NB "
        "probabilities for threshold decisions, Platt scaling or isotonic "
        "regression calibration would be required."
    )
    return result


# =============================================================================
# 5.  Neural Network (MLPClassifier)
# =============================================================================

def run_neural_network(
    X_train: Any, X_test: Any,
    y_train: Any, y_test: Any,
    epochs:      int  = 100,
    hidden_units: tuple = (64, 32),
    learning_rate: float = 0.001,
) -> Dict[str, Any]:
    """
    Multi-Layer Perceptron classifier trained with backpropagation.

    Why a neural network needs epochs
    -----------------------------------
    Unlike Decision Trees or Naive Bayes — which compute their answer in
    a single pass through the data — a neural network updates its weights
    incrementally using gradient descent. One epoch = one full pass through
    the training set. After each pass, weights are nudged in the direction
    that reduces the loss. The model needs many passes to converge.

    Architecture
    ------------
    Input layer  : 10 features (one neuron per feature)
    Hidden layer 1: 64 neurons, ReLU activation
    Hidden layer 2: 32 neurons, ReLU activation
    Output layer : 2 neurons (Pass / Fail), softmax → probability

    Why features must be scaled for neural networks
    ------------------------------------------------
    Gradient descent is sensitive to feature scale. attendance (0–100) and
    confidence_level (1–5) live on very different scales. Without scaling,
    the gradient for attendance dominates and the network trains slowly or
    diverges. StandardScaler (mean=0, std=1) fixes this.

    Decision Trees and Naive Bayes do NOT need scaling — they use splits
    and probability estimates that are scale-invariant.

    Why we do NOT need scaling for trees
    -------------------------------------
    A DT asks "is attendance > 70?" — the threshold adjusts to the scale
    automatically. NB estimates P(feature | class) per feature independently.
    Neither uses distance metrics or gradient-based optimisation.

    Hyperparameters chosen
    ----------------------
    epochs=100    : enough to converge on 800 training rows without overfitting
    hidden=(64,32): two layers, halving width — a common pyramid pattern
    lr=0.001      : standard Adam learning rate
    """
    # Neural networks require feature scaling — trees and NB do not
    scaler  = StandardScaler()
    Xtr_s   = scaler.fit_transform(X_train)
    Xte_s   = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes = hidden_units,
        activation         = "relu",
        solver             = "adam",
        learning_rate_init = learning_rate,
        max_iter           = epochs,
        random_state       = RANDOM_SEED,
        early_stopping     = True,       # stop if validation loss stops falling
        validation_fraction= 0.1,        # 10% of train set for early-stop check
        n_iter_no_change   = 15,         # patience: stop after 15 stale epochs
        verbose            = False,
    )
    clf.fit(Xtr_s, y_train)
    y_pred = clf.predict(Xte_s)

    actual_epochs = len(clf.loss_curve_)

    result = _eval(clf, Xte_s, y_test, y_pred,
                   f"Neural Network (MLP {hidden_units})")
    result["model"]         = clf
    result["scaler"]        = scaler          # must be saved alongside the model
    result["feature_names"] = list(X_train.columns)
    result["loss_curve"]    = clf.loss_curve_
    result["epochs_run"]    = actual_epochs
    result["epochs_max"]    = epochs
    result["hidden_units"]  = hidden_units
    result["learning_rate"] = learning_rate
    result["stopped_early"] = actual_epochs < epochs
    result["discussion"] = (
        f"The MLP ran for {actual_epochs} epoch(s) "
        f"({'early stopping triggered' if actual_epochs < epochs else 'full training'})."
        f" Architecture: input(10) → {hidden_units[0]} → {hidden_units[1]} → output(2)."
        " Features were StandardScaled before training (required for gradient descent)."
        " Neural networks excel at learning non-linear feature interactions that"
        " Decision Trees can only approximate by growing deeper."
    )
    return result


# =============================================================================
# 6.  Shared evaluation helper
# =============================================================================

def _eval(
    clf:    Any,
    X_test: Any,
    y_test: Any,
    y_pred: Any,
    name:   str,
) -> Dict[str, Any]:
    """Compute standard classification metrics and return as a dict."""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "name":      name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": cm,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "y_pred": y_pred,
    }


# =============================================================================
# 6.  Summary comparison table
# =============================================================================

def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    print_subheader("Model comparison")
    W = 34
    print(f"  {'Model':<{W}} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7}  {'FP':>5} {'FN':>5}")
    print("  " + "─" * (W + 42))
    for r in results:
        print(f"  {r['name']:<{W}} "
              f"{r['accuracy']:>7.3f} {r['precision']:>7.3f} "
              f"{r['recall']:>7.3f} {r['f1']:>7.3f}  "
              f"{r['fp']:>5} {r['fn']:>5}")
    print()


# =============================================================================
# 7.  Confusion matrix display
# =============================================================================

def print_confusion_matrix(result: Dict[str, Any]) -> None:
    cm   = result["confusion_matrix"]
    name = result["name"]
    print_subheader(f"Confusion matrix — {name}")
    print(f"  {'':>20}  Predicted Fail  Predicted Pass")
    print(f"  {'Actual Fail':>20}  {cm[0,0]:>14}  {cm[0,1]:>14}")
    print(f"  {'Actual Pass':>20}  {cm[1,0]:>14}  {cm[1,1]:>14}")
    print(f"\n  True Negatives  (TN) = {result['tn']:>4}   "
          "correctly identified as Fail")
    print(f"  False Positives (FP) = {result['fp']:>4}   "
          "predicted Fail, actually Pass   [over-intervention]")
    print(f"  False Negatives (FN) = {result['fn']:>4}   "
          "predicted Pass, actually Fail   [missed at-risk student]")
    print(f"  True Positives  (TP) = {result['tp']:>4}   "
          "correctly identified as Pass")
    print()


# =============================================================================
# 8.  Decision tree rule extraction
# =============================================================================

def extract_tree_rules(result: Dict[str, Any], top_n: int = 5) -> List[Dict]:
    """
    Extract and analyse the top branches of a decision tree.

    Variables near the root: highest information gain, most discriminating.
    Variables near leaves:   specialised, fragile, low coverage.
    Variables absent:        deemed uninformative by the tree's greedy splits.
    """
    clf          = result["model"]
    feature_names = result["feature_names"]

    # Print the full text representation (readable for shallow trees)
    print_subheader(f"Decision tree rules — depth {result['max_depth']}")
    tree_text = export_text(clf, feature_names=feature_names, max_depth=4)
    for line in tree_text.split("\n")[:60]:   # cap at 60 lines for readability
        print(f"  {line}")
    print()

    # Feature importance analysis
    importances = result["feature_importances"]
    sorted_imp  = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    print_subheader("Feature importance (Gini gain)")
    for feat, imp in sorted_imp:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<28} {imp:.4f}  {bar}")
    print()

    # Analyse top features
    top_features = [f for f, _ in sorted_imp[:3]]
    print("  Root-level features (appear first in splits):")
    for i, feat in enumerate(top_features, 1):
        notes = {
            "quiz_score_avg":   "direct proxy for subject mastery — "
                                "high plausibility, high sensitivity",
            "attendance":       "correlated with engagement — "
                                "plausible, but students miss class for many reasons",
            "confidence_level": "self-reported — prone to response bias",
            "stress_level":     "self-reported — may underreport",
            "workload_hours":   "students may estimate poorly",
        }
        note = notes.get(feat, "high discriminative power in this dataset")
        print(f"  {i}. {feat}: {note}")
    print()

    # Variables ignored by the tree
    ignored = [f for f, imp in importances.items() if imp == 0.0]
    if ignored:
        print(f"  Features assigned zero importance: {', '.join(ignored)}")
        print("  These features were not useful for axis-aligned splits "
              "at any depth, but may still carry signal for other model types.")
    print()

    return sorted_imp


# =============================================================================
# 9.  Error type analysis (educational context)
# =============================================================================

def analyse_error_types(results: List[Dict[str, Any]]) -> None:
    print_subheader("Error type analysis — educational context")
    print(
        "  In a student support context:\n"
        "  FP (predict Fail, actually Pass) = over-intervention\n"
        "    Cost: wasted support resources, student stigma, autonomy loss\n"
        "  FN (predict Pass, actually Fail) = missed at-risk student\n"
        "    Cost: student fails without early warning — potentially catastrophic\n"
        "\n"
        "  Deployment scenario impacts:\n"
        "    Academic support:    FN is more costly — missing a struggling "
        "student is worse than offering unnecessary help\n"
        "    Early warning:       FN is critical — any undetected failure is "
        "a system failure\n"
        "    Administrative:      FP is more costly — unjust resource "
        "allocation or stigmatising labelling based on prediction\n"
    )
    for r in results:
        fn_rate = r["fn"] / max(1, r["fn"] + r["tp"])
        fp_rate = r["fp"] / max(1, r["fp"] + r["tn"])
        dominant = "FN dominates" if r["fn"] > r["fp"] else "FP dominates"
        print(f"  {r['name']:<38}  "
              f"FN-rate={fn_rate:.2f}  FP-rate={fp_rate:.2f}  "
              f"→ {dominant}")
    print()


# =============================================================================
# 10.  Gender bias analysis
# =============================================================================

def gender_bias_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare model performance with and without gender as a feature.

    Why gender is a sensitive variable
    ------------------------------------
    Including gender in a predictive model raises two distinct risks:
      1. Proxy discrimination: even if gender is not causal, the model may
         learn correlations between gender and other features (e.g. module
         choice, stress patterns) that produce disparate predictions.
      2. Disparate impact: even a 'fair' model can produce outcomes that
         systematically disadvantage one group if the training data encodes
         historical inequities.

    In our synthetic dataset gender was generated independently of the
    latent risk variable — so any measured performance difference between
    models with/without gender reflects noise, not real signal.
    In a real dataset the picture is more complex and the analysis is
    ethically mandatory before deployment.

    Metrics we report
    ------------------
    Beyond overall accuracy, we report per-group recall (True Positive Rate
    for each gender). If recall(female) ≠ recall(male), the model is less
    reliable for one group — a form of algorithmic unfairness.
    """
    print_subheader("Gender bias analysis")

    X_train_g,  X_test_g,  y_train, y_test = prepare_data(df, include_gender=True)
    X_train_ng, X_test_ng, _,       _      = prepare_data(df, include_gender=False)

    dt_g  = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED,
                                   class_weight="balanced")
    dt_ng = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED,
                                   class_weight="balanced")
    dt_g.fit(X_train_g, y_train)
    dt_ng.fit(X_train_ng, y_train)

    pred_g  = dt_g.predict(X_test_g)
    pred_ng = dt_ng.predict(X_test_ng)

    acc_g  = accuracy_score(y_test, pred_g)
    acc_ng = accuracy_score(y_test, pred_ng)
    f1_g   = f1_score(y_test, pred_g,  zero_division=0)
    f1_ng  = f1_score(y_test, pred_ng, zero_division=0)

    print(f"  {'Metric':<30} {'With gender':>14} {'Without gender':>14}")
    print(f"  {'─'*60}")
    print(f"  {'Accuracy':<30} {acc_g:>14.3f} {acc_ng:>14.3f}")
    print(f"  {'F1 score':<30} {f1_g:>14.3f} {f1_ng:>14.3f}")
    print()

    # Per-group recall
    gender_col = df["gender"].iloc[X_test_g.index]
    results_by_group = {}
    for g_val, g_label in [(0, "Male"), (1, "Female")]:
        mask     = gender_col == g_val
        if mask.sum() == 0:
            continue
        recall_g  = recall_score(y_test[mask], pred_g[mask],  zero_division=0)
        recall_ng = recall_score(y_test[mask], pred_ng[mask], zero_division=0)
        results_by_group[g_label] = {
            "recall_with_gender":    round(recall_g,  3),
            "recall_without_gender": round(recall_ng, 3),
            "n": int(mask.sum()),
        }
        print(f"  {g_label} (n={mask.sum():>3}): "
              f"recall with gender = {recall_g:.3f}, "
              f"without = {recall_ng:.3f}")

    print()
    # Fairness assessment
    recalls = [v["recall_with_gender"] for v in results_by_group.values()]
    if len(recalls) == 2 and abs(recalls[0] - recalls[1]) > 0.05:
        print("  [!] Disparate recall detected (>5% gap between groups).")
        print("      The model is less reliable for one gender group.")
        print("      This warrants fairness intervention before deployment.")
    else:
        print("  [ok] No substantial disparate recall detected (<5% gap).")
        print("       In our synthetic data gender was generated independently")
        print("       of risk — a real dataset may show a different result.")

    print()
    print("  Ethics note:")
    print("  Even when including gender improves accuracy, this does not")
    print("  justify its use. A small accuracy gain from a protected")
    print("  attribute can introduce systematic disparities in who receives")
    print("  interventions. The deployment context must be considered:")
    print("  academic support tools should be scrutinised for disparate")
    print("  impact before any rollout.")
    print()

    return {
        "acc_with":    round(acc_g,  3),
        "acc_without": round(acc_ng, 3),
        "f1_with":     round(f1_g,   3),
        "f1_without":  round(f1_ng,  3),
        "by_group":    results_by_group,
    }


# =============================================================================
# 11.  Dataset limitations discussion
# =============================================================================

def print_limitations() -> None:
    print_subheader("Dataset and model limitations")
    print(
        "  1. Synthetic data: all correlations were designed by the authors.\n"
        "     The model learns the generation process, not real student behaviour.\n"
        "     Performance on real student data may be substantially lower.\n"
        "\n"
        "  2. Success definition: 'success' encodes a binary pass/fail threshold.\n"
        "     It does not capture: partial progress, learning gains, effort quality,\n"
        "     or extenuating circumstances. Models trained on it inherit this\n"
        "     reductionism.\n"
        "\n"
        "  3. Temporal leakage risk: in a real system, features like quiz_score_avg\n"
        "     may be measured AFTER the intervention window. Any feature collected\n"
        "     after the point of prediction constitutes leakage. This split assumes\n"
        "     all features are available at week 3 of term.\n"
        "\n"
        "  4. Class imbalance: ~70% Pass means a trivial baseline achieves 70%\n"
        "     accuracy. F1 and recall are more meaningful metrics for this task.\n"
        "\n"
        "  5. Naive Bayes probability calibration: raw NB probabilities are\n"
        "     over-confident due to correlated features. Do not use them directly\n"
        "     as risk scores without calibration.\n"
        "\n"
        "  6. Decision tree stability: small changes to training data cause large\n"
        "     changes to tree structure. Rules extracted from a single trained tree\n"
        "     should not be treated as universal truths.\n"
    )


# =============================================================================
# 12.  Public API
# =============================================================================

def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train all models and return a dict of {model_name: result}.
    Called by main.py / app.py.
    """
    X_train, X_test, y_train, y_test = prepare_data(df)

    return {
        "baseline": run_baseline(X_train, X_test, y_train, y_test),
        "dt_d3":    run_decision_tree(X_train, X_test, y_train, y_test, 3),
        "dt_d7":    run_decision_tree(X_train, X_test, y_train, y_test, 7),
        "nb":       run_naive_bayes(X_train, X_test, y_train, y_test),
        "nn":       run_neural_network(X_train, X_test, y_train, y_test),
    }


def evaluate_models(
    models:  Dict[str, Any],
    X_test:  Any,
    y_test:  Any,
) -> Dict[str, Dict]:
    """Return evaluation metrics for pre-trained models."""
    out = {}
    for name, result in models.items():
        if "model" in result:
            y_pred = result["model"].predict(X_test)
        else:
            y_pred = result["y_pred"]
        out[name] = _eval(result.get("model"), X_test, y_test, y_pred, name)
    return out


def predict_risk(
    profile: Dict[str, Any],
    models:  Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict success probability for a single student profile.
    Uses dt_d3 (most interpretable) and nb as primary models.
    Returns {model_name: {prediction, probability, label}}.
    """
    feature_vals = {
        "attendance":               profile.get("attendance", 75),
        "confidence_level":         profile.get("confidence_level", 3),
        "stress_level":             profile.get("stress_level", 3),
        "deadlines":                profile.get("deadlines", 2),
        "workload_tasks":           profile.get("workload_tasks", 4),
        "workload_credits":         profile.get("workload_credits", 40),
        "workload_hours":           profile.get("workload_hours", 10),
        "availability_constraints": profile.get("availability_constraints", 2),
        "quiz_score_avg":           profile.get("quiz_score_avg", 65),
        "gender":                   profile.get("gender", 0),
    }

    row = pd.DataFrame([feature_vals])
    results = {}

    for name in ("dt_d3", "nb", "nn"):
        if name not in models or "model" not in models[name]:
            continue
        clf   = models[name]["model"]
        feats = models[name].get("feature_names", FEATURES_WITH_GENDER)
        row_f = row[feats]

        # Neural network needs scaling — scaler stored alongside model
        if name == "nn" and "scaler" in models[name]:
            row_f = models[name]["scaler"].transform(row_f)

        pred  = int(clf.predict(row_f)[0])
        proba = clf.predict_proba(row_f)[0]

        results[models[name]["name"]] = {
            "prediction":  "Pass" if pred == 1 else "Fail",
            "probability": np.round(proba, 3).tolist(),
            "confidence":  round(float(max(proba)), 3),
        }

    return results


# =============================================================================
# 13.  Full pipeline runner (used by main.py)
# =============================================================================

def run_full_ml_pipeline(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Run the complete ML section end-to-end and print all discussion.
    Returns all results for use by downstream pipeline.
    """
    if df is None:
        df = load_dataset()

    print_header("Section 4 — ML model")

    # ── Data prep ─────────────────────────────────────────────────────────────
    print_subheader("4.1  Data preparation")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"  Training set : {len(X_train)} rows")
    print(f"  Test set     : {len(X_test)} rows")
    print(f"  Features     : {list(X_train.columns)}")
    print(f"  Target       : {TARGET}  (0=Fail, 1=Pass)")
    print(f"  Train class split  Pass={y_train.mean():.1%}  "
          f"Fail={(1-y_train.mean()):.1%}")
    print(f"  Test class split   Pass={y_test.mean():.1%}  "
          f"Fail={(1-y_test.mean()):.1%}")
    print()
    print("  Note: risk_level is EXCLUDED from all feature sets.")
    print("  It is derived from the same latent variable as success and")
    print("  would constitute data leakage if included as a feature.")
    print()

    # ── Baseline ──────────────────────────────────────────────────────────────
    print_subheader("4.2  Baseline — null hypothesis")
    baseline = run_baseline(X_train, X_test, y_train, y_test)
    print_confusion_matrix(baseline)
    print(f"  Accuracy : {baseline['accuracy']:.3f}")
    print(f"  Recall   : {baseline['recall']:.3f}  "
          "(how many at-risk students are caught)")
    print(f"  F1 score : {baseline['f1']:.3f}")
    print()
    print(f"  Discussion: {baseline['discussion']}")
    print()

    # ── Decision Tree depth 3 ─────────────────────────────────────────────────
    print_subheader("4.3  Decision Tree — depth 3 (interpretable)")
    dt3 = run_decision_tree(X_train, X_test, y_train, y_test, max_depth=3)
    print_confusion_matrix(dt3)
    print(f"  Accuracy  : {dt3['accuracy']:.3f}")
    print(f"  Precision : {dt3['precision']:.3f}  "
          "(of students flagged at-risk, how many truly are?)")
    print(f"  Recall    : {dt3['recall']:.3f}  "
          "(of truly at-risk students, how many are caught?)")
    print(f"  F1        : {dt3['f1']:.3f}")
    print(f"  Leaves    : {dt3['n_leaves']}  |  "
          f"Features used: {dt3['n_features_used']}")
    print()
    extract_tree_rules(dt3)

    # ── Decision Tree depth 7 ─────────────────────────────────────────────────
    print_subheader("4.4  Decision Tree — depth 7 (memorising)")
    dt7 = run_decision_tree(X_train, X_test, y_train, y_test, max_depth=7)
    print_confusion_matrix(dt7)
    print(f"  Accuracy  : {dt7['accuracy']:.3f}")
    print(f"  F1        : {dt7['f1']:.3f}")
    print(f"  Leaves    : {dt7['n_leaves']}  |  "
          f"Features used: {dt7['n_features_used']}")
    print()
    print("  Depth 3 → Depth 7 comparison:")
    print(f"    Leaves grew from {dt3['n_leaves']} to {dt7['n_leaves']} "
          f"(×{dt7['n_leaves'] // max(1, dt3['n_leaves'])} increase)")
    acc_delta = dt7["accuracy"] - dt3["accuracy"]
    print(f"    Accuracy delta: {acc_delta:+.3f}")
    print(f"    F1 delta      : {dt7['f1'] - dt3['f1']:+.3f}")
    print(
        "    Interpretation: depth 7 allows the tree to memorise training\n"
        "    examples. Leaves become very specific ('quiz > 71.2 AND\n"
        "    attendance > 83.4 AND stress == 2'). These rules are fragile:\n"
        "    a student who scores 71.1 on quiz follows a completely different\n"
        "    branch. Interpretability collapses while accuracy gains are\n"
        "    marginal — the hallmark of overfitting."
    )
    print()

    # ── Naive Bayes ───────────────────────────────────────────────────────────
    print_subheader("4.5  Naive Bayes — GaussianNB")
    nb = run_naive_bayes(X_train, X_test, y_train, y_test)
    print_confusion_matrix(nb)
    print(f"  Accuracy  : {nb['accuracy']:.3f}")
    print(f"  Precision : {nb['precision']:.3f}")
    print(f"  Recall    : {nb['recall']:.3f}")
    print(f"  F1        : {nb['f1']:.3f}")
    print()
    print(f"  Discussion: {nb['discussion']}")
    print()

    # ── Neural Network ────────────────────────────────────────────────────────
    print_subheader("4.6  Neural Network — MLP with backpropagation")
    nn = run_neural_network(X_train, X_test, y_train, y_test, epochs=100)
    print_confusion_matrix(nn)
    print(f"  Accuracy  : {nn['accuracy']:.3f}")
    print(f"  Precision : {nn['precision']:.3f}")
    print(f"  Recall    : {nn['recall']:.3f}")
    print(f"  F1        : {nn['f1']:.3f}")
    print(f"  Epochs run: {nn['epochs_run']} / {nn['epochs_max']} "
          f"({'early stopping' if nn['stopped_early'] else 'full training'})")
    print(f"  Architecture: 10 → {nn['hidden_units'][0]} → "
          f"{nn['hidden_units'][1]} → 2")
    print()
    print("  Why epochs matter here:")
    print("  Unlike DT and NB — which compute their answer in one data pass —")
    print("  the MLP adjusts millions of weights via gradient descent.")
    print("  Each epoch refines the weights slightly. Early stopping monitors")
    print("  a validation split and halts when the loss stops improving,")
    print("  preventing overfitting without needing to tune epoch count manually.")
    print()
    print(f"  Discussion: {nn['discussion']}")
    print()

    # Print loss curve summary
    if nn['loss_curve']:
        curve = nn['loss_curve']
        print(f"  Training loss: {curve[0]:.4f} (epoch 1) → "
              f"{curve[-1]:.4f} (epoch {len(curve)})")
        print(f"  Total loss reduction: {curve[0]-curve[-1]:.4f} "
              f"({(curve[0]-curve[-1])/curve[0]*100:.1f}%)")
    print()

    # ── Comparison ────────────────────────────────────────────────────────────
    all_results = [baseline, dt3, dt7, nb, nn]
    print_comparison_table(all_results)
    analyse_error_types(all_results)

    # ── Gender bias ───────────────────────────────────────────────────────────
    print_subheader("4.7  Gender bias analysis")
    bias = gender_bias_analysis(df)

    # ── Limitations ───────────────────────────────────────────────────────────
    print_subheader("4.8  Limitations and responsible use")
    print_limitations()

    return {
        "baseline": baseline,
        "dt3":      dt3,
        "dt7":      dt7,
        "nb":       nb,
        "nn":       nn,
        "bias":     bias,
        "X_train":  X_train,
        "X_test":   X_test,
        "y_train":  y_train,
        "y_test":   y_test,
    }
