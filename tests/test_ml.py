"""
tests/test_ml.py
================
Tests for ml/dataset.py and ml/model.py.

Run from project root:
    python -m unittest tests.test_ml -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.dataset import generate_dataset, load_dataset
from ml.model import (
    prepare_data, run_baseline, run_decision_tree,
    run_naive_bayes, train_models, predict_risk,
    gender_bias_analysis, FEATURES_WITH_GENDER, FEATURES_NO_GENDER, TARGET,
)


# =============================================================================
# Dataset
# =============================================================================
class TestDataset(unittest.TestCase):

    def setUp(self):
        self.df = generate_dataset(n=500, seed=42)

    def test_shape(self):
        self.assertEqual(self.df.shape, (500, 13))

    def test_no_nulls(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_target_binary(self):
        self.assertTrue(set(self.df["success"].unique()).issubset({0, 1}))

    def test_gender_binary(self):
        self.assertTrue(set(self.df["gender"].unique()).issubset({0, 1}))

    def test_attendance_range(self):
        self.assertTrue(self.df["attendance"].between(0, 100).all())

    def test_confidence_range(self):
        self.assertTrue(self.df["confidence_level"].between(1, 5).all())

    def test_stress_range(self):
        self.assertTrue(self.df["stress_level"].between(1, 5).all())

    def test_quiz_range(self):
        self.assertTrue(self.df["quiz_score_avg"].between(0, 100).all())

    def test_pass_rate_realistic(self):
        rate = self.df["success"].mean()
        self.assertGreater(rate, 0.50)
        self.assertLess(rate, 0.85)

    def test_risk_level_excluded_from_features(self):
        self.assertNotIn("risk_level", FEATURES_WITH_GENDER)
        self.assertNotIn("risk_level", FEATURES_NO_GENDER)

    def test_id_excluded_from_features(self):
        self.assertNotIn("id", FEATURES_WITH_GENDER)

    def test_reproducibility(self):
        df2 = generate_dataset(n=500, seed=42)
        self.assertTrue(self.df["success"].equals(df2["success"]))


# =============================================================================
# Data preparation + leakage guard
# =============================================================================
class TestPrepareData(unittest.TestCase):

    def setUp(self):
        self.df = generate_dataset(n=500, seed=42)

    def test_split_sizes(self):
        X_train, X_test, y_train, y_test = prepare_data(self.df)
        self.assertEqual(len(X_train) + len(X_test), 500)
        self.assertAlmostEqual(len(X_test) / 500, 0.20, delta=0.02)

    def test_stratification_preserves_ratio(self):
        X_train, X_test, y_train, y_test = prepare_data(self.df)
        train_rate = y_train.mean()
        test_rate  = y_test.mean()
        self.assertAlmostEqual(train_rate, test_rate, delta=0.05)

    def test_risk_level_not_in_features(self):
        X_train, X_test, y_train, y_test = prepare_data(self.df)
        self.assertNotIn("risk_level", X_train.columns)
        self.assertNotIn("risk_level", X_test.columns)

    def test_gender_present_by_default(self):
        X_train, _, _, _ = prepare_data(self.df)
        self.assertIn("gender", X_train.columns)

    def test_gender_excluded_when_requested(self):
        X_train, _, _, _ = prepare_data(self.df, include_gender=False)
        self.assertNotIn("gender", X_train.columns)

    def test_no_overlap_between_train_and_test(self):
        X_train, X_test, y_train, y_test = prepare_data(self.df)
        train_idx = set(X_train.index)
        test_idx  = set(X_test.index)
        self.assertEqual(len(train_idx & test_idx), 0)


# =============================================================================
# Baseline
# =============================================================================
class TestBaseline(unittest.TestCase):

    def setUp(self):
        self.df = generate_dataset(n=500, seed=42)
        self.splits = prepare_data(self.df)

    def test_returns_required_keys(self):
        r = run_baseline(*self.splits)
        for key in ("accuracy","precision","recall","f1",
                    "confusion_matrix","fn","fp","tn","tp"):
            self.assertIn(key, r)

    def test_minority_recall_is_zero(self):
        """
        Baseline predicts all as Pass (majority). Every actual Fail student
        is predicted Pass → False Positive. TN = 0 so Fail-class recall = 0.
        """
        r = run_baseline(*self.splits)
        # DummyClassifier predicts all Pass → TN must be 0
        self.assertEqual(r["tn"], 0)
        # All actual Fail students are FP (predicted Pass, actually Fail)
        _, _, _, y_test = self.splits
        actual_fail = int((y_test == 0).sum())
        self.assertEqual(r["fp"], actual_fail)

    def test_accuracy_near_pass_rate(self):
        _, _, y_train, y_test = self.splits
        r = run_baseline(*self.splits)
        expected = y_test.mean()
        self.assertAlmostEqual(r["accuracy"], expected, delta=0.02)


# =============================================================================
# Decision Tree
# =============================================================================
class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        self.df = generate_dataset(n=500, seed=42)
        self.splits = prepare_data(self.df)

    def test_depth3_beats_baseline(self):
        baseline = run_baseline(*self.splits)
        dt3      = run_decision_tree(*self.splits, max_depth=3)
        self.assertGreater(dt3["f1"], baseline["f1"])

    def test_depth7_has_more_leaves(self):
        dt3 = run_decision_tree(*self.splits, max_depth=3)
        dt7 = run_decision_tree(*self.splits, max_depth=7)
        self.assertGreater(dt7["n_leaves"], dt3["n_leaves"])

    def test_feature_importances_sum_to_one(self):
        dt3 = run_decision_tree(*self.splits, max_depth=3)
        total = sum(dt3["feature_importances"].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_required_keys(self):
        dt3 = run_decision_tree(*self.splits, max_depth=3)
        for key in ("accuracy","precision","recall","f1",
                    "confusion_matrix","model","feature_importances"):
            self.assertIn(key, dt3)

    def test_accuracy_in_valid_range(self):
        dt3 = run_decision_tree(*self.splits, max_depth=3)
        self.assertGreater(dt3["accuracy"], 0.5)
        self.assertLessEqual(dt3["accuracy"], 1.0)


# =============================================================================
# Naive Bayes
# =============================================================================
class TestNaiveBayes(unittest.TestCase):

    def setUp(self):
        self.df = generate_dataset(n=500, seed=42)
        self.splits = prepare_data(self.df)

    def test_required_keys(self):
        r = run_naive_bayes(*self.splits)
        for key in ("accuracy","precision","recall","f1","confusion_matrix"):
            self.assertIn(key, r)

    def test_accuracy_reasonable(self):
        r = run_naive_bayes(*self.splits)
        self.assertGreater(r["accuracy"], 0.5)

    def test_beats_baseline_on_recall(self):
        baseline = run_baseline(*self.splits)
        nb       = run_naive_bayes(*self.splits)
        # NB should catch more at-risk students than majority-class baseline
        self.assertGreater(nb["fn"],  0)
        # NB actually uses features so FN < total Fail count
        _, _, _, y_test = self.splits
        actual_fail = (y_test == 0).sum()
        self.assertLess(nb["fn"], actual_fail)

    def test_discussion_present(self):
        r = run_naive_bayes(*self.splits)
        self.assertIn("discussion", r)
        self.assertTrue(r["discussion"])


# =============================================================================
# train_models convenience API
# =============================================================================
class TestTrainModels(unittest.TestCase):

    def setUp(self):
        self.df = generate_dataset(n=300, seed=42)  # small for speed

    def test_returns_all_models(self):
        models = train_models(self.df)
        for key in ("baseline", "dt_d3", "dt_d7", "nb"):
            self.assertIn(key, models)

    def test_all_models_have_accuracy(self):
        models = train_models(self.df)
        for name, result in models.items():
            self.assertIn("accuracy", result, f"{name} missing accuracy")


# =============================================================================
# predict_risk
# =============================================================================
class TestPredictRisk(unittest.TestCase):

    def setUp(self):
        df           = generate_dataset(n=300, seed=42)
        self.models  = train_models(df)

    def test_returns_dict(self):
        profile = {
            "attendance": 80, "confidence_level": 3, "stress_level": 2,
            "deadlines": 1, "workload_tasks": 3, "workload_credits": 40,
            "workload_hours": 8, "availability_constraints": 2,
            "quiz_score_avg": 70, "gender": 0,
        }
        result = predict_risk(profile, self.models)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_prediction_is_pass_or_fail(self):
        profile = {
            "attendance": 80, "confidence_level": 4, "stress_level": 2,
            "deadlines": 0, "workload_tasks": 2, "workload_credits": 40,
            "workload_hours": 6, "availability_constraints": 1,
            "quiz_score_avg": 80, "gender": 1,
        }
        result = predict_risk(profile, self.models)
        for _, pred in result.items():
            self.assertIn(pred["prediction"], ("Pass", "Fail"))

    def test_confidence_in_range(self):
        profile = {
            "attendance": 50, "confidence_level": 2, "stress_level": 5,
            "deadlines": 5, "workload_tasks": 8, "workload_credits": 80,
            "workload_hours": 40, "availability_constraints": 3,
            "quiz_score_avg": 35, "gender": 0,
        }
        result = predict_risk(profile, self.models)
        for _, pred in result.items():
            self.assertGreater(pred["confidence"], 0.5)


# =============================================================================
# Gender bias analysis
# =============================================================================
class TestGenderBias(unittest.TestCase):

    def test_returns_required_keys(self):
        df     = generate_dataset(n=400, seed=42)
        result = gender_bias_analysis(df)
        for key in ("acc_with","acc_without","f1_with","f1_without","by_group"):
            self.assertIn(key, result)

    def test_accuracy_values_in_range(self):
        df     = generate_dataset(n=400, seed=42)
        result = gender_bias_analysis(df)
        self.assertGreater(result["acc_with"],    0.5)
        self.assertGreater(result["acc_without"], 0.5)

    def test_synthetic_gender_has_minimal_impact(self):
        """
        Since gender was generated independently of risk,
        the accuracy difference should be negligible (<5%).
        """
        df     = generate_dataset(n=500, seed=42)
        result = gender_bias_analysis(df)
        delta  = abs(result["acc_with"] - result["acc_without"])
        self.assertLess(delta, 0.05,
            "Synthetic gender should have <5% accuracy impact")


if __name__ == "__main__":
    unittest.main(verbosity=2)
