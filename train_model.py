"""
Train and evaluate a property price estimation model.

IMPORTANT: This prototype uses SYNTHETIC data generated from a fixed formula.
The model's performance on this data does NOT demonstrate real-world pricing
accuracy. It serves as a demonstration of the ML pipeline architecture.

Pricing formula:
    price = sqft * 120 * location_multiplier
          + max(0, bedrooms - 1) * 8_000
          + max(0, bathrooms - 1) * 5_500
          - age * 1_200
          + garage * 15_000
          + noise(mean=0, std=8_000)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SAMPLES = 5_000
MODEL_VERSION = "2.0.0-gbr"

# Paths are relative to THIS file, not the current working directory.
_PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = _PROJECT_DIR / "model.joblib"
METADATA_PATH = _PROJECT_DIR / "model_metadata.json"
EVAL_REPORT_PATH = _PROJECT_DIR / "evaluation_report.txt"

# Feature ranges used during synthetic data generation.
# These are saved in metadata and used at inference time for extrapolation checks.
FEATURE_RANGES = {
    "square_footage": {"min": 300.0, "max": 10_000.0},
    "num_bedrooms": {"min": 1, "max": 9},
    "num_bathrooms": {"min": 1.0, "max": 4.0},
    "age_years": {"min": 0, "max": 79},
    "location_tier": {"values": ["urban", "suburban", "rural"]},
    "has_garage": {"values": [0, 1]},
}

LOCATION_MAP = {"urban": 1.40, "suburban": 1.00, "rural": 0.65}


def generate_synthetic_data(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic property data from a known pricing formula.

    Returns
    -------
    X : ndarray of shape (n_samples, 6)
        Feature matrix with columns:
        [square_footage, num_bedrooms, num_bathrooms, age_years,
         location_tier, has_garage].
    y : ndarray of shape (n_samples,)
        Target prices.
    """
    rng = np.random.default_rng(random_state)

    square_footage = rng.uniform(300, 10_000, n_samples)
    num_bedrooms = rng.integers(1, 10, n_samples)
    num_bathrooms = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], n_samples)
    age_years = rng.integers(0, 80, n_samples)
    location_tier = rng.choice(["urban", "suburban", "rural"], n_samples)
    has_garage = rng.choice([0, 1], n_samples)

    loc_mult = np.array([LOCATION_MAP[t] for t in location_tier])

    price = (
        square_footage * 120 * loc_mult
        + np.maximum(0, num_bedrooms - 1) * 8_000
        + np.maximum(0, num_bathrooms - 1) * 5_500
        - age_years * 1_200
        + has_garage * 15_000
        + rng.normal(0, 8_000, n_samples)
    )

    X = np.column_stack(  # noqa: N806
        [
            square_footage,
            num_bedrooms,
            num_bathrooms,
            age_years,
            location_tier,
            has_garage,
        ]
    )
    return X, price


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Build the full sklearn Pipeline (preprocessing + regressor)."""
    numeric_idx = [0, 1, 2, 3, 5]
    categorical_idx = [4]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_idx),
            (
                "cat",
                OrdinalEncoder(categories=[["rural", "suburban", "urban"]]),
                categorical_idx,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_evaluate() -> dict:
    """Train the model, evaluate against a baseline, and save artifacts.

    Artifacts saved:
    - model.joblib          — serialised sklearn Pipeline
    - model_metadata.json   — version, metrics, feature ranges, sklearn version
    - evaluation_report.txt — human-readable evaluation summary

    Returns
    -------
    dict
        Model metadata dictionary.
    """
    logger.info("Generating synthetic training data (%d samples)...", N_SAMPLES)
    X, y = generate_synthetic_data()  # noqa: N806

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    logger.info("Split: %d train / %d test", len(y_train), len(y_test))

    logger.info("Training pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)

    mae_improvement = (1 - mae / baseline_mae) * 100 if baseline_mae > 0 else 0.0

    logger.info("═══ Evaluation Results ═══")
    logger.info("  Model   MAE  : $%,.2f", mae)
    logger.info("  Model   R²   : %.4f", r2)
    logger.info("  Baseline MAE : $%,.2f  (mean predictor)", baseline_mae)
    logger.info("  Baseline R²  : %.4f  (mean predictor)", baseline_r2)
    logger.info("  Improvement  : %.1f%% MAE reduction vs baseline", mae_improvement)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Model saved → %s", MODEL_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "python_version": sys.version.split()[0],
        "data_source": "synthetic",
        "data_formula": (
            "sqft*120*loc_mult + (beds-1)*8000 + (baths-1)*5500 "
            "- age*1200 + garage*15000 + N(0,8000)"
        ),
        "n_samples_total": N_SAMPLES,
        "n_samples_train": len(y_train),
        "n_samples_test": len(y_test),
        "random_state": RANDOM_STATE,
        "metrics": {
            "mae": round(float(mae), 2),
            "r2": round(float(r2), 4),
            "baseline_mae": round(float(baseline_mae), 2),
            "baseline_r2": round(float(baseline_r2), 4),
            "mae_improvement_pct": round(mae_improvement, 1),
        },
        "feature_ranges": FEATURE_RANGES,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved → %s", METADATA_PATH)

    report_lines = [
        "=" * 60,
        "MODEL EVALUATION REPORT",
        "=" * 60,
        f"Model Version    : {MODEL_VERSION}",
        f"Training Date    : {metadata['trained_at']}",
        f"scikit-learn     : {sklearn.__version__}",
        "Data Source       : SYNTHETIC (not real property data)",
        f"Total Samples     : {N_SAMPLES}",
        f"Train / Test      : {len(y_train)} / {len(y_test)}",
        f"Random State      : {RANDOM_STATE}",
        "",
        "── Model Performance ──",
        f"  MAE  : ${mae:,.2f}",
        f"  R²   : {r2:.4f}",
        "",
        "── Baseline (Mean Predictor) ──",
        f"  MAE  : ${baseline_mae:,.2f}",
        f"  R²   : {baseline_r2:.4f}",
        "",
        "── Baseline (Mean Predictor) ──",
        f"  MAE  : ${baseline_mae:,.2f}",
        f"  R²   : {baseline_r2:.4f}",
        "",
        f"  MAE Improvement : {mae_improvement:.1f}%",
        "",
        "── LIMITATIONS ──",
        "• Training data is SYNTHETIC, generated from a fixed formula.",
        "• Model performance here does NOT indicate real-world accuracy.",
        "• For production use, replace with verified market data.",
        "• See model_metadata.json for feature ranges and formula.",
        "=" * 60,
    ]

    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    logger.info("Evaluation report saved → %s", EVAL_REPORT_PATH)

    logger.info("Verifying model reload + prediction consistency...")
    reloaded = joblib.load(MODEL_PATH)
    sample_input = X_test[:1]
    original_pred = float(pipeline.predict(sample_input)[0])
    reloaded_pred = float(reloaded.predict(sample_input)[0])
    if not np.isclose(original_pred, reloaded_pred):
        raise RuntimeError(
            f"Reload verification FAILED: original={original_pred}, reloaded={reloaded_pred}"
        )
    logger.info("Reload verification PASSED")

    return metadata


if __name__ == "__main__":
    try:
        result = train_and_evaluate()
        print(f"\nTraining complete. Model ready at: {MODEL_PATH}")
        print(f"  MAE : ${result['metrics']['mae']:,.2f}")
        print(f"  R²  : {result['metrics']['r2']:.4f}")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)
