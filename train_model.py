import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
N_SAMPLES = 5_000

rng = np.random.default_rng(RANDOM_STATE)

square_footage = rng.uniform(300, 10_000, N_SAMPLES)
num_bedrooms   = rng.integers(1, 10, N_SAMPLES)
num_bathrooms  = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], N_SAMPLES)
age_years      = rng.integers(0, 80, N_SAMPLES)
location_tier  = rng.choice(["urban", "suburban", "rural"], N_SAMPLES)
has_garage     = rng.choice([0, 1], N_SAMPLES)

LOCATION_MAP = {"urban": 1.40, "suburban": 1.00, "rural": 0.65}
loc_mult = np.array([LOCATION_MAP[t] for t in location_tier])

price = (
    square_footage * 120 * loc_mult
    + np.maximum(0, num_bedrooms - 1) * 8_000
    + np.maximum(0, num_bathrooms - 1) * 5_500
    - age_years * 1_200
    + has_garage * 15_000
    + rng.normal(0, 8_000, N_SAMPLES)
)

X = np.column_stack([
    square_footage,
    num_bedrooms,
    num_bathrooms,
    age_years,
    location_tier,
    has_garage,
])
y = price

NUMERIC_IDX     = [0, 1, 2, 3, 5]
CATEGORICAL_IDX = [4]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_IDX),
        ("cat", OrdinalEncoder(categories=[["rural", "suburban", "urban"]]), CATEGORICAL_IDX),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"Training complete.")
print(f"  MAE : ")
print(f"  R2  : {r2:.4f}")

MODEL_PATH = "model.joblib"
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
