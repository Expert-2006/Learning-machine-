
# DSN Bootcamp Qualification Hackathon 2026 - ML Track
# Robust tabular regression starter / advanced baseline
#
# Expected files in the same folder:
#   train.csv
#   test.csv
#   sample_submission.csv   (recommended)
#
# Target from the competition brief:
#   total_sales
#
# Designed to be adaptable because the exact column names in your files
# should be inspected before final submission.

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_DIR = "."              # Change this if your CSVs are elsewhere
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_FILE = "sample_submission.csv"

TARGET = "total_sales"
RANDOM_STATE = 42
N_SPLITS = 5

train_path = os.path.join(DATA_DIR, TRAIN_FILE)
test_path = os.path.join(DATA_DIR, TEST_FILE)
sample_path = os.path.join(DATA_DIR, SAMPLE_FILE)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("DSN MART SALES PREDICTION")
print("=" * 70)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("\nTrain shape:", train.shape)
print("Test shape :", test.shape)

print("\nTrain columns:")
print(train.columns.tolist())

print("\nFirst 5 rows:")
print(train.head())


# ============================================================
# 3. BASIC DATA CHECK
# ============================================================

if TARGET not in train.columns:
    raise ValueError(
        f"Could not find '{TARGET}' in train.csv.\n"
        f"Available columns are:\n{train.columns.tolist()}"
    )

print("\nMissing values:")
missing = train.isna().sum()
print(missing[missing > 0].sort_values(ascending=False).head(30))

print("\nTarget summary:")
print(train[TARGET].describe())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Conservative, automatic feature engineering.

    - Detects date-like columns and extracts calendar information.
    - Adds row-level missing-value count.
    - Adds log versions of strongly positive numeric features.
    - Adds simple numeric ratios where they are clearly meaningful.
    """

    df = df.copy()

    # Remove target if this function is accidentally called on train
    # after target separation.
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------
    for col in list(df.columns):
        name = str(col).lower()

        if (
            "date" in name
            or "time" in name
            or "timestamp" in name
        ):
            parsed = pd.to_datetime(df[col], errors="coerce")

            if parsed.notna().mean() >= 0.70:
                df[col + "_year"] = parsed.dt.year
                df[col + "_month"] = parsed.dt.month
                df[col + "_day"] = parsed.dt.day
                df[col + "_dayofweek"] = parsed.dt.dayofweek
                df[col + "_weekofyear"] = parsed.dt.isocalendar().week.astype(float)

                # Cyclic encoding helps models represent seasonality.
                df[col + "_month_sin"] = np.sin(
                    2 * np.pi * df[col + "_month"] / 12
                )
                df[col + "_month_cos"] = np.cos(
                    2 * np.pi * df[col + "_month"] / 12
                )
                df[col + "_dow_sin"] = np.sin(
                    2 * np.pi * df[col + "_dayofweek"] / 7
                )
                df[col + "_dow_cos"] = np.cos(
                    2 * np.pi * df[col + "_dayofweek"] / 7
                )

                df = df.drop(columns=[col])

    # --------------------------------------------------------
    # Missingness signal
    # --------------------------------------------------------
    df["__missing_count"] = df.isna().sum(axis=1)

    # --------------------------------------------------------
    # Log transforms for positive numeric columns.
    # This can help tree models capture large-value ranges.
    # --------------------------------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        if col.startswith("__"):
            continue

        values = df[col]
        if len(values) > 0:
            finite = values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite) > 0 and finite.min() >= 0:
                # Only add if there is enough variation.
                if finite.nunique() > 10:
                    df[col + "_log1p"] = np.log1p(
                        values.clip(lower=0)
                    )

    return df


X_raw = train.drop(columns=[TARGET])
y = train[TARGET].astype(float)

X_test_raw = test.copy()

X = engineer_features(X_raw)
X_test = engineer_features(X_test_raw)

# Make sure train/test have exactly the same feature columns.
X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)

print("\nEngineered train shape:", X.shape)
print("Engineered test shape :", X_test.shape)


# ============================================================
# 5. IDENTIFY NUMERIC / CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [
    c for c in X.columns if c not in numeric_features
]

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
                sparse_output=False
            )
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop"
)


# ============================================================
# 7. MODEL 1 — EXTRA TREES
# ============================================================

extra_trees = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=700,
                max_features=0.85,
                min_samples_leaf=2,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 8. MODEL 2 — RANDOM FOREST
# ============================================================

random_forest = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=500,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 9. LOCAL VALIDATION
# ============================================================
# We do not blindly trust a single train/validation split.
# Five-fold CV gives a more stable estimate.

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

models = {
    "ExtraTrees": extra_trees,
    "RandomForest": random_forest,
}

oof_predictions = {}
cv_scores = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    oof = np.zeros(len(X))

    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(X), start=1
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(X_train, y_train)

        pred = model.predict(X_valid)
        pred = np.maximum(pred, 0)

        oof[valid_idx] = pred

        rmse = mean_squared_error(
            y_valid,
            pred,
            squared=False
        )

        print(f"  Fold {fold}: RMSE = {rmse:,.4f}")

    overall_rmse = mean_squared_error(
        y,
        oof,
        squared=False
    )

    oof_predictions[model_name] = oof
    cv_scores[model_name] = overall_rmse

    print(f"  Overall OOF RMSE: {overall_rmse:,.4f}")


print("\nCV SUMMARY")
for name, score in cv_scores.items():
    print(f"{name:15s} -> {score:,.4f}")


# ============================================================
# 10. SIMPLE ENSEMBLE SEARCH
# ============================================================
# Instead of assuming one model is always best, blend their
# out-of-fold predictions. Sometimes the blend is more stable.

best_name = min(cv_scores, key=cv_scores.get)

pred_a = oof_predictions["ExtraTrees"]
pred_b = oof_predictions["RandomForest"]

best_blend = None
best_blend_rmse = np.inf

print("\nSearching blend weights...")

for w in np.arange(0.0, 1.01, 0.05):

    blended = (
        w * pred_a +
        (1.0 - w) * pred_b
    )

    blended = np.maximum(blended, 0)

    score = mean_squared_error(
        y,
        blended,
        squared=False
    )

    if score < best_blend_rmse:
        best_blend_rmse = score
        best_blend = w

print(
    f"Best blend: ExtraTrees={best_blend:.2f}, "
    f"RandomForest={1-best_blend:.2f}"
)
print(f"Blended OOF RMSE: {best_blend_rmse:,.4f}")

print(
    f"\nBest individual model: {best_name} "
    f"(OOF RMSE={cv_scores[best_name]:,.4f})"
)


# ============================================================
# 11. TRAIN FULL MODELS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODELS ON ALL TRAINING DATA")
print("=" * 70)

extra_trees.fit(X, y)
random_forest.fit(X, y)

extra_test_pred = np.maximum(
    extra_trees.predict(X_test),
    0
)

rf_test_pred = np.maximum(
    random_forest.predict(X_test),
    0
)

final_predictions = (
    best_blend * extra_test_pred
    + (1.0 - best_blend) * rf_test_pred
)

final_predictions = np.maximum(final_predictions, 0)


# ============================================================
# 12. CREATE SUBMISSION
# ============================================================

if os.path.exists(sample_path):

    sample = pd.read_csv(sample_path)

    print("\nSample submission columns:")
    print(sample.columns.tolist())

    # Try to identify the prediction column automatically.
    prediction_candidates = [
        c for c in sample.columns
        if c.lower() in {
            TARGET.lower(),
            "prediction",
            "predictions",
            "sales"
        }
    ]

    # If the sample has exactly two columns, use the non-ID column.
    if len(prediction_candidates) == 0 and len(sample.columns) == 2:
        id_candidates = [
            c for c in sample.columns
            if c.lower() in {"id", "row_id", "index"}
        ]
        remaining = [
            c for c in sample.columns
            if c not in id_candidates
        ]
        if len(remaining) == 1:
            prediction_candidates = remaining

    if len(prediction_candidates) == 0:
        raise ValueError(
            "Could not automatically identify the prediction column "
            "in sample_submission.csv. Inspect its columns and set "
            "SUBMISSION_TARGET_COLUMN manually."
        )

    prediction_col = prediction_candidates[0]

    submission = sample.copy()
    submission[prediction_col] = final_predictions

else:
    # Fallback: create a conventional Kaggle submission.
    # If the competition uses another ID column, adjust this section.
    submission = pd.DataFrame({
        "id": test["id"] if "id" in test.columns else np.arange(len(test)),
        TARGET: final_predictions
    })


# ============================================================
# 13. SAVE
# ============================================================

output_file = os.path.join(
    DATA_DIR,
    "submission_advanced_ensemble.csv"
)

submission.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("Submission saved to:")
print(output_file)

print("\nSubmission shape:", submission.shape)

print("\nSubmission preview:")
print(submission.head(10))

print("\nPrediction statistics:")
print(pd.Series(final_predictions).describe())

print(
    "\nIMPORTANT: Before uploading to Kaggle, compare the "
    "submission columns and row count with the competition's "
    "sample_submission.csv."
)
