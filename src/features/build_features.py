"""
build_features.py
-----------------
Builds the preprocessing pipeline for the churn prediction project.

Steps performed:
    1. Load cleaned data
    2. Define numerical and categorical feature columns
    3. Build a ColumnTransformer preprocessor
    4. Split data into train/test sets (80/20 stratified)
    5. Save the fitted preprocessor to disk using joblib

Output:
    - data/processed/X_train.csv
    - data/processed/X_test.csv
    - data/processed/y_train.csv
    - data/processed/y_test.csv
    - models/preprocessor.joblib
"""

import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
DATA_RAW   = BASE_DIR / "data" / "raw"   / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# ── Functions ──────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """
    Load the cleaned Telco churn CSV from disk.

    Parameters
    ----------
    path : Path
        Absolute path to the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all 21 original columns.
    """
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def define_feature_columns():
    """
    Define which columns are numerical and which are categorical.

    TotalCharges is excluded — highly correlated with tenure (0.83),
    identified as redundant during EDA.
    customerID is excluded — it is an identifier, not a feature.

    Returns
    -------
    tuple[list, list]
        (numerical_features, categorical_features)
    """
    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "SeniorCitizen"
    ]

    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    return numerical_features, categorical_features

def build_preprocessor(numerical_features: list,
                        categorical_features: list) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer with two parallel branches:

        Numerical branch:
            Step 1 — SimpleImputer(strategy='median')
                     Fills any missing numerical values with the column median.
                     Median is preferred over mean because it is robust to outliers.
            Step 2 — StandardScaler()
                     Standardizes to mean=0, std=1.

        Categorical branch:
            Step 1 — SimpleImputer(strategy='most_frequent')
                     Fills any missing categorical values with the most common value.
            Step 2 — OneHotEncoder(handle_unknown='ignore')
                     Creates binary dummy columns for each category.
                     handle_unknown='ignore' means if production data has a new
                     category never seen in training, it will not crash —
                     it will just produce all zeros for that column.

    Parameters
    ----------
    numerical_features : list
        Column names for numerical features.
    categorical_features : list
        Column names for categorical features.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor object ready for fit_transform.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline,  numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

def split_and_transform(df: pd.DataFrame,
                         numerical_features: list,
                         categorical_features: list,
                         preprocessor: ColumnTransformer):
    """
    Split data into train/test sets and fit-transform the preprocessor.

    The split is stratified on the target column 'Churn' to preserve
    the 26.6% churn ratio in both train and test sets.

    IMPORTANT: preprocessor is fitted ONLY on X_train, then used to
    transform both X_train and X_test. This prevents data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset.
    numerical_features : list
        Numerical column names.
    categorical_features : list
        Categorical column names.
    preprocessor : ColumnTransformer
        Unfitted preprocessor from build_preprocessor().

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, fitted_preprocessor)
    """
    all_features = numerical_features + categorical_features
    X = df[all_features]
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"[split_and_transform] Train size : {X_train.shape[0]} rows")
    print(f"[split_and_transform] Test  size : {X_test.shape[0]} rows")
    print(f"[split_and_transform] Train churn rate: {y_train.mean():.3f}")
    print(f"[split_and_transform] Test  churn rate: {y_test.mean():.3f}")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def save_outputs(X_train, X_test, y_train, y_test,
                 preprocessor, numerical_features, categorical_features):
    """
    Save processed arrays and the fitted preprocessor to disk.

    Processed feature arrays are saved as CSVs in data/processed/.
    Column names are reconstructed from the preprocessor for readability.
    The fitted preprocessor is saved as a joblib file in models/.

    Parameters
    ----------
    X_train : np.ndarray
        Processed training features.
    X_test : np.ndarray
        Processed test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    preprocessor : ColumnTransformer
        Fitted preprocessor object.
    numerical_features : list
        Numerical column names (for reconstructing column headers).
    categorical_features : list
        Categorical column names (for reconstructing column headers).
    """
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cat_columns = preprocessor.named_transformers_["cat"] \
                              .named_steps["encoder"] \
                              .get_feature_names_out(categorical_features) \
                              .tolist()
    all_columns = numerical_features + cat_columns

    pd.DataFrame(X_train, columns=all_columns).to_csv(
        DATA_PROC / "X_train.csv", index=False)
    pd.DataFrame(X_test,  columns=all_columns).to_csv(
        DATA_PROC / "X_test.csv",  index=False)

    y_train.reset_index(drop=True).to_csv(
        DATA_PROC / "y_train.csv", index=False)
    y_test.reset_index(drop=True).to_csv(
        DATA_PROC / "y_test.csv",  index=False)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    print(f"[save_outputs] Saved X_train, X_test, y_train, y_test to {DATA_PROC}")
    print(f"[save_outputs] Saved preprocessor to {MODELS_DIR / 'preprocessor.joblib'}")


def main():
    """
    Orchestrates the full feature engineering pipeline.

    Execution order:
        1. Load data
        2. Define feature columns
        3. Build preprocessor
        4. Split and transform
        5. Save all outputs
    """
    print("=" * 50)
    print("  Feature Engineering Pipeline — Day 3")
    print("=" * 50)

    df = load_data(DATA_RAW)

    numerical_features, categorical_features = define_feature_columns()

    preprocessor = build_preprocessor(numerical_features, categorical_features)

    X_train, X_test, y_train, y_test, fitted_preprocessor = split_and_transform(
        df, numerical_features, categorical_features, preprocessor
    )

    save_outputs(X_train, X_test, y_train, y_test,
                 fitted_preprocessor, numerical_features, categorical_features)


    print("Pipeline complete. All outputs saved.")



if __name__ == "__main__":
    main()
