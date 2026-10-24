import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_DIR = "data/"
INPUT_FILE = "features"
HOLDOUT_YEAR = 2023

FEATURES = ["grid","qualiPosition","teammateDelta","driverForm","constructorForm"]
TARGET = "positionOrder"

def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR + f"{name}.csv", na_values=r"\N")

def add_start_position(df: pd.DataFrame) -> pd.DataFrame:
    # Same starting position the Phase 4 baseline uses, so we can score both on
    # the exact same holdout rows.
    df["startPosition"] = df["grid"].fillna(df["qualiPosition"])
    return df

def split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The golden rule: split chronologically, never randomly. Train on every
    # season before the holdout, test on the holdout season.
    train = df[df["year"] < HOLDOUT_YEAR]
    test = df[df["year"] == HOLDOUT_YEAR]
    return train, test

def main() -> None:
    df = load_csv(INPUT_FILE)
    df = add_start_position(df)

    train, test = split_by_year(df)

    # HistGradientBoosting handles missing feature values natively, so we don't
    # need to impute the grid/quali/form gaps.
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(train[FEATURES], train[TARGET])

    test = test.copy()
    test["prediction"] = model.predict(test[FEATURES])

    # Score the model and the baseline on the same rows for a fair comparison.
    scored = test.dropna(subset=["startPosition", TARGET])
    model_mae = mean_absolute_error(scored[TARGET], scored["prediction"])
    baseline_mae = mean_absolute_error(scored[TARGET], scored["startPosition"])

    print("MODEL vs BASELINE")
    print("-----------------")
    print(f"Train rows: {len(train)} ({int(train['year'].min())} - {int(train['year'].max())})")
    print(f"Test rows: {len(scored)} ({HOLDOUT_YEAR})")
    print(f"Baseline MAE: {baseline_mae:.3f}")
    print(f"Model MAE: {model_mae:.3f}")
    print(f"Improvement: {baseline_mae - model_mae:.3f}")

if __name__ == "__main__":
    main()





