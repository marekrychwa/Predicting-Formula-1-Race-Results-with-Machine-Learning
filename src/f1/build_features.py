import pandas as pd

DATA_DIR = "data/"
INPUT_FILE = "race_weekend"
OUTPUT_FILE = "data/features.csv"
ROLLING_WINDOW = 5

def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR + f"{name}.csv", na_values=r"\N")

def add_rolling_form(df: pd.DataFrame, group_col: str, output_col: str) -> pd.DataFrame:
    # Average finish position over the previous ROLLING_WINDOW races for each
    # driver or constructor. shift(1) drops the current race, so a row only ever
    # sees races that happened before it (the golden rule).
    df[output_col] = (
        df.groupby(group_col)["positionOrder"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    return df

def add_teammate_delta(df: pd.DataFrame) -> pd.DataFrame:
    # For each row, the qualifying position of the other car(s) in the same team
    # that race. Return NaN when there is no teammate
    group = df.groupby(["raceId", "constructorId"])["qualiPosition"]
    total = group.transform("sum")
    count = group.transform("count")
    df["teammateQuali"] = (total - df["qualiPosition"]) / (count - 1)

    # Positive means this driver qualified worse than their teammate.
    df["teammateDelta"] = df["qualiPosition"] - df["teammateQuali"]
    return df

def build_features() -> pd.DataFrame:
    df = load_csv(INPUT_FILE)

    # F1 is chronological, so order races oldest to newest before any rolling
    df = df.sort_values(["year", "round","driverId"]).reset_index(drop=True)

    df = add_rolling_form(df, "driverId", "driverForm")
    df = add_rolling_form(df, "constructorId", "constructorForm")
    df = add_teammate_delta(df)
    return df

def main() -> None:
    df = build_features()

    columns = [
        "raceId", "year", "round", "race",
        "driverId", "driver", "constructorId", "constructor",
        "grid", "qualiPosition", "teammateDelta",
        "driverForm", "constructorForm",
        "positionOrder",
    ]
    df = df[columns]

    print("FEATURES")
    print("--------")
    print(f"Rows: {len(df)}")
    print(f"Years: {int(df['year'].min())}-{int(df['year'].max())}")

    print("\nMISSING VALUES")
    for column in ["grid", "qualiPosition", "teammateDelta", "driverForm", "constructorForm"]:
        print(f"{column}: {int(df[column].isna().sum())}")

    print("\nSAMPLE (2014 round 5, so rolling form has history)")
    sample = df[(df["year"] == 2014) & (df["round"] ==5)]
    print(sample.head(5).to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

