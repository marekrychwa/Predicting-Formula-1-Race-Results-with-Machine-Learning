import pandas as pd

DATA_DIR = "data/"
INPUT_FILE = "features"

def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR + f"{name}.csv", na_values=r"\N")

def add_start_position(df: pd.DataFrame) -> pd.DataFrame:
    # The baseline predicts the finish from where the driver started. grid is
    # the race grid, but it's missing for the sentinel rows, so fall back to the
    # gualifying position when we can.
    df["startPosition"] = df["grid"].fillna(df["qualiPosition"])
    return df

def main() -> None:
    df = load_csv(INPUT_FILE)
    df = add_start_position(df)
    # We can only score rows that have a starting position and a finish.
    scored = df.dropna(subset=["startPosition","positionOrder"]).copy()

    # The "dumb" baseline: predict every driver finishes exactly where they started
    scored["prediction"] = scored["startPosition"]
    scored["error"] = (scored["prediction"] - scored["positionOrder"]).abs().astype(int)

    mae = scored["error"].mean()

    print("BASELINE: FINISH = START POSITION")
    print("----------------------------------")
    print(f"Rows scored: {len(scored)}")
    print(f"Dropped (no start position): {len(df) - len(scored)}")
    print(f"MAE: {mae:.3f}")

    print("\nERROR DISTRIBUTION")
    for error, count in scored["error"].value_counts().sort_index().items():
        print(f"{error}: {count}")

if __name__ == "__main__":
    main()

