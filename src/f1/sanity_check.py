from pathlib import Path
import pandas as pd

DATA_DIR = "data/"
ERA_START_YEAR = 2014

def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR + f"{name}.csv", na_values=r"\N")

def check_ids_match(
        child: pd.DataFrame,
        child_col: str,
        parent: pd.DataFrame,
        parent_col: str,
) -> None:
    # Report whether every ID in child_col exists in parent_col.
    child_ids = set(child[child_col].dropna().unique())
    parent_ids = set(parent[parent_col].dropna().unique())
    missing = child_ids - parent_ids

    label = f"{child_col} -> {parent_col}"
    if not missing:
        print(f"{label} OK")
    else:
        print(f"{label} MISSING {len(missing)}")

def describe_non_finishes(results: pd.DataFrame) -> None:
    # Show how non-finishes show up in the (era) results table.
    classified = results["position"].notna()

    print("\nHOW ARE NON-FINISHES REPRESENTED")
    print(f"Position set: {int(classified.sum())}")
    print(f"Position blank: {int((~classified).sum())}")

    # A row is a non-finish when position is blank. The positionText column
    # carries a short code for why

    print("\nNON-FINISH positionText CODES")
    for code, count in results.loc[~classified, "positionText"].value_counts().items():
        print(f"{str(code)}: {count}")

    print("\nstatus TEXT FOR NON-FINISHES (TOP 10)")

    for text, count in results.loc[~classified, "status"].value_counts().head(10).items():
        print(f"{str(text)}: {count}")

def describe_grid_sentinel(results: pd.DataFrame) -> None:
    # grid == 0 is not a real position. It means "no grid slot" (withdrew,
    # pit-lane start, or retired before the start), so it must not be read as
    # starting last.
    sentinel = results["grid"] == 0

    print("\nGRID == 0 SENTINEL")
    print(f"Rows with grid == 0: {int(sentinel.sum())}")
    print(f"Real grid range: {int(results.loc[~sentinel, 'grid'].min())}-{int(results.loc[~sentinel, 'grid'].max())}")

def check_qualifying_coverage(results: pd.DataFrame, qualifying: pd.DataFrame) -> None:
    # Every result should have a matching qualifying row. Report the ones
    # don't, since a left join would leave their qualiPosition blank.
    keys = ["raceId", "driverId"]
    matched = results.merge(qualifying[keys].assign(hasQualifying=True), on=keys, how="left")
    missing = matched["hasQualifying"].isna()

    print("\nQUALIFYING COVERAGE")
    print(f"Results without a qualifying row: {int(missing.sum())}")

def main() -> None:
    results = load_csv("results")
    races = load_csv("races")
    drivers = load_csv("drivers")
    constructors = load_csv("constructors")
    qualifying = load_csv("qualifying")
    status = load_csv("status")

    # Attach the season year to every result, then keep only the era we model
    results = results.merge(races[["raceId", "year", "round"]], on="raceId", how="left")
    era = results[results["year"] >= ERA_START_YEAR].copy()

    print("SANIITY CHECK")
    print("-------------")
    print(f"All results rows: {len(results)}")
    print(f"Era results rows:  {len(era)}  ({int(era['year'].min())}-{int(era['year'].max())})")

    print("\nID INTEGRITY")
    check_ids_match(era, "raceId", races, "raceId")
    check_ids_match(era, "driverId", drivers, "driverId")
    check_ids_match(era, "constructorId", constructors, "constructorId")
    check_ids_match(era, "statusId", status, "statusId")

    # Bring in the human-readable status text so DNFs are easy to read
    era = era.merge(status, on="statusId", how="left")
    describe_non_finishes(era)
    describe_grid_sentinel(era)
    check_qualifying_coverage(era, qualifying)

if __name__ == "__main__":
    main()


