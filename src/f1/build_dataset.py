import pandas as pd

DATA_DIR="data/"
ERA_START_YEAR = 2014
OUTPUT_FILE = "data/race_weekend.csv"

def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR + f"{name}.csv", na_values=r"\N")

def build_race_weekend() -> pd.DataFrame:
    results = load_csv("results")
    races = load_csv("races")
    drivers = load_csv("drivers")
    constructors = load_csv("constructors")
    qualifying = load_csv("qualifying")
    status = load_csv("status")

    # Attach race info (year, round, circuit) and keep only the era we model.
    races = races[["raceId", "year", "round", "circuitId", "name"]].rename(
        columns={"name": "race"}
    )
    df = results.merge(races, on="raceId", how="left")
    df = df[df["year"] >= ERA_START_YEAR].copy()

    # Attach human-readable driver and constructor names.
    drivers = drivers.assign(driver=drivers["forename"] + " " + drivers["surname"])
    df = df.merge(drivers[["driverId","driver"]], on="driverId", how="left")

    constructors = constructors[["constructorId", "name"]].rename(
        columns={"name": "constructor"}
    )
    df = df.merge(constructors, on="constructorId", how="left")

    #Attach the pure qualifying result. results.grid is the race grid, so we
    # rename qualifying's position to avoid clashing with the race position
    qualifying = qualifying[["raceId","driverId","position"]].rename(
       columns={"position": "qualiPosition"}
    )
    df = df.merge(qualifying, on=["raceId", "driverId"], how="left")

    # Attach the human-readable status text.
    df = df.merge(status, on="statusId", how="left")

    # grid == 0 is a sentinel for "no grid slot" not a real position, so treat
    # it as missing
    df["grid"] = df["grid"].where(df["grid"] != 0)

    return df

def main() -> None:
    df = build_race_weekend()

    # Keep one row per driver per race, with the columns we actually need.
    columns = [
        "raceId", "year", "round", "race", "circuitId",
        "driverId", "driver", "constructorId", "constructor",
        "grid", "qualiPosition",
        "positionOrder", "positionText", "points", "status",
    ]
    df = df[columns]

    print("RACE WEEKEND DATASET")
    print("--------------------")
    print(f"Rows: {len(df)}")
    print(f"Races: {df['raceId'].nunique()}")
    print(f"Drivers: {df['driverId'].nunique()}")
    print(f"Constructors: {df['constructorId'].nunique()}")
    print(f"Years: {int(df['year'].min())}-{int(df['year'].max())}")

    print("\nMISSING VALUES")
    for column in ["grid", "qualiPosition", "positionOrder", "status"]:
        print(f"{column}: {int(df[column].isna().sum())}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {OUTPUT_FILE}")

if __name__ == "__main__":
    main()






