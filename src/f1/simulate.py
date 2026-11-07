import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = "data/"
INPUT_FILE = "features"
TRAIN_END_YEAR = 2023
SIM_YEAR = 2024
ROLLING_WINDOW = 5
FEATURES = [
    "grid",
    "qualiPosition",
    "teammateDelta",
    "driverForm",
    "constructorForm",
]
TARGET = "positionOrder"

# The points system used throughout the Turbo-Hybrid era.
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR + f"{name}.csv", na_values=r"\N")

def train_model(df: pd.DataFrame) -> HistGradientBoostingRegressor:
    train = df[df["year"] <= TRAIN_END_YEAR]
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(train[FEATURES], train[TARGET])
    return model

def form_for(history: pd.DataFrame, group_col: str, keys: pd.Series) -> pd.Series:
    # Mean of the last ROLLING_WINDOW finishes for each driver or constructor.
    recent = (
        history.groupby(group_col)["positionOrder"]
        .apply(lambda s: s.tail(ROLLING_WINDOW).mean())
    )
    return keys.map(recent)

def teammate_delta(race: pd.DataFrame) -> pd.Series:
    # Same idea as Phase 3, but computed within a single race.
    group = race.groupby("constructorId")["qualiPosition"]
    total = group.transform("sum")
    count = group.transform("count")
    teammate = (total - race["qualiPosition"]) / (count - 1)
    return race["qualiPosition"] - teammate

def build_race_features(race: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    features = race.copy()
    features["driverForm"] = form_for(history, "driverId", race["driverId"])
    features["constructorForm"] = form_for(history, "constructorId", race["constructorId"])
    features["teammateDelta"] = teammate_delta(race)
    return features

def simulate_season(model, season: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # history grows as we go: each race's predicted finishes feed the next
    # race's rolling form, so the season evolves on its own predictions
    results = []

    for _, race in season.groupby("raceId", sort=True):
        features = build_race_features(race, history)
        race = race.copy()
        race["prediction"] = model.predict(features[FEATURES])

        # Rank the field by predicted finish and award championship points
        race = race.sort_values("prediction").reset_index(drop=True)
        race["simPosition"] = race.index + 1
        race["points"] = race["simPosition"].map(POINTS).fillna(0)
        results.append(race)

        # Feed the predicted finishes back in as history for the next race.
        history = pd.concat([
            history,
            race[["driverId","constructorId","simPosition"]].rename(
                columns={"simPosition": "positionOrder"}
            ),
        ])

    return pd.concat(results)

def total_points(results: pd.DataFrame, group_col: str) -> pd.Series:
    return results.groupby(group_col)["points"].sum().sort_values(ascending=False)

def main() -> None:
    df = load_csv(INPUT_FILE)
    model = train_model(df)

    season = df[df["year"] == SIM_YEAR]
    history = df[df["year"] < SIM_YEAR]

    simulated = simulate_season(model, season, history)

    # Actual points, using the same scoring, so the comparison is fair.
    actual = season.copy()
    actual["points"] = actual["positionOrder"].map(POINTS).fillna(0)

    print(f"SIMULATED {SIM_YEAR} SEASON")
    print("----------------------------")
    print(f"Races: {season['raceId'].nunique()}")

    print("\nDRIVERS (simulated vs actual points)")
    sim_drivers = total_points(simulated, "driverId")
    act_drivers = total_points(actual, "driverId")
    names = season.drop_duplicates("driverId").set_index("driverId")["driver"]
    for driver_id in sim_drivers.head(10).index:
        print(f"{names[driver_id]:<22} sim {sim_drivers[driver_id]:>5.0f} actual {act_drivers.get(driver_id,0):>5.0f}")

    print("\nCONSTRUCTORS (simulated vs actual points)")
    sim_teams = total_points(simulated, "constructorId")
    act_teams = total_points(actual, "constructorId")
    team_names = season.drop_duplicates("constructorId").set_index("constructorId")["constructor"]
    for team_id in sim_teams.index:
        print(f"{team_names[team_id]:<22} sim {sim_teams[team_id]:>5.0f} actual {act_teams.get(team_id,0):>5.0f}")

    print(f"\nSimulated champion: {names[sim_drivers.index[0]]}")
    print(f"Actual champion:    {names[act_drivers.index[0]]}")

if __name__ == "__main__":
    main()







