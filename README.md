# 🏎️ F1 Season Simulator

An end-to-end Machine Learning pipeline designed to predict Formula 1 race results and simulate full championship seasons. Unlike static models, this simulator features an **dynamic rolling form feedback loop** — meaning predicted race outcomes directly influence driver and constructor momentum in following events.

---

## 💡 How It Works (The Engine Concept)

Traditional sports prediction models treat events in isolation. In F1, however, momentum matters. Our engine models season dynamics through a multi-stage approach:

1. **Train / Test Split by Year:** Trains exclusively on past seasons ($< N$) to predict target season $N$, eliminating any data leakage.
2. **Dynamic Feature Building:** Generates live driver form, team form, and intra-team qualifying differentials (`teammateDelta`) right before each Grand Prix.
3. **Sequential Simulation:** The model ranks drivers for Race 1, awards standard FIA points, and **feeds the predicted finishes back into the historical dataset** as the new "truth" for Race 2.

---

## 🧱 Component Architecture & Data Flow

The project is structured into modular scripts under `src/f1/`, each handling a specific stage of the pipeline:

```text
[ Data Cleaning & Integrity ] ──► [ Dataset & Features ] ──► [ Model Training & Baseline ] ──► [ Season Engine ]
       sanity_check.py                 dataset.py                 baseline.py / model.py             simulate.py
```

### 🗂️ Module Breakdown

| Module | Core Responsibility | Key Logic / Output                                                                                                 |
| :--- | :--- |:-------------------------------------------------------------------------------------------------------------------|
| **`sanity_check.py`** | Data Integrity & Validation | Audits raw CSV files (2014–2024), validates qualifying coverage, and encodes DNF statuses.                         |
| **`dataset.py`** | Pipeline & Feature Engineering | Merges race weekends, calculates rolling averages (`driverForm`, `constructorForm`), and computes `teammateDelta`. |
| **`baseline.py`** | Benchmark Reference | Establishes a simple baseline model (predicting finish = grid position) to measure ML lift (MAE: 3.605).           |
| **`model.py`** | Model Training & Tuning | Trains the `HistGradientBoostingRegressor` on historical data up to season $N-1$ (Achieves MAE: 3.416).            |
| **`simulate.py`** | Full Season Simulator | Runs the chronological loop over season $N$, updating history on the fly and calculating final WDC/WCC standings.  |

---

## 🔄 Dynamic Feedback Loop in Action

Here is how data flows sequentially during a season simulation inside `simulate.py`:

```text
[ 1. Incoming Race ] ──► build_race_features() ──► Computes driverForm & constructorForm from history
                               │
                               ▼
[ 2. ML Prediction ] ──► model.predict()       ──► Outputs predicted finish score for each driver
                               │
                               ▼
[ 3. Race Outcome ]  ──► sort_values() & PTS   ──► Assigns positions (1–20) & awards standard FIA points
                               │
                               ▼
[ 4. Feedback Loop ] ──► Append to 'history'   ──► Updates form metrics for the NEXT Grand Prix
```

---

## 📊 Benchmark & Results (2024 Season Test)

Evaluating the simulator against the 2024 season (trained on 2014–2023 data):

* **Simulated Champion:** Max Verstappen *(Actual: Max Verstappen)*
* **Top 4 Driver Order:** Verstappen ➔ Norris ➔ Leclerc ➔ Piastri *(100% match with real-world top 4)*

### Drivers' Championship (WDC Top 5)

| Driver | Simulated Points | Actual Points (GP Only) | Delta |
| :--- | :---: | :---: | :---: |
| **Max Verstappen** | **456** | **396** | +60 |
| **Lando Norris** | **375** | **338** | +37 |
| **Charles Leclerc** | **314** | **324** | -10 |
| **Oscar Piastri** | **263** | **265** | -2 |
| **Carlos Sainz** | **225** | **261** | -36 |

> **Note:** Actual points are benchmarked strictly against standard Grand Prix top-10 scoring tables (excluding sprint races and fastest lap bonuses) to ensure a 1:1 metric comparison.

---

## 🛠️ Project Structure

```text
.
├── data/                 # Raw and processed F1 CSV datasets (git-ignored)
├── src/
│   └── f1/
│       ├── sanity_check.py  # Data integrity & validation
│       ├── dataset.py       # Dataset merger & feature pipeline
│       ├── baseline.py      # MAE baseline script
│       ├── model.py         # HistGradientBoosting training logic
│       └── simulate.py      # Main simulation engine
├── .gitignore            # Excludes data/ and virtual environment
├── pyproject.toml        # Project dependencies & environment configuration
└── README.md             # Project documentation
```

---

## 🚀 Quick Start

### 1. Requirements
* Python 3.10+
* Dependencies: `pandas`, `scikit-learn`

### 2. Running the Simulator
To execute the full season simulation with terminal output:

```bash
python src/f1/simulate.py
```

---

## 🔮 Roadmap & Future Improvements

- [ ] **Weather & Track Dynamics:** Inject rain indicators, circuit Safety Car probability, and driver wet-weather performance metrics.
- [ ] **Monte Carlo Simulations:** Add race-day randomness and DNF probabilities to run 10,000 season iterations.
- [ ] **Two-Stage Model Pipeline:** Build a standalone qualifying predictor to enable pre-season forecasts (eliminating real grid dependency).
- [ ] **Interactive Streamlit Web Dashboard:** Allow users to tweak starting grids and simulate custom "What-If" scenarios in real time.