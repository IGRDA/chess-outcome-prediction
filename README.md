# Chess Outcome Prediction — Titled Tuesday

Predict the outcome of a Titled Tuesday blitz game **from White's perspective**
(`white win` / `draw` / `white loss`) using only information available **before
the game starts**. Data comes from Chess.com's public PubAPI.

The end-to-end story — EDA, modeling, evaluation, and the writeup — lives in
**[`notebooks/chess_outcome_model.ipynb`](notebooks/chess_outcome_model.ipynb)**.
The `src/` package is a small, leakage-safe pipeline the notebook reuses:

- `src/chess_api.py`, `src/make_dataset.py` — fetch + cache the PubAPI and build
  the leak-safe `data/processed/base_dataset.csv` (one row per game).
- `src/features/` — the family-organized feature layer (`build_feature_matrix`):
  pre-game ratings, Elo expectation, matchup/stage, in-tournament form,
  cross-event recent form, and head-to-head.
- `src/build_features.py` — apply the feature layer and write
  `data/processed/modeling_dataset.csv` with temporal `split` tags.

## Setup & run from a fresh clone

Uses [uv](https://docs.astral.sh/uv/). The modeling stack (xgboost, scikit-learn,
matplotlib, seaborn) is in a separate `notebook` dependency group. The full raw
cache and generated CSVs are not committed; `make_dataset.py` recreates them from
Chess.com's PubAPI, writing cache files under `data/raw/` and processed datasets
under `data/processed/`. All requests use the `User-Agent` in `src/chess_api.py`
by default; override it with `--user-agent` if needed.

```bash
uv sync --group notebook

# Build the datasets. On a fresh clone this fetches PubAPI responses; on reruns it
# reuses the local cache unless you pass --refresh.
uv run python src/make_dataset.py     # -> data/processed/base_dataset.csv
uv run python src/build_features.py   # -> data/processed/modeling_dataset.csv

# Run the notebook end-to-end:
uv run --group notebook jupyter nbconvert --to notebook --execute --inplace \
    notebooks/chess_outcome_model.ipynb
```

If you are not using uv, install the notebook/runtime dependencies with
`python -m pip install -r requirements.txt`, then run the same Python and
`jupyter nbconvert` commands in your environment.

Checks: `uv run pytest`, `uv run ruff check .`, `uv run mypy .`.

## Modeling in brief

- **Split:** temporal (out-of-time) — train Feb 10/17/24, validate Mar 03, test
  Mar 10; three earlier weeks are history-only feature context.
- **Model:** ordinal (cumulative / Frank–Hall) XGBoost — two heads, `P(y≥draw)`
  and `P(y≥win)`, recombined into a 3-class distribution; hyper-parameters chosen
  by leave-one-event-out cross-validation.
- **Baseline:** "the higher-rated player wins."

## ⚠️ Leakage note

Chess.com's game-object rating columns are recorded **after** the match, not
before it. The pipeline reconstructs true pre-game ratings from each player's
most recent earlier game (`src/features/pregame.py`) and excludes the raw
post-match `white_rating` / `black_rating` columns from the model.

With that correction, the honest model sits right on the "higher-rated player
wins" baseline (~0.70 accuracy): pre-game blitz outcomes are mostly determined by
ratings, so the model's value is calibrated probabilities (decisive-class AUC ≈
0.78), not a better argmax. See the notebook writeup for the full evaluation and
next steps.
