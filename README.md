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

## Setup & run

Uses [uv](https://docs.astral.sh/uv/). The modeling stack (xgboost, scikit-learn,
matplotlib, seaborn) is in a separate `notebook` dependency group.

```bash
uv sync --group notebook

# (Optional) rebuild datasets from the cached raw API responses:
uv run python src/make_dataset.py     # -> data/processed/base_dataset.csv
uv run python src/build_features.py   # -> data/processed/modeling_dataset.csv

# Run the notebook end-to-end:
uv run --group notebook jupyter nbconvert --to notebook --execute --inplace \
    notebooks/chess_outcome_model.ipynb
```

Checks: `uv run pytest`, `uv run ruff check .`, `uv run mypy .`.

## Modeling in brief

- **Split:** temporal (out-of-time) — train Feb 10/17/24, validate Mar 03, test
  Mar 10; three earlier weeks are history-only feature context.
- **Model:** ordinal (cumulative / Frank–Hall) XGBoost — two heads, `P(y≥draw)`
  and `P(y≥win)`, recombined into a 3-class distribution; hyper-parameters chosen
  by leave-one-event-out cross-validation.
- **Baseline:** "the higher-rated player wins."

## ⚠️ Leakage note (worth reading)

Chess.com's game-object ratings are recorded **after** the game. A first model
scored a suspicious **93% test accuracy** by differencing a player's post-game
rating against an earlier-round rating to recover the result. The fix:

1. removed the two in-tournament prior-*rating* features
   (`src/features/tournament_form.py`), and
2. **re-based the whole rating family on reconstructed true pre-game ratings**
   (`src/features/pregame.py`) — each player's rating from their most recent
   *earlier* game (previous round, or previous week for round 1) — and excluded
   the raw post-game ratings from the model.

The proof, the "when," and the before/after are **Section 3 of the notebook**.
Once the leak is gone, the honest model sits right on the baseline (~0.70
accuracy): pre-game blitz outcomes are almost entirely determined by ratings, so
the model's real value is *calibrated probabilities* (decisive-class AUC ≈ 0.78),
and draws are effectively unpredictable pre-game. See the notebook writeup for the
full evaluation and next steps.
