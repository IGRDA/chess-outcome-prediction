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

## Data & split

**Data selection.** I collect **eight** consecutive weekly Titled Tuesday events,
not just the two the assignment names. A game's pre-game features (reconstructed
pre-game ratings, in-tournament and cross-event form, head-to-head) only exist when
a player has *strictly earlier* games to read, so I add three **past** weeks
(Jan 20 / 27, Feb 03) as pure history and the **in-between** weeks (Feb 17 / 24,
Mar 03) between the two required events. The three earliest weeks are
history-only: they feed features but produce no labeled rows.

**Split.** A **temporal (out-of-time)** split over the labeled weeks:

| split | events | role |
|-------|--------|------|
| history | Jan 20 / 27 · Feb 03 | feature context only (no rows) |
| **train** | Feb 10 / 17 / 24 | fit + hyper-parameter tuning |
| **validation** | Mar 03 | early stopping + model selection |
| **test** | Mar 10 | touched **once**, final verdict |

Why this split:

- **Predicts the future from the past → no time travel / leakage.** A row is only
  ever scored from information that predates it, exactly how the model would run
  live.
- **Measures generalization to new users / cold-start.** Each later week's field
  contains players unseen in training, so a future event is an honest read on how
  the model handles **new users** — the regime a random split would contaminate by
  leaking a new player's other same-day games into train.
- **History-only events are feature context, not rows.** The three earliest weeks
  populate form / H2H / pre-game-rating features but emit no labeled rows.

Unlike a random split, a temporal one does not leak *tournament-level* structure
(shared field, interlocking Swiss pairings) across the boundary. The split is
defined once in `src/events.py` and imported by both data collection and the
feature build, so it stays a single source of truth. See the notebook for the full
argument and an event-timeline figure.

## Modeling in brief

- **Model:** ordinal (cumulative / Frank–Hall) XGBoost — two heads, `P(y≥draw)`
  and `P(y≥win)`, recombined into a 3-class distribution; hyper-parameters chosen
  by leave-one-event-out cross-validation.
- **Baseline:** "the higher-rated player wins."

## Writeup: split, model quality, next steps

The full writeup lives in the notebook's **Section 10**; in brief:

- **Split & why.** Temporal, out-of-time (train Feb 10/17/24 · validate Mar 03 ·
  test Mar 10; earlier weeks are history-only feature context). It mirrors live
  use — predict *this* week from *past* weeks — and avoids the same-tournament
  leakage a random split invites (shared pairings, field, and same-day ratings
  straddling the boundary). The test event is touched exactly once.
- **Model quality (honest opinion).** Correct and leak-free, but as a 3-class
  classifier it **ties the "higher-rated wins" baseline** (~0.70 accuracy; the edge
  is ~0.001, within noise): once leakage is removed, a scalar rating already carries
  almost all the pre-game signal. Its real edge is **probabilistic** — win/draw/loss
  scores with decisive-class AUC ≈ 0.78 the baseline can't give; draws stay
  near-unpredictable (AUC ≈ 0.47).
- **Is it "good"? It depends on the goal — which also fixes the metric.** For a
  *decisive label* the baseline is the simpler, equally good choice. For *draw
  detection*, up-weighting the draw class (oversampling / stratified sampling /
  weighted loss, tested in the notebook's §9d) lifts draw precision/recall/F1 from 0
  to ≈ 0.09 / 0.17 / 0.12 and macro-F1 to ≈ 0.50, at the cost of accuracy
  (0.70 → 0.63). For *calibrated probabilities*, judge on log-loss / AUC, not argmax.
- **Next steps.** Define the goal / product use case first — it fixes the metric
  and the model choice (e.g. address class imbalance if raw prediction is the
  goal). Talk to stakeholders, and above all **collect more data** (a full
  season) so form / head-to-head / draw signals become learnable. Plan
  productionization: a scheduled ETL over the PubAPI and a retraining cadence that
  preserves the leakage guard and temporal split. Product usages — explainability,
  matchup previews, broadcast probabilities, upset alerts, fairer pairing — become
  worthwhile once the model earns a probabilistic edge over raw rating.

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
