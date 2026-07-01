# Chess Outcome Prediction — Titled Tuesday

Predict the outcome of a Titled Tuesday blitz game **from White's perspective**
(`white win` / `draw` / `white loss`) using only information available **before
the game starts**. Data comes from Chess.com's public PubAPI. Each row is one
game; the target is White's result.

The end-to-end story — EDA, modeling, evaluation, and the writeup — lives in
**[`notebooks/chess_outcome_model.ipynb`](notebooks/chess_outcome_model.ipynb)**
(rendered copy: `notebooks/chess_outcome_model.html`). The notebook and the
`src/` package are the source of truth; this README summarizes them.

## Repository layout

```
src/
  chess_api.py        PubAPI client: cached fetch + retries, sends a User-Agent header
  parsing.py          Flatten raw game JSON into typed rows; derive the win/draw/loss label
  make_dataset.py     Collect the 8 events -> data/processed/base_dataset.csv (one row per game)
  events.py           The 8 events and their split roles — single source of truth for the split
  features/           Leakage-safe, family-organized feature layer (build_feature_matrix):
    pregame.py          reconstruct true pre-game ratings from strictly earlier games
    rating.py           Elo expected score, rating diff / abs diff / average
    matchup.py          favorite flags, round stage, rating-edge × stage
    tournament_form.py  in-event prior score / games / opponents / streak (earlier rounds only)
    recent_form.py      cross-event last-5 form (strictly earlier games)
    head_to_head.py     prior head-to-head record between the two players
    build.py            applies the families in order; LEAKY_COLUMNS guard
  build_features.py   Apply the feature layer -> data/processed/modeling_dataset.csv (+ split tags)
notebooks/            Modeling + evaluation deliverable (chess_outcome_model.ipynb / .html)
tests/                Fast, network-free unit tests for parsing, features, and the split
data/
  raw/                Cached PubAPI responses (generated; not committed)
  processed/          base_dataset.csv, modeling_dataset.csv, run_summary.json (generated)
```

Data collection, feature creation, the split definition, and modeling are kept
orthogonal: `src/events.py` defines the split once and is imported by both data
collection and the feature build, so it never drifts.

## Setup & reproduce (from a fresh clone)

Uses [uv](https://docs.astral.sh/uv/). The modeling stack (xgboost, scikit-learn,
matplotlib, seaborn) is in a separate `notebook` dependency group. The raw cache
and generated CSVs are **not committed**; the pipeline recreates them from
Chess.com's PubAPI, writing cache files under `data/raw/` and datasets under
`data/processed/`. All requests send the `User-Agent` in `src/chess_api.py` by
default (override with `--user-agent`).

```bash
uv sync --group notebook

# Build the datasets. On a fresh clone this fetches PubAPI responses; on reruns it
# reuses the local cache unless you pass --refresh.
uv run python src/make_dataset.py     # -> data/processed/base_dataset.csv
uv run python src/build_features.py   # -> data/processed/modeling_dataset.csv

# Run the notebook end-to-end (deterministic; RNG seed is fixed):
uv run --group notebook jupyter nbconvert --to notebook --execute --inplace \
    notebooks/chess_outcome_model.ipynb
```

The run is deterministic, so re-executing reproduces the same numbers. If you are
not using uv, install with `python -m pip install -r requirements.txt`, then run
the same Python and `jupyter nbconvert` commands.

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
| **validation** | Mar 03 | model selection + reported metrics |
| **test** | Mar 10 | touched **once**, final verdict |

Why this split:

- **Predicts the future from the past → no time travel / leakage.** A row is only
  ever scored from information that predates it, exactly how the model would run
  live.
- **Measures generalization to new users / cold-start.** Each later week's field
  contains players unseen in training, so a future event is an honest read on how
  the model handles **new users** — the regime a random split would contaminate by
  leaking a new player's other same-day games into train.
- **Three training events enable leave-one-week-out CV.** Model selection
  cross-validates by event, so a chosen config must generalize to an unseen week.

Unlike a random split, a temporal one does not leak *tournament-level* structure
(shared field, interlocking Swiss pairings, the same day's ratings straddling the
boundary) across the split. The split is defined once in `src/events.py`. See the
notebook for the full argument and an event-timeline figure.

## Features

Every model feature is white-perspective and known before move one. The pipeline
uses **26** features across five families (full one-sentence glossary in the
notebook, §0):

- **Pre-game rating** — `white_pregame_rating`, `black_pregame_rating`,
  `rating_diff`, `abs_rating_diff`, `avg_rating`, `white_expected_score`
  (reconstructed pre-game ratings and the Elo expectation derived from them).
- **Matchup / stage** — `white_is_favorite`, `favorite_magnitude`, `round`,
  `round_norm`, `rounds_remaining`, `rating_edge_scaled_by_round`, `is_late_round`
  (who is favored, by how much, and where in the event the game sits).
- **Tournament form** — `prior_*` differences in this event's score, games played,
  average opponent rating, active streak, and score rate, using earlier rounds only.
- **Recent form** — `recent_last5_*` and `current_color_last5_*` differences over
  each player's up-to-5 strictly earlier collected games (overall and same-color).
- **Head-to-head** — `h2h_*`: prior meetings between the two players and White's
  score rate in them, overall and for same-color meetings.

The raw post-game `white_rating` / `black_rating` and the collinear
`black_expected_score` are excluded (see the leakage note).

## Modeling in brief

- **Model:** ordinal (cumulative / Frank–Hall) XGBoost — two heads, `P(y≥draw)`
  and `P(y≥win)`, recombined into a 3-class distribution; features and
  hyper-parameters chosen by leave-one-event-out cross-validation.
- **Baseline:** "the higher-rated player wins."
- **Metrics:** accuracy, macro-F1, per-class report, a confusion matrix, and
  one-vs-rest ROC AUC — because draws are rare, accuracy alone is not enough.

## Writeup: split, model quality, next steps

The full writeup lives in the notebook's **§10**; in brief:

- **Split & why.** Temporal, out-of-time (train Feb 10/17/24 · validate Mar 03 ·
  test Mar 10; earlier weeks are history-only feature context). It mirrors live
  use — predict *this* week from *past* weeks — and avoids the same-tournament
  leakage a random split invites (shared pairings, field, and same-day ratings
  straddling the boundary). The test event is touched exactly once.
- **Model quality (honest opinion).** Correct and leak-free, but as a 3-class
  classifier it **ties the "higher-rated wins" baseline** (~0.70 accuracy /
  ~0.48 macro-F1; the edge is ~0.001 accuracy, within noise): once leakage is
  removed, a scalar rating already carries almost all the pre-game signal. Its real
  edge is **probabilistic** — win/draw/loss scores with decisive-class ROC AUC ≈
  0.78 the baseline can't give. **Draws stay near-unpredictable** (AUC ≈ 0.47,
  below chance) and the model never argmaxes a draw; the thin draw band is largely
  irreducible in blitz between similar-Elo titled players.
- **Is it "good"? It depends on the goal — which also fixes the metric.** For a
  *decisive label* the baseline is the simpler, equally good choice. For *draw
  detection*, up-weighting the draw class (oversampling / stratified sampling /
  weighted loss, tested in §9d) lifts draw precision/recall/F1 from 0 to ≈ 0.09 /
  0.17 / 0.12 and macro-F1 to ≈ 0.50, at the cost of accuracy (0.70 → 0.63). For
  *calibrated probabilities*, judge on log-loss / AUC, not argmax. Accuracy is
  high only when ratings are far apart (0–50 Elo ≈ 0.57 → 400+ ≈ 0.91) and slides
  toward a coin-flip in late rounds as the field converges — the physics of Elo,
  not a fragile regime.
- **Next steps.** Define the goal / product use case first — it fixes the metric
  and the model choice (e.g. address class imbalance if raw prediction is the
  goal). Talk to stakeholders, and above all **collect more data** (a full season)
  so form / head-to-head / draw signals leave cold-start and become learnable. Plan
  productionization: a scheduled ETL over the PubAPI and a retraining cadence that
  preserves the leakage guard and temporal split. Product uses — explainability,
  matchup previews, broadcast probabilities, upset alerts, fairer pairing — become
  worthwhile once the model earns a probabilistic edge over raw rating.

## ⚠️ Leakage note

Chess.com's game-object rating columns (`white_rating` / `black_rating`) are
recorded **after** the match, not before it. The pipeline reconstructs true
pre-game ratings from each player's most recent earlier game
(`src/features/pregame.py`) and excludes the raw post-match columns from the model.
The feature build also drops move/result-derived fields — `pgn`, `fen`, `eco`,
`end_time`, `termination`, and result fields (kept only to build the label) — via
the `LEAKY_COLUMNS` guard in `src/features/build.py`.

With that correction, the honest model sits right on the "higher-rated player
wins" baseline (~0.70 accuracy): pre-game blitz outcomes are mostly determined by
ratings, so the model's value is calibrated probabilities (decisive-class AUC ≈
0.78), not a better argmax. See the notebook writeup for the full evaluation and
next steps.
