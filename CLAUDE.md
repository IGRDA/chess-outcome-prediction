# Agent Notes

This repo is for the Chess.com AI/ML take-home exercise: predict the outcome of
a Titled Tuesday blitz game from White's perspective using only information
available before the game starts.

The highest priority is alignment with the PDF guidance: a clear, runnable,
3-4 hour take-home solution with honest tradeoffs. Good engineering matters, but
only when it helps that goal.

## Keep It Simple

- Favor a small, reproducible Python pipeline over a complex model.
- The target time for the exercise is 3-4 hours; do not overbuild.
- Keep modules focused: API fetching in `src/chess_api.py`, feature logic in
  `src/features.py`, training/evaluation orchestration in `src/train.py`.
- Prefer clear baselines and honest evaluation over leaderboard chasing.
- Do not commit changes unless the user explicitly asks.

## Engineering Practices

- Apply ETC: make changes easier to change later, but avoid speculative
  abstractions.
- Keep logic DRY when duplication hides a decision, not when two lines merely
  look similar.
- Preserve orthogonality: data fetching, feature creation, training, and
  reporting should not know unnecessary details about each other.
- Prefer small functions with clear contracts. Validate important preconditions
  near boundaries and keep invariants obvious in tests.
- Pull complexity downward into helpers so the notebook and training script read
  like the story of the solution.
- Make modules deep enough to be useful: simple interface, contained
  implementation details.
- Prevent software entropy. Fix small broken-window issues while nearby, but do
  not wander into unrelated refactors.
- Use tracing bullets before prototypes here: build the narrow end-to-end path
  first, then improve it.
- Test behavior, edge cases, and invariants. Keep tests fast and network-free.
- Prefer actionable options over defensive explanations when something is
  blocked or weak.

## Assignment Constraints

- Use Chess.com's public PubAPI. Include a `User-Agent` header on requests.
- Suggested events:
  - February 10, 2026:
    `https://api.chess.com/pub/tournament/titled-tuesday-blitz-february-10-2026-6221327`
  - March 10, 2026:
    `https://api.chess.com/pub/tournament/titled-tuesday-blitz-march-10-2026-6277141`
- Each row should represent one game.
- Target label is White's result: `white_win`, `white_loss`, or `draw`.
- Deliverable should be runnable and include a brief explanation of split choice,
  model quality, and what to do next with more time.

## API Endpoints

- Tournament info: `/pub/tournament/{id}` returns tournament metadata, players,
  and round URLs.
- Round traversal: fetch each round URL from the tournament metadata to discover
  group URLs.
- Round/group games: `/pub/tournament/{id}/{round}/{group}` returns the games
  for that round/group.
- Player profiles, optional: `/pub/player/{username}` returns player metadata
  that may help construct pre-game features.
- Player stats, optional: `/pub/player/{username}/stats` returns rating summaries
  across game types that may help construct pre-game features.
- Treat optional player profile/stats fields carefully because some values may be
  current snapshots, not historical values from game time.

## Avoid Leakage

Only use features known before the game starts. Safe starting features include:

- Reconstructed White and Black pre-game ratings from strictly earlier games
- Rating difference and absolute rating difference
- Elo expected score
- Round number
- Tournament state before the current round, such as player score so far,
  games played so far, prior wins/losses/draws, and prior opponent strength

Do not use these as input features:

- `pgn`
- `fen`
- `eco`
- `end_time`
- `termination`
- Final result fields, except to create the target label
- Raw game-object ratings as model inputs; they are post-game snapshots here
- Any feature derived from moves or final position

Be cautious with optional player profile/stats endpoints. Some values may be
current API snapshots rather than values available at game time.

## Evaluation Guidance

- Use a temporal split: train on February 10, 2026 and test on March 10, 2026.
- Include simple baseline: Higher-rated player wins
- Report accuracy, macro F1, and a confusion matrix.
- Draws are rare, so do not rely on accuracy alone.
- State plainly whether the model is useful and where it is weak.

## Local Workflow

Use uv for environment and dependency management:

```bash
uv sync --dev
```

Run checks before handing work back:

```bash
uv run pytest
uv run pytest --cov
uv run ruff check .
uv run ruff format . --check
uv run mypy .
uv run pre-commit run --all-files
```

In a brand-new repo with untracked files, `pre-commit run --all-files` may skip
hooks. In that case, run pre-commit against explicit files before reporting.

## Data Hygiene

- Put optional cached API responses in `data/raw/`.
- Put generated datasets in `data/processed/`.
- Keep generated data out of git unless it is small and useful for reproducible
  review.
- Tests should stay fast and should not hit the network.
