# Ranked leaderboard submissions are blocked

You tried to submit with `--mode leaderboard`. That publishes to the public
competition board and draws on a rate-limit bucket that is separate from the one
your other submissions use — so a search that ranked every candidate would exhaust
it and fill the board with dead ends. Deciding what gets ranked is not yours to make.

Everything you actually need to iterate is available:

- `--test-only` on the scorer — correctness, one submission, fastest.
- The scorer with no flags — correctness *and* the metric you are being ranked on
  within this search.
- `--profile-brev` — Nsight Compute counters from the competition hardware.

Use those. The best kernel this search produces gets submitted for ranking afterwards.
