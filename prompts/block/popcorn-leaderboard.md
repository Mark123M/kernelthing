# Ranked leaderboard submissions are blocked

You tried to submit with `--mode leaderboard`. That publishes to the public
competition board and draws on a rate-limit bucket that is separate from the one
your other submissions use — so a search that ranked every candidate would exhaust
it and fill the board with dead ends. Deciding what gets ranked is not yours to make.

Everything you actually need to iterate is available:

- `--test-only` on the scorer — one unprofiled call at the exact scored shape on
  Modal/B200 with a 120s timeout, followed by Popcorn correctness.
- The scorer with no flags — correctness, the metric you are being ranked on within
  this search, *and* an Nsight Compute profile of the scored shape, captured on the
  competition hardware and printed above the verdict. You do not have to ask for it.

Use those. The best kernel this search produces gets submitted for ranking afterwards.
