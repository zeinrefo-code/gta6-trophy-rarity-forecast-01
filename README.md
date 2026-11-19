# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeinrefo-code/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeinrefo-code/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the week-one earn rates of Grand Theft Auto VI's PlayStation 5
trophies on Exophase. Given a case file holding every recorded read of the
game's trophy page up to a checkpoint — each read is the players-tracked count
and every trophy's earn rate, the percent of tracked players who have earned
it — the program writes, for each of the game's 55 trophies, five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of its earn rate at the case's target, a week
after launch. Scoring is a pinball loss on the logit scale, so the quantiles
have to be honest, not just the median accurate.

## How it works

Exophase's page changes only when Exophase refreshes it, so a history is a
step function: long runs of identical reads, then one jump in which the
tracked-player count and every rate move together. The program first
collapses the history to its **distinct states** — reads between refreshes
carry no new information, and fitting them would weight a stale plateau by
how often the recorder happened to poll it.

For each trophy it then fits a weighted least-squares line of the rate's
**logit on ln(1 + hours since release)** over the distinct states, each state
weighted by sqrt(players tracked). The log-time axis compresses the
extrapolation the way earn curves bend: fast movement in the first days,
flattening as the keenest players finish and the rest catch up.

The extrapolation is **damped** toward the last observed rate,
`median = last + lam * (fitted - last)`, and `lam` is not a hand-set
constant: the program backtests itself inside the case, refitting on every
prefix of the states, forecasting the next state under each candidate
damping, and keeping the one with the lowest pooled absolute logit error.
The spread is a random walk on the logit scale whose per-day volatility is
that backtest's one-step error, plus the fit's own residual in quadrature.
A case whose states carry no trend signal (a single state, like the
launch-night practice cases) falls back to persistence with the baseline
spread — the honest forecast when nothing has moved yet.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic: two runs on the same case
write the same bytes. In the grader's sandbox the directories default to
`/input` and `/output`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=bxjfe4p7&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
