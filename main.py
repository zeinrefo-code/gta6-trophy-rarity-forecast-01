"""Week-one earn-rate forecast for every Grand Theft Auto VI PS5 trophy.

Entry for make.fun's gta6-trophy-rarity-forecast-01 (program/v1, Python 3.13,
standard library only). The grader runs this file once per case with a single
<id>.in in the input directory; for each case it writes <id>.out holding five
quantiles (the case's levels) of every trophy's earn rate -- the percent of
the players Exophase tracks for the game who have earned it -- at the case's
target. Scoring is a pinball loss on the logit scale, so the quantiles have
to be honest, not just the median accurate.

What the data is: the house records Exophase's trophy page on a fixed clock,
but the page itself changes only when Exophase refreshes it. A case's history
is therefore a step function: long runs of identical reads, then one jump in
which the tracked-player count and every rate move together. The answer is
the page's state at the target -- the state after the last refresh before it.

Method (a damped trend over distinct states, calibrated inside each case):

  * Collapse the history to its distinct states: a new state begins when the
    players-tracked count or any rate changes. Reads between refreshes carry
    no new information, and fitting them would weight a stale plateau by how
    often the recorder happened to poll it.
  * For each trophy, fit a weighted least-squares line of the rate's logit
    on ln(1 + hours since release) over the distinct states, each state
    weighted by sqrt(players tracked): a state averaged over more players
    is measured more precisely. The log-time axis compresses the
    extrapolation the way earn curves bend -- fast movement in the first
    days, flattening as the keenest players finish and the rest catch up.
  * Damp the extrapolation toward the last observed rate:
        median_logit = last_logit + lam * (fitted_logit(target) - last_logit)
    lam is not a hand-set constant. The program backtests itself inside the
    case: for every prefix of the states it refits, forecasts the next state
    under each candidate lam, and keeps the lam with the lowest pooled
    absolute logit error. A case whose states carry no trend signal (the
    launch-night practice cases hold a single state) falls back to lam = 0,
    plain persistence.
  * Spread: a random walk on the logit scale. Its per-day volatility is the
    pooled backtest one-step error, rescaled by each step's gap; the fit's
    own residual for the trophy adds variance in quadrature. With no
    backtest the volatility defaults to 0.5 per sqrt-day, the house
    baseline's spread. Quantiles are inv_logit(median + z_tau * spread).

The program is deterministic: no randomness, no clock, no network. Two runs
on the same case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves this file's directory off sys.path; nothing else is
# imported, but keep the entrypoint robust if a helper module joins it later.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Rates are printed to 2 decimals; the scorer clamps percents into this range
# before the logit, and so does every logit taken here.
RATE_FLOOR = 0.01
RATE_CEIL = 99.99
# Candidate damping factors for the in-case backtest, mildest first so a tie
# keeps the more conservative forecast.
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Logit volatility per sqrt-day when a case offers no backtest (the house
# persistence baseline's spread), and the band the backtest estimate is held
# to: never so tight that a refresh jump is priced as impossible, never so
# wide that the quantiles stop saying anything.
SIGMA_DEFAULT = 0.5
SIGMA_MIN = 0.2
SIGMA_MAX = 1.5
SPREAD_MIN = 0.05
SPREAD_MAX = 2.5

_NORMAL = NormalDist()


def parse_instant(text):
    """Parse an ISO 8601 timestamp (the cases use a trailing Z for UTC)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_origin(case):
    """Midnight UTC on the game's release date: hour zero of every curve."""
    raw = (case.get("game") or {}).get("release_date", "")
    try:
        day = datetime.fromisoformat(raw)
        return day.replace(hour=0, minute=0, second=0, microsecond=0,
                           tzinfo=timezone.utc)
    except ValueError:
        first = parse_instant(case["history"][0]["slot"])
        return first.replace(hour=0, minute=0, second=0, microsecond=0)


def hours_since(origin, instant):
    return (instant - origin).total_seconds() / 3600.0


def logit_pct(pct):
    """Logit of a percent, clamped to the scorer's range first."""
    p = min(RATE_CEIL, max(RATE_FLOOR, pct))
    return math.log(p / (100.0 - p))


def inv_logit_pct(value):
    """Inverse logit back to a percent, guarded against overflow."""
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round half up to 4 decimals; whole numbers come back as ints so the
    JSON writes 100 where the reference writes 100, not 100.0."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 10000.0
    return int(result) if float(result).is_integer() else result


def distinct_states(case, origin):
    """The history collapsed to the moments the page actually changed.

    A state is (hours since release, players tracked, rates). A read starts
    a new state when its players-tracked count or any of its rates differs
    from the previous read's; the state's time is that first read's slot.
    """
    states = []
    previous_key = None
    for read in case["history"]:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == previous_key:
            continue
        previous_key = key
        states.append(
            {
                "hours": hours_since(origin, parse_instant(read["slot"])),
                "players": read.get("players_tracked") or 0,
                "rates": list(rates),
            }
        )
    return states


def trophy_points(states, index):
    """One trophy's usable fit points over the given states.

    Each point is (x, logit, weight) with x = ln(1 + hours since release)
    (hours floored so a pre-release read keeps 1 + hours positive) and the
    weight sqrt(players tracked). States where the read lacked this trophy
    contribute nothing.
    """
    points = []
    for state in states:
        if index >= len(state["rates"]):
            continue
        rate = state["rates"][index]
        if rate is None:
            continue
        x = math.log(max(0.05, 1.0 + state["hours"]))
        weight = math.sqrt(max(1, state["players"]))
        points.append((x, logit_pct(rate), weight))
    return points


def fit_line(points):
    """Weighted least squares of logit on x. Returns (intercept, slope,
    residual sd); a single point, or no x-variance, fits a flat line through
    the weighted mean with zero residual."""
    total_w = sum(w for _, _, w in points)
    mean_x = sum(w * x for x, _, w in points) / total_w
    mean_y = sum(w * y for _, y, w in points) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, _, w in points)
    if len(points) < 2 or sxx <= 0:
        return mean_y, 0.0, 0.0
    sxy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in points) / total_w
    )
    return intercept, slope, resid


def backtest(states, n_trophies):
    """Choose the damping factor and the random-walk volatility from the
    case's own states.

    For every prefix of at least two states, each trophy is refit on the
    prefix and asked for the next state under every candidate damping; the
    errors pool over trophies and prefixes. Returns (lam, sigma_per_sqrt_day)
    or None when fewer than three distinct states exist (nothing to test a
    trend against: the caller falls back to persistence).
    """
    if len(states) < 3:
        return None
    abs_errors = {lam: [] for lam in LAMBDAS}
    step_errors = []  # (signed logit error at the winning lam, gap in days)
    pending = []  # per (prefix, trophy): {lam: error}, plus the gap
    for k in range(2, len(states)):
        prefix, nxt = states[:k], states[k]
        gap_days = max((nxt["hours"] - prefix[-1]["hours"]) / 24.0, 1e-9)
        x_next = math.log(max(0.05, 1.0 + nxt["hours"]))
        for index in range(n_trophies):
            actual = nxt["rates"][index] if index < len(nxt["rates"]) else None
            if actual is None:
                continue
            points = trophy_points(prefix, index)
            if len(points) < 2:
                continue
            intercept, slope, _ = fit_line(points)
            last_logit = points[-1][1]
            fitted = intercept + slope * x_next
            errors = {}
            for lam in LAMBDAS:
                err = logit_pct(actual) - (last_logit + lam * (fitted - last_logit))
                errors[lam] = err
                abs_errors[lam].append(abs(err))
            pending.append((errors, gap_days))
    if not pending:
        return None
    lam_star = LAMBDAS[0]
    best = None
    for lam in LAMBDAS:
        errs = abs_errors[lam]
        if not errs:
            continue
        mean_abs = sum(errs) / len(errs)
        if best is None or mean_abs < best:
            best = mean_abs
            lam_star = lam
    for errors, gap_days in pending:
        step_errors.append(errors[lam_star] / math.sqrt(gap_days))
    sigma = math.sqrt(sum(e * e for e in step_errors) / len(step_errors))
    return lam_star, min(SIGMA_MAX, max(SIGMA_MIN, sigma))


def last_rates(states, n_trophies):
    """Each trophy's most recent non-null rate across the states."""
    lasts = [None] * n_trophies
    for state in states:
        for index in range(n_trophies):
            if index < len(state["rates"]) and state["rates"][index] is not None:
                lasts[index] = state["rates"][index]
    return lasts


def forecast_case(case):
    """The forecast object for one case: {'trophies': [{id, q}, ...]}."""
    origin = release_origin(case)
    states = distinct_states(case, origin)
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    target_x = math.log(
        max(0.05, 1.0 + hours_since(origin, parse_instant(case["target"])))
    )
    z_at_level = {level: _NORMAL.inv_cdf(level) for level in case["levels"]}

    calibration = backtest(states, n)
    if calibration is None:
        lam, sigma = 0.0, SIGMA_DEFAULT
    else:
        lam, sigma = calibration

    lasts = last_rates(states, n)
    known = sorted(rate for rate in lasts if rate is not None)
    # A trophy no read ever priced borrows the field's middle rate; with no
    # rates at all, half a percent is as honest a prior as any.
    fallback_rate = known[len(known) // 2] if known else 0.5

    rows = []
    for index, trophy in enumerate(trophies):
        points = trophy_points(states, index)
        if points:
            last_logit = points[-1][1]
            if len(points) >= 2:
                intercept, slope, resid = fit_line(points)
                fitted = intercept + slope * target_x
                median_logit = last_logit + lam * (fitted - last_logit)
            else:
                resid = 0.0
                median_logit = last_logit
        else:
            resid = 0.0
            median_logit = logit_pct(fallback_rate)
        spread = math.sqrt((sigma * sigma + 0.25 * resid * resid) * horizon_days)
        spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))
        quantiles = []
        previous = 0.0
        for level in case["levels"]:
            q = round4(inv_logit_pct(median_logit + z_at_level[level] * spread))
            if q < previous:  # rounding must never break the ordering
                q = previous
            quantiles.append(q)
            previous = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
