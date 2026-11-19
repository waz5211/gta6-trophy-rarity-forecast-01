"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target.

Model, per trophy, in three parts:

1. Trend fit.  Over the case's history, fit logit(rate) against
   ln(1 + hours since the first read) by least squares.  Player progression
   is fast just after release and slows, while the tracked-player base floods
   in and dilutes rates, so the net move is roughly linear in log time and
   the fit extrapolates with built-in damping.  A flat history (launch night)
   gives a flat fit, which is honest: nothing has moved yet.

2. Semantic prior.  Trophy text (name, description, tier) classifies each
   trophy -- story chapter, named mission, side strand, collectible, grind,
   completion, platinum -- and each class gets a week-one target rate: the
   rate a typical trophy of that kind reaches about seven days after release,
   when the early cohort has played through the story and the tracked base
   has grown.  The prior is blended into the fit in logit space, weighted by
   how far the target sits into week one (about nothing for a 90-minute
   practice horizon, about half for the 6.5-day final horizon), so the
   history always keeps a voice.

3. Spread.  Quantiles are the median shifted on the logit scale by the
   standard normal's quantiles times a sigma that grows with the horizon
   (random-walk term), widens when the history is flat or noisy (a refresh
   or dilution surge the history has not shown yet is a real risk), and
   widens a little more for rare, slow classes (platinum, collectibles,
   grinds) whose week-one rates are the least predictable.

Standard library only, no network, deterministic: same input bytes give the
same output bytes.  Reads every <id>.in in $MAKEFUN_INPUT_DIR (default
/input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# A week after release, in hours: the age the semantic priors are written for.
WEEK_HOURS = 7 * 24.0


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, as JavaScript Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9 accurate)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Trophy semantics: classify from the trophy's own text, generically.
# ---------------------------------------------------------------------------

CHAPTER_WORDS = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
                 "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

# Week-one target rate (percent) per class, and how uncertain that class is.
PRIOR = {
    "platinum": (0.15, 0.35),
    "completion": (0.40, 0.35),   # 100% completion
    "chapter": (None, 0.20),      # per-chapter rate filled in below
    "finale": (16.0, 0.20),
    "mission": (18.0, 0.25),
    "strand": (8.0, 0.30),        # final mission for a side character
    "collectible": (4.0, 0.35),
    "grind": (2.5, 0.35),
    "online": (3.0, 0.40),
    "misc": (10.0, 0.30),
}

CHAPTER_PRIOR = {1: 38.0, 2: 30.0, 3: 24.0, 4: 19.0, 5: 15.0, 6: 12.0}

COLLECTIBLE_RE = re.compile(
    r"collectible|card set|waink|discover|explore|meditat|every spot|"
    r"all of leonida|register \d+ vehicles|purchase all|garage propert")
GRIND_RE = re.compile(
    r"\d+ stunts|gold medal|complete \d+ challenges|win \d+ different|"
    r"win a .* race at every|earn \$\d|spend \$\d|launder")
ONLINE_RE = re.compile(r"online|multiplayer|co-op|with (other|another) player")


def classify(trophy: dict) -> tuple[str, float, float]:
    """Return (class, week-one prior percent, class sigma bump)."""
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        rate, bump = PRIOR["platinum"]
        return "platinum", rate, bump
    if "100% completion" in text or "100% complet" in text:
        rate, bump = PRIOR["completion"]
        return "completion", rate, bump
    m = re.search(r"complete chapter (\w+)", text)
    if m and m.group(1) in CHAPTER_WORDS:
        ch = CHAPTER_WORDS[m.group(1)]
        return "chapter", CHAPTER_PRIOR.get(ch, 15.0), PRIOR["chapter"][1]
    if "final mission for" in text:
        rate, bump = PRIOR["strand"]
        return "strand", rate, bump
    if "final mission" in text and ("complete the final mission" in text
                                    or "complete the final" in text):
        rate, bump = PRIOR["finale"]
        return "finale", rate, bump
    if ONLINE_RE.search(text):
        rate, bump = PRIOR["online"]
        return "online", rate, bump
    if COLLECTIBLE_RE.search(text):
        rate, bump = PRIOR["collectible"]
        return "collectible", rate, bump
    if GRIND_RE.search(text):
        rate, bump = PRIOR["grind"]
        return "grind", rate, bump
    if text.startswith("complete ") or "complete " in text:
        rate, bump = PRIOR["mission"]
        return "mission", rate, bump
    rate, bump = PRIOR["misc"]
    return "misc", rate, bump


# ---------------------------------------------------------------------------
# Per-trophy forecast.
# ---------------------------------------------------------------------------

def fit_logtime(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least squares of logit(rate) on ln(1 + hours); returns (a, b, resid_sd).

    Falls back to a flat fit when there is too little to fit on.
    """
    if len(points) < 3:
        ys = [y for _, y in points] or [logit(0.5)]
        return sum(ys) / len(ys), 0.0, 0.0
    xs = [math.log1p(t) for t, _ in points]
    ys = [y for _, y in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, len(resid) - 2))
    return a, b, sd


def forecast_trophy(trophy: dict, history: list[dict], case: dict,
                    first_at: datetime, target_at: datetime,
                    horizon_days: float) -> list[float | int]:
    idx = trophy["_index"]
    # Last non-null rate, walking back; and the (time, logit rate) series.
    series: list[tuple[float, float]] = []
    last_rate: float | None = None
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates):
            continue
        r = rates[idx]
        if r is None:
            continue
        t = (parse_time(read["at"]) - first_at).total_seconds() / 3600.0
        if t < 0:
            t = 0.0
        series.append((t, logit(float(r))))
        last_rate = float(r)
    cls, prior_rate, class_bump = classify(trophy)
    if last_rate is None:
        # No read ever listed this trophy: fall back to the class prior.
        median = prior_rate
        sigma = 1.6
    else:
        if last_rate >= 95.0:
            # An opening-mission trophy everyone already has: it stays ~full.
            median = min(99.5, last_rate + 0.5 * (100.0 - last_rate))
            sigma = 0.15
            return quantiles(median, sigma)
        a, b, resid_sd = fit_logtime(series)
        t_target = (target_at - first_at).total_seconds() / 3600.0
        fit_logit = a + b * math.log1p(max(0.0, t_target))
        # Never let a steep early slope run the median away: cap the fit's
        # move from the last read to +/-3 logit units (about a 20x move).
        last_logit = logit(last_rate)
        fit_logit = min(last_logit + 3.0, max(last_logit - 3.0, fit_logit))
        # How far the target sits into week one, from the checkpoint's age.
        t_check = (parse_time(case["at"]) - first_at).total_seconds() / 3600.0
        span = WEEK_HOURS - max(0.0, t_check)
        frac = 0.0 if span <= 0 else min(1.0, max(0.0, (t_target - t_check) / span))
        # Blend fit and prior; the prior never takes more than half the say.
        w_prior = 0.5 * frac
        med_logit = (1.0 - w_prior) * fit_logit + w_prior * logit(prior_rate)
        # Do not let the blend drag a trophy far below where it already is
        # when the history is rising, or far above its class ceiling.
        ceiling = {"platinum": 2.0, "completion": 5.0}.get(cls)
        median = sigmoid(med_logit)
        if ceiling is not None:
            median = min(median, ceiling)
        median = clamp_pct(median)
        # Spread: horizon random walk + flat/noisy-history doubt + class.
        flat = len({round(y, 9) for _, y in series}) <= 1
        sigma = math.sqrt(
            (0.45 * math.sqrt(horizon_days)) ** 2
            + (0.8 if flat else 0.0) ** 2
            + (min(0.5, resid_sd) if not flat else 0.0) ** 2
            + class_bump ** 2
        )
        sigma = min(2.0, max(0.5, sigma))
    return quantiles(median, sigma)


def quantiles(median: float, sigma: float) -> list[float | int]:
    med_logit = logit(median)
    out: list[float | int] = []
    prev = 0.0
    for level in LEVELS:
        q = round4(clamp_pct(sigmoid(med_logit + norm_ppf(level) * sigma)))
        q = max(q, prev)  # rounding must never break monotonicity
        out.append(q)
        prev = q
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    first_at = parse_time(history[0]["at"])
    target_at = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    trophies = case["trophies"]
    for i, trophy in enumerate(trophies):
        trophy["_index"] = i
    rows = []
    for trophy in trophies:
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(trophy, history, case, first_at,
                                          target_at, horizon_days)})
    for trophy in trophies:
        trophy.pop("_index", None)
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
