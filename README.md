# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/waz5211/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/waz5211/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PS5 trophy's earn rate on Exophase at 2026-11-26T00:00Z. `main.py` (Python 3.13, standard library only) fits, per trophy, a least-squares line of logit(rate) against log time over the case history — progression is fast just after release and slows, while the tracked-player base floods in and dilutes rates, so the net move is roughly linear in log time — then blends that fit in logit space toward a semantic week-one prior: trophy text classifies each trophy as story chapter, named mission, side strand, collectible, grind, 100% completion or platinum, each with its own week-one target rate. The prior's weight grows with how far the target sits into week one (near zero for a 90-minute practice horizon, about half for the 6.5-day final). Quantile spreads are honest logit-scale bands that widen with the horizon, widen further when the history is flat or noisy (a refresh the history has not shown yet is a real risk), and widen a little more for the least predictable classes. Deterministic: same input bytes give the same output bytes.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref={referral_code}&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
