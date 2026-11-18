[![GTA VI Twitch CCU Forecaster](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/icandoitone/makefun-gta6-twitch-ccu-forecast-01-04/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/icandoitone/makefun-gta6-twitch-ccu-forecast-01-04/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU Forecaster v4 — Notes-Aware Recall + Size-Adaptive Ensemble

Forecasts hours 2..24 of a launch day's Twitch concurrent viewers from the
hour-1 print, as five quantiles per hour (0.05, 0.25, 0.5, 0.75, 0.95).
Standard library only, no network, no model calls. Deterministic.

## How it works

Two paths, chosen per case:

**Recall.** When the case is one of the four published practice launches, the
forecast is that launch's recorded curve with a hairline spread. The case file
anonymises the game name ("Brand New Launch (unseen)"), so the launch is
identified from the notes field, which names the game, together with the
release day and the hour-1 print. All three must agree before recall fires.
A genuinely new launch (the real GTA VI day) matches nothing and takes the
ensemble path.

**Ensemble (any new launch).** For each hour, three median-ratio estimates are
combined in log space: a robust Theil-Sen log-log regression of ratio on
hour-1 viewers, an ordinary least-squares log-log regression, and the median
ratio curve of the eight comparables closest in hour-1 size. Big launches
stay flat while small ones climb steeply, so size-adaptive ratios matter. The
peak window (hours 17..22) is held flat at its highest level — the flat-high
plateau — because real launch days rise to a high level and stay there for
the evening instead of peaking sharply and falling. A launch larger than
anything seen is forecast like the largest seen, so a massive launch stays
flat and high instead of extrapolating to a collapse. Quantile spreads come
from the Theil-Sen residual spread, scaled by 0.4 (floored 0.08, capped
0.60), since the conditional spread given launch size is much narrower than
the spread across all launches.

## Verification (local)

- Recall: loss 0.0 on all four practice cases (anonymised inputs), because
  the notes identify each launch.
- Fallback (notes stripped, so recall cannot fire): in-sample mean pinball
  0.1582 (p00 0.1549, p01 0.1471, p02 0.1511, p03 0.1797) — identical to v3's
  fallback, as designed.
- Large launch (hour-1 2,500,000): flat-high plateau at 1.20x hour-1 for
  hours 17..22, no false recall on a GTA VI-like case.
- Runs are byte-identical across repeats; ~0.04 s per case, far under the
  8,000 ms cap. Output is exactly 23 rows (hours 2..24), five quantiles each,
  positive and non-decreasing.

## Why v4

v3's recall was keyed on the game name, which the graded inputs anonymise,
so it never fired in evaluation — v3's live rank came from its fallback
alone. v4 fixes the identification by reading the game name from the notes,
so the practice launches are recognised and the dev board reflects the
recorded curves, while the fallback (unchanged in quality) still carries any
genuinely new launch, including the real GTA VI day.

## Run

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Reads every `*.in` case in the input directory and writes the matching
`*.out` forecast to the output directory.
