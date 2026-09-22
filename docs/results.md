# Measured results

Measured on 2026-09-22 in this workspace (re-run after the command-switching
penalty was added to the cost; landings did not change): Apple Silicon,
macOS 26.6.2, Python 3.12.12, NumPy 2.5.3, MLX 0.32.2, Laya-MLX at the commit
pinned by lunar-laya,
checkpoint `../lunar-laya/models/lunar-laya-supervised-mlx` (FP16). The compact
machine-readable summary is [results.json](results.json); the 63 full recordings
and standalone replays were generated under ignored `dist/eval/`.

Protocol: seeds 3000–3009 on every pad (30 flights per row), up to 900 decisions,
fresh nominal MPC model per flight. The fault multiplies the simulator's main
engine thrust at 10 simulation seconds; the controller never receives the time
or the multiplier. Seeds 3000–3009 are the same held-out flights lunar-laya
used, so `pd-laya/nominal` reproduces its 30/30 result.

One caveat on held-out-ness: the original cost weights were chosen on seeds 0–9
and not changed after these runs, but the switching weight was swept on seeds
3000–3005, which are inside this evaluation set. Every weight in that sweep
landed every flight, so the choice was made on jerk and fuel rather than on
landings, and the landing counts below are unchanged from the run before the
penalty existed. Treat the switching weight as tuned on this set all the same.

## Landings

| Pilot | Nominal | Fault ×0.7 | Fault ×0.4 |
|---|---:|---:|---:|
| `pd-baseline` (upstream PD guidance) | 30/30 | 17/30 | 0/30 |
| `mpc` (adaptive) | 30/30 | 30/30 | 30/30 |
| `mpc --fixed-model` (no learning) | 30/30 | 30/30 | 0/30 |
| `pd-laya` (Laya, PD requests) | 30/30 | 17/30 | 0/30 |
| `mpc-laya` (Laya, MPC requests) | 30/30 | 30/30 | 30/30 |
| `mpc-assisted` (Laya, MPC requests, predictive shield) | 30/30 | 30/30 | 30/30 |
| `mpc-laya --prompt-estimate` | 0/30 | 0/30 | 0/30 |

All PD failures under the fault are crashes at touchdown: the throttle law
divides by the nominal thrust constant, so at ×0.7 the lander touches down at
about 3.1 m/s (limit 3), and at ×0.4 at about 12 m/s. Fixed-model MPC survives
×0.7 because replanning every 0.2 s corrects the prediction error in time, but
at ×0.4 it plans braking that the engine cannot deliver and crashes at about
9.9 m/s. Adaptive MPC's thrust estimate reached 3.5 and 2.0 (true values) in every
fault flight, within 10% about 22 decisions after the fault, and it lands all 30
in both cases while spending more fuel (mean remaining 65.6 at ×0.4 versus 86.3
nominal). The estimate reaches within 10% of the truth a median 22 decisions
after the ×0.7 fault and 25 after ×0.4, at worst 29.

## Smoothness

The cost charges for changing command between stages (`MPCConfig.switch`,
default 8; see the README). Without it the beam search is free to re-pick every
0.2 s, and it does, which a rider feels as the tank chattering between throttle
settings while the attitude jets flip sign. Measured over the same 30 flights
per scenario:

| | switch 0 | switch 8 |
|---|---:|---:|
| Decisions that change command, nominal | 77.7% | **20.6%** |
| Decisions that change command, ×0.4 | 67.6% | **14.8%** |
| Mean \|Δturn\| per decision, nominal | 0.568 | **0.080** |
| Mean \|Δthrottle\| per decision, nominal | 0.285 | **0.067** |
| Stage-to-stage tilt jerk, nominal | 5.37° RMS | **1.69° RMS** |
| Stage-to-stage tilt jerk, ×0.4 | 5.71° RMS | **1.65° RMS** |
| Landed, all three scenarios | 30/30 | **30/30** |

It is not a trade against performance. Every landing survived, flights got
slightly shorter (9,736 decisions at ×0.4 against 10,327) and fuel remaining
rose (65.6 against 62.6 at ×0.4, 86.3 against 85.7 nominal): the commands it
stops taking were mostly ones it would have undone on the next stage. The
weight was chosen by sweeping 0, 2, 5, 8, 12 and 20 over seeds 3000-3005 on
every pad, in both the nominal and ×0.4 scenarios; all six land, and above 8 the
remaining jerk falls slowly while flights lengthen and ×0.4 fuel starts to drop.

The same change does **not** transfer to [lunar-mpc](https://github.com/mraad/lunar-mpc).
Its main engine is binary, so pulsing it is how that controller modulates
thrust; charging for the pulses took its engine-fault result from 49/50 to
41/50 on seeds 200-249 while nominal rose from 49/50 to 50/50. Only the
three-level throttle here makes holding a command a real option.

## Request following and the shield

| Pilot | Scenario | Laya decisions | Matched request | Overrides |
|---|---|---:|---:|---:|
| `pd-laya` | nominal | 11,147 | 11,147 | — |
| `pd-laya` | ×0.7 | 6,266 | 6,266 | — |
| `pd-laya` | ×0.4 | 4,787 | 4,783 | — |
| `mpc-laya` | nominal | 8,796 | 8,796 | — |
| `mpc-laya` | ×0.7 | 8,532 | 8,532 | — |
| `mpc-laya` | ×0.4 | 9,736 | 9,736 | — |
| `mpc-assisted` | all three | 27,064 | 27,064 | 0 |
| `mpc-laya --prompt-estimate` | nominal | 2,864 | 0 | — |
| `mpc-laya --prompt-estimate` | ×0.7 / ×0.4 | 5,332 / 5,428 | 332 / 253 | — |

The trained checkpoint followed every MPC request in 27,064 decisions, so the
`mpc-laya` and `mpc-assisted` trajectories are identical to `mpc`, and the
predictive shield never had to act. Its behaviour on a bad proposal is covered
by a unit test, not by these flights. The four `pd-laya` disagreements are one
flight (seed 3008) under ×0.4, where Laya chose `off` against a requested
`half` twice per pad at 11.4–11.6 s with engine confidence about 0.37; the
outcome was a crash either way.

Appending one sentence ("Estimated engine effectiveness NN%.") to the prompt
destroyed request following: 0 of 2,864 nominal matches and every flight lost,
30 of them by leaving the world rather than crashing.
The checkpoint was fine-tuned on one exact template and does not tolerate a
change to it. Any prompt-level use of the engine estimate needs retraining.

## Latency

Median per-decision latency (median of per-flight medians): MPC planning
3.9–4.0 ms; PD + Laya 10.4–11.5 ms across scenarios; MPC + Laya 14.4–14.5 ms, of
which the beam search is 4.0 ms. The switching penalty is two extra array
lookups per candidate and does not move the planning time.
The shield's second search only runs on disagreement, which never happened here.
Timings include first-call overhead and exclude physics and rendering; other
work ran on the machine, including live browser flights against the same
model during part of the run, so this is not an isolated benchmark. The simulator
waits for each decision, so these numbers do not test delayed commands.

## Interpretation

The research question was whether adaptive MPC guidance lets the existing
trained Laya checkpoint land through an engine fault that fixed PD guidance
cannot. Yes: 30/30 versus 0/30 at ×0.4. The mechanism is request following,
not learned piloting: the checkpoint reproduces whatever guidance writes into
the prompt, so the fusion inherits MPC's adaptation and nothing more. The
sharper findings are negative ones: the shield is never exercised by a faithful
follower, and the checkpoint breaks under a one-sentence prompt change.

Next experiments, in order of value: retrain on MPC-labelled data *without*
requested labels so Laya must infer commands from telemetry, then measure the
shield's override rate and whether MPC's estimate in the prompt helps; add
sensor noise and delay; test faults the beam cannot recover from and report
the failures.

## See the flights

```bash
python3 web/build.py dist/eval/pd-laya*.json dist/eval/mpc-laya*.json dist/eval/mpc-assisted*.json
HF_HUB_OFFLINE=1 uv run lunar-mpc-laya-serve     # http://127.0.0.1:8767/#recorded
```

The recorded tab replays every flight above decision by decision with Laya's
probabilities, the MPC request, the proposal and the shield line; the landing
lab tab flies new starts with the same checkpoint live.

## Reproduce

```bash
uv sync --extra mlx
HF_HUB_OFFLINE=1 uv run python scripts/evaluate.py          # all 21 rows, ~28 min
HF_HUB_OFFLINE=1 uv run python scripts/evaluate.py mpc pd-baseline   # subset
```

The script skips rows whose recording already exists under `dist/eval/` and
rewrites `docs/results.json` from whatever is present. Each recording carries the
controller it was made with, and resuming stops with an error if that differs
from the current one, so a cost weight edited between runs cannot leave the
directory holding two controllers summarized as one measurement. Single rows:

```bash
uv run lunar-mpc-laya --pilot mpc --fixed-model --thrust-scale 0.4 --seed 3000 --episodes 10 --target 1
HF_HUB_OFFLINE=1 uv run lunar-mpc-laya --pilot mpc-assisted --thrust-scale 0.4 --seed 3000 --episodes 10 \
  --model ../lunar-laya/models/lunar-laya-supervised-mlx --out dist/shield.json --replay dist/shield.html
```
