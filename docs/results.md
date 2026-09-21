# Measured results

Measured on 2026-09-21 in this workspace (re-run after the off-pad altitude hold was added to the cost; landings did not change): Apple Silicon, macOS 26.6.2, Python
3.12.12, NumPy 2.5.3, MLX 0.32.2, Laya-MLX at the commit pinned by lunar-laya,
checkpoint `../lunar-laya/models/lunar-laya-supervised-mlx` (FP16). The compact
machine-readable summary is [results.json](results.json); the 63 full recordings
and standalone replays were generated under ignored `dist/eval/`.

Protocol: seeds 3000–3009 on every pad (30 flights per row), up to 900 decisions,
fresh nominal MPC model per flight. The fault multiplies the simulator's main
engine thrust at 10 simulation seconds; the controller never receives the time
or the multiplier. Cost weights were chosen on seeds 0–9 and not changed after
these runs. Seeds 3000–3009 are the same held-out flights lunar-laya used, so
`pd-laya/nominal` reproduces its 30/30 result.

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
9 m/s. Adaptive MPC's thrust estimate reached 3.5 and 2.0 (true values) in every
fault flight, within 10% about 22 decisions after the fault, and it lands all 30
in both cases while spending more fuel (mean remaining 62.6 at ×0.4 versus 85.7
nominal).

## Request following and the shield

| Pilot | Scenario | Laya decisions | Matched request | Overrides |
|---|---|---:|---:|---:|
| `pd-laya` | nominal | 11,147 | 11,147 | — |
| `pd-laya` | ×0.7 | 6,266 | 6,266 | — |
| `pd-laya` | ×0.4 | 4,787 | 4,783 | — |
| `mpc-laya` | nominal | 8,815 | 8,815 | — |
| `mpc-laya` | ×0.7 | 8,884 | 8,884 | — |
| `mpc-laya` | ×0.4 | 10,327 | 10,327 | — |
| `mpc-assisted` | all three | 28,026 | 28,026 | 0 |
| `mpc-laya --prompt-estimate` | nominal | 2,864 | 0 | — |
| `mpc-laya --prompt-estimate` | ×0.7 / ×0.4 | 5,276 / 5,271 | 233 / 212 | — |

The trained checkpoint followed every MPC request in 28,026 decisions, so the
`mpc-laya` and `mpc-assisted` trajectories are identical to `mpc`, and the
predictive shield never had to act. Its behaviour on a bad proposal is covered
by a unit test, not by these flights. The four `pd-laya` disagreements are one
flight (seed 3008) under ×0.4, where Laya chose `off` against a requested
`half` twice per pad at 11.4–11.6 s with engine confidence about 0.37; the
outcome was a crash either way.

Appending one sentence ("Estimated engine effectiveness NN%.") to the prompt
destroyed request following: 0 of 2,864 nominal matches and every flight lost.
The checkpoint was fine-tuned on one exact template and does not tolerate a
change to it. Any prompt-level use of the engine estimate needs retraining.

## Latency

Median per-decision latency (median of per-flight medians): MPC planning 3.8 ms;
PD + Laya 10.3–14.6 ms across scenarios; MPC + Laya 14.2–14.4 ms, of which the
beam search is 3.8–4.0 ms.
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
HF_HUB_OFFLINE=1 uv run python scripts/evaluate.py          # all 21 rows, ~35 min
HF_HUB_OFFLINE=1 uv run python scripts/evaluate.py mpc pd-baseline   # subset
```

The script skips rows whose recording already exists under `dist/eval/` and
rewrites `docs/results.json` from whatever is present. Single rows:

```bash
uv run lunar-mpc-laya --pilot mpc --fixed-model --thrust-scale 0.4 --seed 3000 --episodes 10 --target 1
HF_HUB_OFFLINE=1 uv run lunar-mpc-laya --pilot mpc-assisted --thrust-scale 0.4 --seed 3000 --episodes 10 \
  --model ../lunar-laya/models/lunar-laya-supervised-mlx --out dist/shield.json --replay dist/shield.html
```
