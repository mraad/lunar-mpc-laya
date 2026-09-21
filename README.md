# Lunar MPC + Laya

**Adaptive model predictive control as the guidance and shield for a Laya
decision model**, on the arcade lunar lander from [lunar-laya](https://github.com/mraad/lunar-laya).

[lunar-mpc](https://github.com/mraad/lunar-mpc) learns its engine during flight and replans by beam
search, but has no learned policy. [lunar-laya](https://github.com/mraad/lunar-laya) runs a Laya
typed-decision model on Apple MLX, but its prompt labels come from a fixed-gain
PD `guidance()` that assumes nominal engine thrust and never adapts. This
project puts the first inside the second:

1. **Guidance**: an adaptive MPC (recursive-least-squares thrust estimate +
   NumPy beam search over the nine discrete commands) replaces `guidance()` as
   the source of the *requested rotation/power labels* in Laya's prompt. The
   prompt template is upstream's, byte for byte, so the existing trained
   checkpoint needs no retraining.
2. **Shield**: instead of overriding every disagreement, MPC evaluates Laya's
   proposal by rolling it out and overrides only when the best continuation
   after it is worse than MPC's own plan by more than a margin.
3. **Scenario**: an unannounced main-engine power loss mid-flight, which the
   controller only sees through the resulting motion.

Research question: *does adaptive MPC guidance let the existing trained Laya
checkpoint land through an engine fault that fixed PD guidance cannot?*

| Document | Contents |
|---|---|
| [How MPC and Laya combine](docs/fusion.md) | What each side contributes, the three couplings, why the combination is stronger than either, and what comes next |
| [Measured results](docs/results.md) | Landings, request following, shield activity and latency for every pilot and fault scenario, with reproduction commands |
| [Landing lab](web/README.md) | Interactive VanillaJS page running the adaptive MPC live in the browser |

## Landing lab in the browser

```bash
HF_HUB_OFFLINE=1 uv run lunar-mpc-laya-serve          # live Laya + page at http://127.0.0.1:8767/
python3 -m http.server 8767 --bind 127.0.0.1 --directory web   # page only, no live Laya
```

Click the sky to place the lander, choose a pad and an engine fault, and launch.
**Browser MPC** runs a line-for-line JavaScript port of the physics and the
adaptive MPC (a Node check verifies it reproduces every Python-recorded
command). **Live Laya** pilots send each decision to the local server, where
the real Laya checkpoint on MLX answers the prompt and the shield checks it.
A second tab replays the recorded evaluation flights decision by decision after
`python3 web/build.py dist/eval/*laya*.json dist/eval/mpc-assisted*.json`.
See [web/README.md](web/README.md).

## Quick start

Requires the sibling `../lunar-laya` checkout (path dependency, like lunar-laya
depends on laya-mlx) and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/mraad/lunar-laya.git
git clone https://github.com/mraad/lunar-mpc-laya.git
cd lunar-mpc-laya
```
 The MPC modes need
only NumPy; the Laya modes need an Apple Silicon Mac and the `mlx` extra.

```bash
uv sync                                   # MPC-only environment
uv run python -m unittest discover -s tests -v
uv run lunar-mpc-laya --pilot mpc --episodes 10 --seed 3000 --thrust-scale 0.4 \
  --out dist/mpc-fault.json --replay dist/mpc-fault.html
open dist/mpc-fault.html                  # macOS; upstream's standalone replay

uv sync --extra mlx                       # adds laya-mlx and MLX
HF_HUB_OFFLINE=1 uv run lunar-mpc-laya --pilot mpc-assisted \
  --model ../lunar-laya/models/lunar-laya-supervised-mlx \
  --episodes 3 --seed 3000 --thrust-scale 0.4 \
  --out dist/fusion.json --replay dist/fusion.html
```

Pilots: `pd-baseline` and `pd-laya` are upstream's PD guidance for comparison;
`mpc` is MPC alone; `mpc-laya` executes Laya's choices directly with MPC
requests in the prompt; `mpc-assisted` adds the predictive shield. `--fixed-model`
disables learning (ablation), `--thrust-scale` and `--fault-at` set the fault
(default: no fault), `--margin` sets the shield tolerance, `--prompt-estimate`
appends the engine estimate to the prompt (changes the trained template; measured
separately). `uv run lunar-mpc-laya --help` lists the rest.

```text
lunar_mpc_laya/mpc.py     Dynamics (scalar RLS on thrust gain) and AdaptiveMPC (beam search)
lunar_mpc_laya/pilot.py   MPCPilot: upstream Pilot with MPC reference and predictive shield
lunar_mpc_laya/cli.py     Session (episode loop with fault injection), upstream JSON schema and replay
lunar_mpc_laya/serve.py   local server: page + live /reset and /step decisions from the Python pilot
scripts/evaluate.py       records every pilot/scenario on seeds 3000-3009 and writes docs/results.json
scripts/parity.py         records Python flights that the JavaScript port must reproduce
tests/test_fusion.py      dependency-free checks (fake agents; no MLX)
web/                      landing lab: lander.js (physics + MPC port), app.js, build.py, Node test
```

## How it works

Every 0.2 s decision the controller reads `Game.snapshot()` and the public map
geometry, never the fault timing or true engine multiplier. `Dynamics.predict`
integrates a constant command with the same ten substeps as `Game.step`, using
the *estimated* thrust gain. `AdaptiveMPC.act` expands every retained candidate
by the nine commands for 15 stages (3 s of foresight), keeps the best 32 by
accumulated cost, and returns the plan, the predicted path and the cost. Safe
touchdown inside the target pad is an absorbing zero-cost state; any other
contact or leaving the world is an absorbing 1000-unit penalty. The cost tracks
pad offset, a lateral-velocity reference, an angle reference, a descent-speed
reference capped by the braking distance the estimated thrust allows (0.7
design margin, as in lunar-mpc), a hold at 120 m above the pad while off it,
plus terrain clearance while off the pad and a small throttle cost.

After each executed command `Dynamics.observe` compares the measured velocity
change with the prediction and updates the thrust gain by scalar recursive
least squares (forgetting 0.97, clipped to [1, 10]). Gravity and turn rate are
treated as known constants: learning gravity too made thrust and gravity
collinear during a sustained upright burn, and the estimates drifted along the
unobservable direction. Coasting, terminal and fuel-starved transitions are
excluded.

The replay is upstream's `replay.html`; its banner reads upstream's `mode`
(`baseline`/`laya`/`assisted`), so the JSON provenance keeps that field and adds
`pilot` (`mpc`, `mpc-laya`, `mpc-assisted`) and a `controller` block. Each
decision record keeps upstream's keys and adds `plan`, `prediction`,
`plan_cost`, `proposal_cost`, `model`, `prediction_error` and `solve_ms`.

## Checks

```bash
uv run python -m unittest discover -s tests -v      # controller, shield, fault harness, server
node --check web/app.js && node web/app.test.cjs   # JS parity with Python, page behaviour
uv run python scripts/parity.py            # after any controller change, then rerun the Node check
```

## Limits

This is an exploration, not a flight-certified controller. The beam search is
incomplete, the cost weights are hand-picked on development seeds 0–9, the
landing model checks contact at stage ends only, fuel starvation is not
predicted, and the shield margin is a design choice, not a certified safe set.
The trained checkpoint still receives requested labels in every prompt: the
Laya results demonstrate faithful request following under MPC guidance, not
independent learned piloting, and the shield never fired in the recorded
flights. Appending the engine estimate to the prompt broke request following
completely, so that flag is a negative result, not a feature. The fault is a single step change in main-engine
thrust; sensor noise, delays and other disturbances are untested.

## Related

[lunar-mpc](https://github.com/mraad/lunar-mpc) (adaptive MPC on Box2D),
[lunar-laya](https://github.com/mraad/lunar-laya) (Laya on the arcade lander),
[Laya](https://github.com/NandhaKishorM/laya), [Laya-MLX](https://github.com/mizorewww/laya-mlx).

## License

[Apache-2.0](LICENSE). Copyright 2026 mraad. [NOTICE](NOTICE) records the
attribution; Laya and Laya-MLX retain their upstream terms. The trained
checkpoint lives in lunar-laya and is not redistributed here.
