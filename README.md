# Lunar MPC + Laya

A lunar lander that lands itself, two different ways, and then both ways at
once.

**The game.** An Atari-style lander with three pads, finite fuel and an engine
that may silently lose thrust mid-flight. Every 0.2 s the pilot picks one of
nine commands (tilt left/hold/right × engine off/half/full). Land upright,
slowly, inside a pad.

**Two pilots that already existed.**

- **MPC** (model predictive control, from [lunar-mpc](https://github.com/mraad/lunar-mpc)):
  knows the physics. Each decision it simulates thousands of command sequences
  three seconds ahead, picks the cheapest, executes the first command, and
  replans. It also measures how the engine actually responds and updates its
  thrust estimate in flight. Hand-written; no training data.
- **Laya** (a small typed-decision model on Apple MLX, from [lunar-laya](https://github.com/mraad/lunar-laya)):
  reads a short text prompt about the flight and answers two multiple-choice
  questions. Learned from examples; knows no physics. Its prompt used to carry
  requests from a simple feedback law that assumed a healthy engine.

**What this project does with them.**

1. **MPC writes Laya's prompt.** The requested tilt and power in the prompt
   now come from adaptive MPC instead of the fixed feedback law. Same Laya
   weights, same prompt template. Result: through a 60% engine loss, Laya with
   the old requests lands 0 of 30; with MPC's requests, 30 of 30.
2. **MPC checks Laya's answer.** Before executing Laya's choice, MPC simulates
   it forward. If the outcome is predicted worse than MPC's own plan, MPC's
   command runs instead. Laya keeps its freedom; physics keeps a veto.
3. **Laya retrained to decide alone.** A second checkpoint learned from MPC's
   decisions with the requests removed from the prompt. By itself it lands
   0 of 90 (small mistakes compound). With MPC checking it, 90 of 90, while
   about two thirds of the executed commands are Laya's own.

**Try it.** A browser page runs the MPC live (click anywhere in the sky, pick
a pad, add a fault, launch). With the local server it also flies the real Laya
checkpoints and replays every recorded decision with the model's probabilities.

| Document | Contents |
|---|---|
| [How MPC and Laya combine](docs/fusion.md) | What each side contributes, the three couplings, why the combination is stronger than either, and what comes next |
| [Measured results](docs/results.md) | Landings, request following, shield activity and latency for every pilot and fault scenario, with reproduction commands |
| [Distillation](docs/distillation.md) | Retraining Laya on MPC labels with a telemetry-only prompt: raw 0/90, shielded 90/90 with Laya's choice kept two thirds of the time |
| [Landing lab](web/README.md) | The browser page: live MPC, live Laya pilots through the local server, recorded flights |

## Landing lab in the browser

```bash
HF_HUB_OFFLINE=1 uv run lunar-mpc-laya-serve          # http://127.0.0.1:8767/
```

Pilots on the page: **Browser MPC** (JavaScript port of the controller, no
server needed), **Live Laya** with MPC requests, raw or shielded, and **Live
distilled Laya**, telemetry only, raw or shielded. The recorded tab replays the
evaluation flights after `python3 web/build.py dist/eval/*laya*.json dist/eval/mpc-assisted*.json dist/distilled-evaluation-*.json`.
Without the server, `python3 -m http.server 8767 --bind 127.0.0.1 --directory web`
gives Browser MPC only. Details in [web/README.md](web/README.md).

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
lunar_mpc_laya/distilled.py  telemetry-only prompt and the distilled pilot (Laya decides, MPC shields)
lunar_mpc_laya/serve.py   local server: page + live /reset and /step decisions from the Python pilots
scripts/evaluate.py       records every pilot/scenario on seeds 3000-3009 and writes docs/results.json
scripts/parity.py         records Python flights that the JavaScript port must reproduce
training/                 telemetry-only prompt, MPC-labelled data, CUDA trainer, distilled pilot, MLX verification
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
Some starts are unrecoverable for every pilot: for example x = 300 m heading
for the right-hand pad with a ×0.4 fault crashes under MPC alone, so it crashes
shielded too. The shield can only be as good as MPC's own plan.
The original checkpoint receives requested labels in every prompt, so those
Laya results demonstrate request following, not independent piloting, and the
shield never fired. The distilled checkpoint ([docs/distillation.md](docs/distillation.md))
decides from telemetry alone: it cannot land unshielded (0/90) and lands every
flight shielded (90/90) with about a third of its proposals overridden. Appending the engine estimate to the prompt broke request following
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
