# Landing lab (VanillaJS)

Two tabs, one page, no npm, no bundler, no external assets.

- **Landing lab**: click the sky to place the lander, set speeds and tilt,
  choose a pad and an engine fault, launch. Pilot **Browser MPC** runs the
  adaptive MPC in JavaScript. The **Live** pilots send each decision to the
  local Python server, where a real Laya checkpoint answers on Apple MLX and
  the MPC shield checks the answer; the browser only draws the returned frames.
  *Live Laya* is the original checkpoint reading MPC's requests; *Live
  distilled Laya* is the retrained one deciding from telemetry alone.
- **Recorded Laya flights**: Python-recorded flights from `scripts/evaluate.py`
  (held-out seeds 3000–3009) with every decision's probabilities, MPC request,
  Laya proposal, executed command and shield activity.

## Serve

```bash
# Live Laya + static page (Apple Silicon, mlx extra installed):
HF_HUB_OFFLINE=1 uv run lunar-mpc-laya-serve            # http://127.0.0.1:8767/
uv run lunar-mpc-laya-serve --no-model                   # MPC-only live server
# Static page only (Browser MPC and recorded flights, no live Laya):
python3 -m http.server 8767 --bind 127.0.0.1 --directory web
```

The server binds 127.0.0.1, loads the models once (`--model`, default
`../lunar-laya/models/lunar-laya-supervised-mlx`; `--distilled`, default
`models/lunar-mpc-laya-distilled-mlx`, skipped when absent), serves `dist/web` when built
and `web/` otherwise (`--root` overrides), and exposes three JSON endpoints:
`GET /status` (available pilots, model provenance), `POST /reset` (start state,
target, fault, adaptive, mode, margin) and `POST /step` (`n` decisions, returns
frames with the full decision record). One flight at a time; a reset replaces it.

## Build the recorded tab

```bash
python3 web/build.py dist/eval/pd-laya*.json dist/eval/mpc-laya*.json dist/eval/mpc-assisted*.json \
  dist/distilled-evaluation-*.json
uv run lunar-mpc-laya-serve            # now serves dist/web
```

`build.py` copies the page into `dist/web`, strips prompts and predicted paths
from the recordings, rounds values and groups flights by pilot and scenario into
`runs/*.json` with a `manifest.json`. Without a build the tab explains how to
make one.

## Files

| File | Role |
|---|---|
| `lander.js` | Line-for-line port of `lunar_laya.game` and `lunar_mpc_laya.mpc`, plus the shared lander art; loads in the browser and in Node |
| `app.js` | Tabs, controls, live loop (browser or server), recorded playback, canvas drawing |
| `index.html`, `styles.css` | Page and styling |
| `build.py` | Static builder for the recorded tab |
| `parity.json` | Six Python-recorded flights the JS port must reproduce; regenerate with `uv run python scripts/parity.py` |
| `app.test.cjs` | Node check: parity, browser flight, fault toggle, live-server pilot against a stub, recorded playback |

```bash
node --check web/app.js && node web/app.test.cjs
```

## Reading the page

Dashed amber = current 3 s plan (browser or server MPC). Solid trail = executed
path. "Engine estimate" = learned thrust gain relative to nominal. In the Laya
card, bars are the model's option probabilities for the two questions, with the
MPC request, Laya's proposal and the executed command underneath; the shield
line says whether Laya matched, deviated and was accepted, or was overridden.

## What it is not

Browser timings are not the Python timings in `docs/results.md`. Live flights
run one at a time on the local machine. Some starts are unrecoverable and the
search has no safety guarantee; a crash is a real controller limitation.

## Drawing

`Lander.drawLander` is one chamfered-box lander shared byte for byte with
lunar-mpc and lunar-laya. Its coordinates are in units of `RADIUS / 8` with y
pointing down, so the footpads land exactly one hull radius below the centre
and rest on the surface at touchdown; `flip: -1` draws it into a y-up frame.

A control stage is 0.2 s, so drawing only when a stage ends animates the flight
at 5 fps. `pose()` mixes the state a stage ended in back into the one it started
from and the canvas redraws every animation frame. The drawn lander is one stage
behind the telemetry panel, which is not visible at 0.2 s, and the panel itself
still shows exact decision states.

The stage that ends in touchdown is interpolated like the rest, so the final
approach does not snap; `app.js` and the `terminal blend` assertion in
`app.test.cjs` carry the detail.
