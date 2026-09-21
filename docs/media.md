# Animated flight

![Distilled Laya with the MPC shield landing through a 60% engine loss](assets/distilled-shield.gif)

One actual recorded flight: seed 3000, central pad, the **distilled Laya**
checkpoint deciding from telemetry alone with the **adaptive MPC shield**, and
a main-engine fault to ×0.4 thrust at 10 s. The flight lands at 69.6 s after
348 decisions; the shield replaced 109 of Laya's proposals (31%). The engine
estimate falls from 100% to 40% within a few seconds of the fault and the
descent slows accordingly. The panel shows Laya's option probabilities, MPC's
request, Laya's proposal, the executed command, and whether the shield fired.

## Reproduce

```bash
uv sync --extra mlx --extra media
HF_HUB_OFFLINE=1 uv run python scripts/record_gif_flight.py 3000    # writes dist/gif-flight.json
uv run python -m lunar_mpc_laya.gif dist/gif-flight.json --output docs/assets/distilled-shield.gif --speed 4
```

The recording step needs Apple Silicon and the distilled checkpoint
(`models/lunar-mpc-laya-distilled-mlx`, see [distillation.md](distillation.md)).
The renderer needs only Pillow and the JSON. Pass another recording (any
`lunar-mpc-laya --out` file works) and `--episode` to animate a different flight.

## Rendering

`lunar_mpc_laya/gif.py` draws recorded states only; nothing is interpolated.
The amber dashed line is MPC's plan at that decision; the solid trail is the
executed path. The GIF is 1000 × 620, 64 colours per frame, 10 fps, at 4×
speed one frame per 0.4 simulated seconds, first frame held 0.8 s and the last
2.5 s. This export has 175 frames and a 20.6 s loop, about 2.2 MB.
