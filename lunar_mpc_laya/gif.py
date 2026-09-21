"""Render a recorded flight as a documentation GIF: path, plan, fault, estimate, Laya's answer, shield.

Draws only recorded states; nothing is interpolated. Modelled on lunar_laya.gif.
"""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lunar_laya.game import PADS, RADIUS, TERRAIN, THRUST

BG, PANEL, GRID = "#0b111b", "#101d29", "#203140"
TEXT, MUTED, MINT, AMBER, RED = "#e6f4f4", "#94aabd", "#90edd0", "#f3c583", "#e7ac83"
FONT = {size: ImageFont.load_default(size=size) for size in (12, 14, 17, 22, 28)}
W, H = 1000, 620


def point(x, y):
    return (38 + x * .65, 509 - y * .43)


def word(c):
    return f"{'right' if c['turn'] > 0 else 'left' if c['turn'] < 0 else 'hold'} · {'full' if c['throttle'] >= 1 else 'half' if c['throttle'] > 0 else 'off'}"


def render(record, episode, index):
    frames, summary = episode["frames"], episode["summary"]
    terminal = index >= len(frames)
    frame = frames[min(index, len(frames) - 1)]
    state = summary if terminal else frame["before"]
    d = frame["decision"]
    fault = record.get("fault") or {}
    faulted = fault.get("thrust_scale", 1) < 1 and state["time"] >= fault.get("fault_at", 10)
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=14, color=TEXT):
        draw.text((x, y), str(value), font=FONT[size], fill=color)

    text(26, 18, "LUNAR MPC + LAYA", 28, MINT)
    text(26, 56, "adaptive MPC labels and shields · distilled Laya decides from telemetry · live MLX inference", 14, MUTED)
    badge = f"FAULT · THRUST x{fault['thrust_scale']}" if faulted else "NOMINAL ENGINE"
    text(760, 24, badge, 17, AMBER if faulted else MINT)
    text(760, 50, str(record["pilot"].get("pilot", "")).upper(), 14)
    draw.line((24, 89, 976, 89), fill=GRID)
    pad = PADS[summary["target"]]
    thrust = d["model"]["thrust"] if d.get("model") else THRUST
    metrics = [("PAD ALTITUDE", f"{max(0, state['y'] - pad[2] - RADIUS):.0f} m"),
               ("VERTICAL", f"{state['vy']:+.1f} m/s"), ("HORIZONTAL", f"{state['vx']:+.1f} m/s"),
               ("FUEL", f"{state['fuel']:.0f}"), ("ENGINE ESTIMATE", f"{100 * thrust / THRUST:.0f}%")]
    for x, (label, value) in zip((26, 176, 316, 456, 560), metrics):
        text(x, 102, label, 12, MUTED)
        text(x, 122, value, 22, AMBER if label.startswith("ENGINE") and faulted else TEXT)
    draw.rounded_rectangle((24, 165, 704, 542), radius=10, fill=PANEL, outline=GRID)
    draw.rounded_rectangle((724, 102, 976, 542), radius=10, fill=PANEL, outline=GRID)
    for i in range(55):
        draw.point((40 + i * 173 % 650, 182 + i * 79 % 240), fill=MUTED if i % 7 == 0 else GRID)
    for y in (200, 400, 600):
        draw.line([point(0, y), point(1000, y)], fill=GRID)
    terrain = [point(x, y) for x, y in TERRAIN]
    draw.polygon([(38, 529), *terrain, (688, 529)], fill="#182b37")
    draw.line(terrain, fill=MUTED, width=2)
    for i, p in enumerate(PADS):
        selected = i == summary["target"]
        draw.line([point(p[0], p[2]), point(p[1], p[2])], fill=MINT if selected else MUTED, width=4 if selected else 2)
    path = [point(f["before"]["x"], f["before"]["y"]) for f in frames[:index + 1]]
    if terminal:
        path.append(point(state["x"], state["y"]))
    if len(path) > 1:
        draw.line(path, fill="#3f8a7a", width=2)
    if not terminal and d.get("prediction"):
        plan = [point(v[0], v[1]) for v in d["prediction"]]
        for a, b in zip(plan, plan[1:]):
            draw.line([a, b], fill=AMBER, width=1)
    x, y = point(state["x"], state["y"])
    angle = math.radians(state["angle"])

    def ship(points):
        return [(x + px * math.cos(angle) - py * math.sin(angle), y + px * math.sin(angle) + py * math.cos(angle)) for px, py in points]

    crashed = state["status"] in ("crashed", "out_of_bounds")
    # The game treats the hull as a circle of RADIUS around (x, y); it sits on the
    # surface when y - RADIUS == ground. Feet are drawn exactly RADIUS below the
    # centre in map units (vertical scale .43 px/m) so they touch, never sink.
    foot = RADIUS * .43
    hull = [(0, -foot * 1.1), (foot * .8, -foot * .3), (foot * .7, foot * .2), (-foot * .7, foot * .2), (-foot * .8, -foot * .3)]
    scale = 3.2  # visual size; legs still end at exactly `foot` below centre
    draw.polygon(ship([(px * scale, py * scale) for px, py in hull]), fill=PANEL, outline=RED if crashed else TEXT)
    draw.line(ship([(-foot * .7 * scale, foot * .2 * scale), (-foot * 1.1 * scale, foot)]), fill=TEXT)
    draw.line(ship([(foot * .7 * scale, foot * .2 * scale), (foot * 1.1 * scale, foot)]), fill=TEXT)
    if not terminal and d["executed"]["throttle"] > 0 and state["fuel"] > 0:
        base = foot * .2 * scale
        draw.polygon(ship([(-3, base), (0, base + 4 + 14 * d["executed"]["throttle"]), (3, base)]), fill=AMBER)
    status = state["status"].upper().replace("_", " ") if terminal else "FLYING"
    text(38, 516, f"{status}   T+ {state['time']:5.1f} s   tilt {state['angle']:+.0f}°", 12, RED if crashed else MINT)

    # Decision panel.
    text(740, 114, "LAYA · MLX DECISION", 12, MUTED)
    yy = 134
    for q, answer in (d.get("answers") or {}).items():
        text(740, yy, q.upper(), 12, MUTED); yy += 16
        for label, p in answer["probabilities"].items():
            chosen = label == answer["choice"]
            text(740, yy, label, 12, MINT if chosen else MUTED)
            draw.rectangle((790, yy + 4, 940, yy + 10), fill=GRID)
            draw.rectangle((790, yy + 4, 790 + 150 * p, yy + 10), fill=MINT if chosen else "#476376")
            text(946, yy, f"{100 * p:.0f}", 12, MUTED); yy += 16
        yy += 6
    yy = max(yy, 340)
    for label, value, color in (("MPC request", word(d["requested"]), TEXT), ("Laya proposal", word(d["proposed"]), TEXT),
                                ("Executed", word(d["executed"]), TEXT)):
        text(740, yy, label, 12, MUTED); text(850, yy, value, 12, color); yy += 20
    yy += 6
    if d.get("intervened"):
        draw.rounded_rectangle((740, yy, 960, yy + 40), radius=6, fill="#2c241b", outline="#5b4630")
        text(750, yy + 5, "SHIELD OVERRIDE", 12, AMBER)
        text(750, yy + 22, f"rollout cost {d['proposal_cost']:.2f} > plan {d['plan_cost']:.2f}", 12, AMBER)
    else:
        draw.rounded_rectangle((740, yy, 960, yy + 40), radius=6, fill="#142a25", outline="#315349")
        text(750, yy + 5, "PROPOSAL ACCEPTED" if d["proposed"] != d["requested"] else "MATCHES MPC", 12, MINT)
        text(750, yy + 22, f"plan cost {d['plan_cost']:.2f}", 12, MINT)
    yy += 52
    overrides = sum(f["decision"]["intervened"] for f in frames[:index + 1])
    text(740, yy, f"overrides so far {overrides} / {min(index + 1, len(frames))}", 12, MUTED); yy += 18
    text(740, yy, f"decision {d['latency_ms']:.0f} ms · beam {d['solve_ms']:.1f} ms", 12, MUTED)
    text(740, 522, "github.com/mraad/lunar-mpc-laya", 12, MUTED)
    return image


def export_gif(source, output, episode_index=0, speed=4):
    record = json.loads(Path(source).read_text())
    episode = record["episodes"][episode_index]
    n = len(episode["frames"])
    stride = max(1, round(speed / 2))          # 10 fps; at 4x every 2nd decision, 0.4 simulated s per frame
    indices = list(range(0, n, stride)) + [n]
    images = [render(record, episode, i).quantize(colors=64) for i in indices]
    durations = [800] + [100] * (len(images) - 2) + [2500]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    return {"frames": len(images), "seconds": sum(durations) / 1000, "bytes": output.stat().st_size,
            "summary": episode["summary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/distilled-shield.gif"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--speed", type=float, default=4)
    args = parser.parse_args()
    print(json.dumps(export_gif(args.recording, args.output, args.episode, args.speed)))


if __name__ == "__main__":
    main()
