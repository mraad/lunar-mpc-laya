"""Build dist/web: the landing lab plus slimmed recordings for the recorded-flights tab."""

import argparse
import json
from pathlib import Path
import shutil

KEEP = ("requested", "proposed", "executed", "intervened", "answers", "model", "latency_ms",
        "plan_cost", "proposal_cost", "solve_ms")


def compact(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def slim(record):
    episodes = []
    for episode in record["episodes"]:
        frames = [{"before": f["before"], "decision": {k: f["decision"].get(k) for k in KEEP}}
                  for f in episode["frames"]]
        episodes.append({"summary": episode["summary"], "frames": frames,
                         "final": episode["frames"][-1]["after"] if episode["frames"] else None})
    return compact({"pilot": record["pilot"], "fault": record.get("fault"), "episodes": episodes})


def pilot_name(record):
    """`pd-laya` for the upstream pilot, the fusion name otherwise, plus the prompt-estimate variant."""
    pilot = record["pilot"]
    name = pilot.get("pilot") or f"pd-{pilot['mode']}"
    if pilot.get("controller", {}).get("prompt_estimate"):
        name += " (estimate in prompt)"
    return name


def label(record, episode=None):
    fault = dict(record.get("fault") or {})
    if episode and "thrust_scale" in episode:      # verify_mlx records mix scenarios in one file
        fault["thrust_scale"] = episode["thrust_scale"]
    scale = fault.get("thrust_scale", 1)
    return f"{pilot_name(record)} · " + ("nominal" if scale == 1 else f"thrust ×{scale} at {fault.get('fault_at', 10):g} s")


def build(recordings, out):
    out = Path(out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    web = Path(__file__).parent
    for name in ("index.html", "app.js", "lander.js", "styles.css"):
        shutil.copy2(web / name, out / name)
    runs = {}
    for path in map(Path, recordings):
        record = json.loads(path.read_text())
        for raw, episode in zip(record["episodes"], slim(record)["episodes"]):
            key = label(record, raw)
            run = runs.setdefault(key, {"id": len(runs), "label": key, "file": f"runs/{len(runs)}.json",
                                        "pilot": pilot_name(record), "episodes": []})
            run["episodes"].append(episode)
    manifest = []
    for run in runs.values():
        (out / run["file"]).write_text(json.dumps({"label": run["label"], "pilot": run["pilot"],
                                                   "episodes": run["episodes"]}, separators=(",", ":")))
        manifest.append({k: run[k] for k in ("id", "label", "file", "pilot")}
                        | {"flights": len(run["episodes"]),
                           "landed": sum(e["summary"]["status"] == "landed" for e in run["episodes"])})
    (out / "manifest.json").write_text(json.dumps({"version": 1, "runs": manifest}, indent=1))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("dist/web"))
    args = parser.parse_args()
    for run in build(args.recordings, args.out):
        print(f"{run['label']:48} {run['landed']}/{run['flights']} landed -> {run['file']}")
    print(f"Serve with: uv run lunar-mpc-laya-serve   (or python3 -m http.server 8767 --bind 127.0.0.1 --directory {args.out})")
