"""Record every pilot/fault combination on seeds 3000-3009, all pads; summarize to docs/results.json."""

from collections import Counter
import json
from pathlib import Path
import statistics
import sys

from lunar_mpc_laya.cli import main
from lunar_mpc_laya.mpc import MPCConfig

MODEL = "../lunar-laya/models/lunar-laya-supervised-mlx"
CONFIGS = {
    "pd-baseline": ["--pilot", "pd-baseline"],
    "mpc": ["--pilot", "mpc"],
    "mpc-fixed": ["--pilot", "mpc", "--fixed-model"],
    "pd-laya": ["--pilot", "pd-laya", "--model", MODEL],
    "mpc-laya": ["--pilot", "mpc-laya", "--model", MODEL],
    "mpc-assisted": ["--pilot", "mpc-assisted", "--model", MODEL],
    "mpc-laya-estimate": ["--pilot", "mpc-laya", "--model", MODEL, "--prompt-estimate"],
}
FAULTS = {"nominal": "1.0", "fault-0.7": "0.7", "fault-0.4": "0.4"}
OUT = Path("dist/eval")


def controller(flags):
    """The controller block a fresh run with these flags records; None for PD pilots."""
    if flags[flags.index("--pilot") + 1].startswith("pd-"):
        return None
    return {"type": "adaptive-mpc", **vars(MPCConfig(adaptive="--fixed-model" not in flags)),
            "shield_margin": 0., "prompt_estimate": "--prompt-estimate" in flags}


def run(names):
    for name in names:
        expected = controller(CONFIGS[name])
        for fault, scale in FAULTS.items():
            for target in range(3):
                stem = f"{OUT}/{name}-{fault}-t{target}"
                if Path(stem + ".json").exists():
                    # Resuming is only safe while the controller is unchanged. A
                    # cost weight edited between runs would otherwise leave this
                    # directory holding two different controllers and summarize
                    # them as one measurement.
                    cached = json.loads(Path(stem + ".json").read_text())["pilot"].get("controller")
                    if cached != expected:
                        raise SystemExit(
                            f"{stem}.json was recorded with a different controller.\n"
                            f"  cached:   {cached}\n  current:  {expected}\n"
                            f"Delete {OUT}/ and re-record, or move it aside to keep both.")
                    continue
                main([*CONFIGS[name], "--seed", "3000", "--episodes", "10", "--target", str(target),
                      "--thrust-scale", scale, "--fault-at", "10",
                      "--out", stem + ".json", "--replay", stem + ".html"])


def summarize():
    rows = {}
    for name in CONFIGS:
        for fault in FAULTS:
            eps = [e["summary"] for target in range(3)
                   for e in json.loads((OUT / f"{name}-{fault}-t{target}.json").read_text())["episodes"]
                   if (OUT / f"{name}-{fault}-t{target}.json").exists()]
            if not eps:
                continue
            rows[f"{name}/{fault}"] = {
                "pilot": name, "scenario": fault, "flights": len(eps),
                "landed": sum(e["status"] == "landed" for e in eps),
                "outcomes": dict(Counter(e["status"] for e in eps)),
                "decisions": sum(e["decisions"] for e in eps),
                "agreements": sum(e["agreements"] for e in eps),
                "laya_decisions": sum(e["decisions"] for e in eps if e["latency_p50_ms"] > 8),
                "interventions": sum(e["interventions"] for e in eps),
                "fuel_mean": round(statistics.mean(e["fuel"] for e in eps), 2),
                "latency_p50_ms": round(statistics.median(e["latency_p50_ms"] for e in eps), 3),
                "solve_p50_ms": round(statistics.median(e["solve_p50_ms"] for e in eps), 3)
                if "solve_p50_ms" in eps[0] else None,
                "thrust_estimate_final": [round(e["model"]["thrust"], 3) for e in eps][:3]
                if "model" in eps[0] else None}
    return rows


if __name__ == "__main__":
    run(sys.argv[1:] or CONFIGS)
    summary = summarize()
    Path("docs/results.json").write_text(json.dumps(summary, indent=2) + "\n")
    for key, row in summary.items():
        print(f"{key:30} landed {row['landed']:2}/{row['flights']} {row['outcomes']} decisions {row['decisions']:5} "
              f"agree {row['agreements']:5} interventions {row['interventions']}")
