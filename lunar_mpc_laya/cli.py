"""Finite runs with an unannounced engine fault; JSON and portable HTML replay."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter

import lunar_laya.game as game_module
from lunar_laya.cli import export_replay, new_recording, positive_int
from lunar_laya.game import Game
from lunar_laya.pilot import Pilot, guidance

from .mpc import MPCConfig, AdaptiveMPC
from .pilot import MPCPilot, UPSTREAM

PILOTS = ("pd-baseline", "pd-laya", "mpc", "mpc-laya", "mpc-assisted")


class PDPilot(Pilot):
    """Upstream pilot that also records its requested command, like MPCPilot."""

    def decide(self, game):
        reference = guidance(game)[0]
        executed, decision = super().decide(game)
        decision["requested"] = {"turn": reference.turn, "throttle": reference.throttle}
        return executed, decision


class Session:
    """One flight stepped on demand; the pilot never sees the fault time or scale."""

    def __init__(self, pilot, game, thrust_scale=1., fault_at=10.):
        if not 0 < thrust_scale <= 1 or fault_at < 0:
            raise ValueError("thrust_scale in (0, 1], fault_at nonnegative")
        self.pilot, self.game, self.thrust_scale, self.fault_at = pilot, game, thrust_scale, fault_at
        if hasattr(pilot, "reset"):
            pilot.reset()

    def step(self, n=1):
        frames = []
        nominal = game_module.THRUST
        for _ in range(n):
            if self.game.state.status != "flying":
                break
            before = self.game.snapshot()
            command, decision = self.pilot.decide(self.game)
            try:
                if before["time"] >= self.fault_at:
                    game_module.THRUST = nominal * self.thrust_scale
                after = self.game.step(command)
            finally:
                game_module.THRUST = nominal
            if hasattr(self.pilot, "observe"):
                self.pilot.observe(before, command, after)
            frames.append({"before": before, "decision": decision, "after": after})
        return frames


def run_episode(pilot, seed, target, steps, thrust_scale=1., fault_at=10.):
    """Upstream loop plus a process-local main-engine fault the pilot never sees."""
    if steps < 1:
        raise ValueError("steps must be positive")
    session = Session(pilot, Game(seed, target), thrust_scale, fault_at)
    frames = session.step(steps)
    terminal = session.game.snapshot()
    if terminal["status"] == "flying":
        terminal["status"] = "truncated"
    times = sorted(f["decision"]["latency_ms"] for f in frames)
    summary = {"seed": seed, "target": target, **terminal, "decisions": len(frames),
               "interventions": sum(f["decision"]["intervened"] for f in frames),
               "agreements": sum(f["decision"]["proposed"] == f["decision"]["requested"]
                                 for f in frames if f["decision"]["answers"] is not None),
               "latency_p50_ms": statistics.median(times),
               "latency_p95_ms": times[math.ceil(len(times) * .95) - 1]}
    solves = sorted(f["decision"]["solve_ms"] for f in frames if "solve_ms" in f["decision"])
    if solves:
        summary.update(solve_p50_ms=statistics.median(solves), solve_max_ms=solves[-1],
                       model=frames[-1]["decision"]["model"])
    return {"summary": summary, "frames": frames}


def make_pilot(args):
    if args.pilot.startswith("pd-"):
        return PDPilot("laya" if args.pilot == "pd-laya" else "baseline", args.model, args.revision)
    cfg = MPCConfig(args.horizon, args.beam, not args.fixed_model)
    return MPCPilot(args.pilot, args.model, args.revision, mpc=AdaptiveMPC(cfg),
                    margin=args.margin, prompt_estimate=args.prompt_estimate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", choices=PILOTS, default="mpc")
    parser.add_argument("--model", default="aac6fef/laya-multilingual-mlx")
    parser.add_argument("--revision")
    parser.add_argument("--episodes", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=int, choices=range(3), default=1)
    parser.add_argument("--steps", type=positive_int, default=900)
    parser.add_argument("--thrust-scale", type=float, default=1., help="main-engine multiplier after the fault")
    parser.add_argument("--fault-at", type=float, default=10., help="simulation seconds before the fault")
    parser.add_argument("--fixed-model", action="store_true", help="disable MPC parameter learning")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--beam", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0., help="shield cost margin before overriding Laya")
    parser.add_argument("--prompt-estimate", action="store_true", help="append the engine estimate to the prompt")
    parser.add_argument("--out", type=Path, default=Path("dist/run.json"))
    parser.add_argument("--replay", type=Path, default=Path("dist/replay.html"))
    args = parser.parse_args(argv)
    if args.out.resolve() == args.replay.resolve():
        parser.error("--out and --replay must be different files")
    if not 0 < args.thrust_scale <= 1 or args.fault_at < 0:
        parser.error("thrust-scale in (0, 1] and fault-at nonnegative")
    try:
        start = perf_counter()
        pilot = make_pilot(args)
        record = new_recording(pilot)
        record["runtime"]["load_seconds"] = perf_counter() - start
        record["fault"] = {"thrust_scale": args.thrust_scale, "fault_at": args.fault_at}
        for seed in range(args.seed, args.seed + args.episodes):
            episode = run_episode(pilot, seed, args.target, args.steps, args.thrust_scale, args.fault_at)
            record["episodes"].append(episode)
            print(json.dumps(episode["summary"], allow_nan=False), flush=True)
        landed = sum(e["summary"]["status"] == "landed" for e in record["episodes"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        export_replay(record, args.replay)
        print(f"Landed {landed}/{args.episodes}\nRecording: {args.out}\nReplay: {args.replay}", file=sys.stderr)
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"lunar-mpc-laya: {exc}\n")


if __name__ == "__main__":
    main()
