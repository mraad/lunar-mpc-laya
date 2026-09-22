"""Generate MPC-labelled, telemetry-only decisions with disjoint seed ranges and fault scenarios."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from lunar_laya.game import Game, PADS, State
from lunar_laya.pilot import THROTTLES, TURNS
from lunar_mpc_laya.cli import Session
from lunar_mpc_laya.mpc import COMMANDS, AdaptiveMPC, command_index
from lunar_mpc_laya.pilot import MPCPilot

from .prompt import QUESTIONS, observation

FAULTS = (1., .7, .4)


def labels(command):
    return {"rotation": next(k for k, v in TURNS.items() if v == command.turn),
            "engine": next(k for k, v in THROTTLES.items() if v == command.throttle)}


def examples(game, command, split, source, extra, previous=None):
    state = observation(game, previous)
    lab = labels(command)
    return [{"state": state, "question": q, "label": lab[q], "split": split,
             "seed": game.seed, "target": game.target, "source": source, **extra} for q in QUESTIONS]


def generate(output, train_seeds=32, synthetic=3000, every=4, validation_seeds=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"teacher": "lunar_mpc_laya.pilot.MPCPilot (adaptive MPC)", "questions": QUESTIONS,
                "prompt_contains_requested_labels": False, "prompt_contains_flown_command": True,
                "faults": FAULTS, "every": every, "splits": {}}
    for split, start, count in (("train", 1000, train_seeds), ("validation", 2000, validation_seeds)):
        rows = []
        for seed in range(start, start + count):
            for target in range(3):
                for scale in FAULTS:
                    pilot = MPCPilot("mpc")
                    session = Session(pilot, Game(seed, target), scale, 10.)
                    step = 0
                    while session.game.state.status == "flying":
                        before = session.game.snapshot()
                        # Captured before the step, because observe() advances it.
                        previous = None if pilot.previous is None else COMMANDS[pilot.previous]
                        frames = session.step(1)
                        if step % every == 0:
                            game = Game(seed, target)
                            game.state = State(**before)
                            command = COMMANDS[frames[0]["decision"]["plan"][0]]
                            rows.extend(examples(game, command, split, "mpc_rollout",
                                                 {"thrust_scale": scale}, previous))
                        step += 1
        if synthetic:
            rng = random.Random(73019 if split == "train" else 81019)
            for i in range(synthetic if split == "train" else 500):
                target = i % 3
                pad = PADS[target]
                game = Game((10000 if split == "train" else 20000) + i, target)
                game.state = State((pad[0] + pad[1]) / 2 + rng.uniform(-150, 150),
                                   rng.uniform(80, 650), rng.uniform(-18, 18),
                                   rng.uniform(-20, 8), rng.uniform(-40, 40), rng.uniform(5, 100))
                # Label from the nominal model; the estimate is not in the prompt, so the
                # synthetic teacher sees the same information as the student. The flown
                # command is sampled, because a synthetic state has no flight behind it
                # and the student must learn to hold or break from any of the nine.
                previous = COMMANDS[rng.randrange(len(COMMANDS))]
                plan = AdaptiveMPC().act(game.snapshot(), target, previous=command_index(previous))
                rows.extend(examples(game, COMMANDS[plan["plan"][0]], split, "synthetic_recovery",
                                     {"thrust_scale": 1.}, previous))
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["splits"][split] = {
            "rows": len(rows), "rollout_seeds": [start, start + count - 1],
            "labels": dict(Counter(f"{r['question']}:{r['label']}" for r in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, default=32)
    parser.add_argument("--synthetic", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.train_seeds, args.synthetic), indent=2))
