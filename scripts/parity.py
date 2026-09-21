"""Write web/parity.json: Python start states, MPC commands and outcomes for the JS port to match."""

import json
from pathlib import Path

from lunar_laya.game import Command, Game
from lunar_mpc_laya.cli import run_episode
from lunar_mpc_laya.mpc import command_index
from lunar_mpc_laya.pilot import MPCPilot

cases = []
for target in range(3):
    for seed, scale in ((3000, 1.), (3001, .4)):
        start = Game(seed, target).snapshot()
        episode = run_episode(MPCPilot("mpc"), seed, target, 900, scale, 10.)
        frames = episode["frames"]
        cases.append({"seed": seed, "target": target, "thrust_scale": scale, "start": start,
                      "commands": [command_index(Command(**f["decision"]["executed"])) for f in frames],
                      "final": {k: episode["summary"][k] for k in ("status", "time", "x", "fuel")},
                      "thrust_estimate": episode["summary"]["model"]["thrust"]})
Path("web/parity.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
print(len(cases), "cases", [c["final"]["status"] for c in cases])
