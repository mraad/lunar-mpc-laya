"""Record one distilled-Laya shielded flight through the x0.4 fault for the documentation GIF."""

import json
from pathlib import Path
import sys

from lunar_laya.cli import new_recording
from lunar_laya.pilot import Pilot
from lunar_mpc_laya.cli import run_episode
from lunar_mpc_laya.distilled import DistilledPilot

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
loader = Pilot("laya", "models/lunar-mpc-laya-distilled-mlx")
pilot = DistilledPilot("distilled-assisted", agent=loader.agent)
record = new_recording(pilot)
record["fault"] = {"thrust_scale": .4, "fault_at": 10.}
record["episodes"].append(run_episode(pilot, seed, 1, 900, .4, 10.))
Path("dist").mkdir(exist_ok=True)
Path("dist/gif-flight.json").write_text(json.dumps(record, separators=(",", ":")) + "\n")
print(json.dumps(record["episodes"][0]["summary"]))
