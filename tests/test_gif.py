"""The GIF renderer draws recorded frames and the terminal summary without touching physics."""

import json
from pathlib import Path
import tempfile
import unittest

from lunar_laya.cli import new_recording
from lunar_mpc_laya.cli import run_episode
from lunar_mpc_laya.distilled import DistilledPilot
from tests.test_training import Fixed

try:
    from lunar_mpc_laya.gif import export_gif, render
except ImportError:  # Pillow is an optional extra
    export_gif = None


@unittest.skipIf(export_gif is None, "media extra not installed")
class Gif(unittest.TestCase):
    def test_export(self):
        pilot = DistilledPilot("distilled-assisted", agent=Fixed("hold", "off"))
        record = new_recording(pilot)
        record["fault"] = {"thrust_scale": .4, "fault_at": 10.}
        record["episodes"].append(run_episode(pilot, 3000, 1, 60, .4, 10.))
        image = render(record, record["episodes"][0], len(record["episodes"][0]["frames"]))
        self.assertEqual(image.size, (1000, 620))
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "r.json", Path(tmp) / "out.gif"
            source.write_text(json.dumps(record))
            info = export_gif(source, output, speed=16)
            self.assertEqual(info["frames"], 60 // 8 + 1 + 1)
            self.assertGreater(output.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
