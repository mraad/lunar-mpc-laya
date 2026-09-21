"""Telemetry-only prompt, MPC-labelled data, and the distilled pilot's shield with fake students."""

import json
from pathlib import Path
import tempfile
import unittest

from lunar_laya.game import Game
from lunar_mpc_laya.cli import run_episode
from lunar_mpc_laya.mpc import COMMANDS
from training.data import generate, labels
from training.pilot import DistilledPilot
from training.prompt import QUESTIONS, observation


class Fixed:
    def __init__(self, rotation, engine):
        self.answers = {"rotation": rotation, "engine": engine}

    def predict(self, prompt, questions):
        return {"answers": {q: {"type": "choice", "choice": self.answers[q], "confidence": 1.,
                                "probabilities": {self.answers[q]: 1.}} for q in questions}}


class Training(unittest.TestCase):
    def test_prompt_has_no_requested_labels(self):
        text = observation(Game(0, 1))
        self.assertNotIn("Requested", text)
        self.assertIn("Altitude", text)
        for q in QUESTIONS.values():
            self.assertNotIn("requested", q["instructions"].lower())

    def test_labels_cover_all_commands(self):
        seen = {tuple(labels(c).values()) for c in COMMANDS}
        self.assertEqual(len(seen), 9)

    def test_generate_small_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate(Path(tmp) / "d", train_seeds=1, synthetic=6, validation_seeds=1)
            rows = [json.loads(s) for s in (Path(tmp) / "d" / "train.jsonl").read_text().splitlines()]
            with self.assertRaises(FileExistsError):
                generate(Path(tmp) / "d", train_seeds=1, synthetic=0)
        self.assertFalse(manifest["prompt_contains_requested_labels"])
        self.assertEqual({r["thrust_scale"] for r in rows if r["source"] == "mpc_rollout"}, {1., .7, .4})
        self.assertTrue(all("Requested" not in r["state"] for r in rows))
        self.assertEqual({r["question"] for r in rows}, {"rotation", "engine"})

    def test_shield_rescues_a_bad_student(self):
        raw = run_episode(DistilledPilot("distilled-laya", agent=Fixed("hold", "off")), 3000, 1, 900)["summary"]
        shielded = run_episode(DistilledPilot("distilled-assisted", agent=Fixed("hold", "off")), 3000, 1, 900)["summary"]
        self.assertEqual(raw["status"], "crashed")
        self.assertEqual(shielded["status"], "landed")
        self.assertGreater(shielded["interventions"], shielded["decisions"] // 2)
        self.assertEqual(raw["interventions"], 0)


if __name__ == "__main__":
    unittest.main()
