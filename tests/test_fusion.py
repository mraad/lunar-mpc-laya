"""Dependency-free checks: MPC landings, adaptation, shield, fault harness, replay."""

import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

import lunar_laya.game as game_module
from lunar_laya.cli import export_replay, new_recording
from lunar_laya.game import Command, Game
from lunar_mpc_laya.cli import run_episode
from lunar_mpc_laya.mpc import COMMANDS, AdaptiveMPC, MPCConfig, command_index, surface
from lunar_mpc_laya.pilot import MPCPilot

REQUEST = re.compile(r"Requested tilt correction: (\w+)\. Requested engine power: (\w+)\.")


def answers(rotation, engine):
    return {"answers": {"rotation": {"type": "choice", "choice": rotation, "confidence": 1.,
                                     "probabilities": {rotation: 1.}},
                        "engine": {"type": "choice", "choice": engine, "confidence": 1.,
                                   "probabilities": {engine: 1.}}}}


class EchoAgent:
    """Follows the requested labels exactly, like the trained checkpoint."""
    def predict(self, prompt, questions):
        return answers(*REQUEST.search(prompt).groups())


class ClimbAgent:
    """Always proposes right + full throttle."""
    def predict(self, prompt, questions):
        return answers("right", "full")


class Fusion(unittest.TestCase):
    def test_surface_matches_game(self):
        game = Game(0, 1)
        for x in np.linspace(0, 1000, 201):
            game.state.x = float(x)
            self.assertAlmostEqual(game.surface_height(), surface([x])[0], places=9)

    def test_command_index_roundtrip(self):
        for i, command in enumerate(COMMANDS):
            self.assertEqual(command_index(command), i)

    def test_mpc_lands_nominal(self):
        for target in range(3):
            for seed in range(5):
                summary = run_episode(MPCPilot("mpc"), seed, target, 900)["summary"]
                self.assertEqual(summary["status"], "landed", (target, seed, summary))
                self.assertLess(summary["solve_max_ms"], 50)

    def test_mpc_lands_from_far_start(self):
        pilot = MPCPilot("mpc")
        pilot.reset()
        game = Game(0, 2)
        game.state.x, game.state.y, game.state.vx, game.state.angle = 30., 300., 0., 0.
        while game.state.status == "flying":
            before = game.snapshot()
            command, _ = pilot.decide(game)
            pilot.observe(before, command, game.step(command))
        self.assertEqual(game.state.status, "landed", game.snapshot())

    def test_adaptation_tracks_fault(self):
        episode = run_episode(MPCPilot("mpc"), 3000, 1, 900, thrust_scale=.4, fault_at=10.)
        self.assertEqual(episode["summary"]["status"], "landed")
        after = [f for f in episode["frames"] if f["before"]["time"] >= 10.]
        self.assertLess(abs(after[30]["decision"]["model"]["thrust"] - 2.), .2)
        self.assertAlmostEqual(episode["summary"]["model"]["thrust"], 2., places=3)
        fixed = MPCPilot("mpc", mpc=AdaptiveMPC(MPCConfig(adaptive=False)))
        episode = run_episode(fixed, 3000, 1, 900, thrust_scale=.4, fault_at=10.)
        self.assertEqual(episode["summary"]["model"]["thrust"], 5.)
        self.assertEqual(episode["summary"]["status"], "crashed")

    def test_model_resets_between_episodes(self):
        pilot = MPCPilot("mpc")
        first = run_episode(pilot, 3000, 1, 900, thrust_scale=.4, fault_at=10.)
        second = run_episode(pilot, 3001, 1, 900, thrust_scale=.4, fault_at=10.)
        self.assertAlmostEqual(first["summary"]["model"]["thrust"], 2., places=3)
        self.assertEqual(second["frames"][0]["decision"]["model"], {"thrust": 5., "updates": 0})

    def test_fault_restores_thrust(self):
        run_episode(MPCPilot("mpc"), 0, 1, 20, thrust_scale=.5, fault_at=0.)
        self.assertEqual(game_module.THRUST, 5.)

    def test_echo_agent_reproduces_mpc(self):
        raw = run_episode(MPCPilot("mpc-laya", agent=EchoAgent()), 0, 1, 900)
        plain = run_episode(MPCPilot("mpc"), 0, 1, 900)
        self.assertEqual([f["after"] for f in raw["frames"]], [f["after"] for f in plain["frames"]])
        self.assertEqual(raw["summary"]["agreements"], raw["summary"]["decisions"])
        self.assertEqual(raw["summary"]["interventions"], 0)
        climb = run_episode(MPCPilot("mpc-laya", agent=ClimbAgent()), 0, 1, 20)
        self.assertEqual(climb["summary"]["agreements"], 0)
        self.assertEqual(run_episode(MPCPilot("mpc"), 0, 1, 5)["summary"]["agreements"], 0)

    def test_pd_pilot_records_request(self):
        from lunar_mpc_laya.cli import PDPilot
        episode = run_episode(PDPilot("baseline"), 0, 1, 5)
        self.assertEqual(episode["frames"][0]["decision"]["requested"], episode["frames"][0]["decision"]["executed"])

    def test_shield_overrides_only_worse_proposals(self):
        game = Game(0, 1)
        game.state.x, game.state.y, game.state.vx, game.state.vy, game.state.angle = 500., 40., 0., -1., 0.
        executed, decision = MPCPilot("mpc-assisted", agent=ClimbAgent()).decide(game)
        self.assertTrue(decision["intervened"])
        self.assertGreater(decision["proposal_cost"], decision["plan_cost"])
        self.assertEqual(executed, Command(*decision["executed"].values()))
        _, decision = MPCPilot("mpc-assisted", agent=EchoAgent()).decide(game)
        self.assertFalse(decision["intervened"])
        self.assertIsNone(decision["proposal_cost"])
        for key in ("requested", "proposed", "executed", "intervened", "answers", "guidance", "prompt",
                    "latency_ms", "plan", "prediction", "plan_cost", "model", "solve_ms"):
            self.assertIn(key, decision)

    def test_replay_export(self):
        pilot = MPCPilot("mpc-laya", agent=EchoAgent())
        record = new_recording(pilot)
        record["episodes"].append(run_episode(pilot, 0, 1, 5))
        self.assertEqual((record["pilot"]["mode"], record["pilot"]["pilot"]), ("laya", "mpc-laya"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.html"
            export_replay(record, path)
            html = path.read_text()
        self.assertIn('"type":"adaptive-mpc"', html)
        json.dumps(record, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
