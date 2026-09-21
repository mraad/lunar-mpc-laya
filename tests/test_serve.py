"""Live server: session semantics with a fake agent, and the HTTP round trip without a model."""

from functools import partial
from http.server import HTTPServer
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import lunar_laya.game as game_module
from lunar_mpc_laya.serve import Handler, Live
from tests.test_fusion import EchoAgent


def post(url, payload):
    request = Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urlopen(request) as response:
        return json.loads(response.read())


class Serve(unittest.TestCase):
    def test_live_session_with_fake_laya(self):
        live = Live(agent=EchoAgent(), provenance={"model": "fake"})
        self.assertEqual(live.status()["modes"], ["mpc", "mpc-laya", "mpc-assisted"])
        with self.assertRaises(ValueError):
            live.step({"n": 1})
        start = {"x": 500, "y": 450, "vx": 0, "vy": -8, "angle": 0}
        live.reset({"mode": "mpc-assisted", "start": start, "target": 1, "thrust_scale": .4, "adaptive": True})
        frames = []
        while live.session.game.state.status == "flying":
            frames += live.step({"n": 20})["frames"]
        self.assertEqual(frames[-1]["after"]["status"], "landed")
        self.assertTrue(all(f["decision"]["answers"] is not None for f in frames))
        self.assertEqual(sum(f["decision"]["intervened"] for f in frames), 0)
        self.assertAlmostEqual(frames[-1]["decision"]["model"]["thrust"], 2., places=2)
        self.assertEqual(game_module.THRUST, 5.)
        self.assertEqual(live.step({"n": 5})["frames"], [])
        # Second reset starts nominal again.
        live.reset({"mode": "mpc-laya", "start": start, "target": 0})
        self.assertEqual(live.step({"n": 1})["frames"][0]["decision"]["model"], {"thrust": 5., "updates": 0})
        for bad in ({"mode": "nope"}, {"start": {"x": "a"}}, {"target": 7}, {"thrust_scale": 0}):
            with self.assertRaises(ValueError):
                live.reset({"start": start, **bad})

    def test_http_round_trip_without_model(self):
        server = HTTPServer(("127.0.0.1", 0), partial(Handler, live=Live(), directory="web"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/status") as response:
                self.assertEqual(json.loads(response.read())["modes"], ["mpc"])
            with urlopen(base + "/index.html") as response:
                self.assertIn(b"Landing lab", response.read())
            with self.assertRaises(HTTPError) as error:
                post(base + "/reset", {"mode": "mpc-laya", "start": {"x": 500, "y": 300, "vx": 0, "vy": -8, "angle": 0}})
            self.assertEqual(error.exception.code, 400)
            state = post(base + "/reset", {"mode": "mpc", "start": {"x": 500, "y": 300, "vx": 0, "vy": -8, "angle": 0}, "target": 2})["state"]
            self.assertEqual(state["status"], "flying")
            frames = post(base + "/step", {"n": 3})["frames"]
            self.assertEqual(len(frames), 3)
            self.assertIn("prediction", frames[0]["decision"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
