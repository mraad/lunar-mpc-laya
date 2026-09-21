"""Local server: the landing lab page plus live MPC + Laya decisions for the browser.

Binds 127.0.0.1 only. Static files come from dist/web when built, else web/.
POST /reset starts a flight from a browser-chosen state; POST /step advances it
with the same Python pilot the CLI uses, so live Laya flights are real inference.
"""

import argparse
from functools import partial
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import math
from pathlib import Path

from lunar_laya.game import PADS, Game, State
from lunar_laya.pilot import Pilot

from .cli import Session
from .mpc import AdaptiveMPC, MPCConfig
from .pilot import MPCPilot, UPSTREAM

FIELDS = ("x", "y", "vx", "vy", "angle")


class Live:
    """One flight at a time; the Laya agent is loaded once and shared across resets."""

    def __init__(self, agent=None, provenance=None):
        self.agent, self.provenance, self.session = agent, provenance or {}, None

    def status(self):
        modes = ["mpc"] + (["mpc-laya", "mpc-assisted"] if self.agent else [])
        return {"live": True, "modes": modes, "model": self.provenance}

    def reset(self, request):
        mode = request.get("mode", "mpc")
        if mode not in UPSTREAM or (mode != "mpc" and self.agent is None):
            raise ValueError(f"pilot {mode!r} unavailable; start the server with a model")
        start = request.get("start", {})
        values = [float(start.get(k, math.nan)) for k in FIELDS]
        if not all(math.isfinite(v) for v in values) or not 0 <= values[1] <= 750:
            raise ValueError("start needs finite x, y, vx, vy, angle with y in [0, 750]")
        target = int(request.get("target", 1))
        if target not in range(len(PADS)):
            raise ValueError("target must be 0, 1 or 2")
        cfg = MPCConfig(adaptive=bool(request.get("adaptive", True)))
        pilot = MPCPilot(mode, agent=self.agent, mpc=AdaptiveMPC(cfg), margin=float(request.get("margin", 0.)))
        game = Game(0, target)
        game.state = State(*values)
        self.session = Session(pilot, game, float(request.get("thrust_scale", 1.)), float(request.get("fault_at", 10.)))
        return {"state": game.snapshot(), "pilot": pilot.provenance}

    def step(self, request):
        if self.session is None:
            raise ValueError("reset first")
        n = int(request.get("n", 1))
        if not 1 <= n <= 50:
            raise ValueError("n must be in [1, 50]")
        return {"frames": self.session.step(n)}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, live, **kwargs):
        self.live = live
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def reply(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            return self.reply(HTTPStatus.OK, self.live.status())
        return super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 65536:
            return self.reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/reset":
                return self.reply(HTTPStatus.OK, self.live.reset(request))
            if self.path == "/step":
                return self.reply(HTTPStatus.OK, self.live.step(request))
            return self.reply(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            return self.reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="../lunar-laya/models/lunar-laya-supervised-mlx")
    parser.add_argument("--revision")
    parser.add_argument("--no-model", action="store_true", help="serve MPC-only live flights")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--root", type=Path, help="static directory (default dist/web if built, else web)")
    args = parser.parse_args(argv)
    root = args.root or (Path("dist/web") if Path("dist/web/index.html").exists() else Path("web"))
    live = Live()
    if not args.no_model:
        try:
            loader = Pilot("laya", args.model, args.revision)
            live = Live(loader.agent, loader.provenance)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Laya unavailable ({exc}); serving MPC-only live flights", flush=True)
    server = HTTPServer(("127.0.0.1", args.port), partial(Handler, live=live, directory=str(root)))
    print(f"Serving {root} with live pilots {live.status()['modes']} at http://127.0.0.1:{args.port}/ (Ctrl+C stops)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
