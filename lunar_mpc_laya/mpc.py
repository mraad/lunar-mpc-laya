"""Adaptive finite-control-set MPC for the arcade lander in ``lunar_laya.game``.

Modelled on lunar-mpc: a small parametric model updated by bounded recursive
least squares, and a NumPy beam search over the nine discrete commands. The
controller sees only ``Game.snapshot()`` values and the public map geometry;
fault timing and true engine strength stay in the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from lunar_laya.game import (CONTROL_STEPS, DT, GRAVITY, PADS, RADIUS, TERRAIN, THRUST,
                             TURN_RATE, Command)

STAGE = DT * CONTROL_STEPS  # one decision = 0.2 s
TURN = np.repeat([-1, 0, 1], 3)
THROTTLE = np.tile([0., .5, 1.], 3)
COMMANDS = [Command(int(t), float(p)) for t, p in zip(TURN, THROTTLE)]
FIELDS = ("x", "y", "vx", "vy", "angle", "fuel")
_TX = np.array([p[0] for p in TERRAIN], float)
_TY = np.array([p[1] for p in TERRAIN], float)


def command_index(command):
    return int((command.turn + 1) * 3 + round(command.throttle * 2))


def state_vector(snapshot):
    return np.array([float(snapshot[k]) for k in FIELDS])


def surface(x):
    """Vectorized ``Game.surface_height``: hull footprint plus vertices under it."""
    x = np.asarray(x, float)
    edges = np.maximum(np.interp(x - RADIUS, _TX, _TY), np.interp(x + RADIUS, _TX, _TY))
    under = np.abs(_TX[None, :] - x[:, None]) <= RADIUS
    return np.maximum(edges, np.where(under, _TY[None, :], -np.inf).max(axis=1))


@dataclass
class MPCConfig:
    horizon: int = 15
    beam: int = 32
    adaptive: bool = True
    forgetting: float = 0.97

    def __post_init__(self):
        if min(self.horizon, self.beam) < 1 or not 0 < self.forgetting <= 1:
            raise ValueError("horizon and beam must be positive; forgetting in (0, 1]")


class Dynamics:
    """Thrust gain learned by scalar recursive least squares; gravity and turn
    rate are known constants of this world.

    Learning gravity too makes thrust and gravity collinear during a sustained
    upright burn, and the forgetting factor then drifts both along the
    unobservable direction; the fault scenario only changes engine strength.
    """

    def __init__(self, forgetting=0.97):
        self.thrust = float(THRUST)
        self.gravity = float(GRAVITY)
        self.turn_rate = float(TURN_RATE)
        self.p = 4.
        self.forgetting = forgetting
        self.updates = 0
        self.last_error = 0.

    def predict(self, states, commands):
        """Advance states one stage under constant commands, same substeps as Game.step.

        Fuel decreases nominally; the fuel-starvation fraction is not modelled.
        """
        s = np.array(states, float)
        commands = np.asarray(commands)
        turn, thrust = TURN[commands], THROTTLE[commands] * self.thrust
        step = turn * self.turn_rate * DT
        for _ in range(CONTROL_STEPS):
            s[:, 4] = (s[:, 4] + step + 180) % 360 - 180
            r = np.radians(s[:, 4])
            s[:, 2] += np.sin(r) * thrust * DT
            s[:, 3] += (np.cos(r) * thrust - self.gravity) * DT
            s[:, 0] += s[:, 2] * DT
            s[:, 1] += s[:, 3] * DT
        s[:, 5] -= (THROTTLE[commands] * .65 + np.abs(turn) * .04) * STAGE
        return s

    def _fit(self, phi, target):
        projected = self.p * phi
        gain = projected / (self.forgetting + phi * projected)
        self.thrust += gain * (target - phi * self.thrust)
        self.p = (self.p - gain * projected) / self.forgetting

    def observe(self, before, command, after, learn=True):
        # Terminal contact and an emptied tank are not evidence about the engine.
        if after["status"] != "flying" or after["fuel"] <= 0:
            return
        elapsed = after["time"] - before["time"]
        if elapsed <= 0:
            return
        predicted = self.predict(state_vector(before)[None], [command_index(command)])[0]
        self.last_error = math.hypot(after["vx"] - predicted[2], after["vy"] - predicted[3])
        if not learn or not command.throttle:
            return
        delta = (after["angle"] - before["angle"] + 180) % 360 - 180
        samples = np.radians(before["angle"] + delta * np.arange(1, CONTROL_STEPS + 1) / CONTROL_STEPS)
        ax, ay = (after["vx"] - before["vx"]) / elapsed, (after["vy"] - before["vy"]) / elapsed
        self._fit(command.throttle * float(np.sin(samples).mean()), ax)
        self._fit(command.throttle * float(np.cos(samples).mean()), ay + self.gravity)
        self.thrust = float(np.clip(self.thrust, 1., 10.))
        self.updates += 1


class AdaptiveMPC:
    CRASH = 1000.

    def __init__(self, config=None):
        self.cfg = config or MPCConfig()
        self.model = Dynamics(self.cfg.forgetting)

    def observe(self, before, command, after):
        self.model.observe(before, command, after, learn=self.cfg.adaptive)

    def advance(self, states, commands, done, pad):
        predicted = self.model.predict(states, commands)
        predicted[done] = states[done]
        x, y, vx, vy, angle = predicted[:, :5].T
        contact = y - RADIUS <= surface(x)
        on_pad = (x >= pad[0] + RADIUS) & (x <= pad[1] - RADIUS)
        safe = contact & on_pad & (np.abs(vx) <= 2) & (vy >= -3) & (vy <= 0) & (np.abs(angle) <= 8)
        out = (x < RADIUS) | (x > 1000 - RADIUS) | (y > 750)
        crashed = ~done & ((contact & ~safe) | out)
        # Safe contact is absorbing with zero cost; this mirrors Game's landing
        # rule at stage ends only, not at every physics substep.
        return predicted, done | contact | out, crashed

    def cost(self, states, commands, pad):
        x, y, vx, vy, angle = states[:, :5].T
        px = x - (pad[0] + pad[1]) / 2
        altitude = np.maximum(0, y - pad[2] - RADIUS)
        vx_ref = np.clip(-.15 * px, -10, 10)
        angle_ref = np.clip(12 * (vx_ref - vx), -30, 30)
        vy_ref = -np.clip(.12 * altitude, .7, 12)
        # Hold at least 120 m above the pad while off it; the 3 s horizon cannot
        # see the mountains a 10 m/s traverse reaches.
        off_pad = np.abs(px) > (pad[1] - pad[0]) / 2 - RADIUS
        vy_ref = np.where(off_pad, np.maximum(vy_ref, (120 - altitude) * .15), vy_ref)
        # Lower estimated thrust means braking earlier; .7 is a design margin,
        # not a certified uncertainty bound (same convention as lunar-mpc).
        upward = np.maximum(.2, .7 * self.model.thrust * np.cos(np.radians(angle)) - self.model.gravity)
        vy_ref = np.maximum(vy_ref, -np.sqrt(2 * upward * np.maximum(.05, altitude)))
        clearance = y - RADIUS - surface(x)
        return (.005 * px**2 + .4 * (vx - vx_ref)**2 + 2 * (vy - vy_ref)**2
                + .02 * (angle - angle_ref)**2 + .05 * THROTTLE[commands]
                + .05 * off_pad * np.maximum(0, 40 - clearance)**2)

    def act(self, snapshot, target, first=None):
        """Beam search; ``first`` forces the first command (used by the shield)."""
        begin = time.perf_counter()
        pad = PADS[target]
        state = state_vector(snapshot)
        states, done, costs = state[None], np.zeros(1, bool), np.zeros(1)
        paths = np.zeros((1, 0), int)
        for stage in range(self.cfg.horizon):
            width = 1 if stage == 0 and first is not None else 9
            commands = np.array([first]) if width == 1 else np.tile(np.arange(9), len(states))
            states, done, crashed = self.advance(np.repeat(states, width, 0), commands,
                                                 np.repeat(done, width), pad)
            stage_cost = np.where(done, 0., self.cost(states, commands, pad)) * STAGE
            costs = np.repeat(costs, width) + stage_cost + self.CRASH * crashed
            paths = np.column_stack((np.repeat(paths, width, 0), commands))
            keep = np.argsort(costs, kind="stable")[:self.cfg.beam]
            states, done, costs, paths = states[keep], done[keep], costs[keep], paths[keep]
        best = int(np.argmin(costs))
        plan = paths[best].tolist()
        predicted, settled, trace = state[None], np.zeros(1, bool), [state.tolist()]
        for command in plan:
            predicted, settled, _ = self.advance(predicted, np.array([command]), settled, pad)
            trace.append(predicted[0].tolist())
        return {"plan": plan, "cost": float(costs[best]), "prediction": trace,
                "solve_ms": (time.perf_counter() - begin) * 1000}
