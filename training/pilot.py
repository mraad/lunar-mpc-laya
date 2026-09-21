"""Distilled pilot: Laya decides from telemetry alone; MPC only requests (for scoring) and shields."""

from time import perf_counter

from lunar_laya.game import PADS, RADIUS
from lunar_laya.pilot import THROTTLES, TURNS

from lunar_mpc_laya.mpc import COMMANDS, command_index
from lunar_mpc_laya.pilot import MPCPilot

from .prompt import QUESTIONS, observation

UPSTREAM = {"distilled-laya": "mpc-laya", "distilled-assisted": "mpc-assisted"}


class DistilledPilot(MPCPilot):
    def __init__(self, mode="distilled-assisted", model=None, revision=None, agent=None, mpc=None, margin=0.):
        if mode not in UPSTREAM:
            raise ValueError("unknown pilot mode")
        super().__init__(UPSTREAM[mode], model, revision, agent, mpc, margin)
        self.mode_name = mode
        self.provenance.update(pilot=mode, prompt="telemetry-only", questions=QUESTIONS)

    def decide(self, game):
        start = perf_counter()
        snapshot = game.snapshot()
        plan = self.mpc.act(snapshot, game.target)
        reference = COMMANDS[plan["plan"][0]]
        pad = PADS[game.target]
        prompt = observation(game)
        answers = self.agent.predict(prompt, QUESTIONS)["answers"]
        proposed = COMMANDS[command_index(type(reference)(
            TURNS[answers["rotation"]["choice"]], THROTTLES[answers["engine"]["choice"]]))]
        alternative, intervened = None, False
        if self.mode_name == "distilled-assisted" and proposed != reference:
            alternative = self.mpc.act(snapshot, game.target, first=command_index(proposed))["cost"]
            intervened = alternative > plan["cost"] + self.margin
        executed = reference if intervened else proposed
        return executed, {"requested": {"turn": reference.turn, "throttle": reference.throttle},
                          "proposed": {"turn": proposed.turn, "throttle": proposed.throttle},
                          "executed": {"turn": executed.turn, "throttle": executed.throttle},
                          "intervened": intervened, "answers": answers,
                          "guidance": {"target_dx": (pad[0] + pad[1]) / 2 - snapshot["x"],
                                       "altitude": max(0, snapshot["y"] - pad[2] - RADIUS),
                                       "desired_angle": plan["prediction"][1][4], "desired_vy": plan["prediction"][1][3]},
                          "prompt": prompt, "latency_ms": (perf_counter() - start) * 1000,
                          "plan": plan["plan"], "prediction": plan["prediction"], "plan_cost": plan["cost"],
                          "proposal_cost": alternative,
                          "model": {"thrust": self.mpc.model.thrust, "updates": self.mpc.model.updates},
                          "prediction_error": self.mpc.model.last_error, "solve_ms": plan["solve_ms"]}
