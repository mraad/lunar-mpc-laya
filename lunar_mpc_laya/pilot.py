"""Adaptive MPC as Laya's guidance reference and predictive shield."""

from time import perf_counter

from lunar_laya.game import PADS, RADIUS, THRUST
from lunar_laya.pilot import Pilot, QUESTIONS, THROTTLES, TURNS, observation

from .mpc import COMMANDS, AdaptiveMPC, command_index

UPSTREAM = {"mpc": "baseline", "mpc-laya": "laya", "mpc-assisted": "assisted"}


class MPCPilot(Pilot):
    """Upstream Pilot with ``guidance()`` replaced by adaptive MPC.

    The prompt template is upstream ``observation()`` byte for byte so the
    trained checkpoint sees the distribution it was trained on.
    """

    def __init__(self, mode="mpc-assisted", model="aac6fef/laya-multilingual-mlx", revision=None,
                 agent=None, mpc=None, margin=0., prompt_estimate=False):
        if mode not in UPSTREAM:
            raise ValueError("unknown pilot mode")
        super().__init__(UPSTREAM[mode], model, revision, agent)
        self.mode = mode
        self.mpc = mpc or AdaptiveMPC()
        self.margin, self.prompt_estimate = margin, prompt_estimate
        # Keep upstream's mode so its replay banner stays truthful; name the fusion pilot separately.
        self.provenance.update(pilot=mode, controller={"type": "adaptive-mpc", **vars(self.mpc.cfg),
                                           "shield_margin": margin, "prompt_estimate": prompt_estimate})

    def decide(self, game):
        start = perf_counter()
        snapshot = game.snapshot()
        plan = self.mpc.act(snapshot, game.target)
        reference = COMMANDS[plan["plan"][0]]
        pad, nxt = PADS[game.target], plan["prediction"][1]
        metrics = {"target_dx": (pad[0] + pad[1]) / 2 - snapshot["x"],
                   "altitude": max(0, snapshot["y"] - pad[2] - RADIUS),
                   "desired_angle": nxt[4], "desired_vy": nxt[3]}
        prompt = observation(game, reference, metrics)
        if self.prompt_estimate:
            prompt += f" Estimated engine effectiveness {100 * self.mpc.model.thrust / THRUST:.0f}%."
        answers, proposed, alternative = None, reference, None
        if self.mode != "mpc":
            result = self.agent.predict(prompt, QUESTIONS)
            answers = result["answers"]
            proposed = COMMANDS[command_index(type(reference)(
                TURNS[answers["rotation"]["choice"]], THROTTLES[answers["engine"]["choice"]]))]
        intervened = False
        if self.mode == "mpc-assisted" and proposed != reference:
            # Predictive shield: keep the proposal unless MPC's best continuation
            # after it is worse than its own plan by more than the margin.
            alternative = self.mpc.act(snapshot, game.target, first=command_index(proposed))["cost"]
            intervened = alternative > plan["cost"] + self.margin
        executed = reference if intervened else proposed
        return executed, {"requested": {"turn": reference.turn, "throttle": reference.throttle},
                          "proposed": {"turn": proposed.turn, "throttle": proposed.throttle},
                          "executed": {"turn": executed.turn, "throttle": executed.throttle},
                          "intervened": intervened, "answers": answers, "guidance": metrics,
                          "prompt": prompt, "latency_ms": (perf_counter() - start) * 1000,
                          "plan": plan["plan"], "prediction": plan["prediction"],
                          "plan_cost": plan["cost"], "proposal_cost": alternative,
                          "model": {"thrust": self.mpc.model.thrust, "updates": self.mpc.model.updates},
                          "prediction_error": self.mpc.model.last_error, "solve_ms": plan["solve_ms"]}

    def observe(self, before, command, after):
        self.mpc.observe(before, command, after)

    def reset(self):
        """Fresh nominal model per episode; estimates never carry between flights."""
        self.mpc = AdaptiveMPC(self.mpc.cfg)
