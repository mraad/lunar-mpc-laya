"""Telemetry-only questions and prompt for the distilled pilot: no requested labels."""

from lunar_laya.game import PADS, RADIUS

QUESTIONS = {
    "rotation": {
        "type": "choice",
        "instructions": "Control lunar lander tilt. Choose the tilt correction that lands safely on the target pad.",
        "criteria": {"left": "Decrease tilt angle", "hold": "Keep current tilt", "right": "Increase tilt angle"},
    },
    "engine": {
        "type": "choice",
        "instructions": "Control lunar lander engine. Choose the engine power that lands safely on the target pad.",
        "criteria": {"off": "Zero thrust", "half": "Half thrust", "full": "Full thrust"},
    },
}


def observation(game):
    s = game.state
    pad = PADS[game.target]
    dx = (pad[0] + pad[1]) / 2 - s.x
    altitude = max(0, s.y - pad[2] - RADIUS)
    return (f"Lunar landing. Altitude {altitude:.1f} m. Pad offset {dx:.1f} m. "
            f"Horizontal velocity {s.vx:.1f} m/s. Vertical velocity {s.vy:.1f} m/s. "
            f"Tilt {s.angle:.1f} degrees. Fuel {s.fuel:.1f}. "
            "Positive x is right, positive y is up, positive tilt is right. "
            "Land upright with horizontal speed <=2 and downward speed <=3 m/s.")
