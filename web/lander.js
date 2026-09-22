// Port of lunar_laya.game and lunar_mpc_laya.mpc (adaptive MPC only, no Laya).
// Same constants, substeps, landing rules, cost and beam search as the Python.
"use strict";
const GRAVITY = 1.62, THRUST = 5.0, TURN_RATE = 30.0, DT = 0.02, CONTROL_STEPS = 10, RADIUS = 8.0;
const STAGE = DT * CONTROL_STEPS;
const TERRAIN = [[0, 85], [80, 120], [150, 30], [240, 30], [300, 105], [370, 70], [440, 20],
  [560, 20], [640, 100], [720, 55], [775, 40], [825, 40], [900, 130], [1000, 90]];
const PADS = [[150, 240, 30, 2], [440, 560, 20, 1], [775, 825, 40, 4]];
const COMMANDS = [];
for (let turn = -1; turn <= 1; turn++) for (const throttle of [0, 0.5, 1]) COMMANDS.push({ turn, throttle });

function pymod(v, m) { const r = v % m; return r && (r < 0) !== (m < 0) ? r + m : r; }
function wrap(angle) { return pymod(angle + 180, 360) - 180; }
function clip(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function ground(x) {
  x = clip(x, 0, 1000);
  for (let i = 1; i < TERRAIN.length; i++) {
    const [x0, y0] = TERRAIN[i - 1], [x1, y1] = TERRAIN[i];
    if (x <= x1) return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  }
  return TERRAIN[TERRAIN.length - 1][1];
}

function surface(x) {
  let h = Math.max(ground(x - RADIUS), ground(x + RADIUS));
  for (const [px, py] of TERRAIN) if (x - RADIUS <= px && px <= x + RADIUS && py > h) h = py;
  return h;
}

// Game.step: advance up to CONTROL_STEPS substeps; terminal states absorb.
function step(s, command, thrustScale = 1) {
  for (let i = 0; i < CONTROL_STEPS; i++) {
    if (s.status !== "flying") break;
    const cost = (command.throttle * 0.65 + Math.abs(command.turn) * 0.04) * DT;
    const fraction = cost ? Math.min(1, s.fuel / cost) : 1;
    s.fuel = Math.max(0, s.fuel - cost);
    s.angle = wrap(s.angle + command.turn * TURN_RATE * DT * fraction);
    const thrust = command.throttle * THRUST * thrustScale * fraction, r = s.angle * Math.PI / 180;
    s.vx += Math.sin(r) * thrust * DT;
    s.vy += (Math.cos(r) * thrust - GRAVITY) * DT;
    s.x += s.vx * DT;
    s.y += s.vy * DT;
    s.time += DT;
    if (s.x < RADIUS || s.x > 1000 - RADIUS || s.y > 750) s.status = "out_of_bounds";
    else if (s.y - RADIUS <= surface(s.x)) {
      const pad = PADS.find(p => p[0] + RADIUS <= s.x && s.x <= p[1] - RADIUS);
      const safe = pad && Math.abs(s.vx) <= 2 && -3 <= s.vy && s.vy <= 0 && Math.abs(s.angle) <= 8;
      s.status = safe ? "landed" : "crashed";
      s.score = safe ? 50 * pad[3] : 0;
    } else if (s.time >= 180) s.status = "timeout";
  }
  return s;
}

class Dynamics {
  constructor(forgetting = 0.97) { this.thrust = THRUST; this.p = 4; this.forgetting = forgetting; this.updates = 0; }
  // One 0.2 s stage on a state vector [x, y, vx, vy, angle, fuel], written in place.
  predict(v, c) {
    const { turn, throttle } = COMMANDS[c], thrust = throttle * this.thrust, st = turn * TURN_RATE * DT;
    for (let i = 0; i < CONTROL_STEPS; i++) {
      v[4] = wrap(v[4] + st);
      const r = v[4] * Math.PI / 180;
      v[2] += Math.sin(r) * thrust * DT;
      v[3] += (Math.cos(r) * thrust - GRAVITY) * DT;
      v[0] += v[2] * DT;
      v[1] += v[3] * DT;
    }
    v[5] -= (throttle * 0.65 + Math.abs(turn) * 0.04) * STAGE;
    return v;
  }
  fit(phi, target) {
    const projected = this.p * phi, gain = projected / (this.forgetting + phi * projected);
    this.thrust += gain * (target - phi * this.thrust);
    this.p = (this.p - gain * projected) / this.forgetting;
  }
  observe(before, command, after, learn = true) {
    if (after.status !== "flying" || after.fuel <= 0 || !learn || !command.throttle) return;
    const elapsed = after.time - before.time;
    if (elapsed <= 0) return;
    const delta = wrap(after.angle - before.angle);
    let sbar = 0, cbar = 0;
    for (let k = 1; k <= CONTROL_STEPS; k++) {
      const r = (before.angle + delta * k / CONTROL_STEPS) * Math.PI / 180;
      sbar += Math.sin(r) / CONTROL_STEPS; cbar += Math.cos(r) / CONTROL_STEPS;
    }
    this.fit(command.throttle * sbar, (after.vx - before.vx) / elapsed);
    this.fit(command.throttle * cbar, (after.vy - before.vy) / elapsed + GRAVITY);
    this.thrust = clip(this.thrust, 1, 10);
    this.updates++;
  }
}

// ---------- shared lander art ----------
// One chamfered-box lander, shared byte for byte with lunar-mpc and lunar-laya.
// Coordinates are in units of RADIUS/8 with y pointing down, so the footpads sit
// exactly RADIUS below the hull centre and rest on the surface at touchdown.
const HULL = [[-6, -7], [-4, -9], [4, -9], [6, -7], [6, 1], [4, 3], [-4, 3], [-6, 1]];
const NOZZLE = [[-1.7, 3], [1.7, 3], [1.1, 5.2], [-1.1, 5.2]];
const STRUTS = [[-4, 3, -7.2, 8], [4, 3, 7.2, 8]];
const PADS_ART = [[-8.6, 8, -5.8, 8], [5.8, 8, 8.6, 8]];

// Draws into the caller's frame: translate to the hull centre and rotate first.
// `u` is one eighth of RADIUS in canvas pixels; `flip` is -1 for a y-up frame.
function drawLander(ctx, u, { body = "#d6e6de", trim = "#f0f5eb", glass = "#3a6265", flip = 1 } = {}) {
  const path = points => {
    ctx.beginPath();
    points.forEach(([x, y], i) => i ? ctx.lineTo(x * u, y * u * flip) : ctx.moveTo(x * u, y * u * flip));
    ctx.closePath();
  };
  ctx.lineJoin = "miter";
  path(NOZZLE); ctx.fillStyle = glass; ctx.fill();
  path(HULL); ctx.fillStyle = body; ctx.fill();
  ctx.strokeStyle = trim; ctx.lineWidth = Math.max(0.8, 0.35 * u); ctx.stroke();
  ctx.fillStyle = glass; ctx.fillRect(-2.2 * u, (flip > 0 ? -6.4 : 2) * u, 4.4 * u, 4.4 * u);
  ctx.strokeStyle = trim; ctx.lineWidth = Math.max(0.7, 0.25 * u);
  ctx.beginPath();
  ctx.moveTo(-6 * u, -1.2 * u * flip); ctx.lineTo(6 * u, -1.2 * u * flip);
  for (const [x0, y0, x1, y1] of STRUTS) { ctx.moveTo(x0 * u, y0 * u * flip); ctx.lineTo(x1 * u, y1 * u * flip); }
  ctx.stroke();
  // Footpads carry the weight, so they read heavier than the struts.
  ctx.lineWidth = Math.max(1.2, 0.5 * u); ctx.lineCap = 'butt';
  ctx.beginPath();
  for (const [x0, y0, x1, y1] of PADS_ART) { ctx.moveTo(x0 * u, y0 * u * flip); ctx.lineTo(x1 * u, y1 * u * flip); }
  ctx.stroke();
}

const CRASH = 1000;
class AdaptiveMPC {
  constructor({ horizon = 15, beam = 32, adaptive = true, forgetting = 0.97, switchCost = 8 } = {}) {
    this.horizon = horizon; this.beam = beam; this.adaptive = adaptive; this.switchCost = switchCost;
    this.model = new Dynamics(forgetting);
  }
  observe(before, command, after) { this.model.observe(before, command, after, this.adaptive); }
  // Returns {done, crashed} after advancing v in place; safe touchdown absorbs at zero cost.
  advance(v, c, done, pad) {
    if (done) return { done, crashed: false };
    this.model.predict(v, c);
    const [x, y, vx, vy, angle] = v;
    const contact = y - RADIUS <= surface(x);
    const onPad = x >= pad[0] + RADIUS && x <= pad[1] - RADIUS;
    const safe = contact && onPad && Math.abs(vx) <= 2 && vy >= -3 && vy <= 0 && Math.abs(angle) <= 8;
    const out = x < RADIUS || x > 1000 - RADIUS || y > 750;
    return { done: contact || out, crashed: (contact && !safe) || out };
  }
  // `previous` is the command index this candidate came from, or -1 with no history.
  cost(v, c, pad, previous) {
    const [x, y, vx, vy, angle] = v, px = x - (pad[0] + pad[1]) / 2;
    const altitude = Math.max(0, y - pad[2] - RADIUS);
    const vxRef = clip(-0.15 * px, -10, 10), angleRef = clip(12 * (vxRef - vx), -30, 30);
    let vyRef = -clip(0.12 * altitude, 0.7, 12);
    const offPad = Math.abs(px) > (pad[1] - pad[0]) / 2 - RADIUS ? 1 : 0;
    if (offPad) vyRef = Math.max(vyRef, (120 - altitude) * 0.15);
    const upward = Math.max(0.2, 0.7 * this.model.thrust * Math.cos(angle * Math.PI / 180) - GRAVITY);
    vyRef = Math.max(vyRef, -Math.sqrt(2 * upward * Math.max(0.05, altitude)));
    const clearance = y - RADIUS - surface(x);
    // Both differences are normalized to [0, 1] so a single knob tunes them.
    const change = previous < 0 ? 0 : 0.5 * Math.abs(COMMANDS[c].turn - COMMANDS[previous].turn)
      + Math.abs(COMMANDS[c].throttle - COMMANDS[previous].throttle);
    return 0.005 * px * px + 0.4 * (vx - vxRef) ** 2 + 2 * (vy - vyRef) ** 2 + 0.02 * (angle - angleRef) ** 2
      + 0.05 * COMMANDS[c].throttle + 0.05 * offPad * Math.max(0, 40 - clearance) ** 2 + this.switchCost * change;
  }
  // `previous` is the command actually flown last stage; null means no history yet.
  act(s, target, previous = null) {
    const begin = (typeof performance !== "undefined" ? performance : Date).now();
    const pad = PADS[target], start = [s.x, s.y, s.vx, s.vy, s.angle, s.fuel];
    let beam = [{ v: start, done: false, cost: 0, path: [], prior: previous === null ? -1 : previous }];
    for (let stage = 0; stage < this.horizon; stage++) {
      const next = [];
      for (const cand of beam) for (let c = 0; c < 9; c++) {
        const v = cand.v.slice();
        const { done, crashed } = this.advance(v, c, cand.done, pad);
        const stageCost = (done ? 0 : this.cost(v, c, pad, cand.prior)) * STAGE + (crashed ? CRASH : 0);
        next.push({ v, done, cost: cand.cost + stageCost, path: cand.path.concat(c), prior: c });
      }
      next.sort((a, b) => a.cost - b.cost);
      beam = next.slice(0, this.beam);
    }
    let best = beam[0];
    for (const cand of beam) if (cand.cost < best.cost) best = cand;
    const prediction = [start.slice()];
    let v = start.slice(), done = false;
    for (const c of best.path) { done = this.advance(v, c, done, pad).done; prediction.push(v.slice()); }
    return { plan: best.path, cost: best.cost, prediction,
      solveMs: (typeof performance !== "undefined" ? performance : Date).now() - begin };
  }
}

const Lander = { GRAVITY, THRUST, RADIUS, TERRAIN, PADS, COMMANDS, ground, surface, step, wrap, drawLander, Dynamics, AdaptiveMPC };
if (typeof module !== "undefined") module.exports = Lander; else globalThis.Lander = Lander;
