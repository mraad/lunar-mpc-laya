"use strict";
const L = globalThis.Lander;
const $ = id => document.getElementById(id);
const FAULT_AT = 10;
const lab = { canvas: $("canvas"), start: { x: 500, y: 450, vx: 0, vy: -8, angle: 0 }, target: 1, thrustScale: 1, pilot: "mpc",
  s: null, mpc: null, thrust: L.THRUST, trail: [], plan: null, command: null, decision: null, playing: false, acc: 0, busy: false };
const rec = { canvas: $("rec-canvas"), runs: [], run: null, episode: null, frame: 0, playing: false, acc: 0, loaded: false };
lab.ctx = lab.canvas.getContext("2d");
rec.ctx = rec.canvas.getContext("2d");
let last = null;

const notice = text => { $("notice").textContent = text || ""; };
const word = c => c ? `${c.turn > 0 ? "right" : c.turn < 0 ? "left" : "hold"} · ${c.throttle >= 1 ? "full" : c.throttle > 0 ? "half" : "off"}` : "—";
async function api(path, body) {
  const response = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

// ---------- drawing (shared by both tabs) ----------
function draw(view, s, trail, prediction, target, command, placing) {
  const { canvas, ctx } = view, W = canvas.width, H = canvas.height, k = W / 1100, pad = L.PADS[target];
  const px = x => (50 + x) * k, py = y => H - 38 * k - y * (H - 58 * k) / 750;
  ctx.fillStyle = "#070e17"; ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < 100; i++) { ctx.fillStyle = i % 7 ? "#263847" : "#687e8c"; ctx.fillRect((i * 173 + 19) % W, (i * 79 + 31) % (H - 100), 1.4, 1.4); }
  ctx.strokeStyle = "#142333"; ctx.lineWidth = 1;
  for (let x = 0; x <= 1000; x += 100) { ctx.beginPath(); ctx.moveTo(px(x), 0); ctx.lineTo(px(x), H - 38 * k); ctx.stroke(); }
  for (let y = 100; y <= 700; y += 100) {
    ctx.beginPath(); ctx.moveTo(px(0), py(y)); ctx.lineTo(px(1000), py(y)); ctx.stroke();
    ctx.fillStyle = "#3b5366"; ctx.font = "10px monospace"; ctx.fillText(String(y), 13, py(y) + 3);
  }
  const line = (points, color, width, dash) => {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
    points.forEach(([x, y], i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)));
    ctx.stroke(); ctx.setLineDash([]);
  };
  ctx.beginPath(); ctx.moveTo(px(0), H); L.TERRAIN.forEach(([x, y]) => ctx.lineTo(px(x), py(y)));
  ctx.lineTo(px(1000), H); ctx.closePath(); ctx.fillStyle = "#14212d"; ctx.fill();
  line(L.TERRAIN, "#7b929e", 1.5);
  L.PADS.forEach((p, i) => {
    const selected = i === target;
    line([[p[0], p[2]], [p[1], p[2]]], selected ? "#90edd0" : "#678397", selected ? 4 : 2);
    ctx.font = "11px monospace"; ctx.fillStyle = selected ? "#90edd0" : "#6d8596";
    ctx.fillText(`0${i + 1} / ${p[3]}×`, px(p[0]), py(p[2]) + 22);
  });
  if (trail.length) line(trail.concat([[s.x, s.y]]), "#316459", 1.5);
  if (prediction && s.status === "flying") line(prediction.map(v => [v[0], v[1]]), "#efbf7d", 1.2, [4, 5]);
  const x = px(s.x), y = py(s.y);
  ctx.strokeStyle = "#244c46"; ctx.setLineDash([3, 7]); ctx.beginPath(); ctx.moveTo(x, y + 24); ctx.lineTo(x, py(pad[2])); ctx.stroke(); ctx.setLineDash([]);
  ctx.save(); ctx.translate(x, y); ctx.rotate(s.angle * Math.PI / 180);
  ctx.strokeStyle = s.status === "crashed" || s.status === "out_of_bounds" ? "#e7ac83" : "#d9fff0";
  ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(0, -10); ctx.lineTo(7, -4); ctx.lineTo(6, 4); ctx.lineTo(-6, 4); ctx.lineTo(-7, -4); ctx.closePath();
  ctx.moveTo(-5, 4); ctx.lineTo(-6, 8); ctx.lineTo(-8, 8); ctx.moveTo(5, 4); ctx.lineTo(6, 8); ctx.lineTo(8, 8); ctx.stroke();
  if (s.status === "flying" && s.fuel > 0 && command && command.throttle > 0) {
    ctx.strokeStyle = "#edbf7f"; ctx.beginPath(); ctx.moveTo(-3, 6); ctx.lineTo(0, 10 + 15 * command.throttle); ctx.lineTo(3, 6); ctx.stroke();
  }
  ctx.restore();
  if (placing) { ctx.fillStyle = "#90edd0"; ctx.font = "11px monospace"; ctx.fillText("START · click the sky to move", x + 14, y + 4); }
  ctx.fillStyle = "#829eab"; ctx.font = "10px monospace"; ctx.fillText(view === lab ? "LUNAR SURFACE / LIVE FLIGHT" : "LUNAR SURFACE / RECORDED FLIGHT", 24, H - 12);
}

function showAnswers(prefix, d) {
  const box = $(`${prefix}-probabilities`);
  box.replaceChildren();
  for (const [question, answer] of Object.entries((d && d.answers) || {})) {
    const title = document.createElement("div"); title.className = "prob-title"; title.textContent = question.toUpperCase(); box.append(title);
    for (const [label, p] of Object.entries(answer.probabilities)) {
      const row = document.createElement("div"); row.className = "prob-row" + (label === answer.choice ? " selected" : "");
      const name = document.createElement("span"); name.textContent = label;
      const bar = document.createElement("div"); bar.className = "prob-bar"; const fill = document.createElement("i"); fill.style.width = `${Math.max(0, Math.min(100, p * 100))}%`; bar.append(fill);
      const value = document.createElement("span"); value.textContent = `${(p * 100).toFixed(1)}%`;
      row.append(name, bar, value); box.append(row);
    }
  }
  $(`${prefix}-requested`).textContent = d ? word(d.requested) : "—";
  $(`${prefix}-proposed`).textContent = d && d.answers ? word(d.proposed) : "—";
  $(`${prefix}-executed`).textContent = d ? word(d.executed) : "—";
  $(`${prefix}-latency`).textContent = d ? `${d.latency_ms.toFixed(1)} ms` : "—";
  const shield = $(`${prefix}-shield`);
  shield.textContent = !d || !d.answers ? "" : d.intervened ? `SHIELD OVERRIDE · proposal cost ${d.proposal_cost.toFixed(1)} > plan ${d.plan_cost.toFixed(1)}` :
    d.proposed && d.requested && (d.proposed.turn !== d.requested.turn || d.proposed.throttle !== d.requested.throttle) ? `Laya deviated; rollout cost ${d.proposal_cost == null ? "not checked (raw mode)" : d.proposal_cost.toFixed(1) + " accepted"}` : "Laya matched the MPC request.";
  shield.className = "shield" + (d && d.intervened ? " warning" : "");
}

// ---------- landing lab ----------
function reset() {
  lab.s = { ...lab.start, fuel: 100, time: 0, status: "flying", score: 0 };
  lab.mpc = new L.AdaptiveMPC({ adaptive: $("adaptive").checked });
  lab.thrust = L.THRUST; lab.trail = []; lab.plan = null; lab.command = null; lab.decision = null; lab.playing = false; lab.acc = 0;
  $("log").textContent = ""; notice("");
  renderLab();
}
const log = line => { $("log").textContent += line + "\n"; };

function finish(s) {
  lab.playing = false;
  log(`T+${s.time.toFixed(1)} s  ${s.status.toUpperCase()}  vx ${s.vx.toFixed(2)}  vy ${s.vy.toFixed(2)}  tilt ${s.angle.toFixed(1)}°  estimate ${lab.thrust.toFixed(2)}`);
}

function decide() {
  const s = lab.s, before = { ...s };
  const result = lab.mpc.act(s, lab.target);
  const command = L.COMMANDS[result.plan[0]];
  L.step(s, command, before.time >= FAULT_AT ? lab.thrustScale : 1);
  lab.mpc.observe(before, command, s);
  lab.trail.push([before.x, before.y]);
  lab.plan = result; lab.command = command; lab.thrust = lab.mpc.model.thrust;
  if (Math.abs(before.time - FAULT_AT) < 1e-9 && lab.thrustScale < 1) log(`T+10.0 s  engine thrust ×${lab.thrustScale} (controller not told)`);
  if (s.status !== "flying") finish(s);
}

function applyFrames(frames) {
  for (const f of frames) {
    lab.trail.push([f.before.x, f.before.y]);
    lab.s = { ...f.after }; lab.decision = f.decision; lab.command = f.decision.executed; lab.thrust = f.decision.model.thrust;
    lab.plan = { prediction: f.decision.prediction, cost: f.decision.plan_cost, solveMs: f.decision.solve_ms };
    if (Math.abs(f.before.time - FAULT_AT) < 1e-9 && lab.thrustScale < 1) log(`T+10.0 s  engine thrust ×${lab.thrustScale} (controller not told)`);
    if (lab.s.status !== "flying") finish(lab.s);
  }
  if (!frames.length && lab.s.status === "flying") { lab.playing = false; notice("Server returned no frames; reset and launch again."); }
}

async function launch() {
  if (lab.s.status !== "flying") reset();
  if (lab.playing) { lab.playing = false; renderLab(); return; }
  if (lab.pilot !== "mpc" && !lab.trail.length) {
    try {
      lab.busy = true;
      await api("/reset", { mode: lab.pilot, start: lab.start, target: lab.target, thrust_scale: lab.thrustScale, fault_at: FAULT_AT, adaptive: $("adaptive").checked });
    } catch (error) { notice(`Live Laya unavailable: ${error.message}. Start it with: uv run lunar-mpc-laya-serve`); lab.busy = false; renderLab(); return; }
    lab.busy = false;
  }
  lab.playing = true; lab.acc = 0;
  renderLab();
}

function renderLab() {
  const s = lab.s, pad = L.PADS[lab.target];
  draw(lab, s, lab.trail, lab.plan && lab.plan.prediction, lab.target, lab.command, !lab.trail.length);
  $("altitude").textContent = `${Math.max(0, s.y - pad[2] - L.RADIUS).toFixed(1)} m`;
  $("vertical").textContent = `${s.vy.toFixed(2)} m/s`;
  $("horizontal").textContent = `${s.vx.toFixed(2)} m/s`;
  $("tilt").textContent = `${s.angle.toFixed(1)}°`;
  $("fuel").textContent = s.fuel.toFixed(1);
  $("estimate").textContent = `${(100 * lab.thrust / L.THRUST).toFixed(0)}%`;
  $("time").textContent = `T+ ${s.time.toFixed(1)} s`;
  const fault = lab.thrustScale < 1 && s.time >= FAULT_AT;
  $("status").textContent = s.status === "flying" ? (lab.trail.length ? (lab.playing ? "FLYING" : "PAUSED") : "READY · CLICK THE SKY TO PLACE THE LANDER") : s.status.toUpperCase().replace("_", " ");
  $("status").className = s.status === "crashed" || s.status === "out_of_bounds" ? "warning" : "";
  $("badge").textContent = s.status !== "flying" ? `● \u00a0 ${s.status.toUpperCase().replace("_", " ")}` : fault ? `● \u00a0 FAULT ACTIVE · THRUST ×${lab.thrustScale}` : lab.pilot === "mpc" ? "● \u00a0 BROWSER MPC" : "● \u00a0 LIVE LAYA / MLX";
  $("solve").textContent = lab.plan ? `PLAN ${lab.plan.solveMs.toFixed(1)} ms` : "PLAN —";
  $("command").textContent = word(lab.command);
  $("cost").textContent = lab.plan ? lab.plan.cost.toFixed(1) : "—";
  $("outcome").textContent = s.status === "flying" ? "—" : s.status.replace("_", " ");
  $("score").textContent = s.status === "flying" ? "—" : String(s.score);
  $("position").textContent = `x ${lab.start.x.toFixed(0)} · y ${lab.start.y.toFixed(0)}`;
  $("launch").textContent = s.status !== "flying" ? "Fly again" : lab.playing ? "Pause" : lab.trail.length ? "Resume" : "Launch";
  $("lab-laya").hidden = !(lab.decision && lab.decision.answers);
  if (!$("lab-laya").hidden) showAnswers("lab", lab.decision);
}

function place(event) {
  const c = lab.canvas, rect = c.getBoundingClientRect();
  const cx = (event.clientX - rect.left) * c.width / rect.width, cy = (event.clientY - rect.top) * c.height / rect.height;
  const k = c.width / 1100, H = c.height;
  const x = Math.min(980, Math.max(20, cx / k - 50));
  lab.start.x = x; lab.start.y = Math.min(740, Math.max(L.surface(x) + 30, (H - 38 * k - cy) * 750 / (H - 58 * k)));
  reset();
}
function slider(id, digits) {
  $(id).oninput = () => { lab.start[id] = Number($(id).value); $(`${id}-label`).textContent = Number($(id).value).toFixed(digits); reset(); };
}
lab.canvas.onclick = place;
slider("vx", 1); slider("vy", 1); slider("angle", 0);
$("random").onclick = () => {
  const pad = L.PADS[lab.target], r = (lo, hi) => lo + Math.random() * (hi - lo);
  Object.assign(lab.start, { x: (pad[0] + pad[1]) / 2 + r(-100, 100), y: 450, vx: Math.round(r(-8, 8) * 2) / 2, vy: -8, angle: Math.round(r(-12, 12)) });
  for (const [id, digits] of [["vx", 1], ["vy", 1], ["angle", 0]]) { $(id).value = String(lab.start[id]); $(`${id}-label`).textContent = lab.start[id].toFixed(digits); }
  reset();
};
for (const target of [0, 1, 2]) $(`pad-${target}`).onclick = () => {
  lab.target = target;
  for (const t of [0, 1, 2]) $(`pad-${t}`).classList.toggle("active", t === target);
  reset();
};
$("fault").onchange = () => { lab.thrustScale = Number($("fault").value); reset(); };
$("pilot").onchange = () => { lab.pilot = $("pilot").value; reset(); };
$("adaptive").onchange = reset;
$("reset").onclick = reset;
$("launch").onclick = launch;

// ---------- recorded flights ----------
async function loadManifest() {
  rec.loaded = true;
  try {
    const manifest = await api("manifest.json");
    rec.runs = manifest.runs || [];
    if (!rec.runs.length) throw new Error("manifest lists no runs");
  } catch (error) {
    $("rec-status").textContent = "NO RECORDINGS IN THIS BUILD";
    notice(`Recorded flights are not built (${error.message}). Run: python3 web/build.py dist/eval/*laya*.json dist/eval/mpc-assisted*.json`);
    return;
  }
  $("run").replaceChildren(...rec.runs.map(run => { const o = document.createElement("option"); o.value = String(run.id); o.textContent = run.label; return o; }));
  $("rec-table").replaceChildren(...rec.runs.map(run => { const row = document.createElement("div"); row.textContent = `${run.label}   ${run.landed}/${run.flights} landed`; return row; }));
  await loadRun(rec.runs[0]);
}
async function loadRun(run) {
  try { rec.run = await api(run.file); } catch (error) { notice(`Cannot load ${run.file}: ${error.message}`); return; }
  $("flight").replaceChildren(...rec.run.episodes.map((e, i) => { const o = document.createElement("option"); o.value = String(i); o.textContent = `seed ${e.summary.seed} · pad 0${e.summary.target + 1} · ${e.summary.status}`; return o; }));
  showFlight(0);
}
function showFlight(index) {
  rec.episode = rec.run.episodes[index]; rec.frame = 0; rec.playing = false; rec.acc = 0;
  $("rec-timeline").max = String(rec.episode.frames.length); $("rec-timeline").disabled = false;
  $("rec-play").disabled = false; $("rec-restart").disabled = false;
  renderRec();
}
function renderRec() {
  const e = rec.episode;
  if (!e) return;
  const frames = e.frames, end = rec.frame >= frames.length, i = Math.min(rec.frame, frames.length - 1);
  const s = end ? e.final : frames[i].before, d = frames[i].decision;
  draw(rec, s, frames.slice(0, i).map(f => [f.before.x, f.before.y]), null, e.summary.target, d.executed, false);
  $("rec-vertical").textContent = `${s.vy.toFixed(2)} m/s`;
  $("rec-horizontal").textContent = `${s.vx.toFixed(2)} m/s`;
  $("rec-tilt").textContent = `${s.angle.toFixed(1)}°`;
  $("rec-estimate").textContent = d.model ? `${(100 * d.model.thrust / L.THRUST).toFixed(0)}%` : "—";
  $("rec-time").textContent = `T+ ${s.time.toFixed(1)} s`;
  $("rec-frame").textContent = `DECISION ${Math.min(rec.frame + 1, frames.length)} / ${frames.length}`;
  $("rec-status").textContent = end ? s.status.toUpperCase().replace("_", " ") : rec.playing ? "PLAYING" : "PAUSED";
  $("rec-status").className = end && s.status !== "landed" ? "warning" : "";
  $("rec-timeline").value = String(rec.frame);
  $("rec-play").textContent = rec.playing ? "Pause" : "Play";
  showAnswers("rec", d);
  $("rec-outcome").textContent = e.summary.status;
  $("rec-agreements").textContent = `${e.summary.agreements} / ${e.summary.decisions}`;
  $("rec-interventions").textContent = String(e.summary.interventions);
  $("rec-decisions").textContent = String(e.summary.decisions);
  $("rec-fuel").textContent = e.summary.fuel.toFixed(1);
}
$("run").onchange = () => loadRun(rec.runs[Number($("run").value)]);
$("flight").onchange = () => showFlight(Number($("flight").value));
$("rec-timeline").oninput = () => { rec.frame = Number($("rec-timeline").value); rec.playing = false; renderRec(); };
$("rec-play").onclick = () => { if (rec.frame >= rec.episode.frames.length) rec.frame = 0; rec.playing = !rec.playing; rec.acc = 0; renderRec(); };
$("rec-restart").onclick = () => { rec.frame = 0; rec.playing = false; renderRec(); };

// ---------- loop, routing, startup ----------
function tick(now) {
  const dt = last === null ? 0 : Math.max(0, now - last) / 1000;
  last = now;
  if (lab.playing) {
    lab.acc += dt * Number($("speed").value);
    if (lab.pilot === "mpc") {
      for (let n = 0; n < 40 && lab.playing && lab.acc >= 0.2; n++) { decide(); lab.acc -= 0.2; }
    } else if (!lab.busy && lab.acc >= 0.2) {
      const n = Math.min(20, Math.floor(lab.acc / 0.2));
      lab.acc -= n * 0.2; lab.busy = true;
      api("/step", { n }).then(data => { applyFrames(data.frames); renderLab(); })
        .catch(error => { lab.playing = false; notice(`Live flight stopped: ${error.message}`); renderLab(); })
        .finally(() => { lab.busy = false; });
    }
    renderLab();
  }
  if (rec.playing && rec.episode) {
    rec.acc += dt * Number($("rec-speed").value);
    const steps = Math.floor(rec.acc / 0.2);
    if (steps > 0) { rec.acc -= steps * 0.2; rec.frame = Math.min(rec.frame + steps, rec.episode.frames.length); if (rec.frame >= rec.episode.frames.length) rec.playing = false; renderRec(); }
  }
  requestAnimationFrame(tick);
}
function route() {
  const page = location.hash === "#recorded" ? "recorded" : "lab";
  $("lab-page").hidden = page !== "lab"; $("recorded-page").hidden = page !== "recorded";
  $("page-title").textContent = page === "lab" ? "Landing lab" : "Recorded Laya flights";
  for (const a of document.querySelectorAll("nav a")) a.classList.toggle("active", a.dataset.page === page);
  if (page === "recorded" && !rec.loaded) loadManifest();
}
window.addEventListener("hashchange", route);
window.addEventListener("keydown", event => { if (event.code === "Space" && event.target === document.body) { event.preventDefault(); location.hash === "#recorded" ? $("rec-play").onclick() : launch(); } });
api("/status").then(status => {
  for (const option of $("pilot").options) option.disabled = !status.modes.includes(option.value);
  const model = status.model && status.model.model;
  $("pilot-note").textContent = status.modes.length > 1 ? `Live server ready · model ${model || "loaded"}` : "Live server running without a model: browser MPC and server MPC only.";
}).catch(() => { $("pilot-note").textContent = "Live Laya needs the local server: uv run lunar-mpc-laya-serve"; });
reset();
route();
requestAnimationFrame(tick);
