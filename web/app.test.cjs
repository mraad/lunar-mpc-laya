// Dependency-free checks: JS controller matches Python-recorded decisions; the page flies, lands,
// switches pilots through the live server API, and replays recorded flights.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const L = require("./lander.js");

for (const c of require("./parity.json")) {
  const s = { ...c.start }, mpc = new L.AdaptiveMPC();
  let previous = null;
  for (const expected of c.commands) {
    const plan = mpc.act(s, c.target, previous).plan;
    previous = plan[0];
    assert.equal(plan[0], expected, `seed ${c.seed} pad ${c.target}: JS command differs from Python at t=${s.time}`);
    const before = { ...s }, command = L.COMMANDS[expected];
    L.step(s, command, before.time >= 10 ? c.thrust_scale : 1);
    mpc.observe(before, command, s);
  }
  assert.equal(s.status, c.final.status);
  assert.ok(Math.abs(s.time - c.final.time) < 1e-9 && Math.abs(s.x - c.final.x) < 1e-6);
  assert.ok(Math.abs(mpc.model.thrust - c.thrust_estimate) < 1e-9);
}
console.log("parity: every Python-recorded command reproduced by the JS port");

async function main() {
  const elements = new Map(), listeners = {}, calls = [];
  const ctx = new Proxy({}, { get: () => () => {} });
  function element(id) {
    return { id, value: id === "speed" || id === "rec-speed" ? "16" : id === "fault" ? "1" : "0", checked: true, textContent: "", className: "",
      hidden: false, disabled: false, children: [], options: [], dataset: {}, style: {}, classList: { toggle() {} }, width: 1100, height: 640,
      append(...items) { this.children.push(...items); }, replaceChildren(...items) { this.children = items; if (id === "pilot") this.options = items; },
      getContext: () => ctx, getBoundingClientRect: () => ({ left: 0, top: 0, width: 1100, height: 640 }) };
  }
  const pilotOptions = ["mpc", "mpc-laya", "mpc-assisted"].map(value => ({ value, disabled: true }));
  // Server stub: /status advertises Laya; /reset and /step run the JS controller as a stand-in for the Python pilot.
  let live = null;
  const decision = f => ({ ...f, answers: { rotation: { choice: "hold", probabilities: { left: 0.01, hold: 0.98, right: 0.01 } }, engine: { choice: "half", probabilities: { off: 0.1, half: 0.8, full: 0.1 } } } });
  const record = { label: "mpc-assisted · thrust ×0.4 at 10 s", pilot: "mpc-assisted", episodes: [{ summary: { seed: 3000, target: 1, status: "landed", agreements: 2, decisions: 2, interventions: 0, fuel: 90 },
    frames: [{ before: { x: 500, y: 40, vx: 0, vy: -1, angle: 0, fuel: 90, time: 0, status: "flying" }, decision: decision({ requested: { turn: 0, throttle: 0.5 }, proposed: { turn: 0, throttle: 0.5 }, executed: { turn: 0, throttle: 0.5 }, intervened: false, model: { thrust: 5 }, latency_ms: 14, plan_cost: 3, proposal_cost: null }) },
            { before: { x: 500, y: 30, vx: 0, vy: -1, angle: 0, fuel: 89, time: 0.2, status: "flying" }, decision: decision({ requested: { turn: 0, throttle: 0.5 }, proposed: { turn: 1, throttle: 1 }, executed: { turn: 0, throttle: 0.5 }, intervened: true, model: { thrust: 5 }, latency_ms: 14, plan_cost: 3, proposal_cost: 40 }) }],
    final: { x: 500, y: 28, vx: 0, vy: -1, angle: 0, fuel: 88, time: 0.4, status: "landed", score: 50 } }] };
  async function fetch(url, init) {
    calls.push(url);
    const body = init && init.body ? JSON.parse(init.body) : null;
    const ok = data => ({ ok: true, status: 200, json: async () => data });
    if (url === "/status") return ok({ live: true, modes: ["mpc", "mpc-laya", "mpc-assisted"], model: { model: "stub" } });
    if (url === "/reset") { live = { s: { ...body.start, fuel: 100, time: 0, status: "flying", score: 0 }, target: body.target, scale: body.thrust_scale, mpc: new L.AdaptiveMPC({ adaptive: body.adaptive }) }; return ok({ state: live.s }); }
    if (url === "/step") {
      const frames = [];
      for (let i = 0; i < body.n && live.s.status === "flying"; i++) {
        const before = { ...live.s }, r = live.mpc.act(live.s, live.target), command = L.COMMANDS[r.plan[0]];
        L.step(live.s, command, before.time >= 10 ? live.scale : 1); live.mpc.observe(before, command, live.s);
        frames.push({ before, after: { ...live.s }, decision: decision({ requested: command, proposed: command, executed: command, intervened: false, model: { thrust: live.mpc.model.thrust }, latency_ms: 14, plan_cost: r.cost, proposal_cost: null, prediction: r.prediction, solve_ms: r.solveMs }) });
      }
      return ok({ frames });
    }
    if (url === "manifest.json") return ok({ version: 1, runs: [{ id: 0, label: record.label, file: "runs/0.json", pilot: "mpc-assisted", flights: 1, landed: 1 }] });
    if (url === "runs/0.json") return ok(record);
    return { ok: false, status: 404, json: async () => ({}) };
  }
  const context = vm.createContext({ console, Lander: L, performance, Math, fetch, location: { hash: "#lab" },
    document: { body: {}, getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); }, createElement: element, querySelectorAll: () => [] },
    window: { addEventListener(name, fn) { listeners[name] = fn; } }, requestAnimationFrame() {} });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "app.js"), "utf8"), context);
  const $ = id => elements.get(id);
  $("pilot").options = pilotOptions;
  await new Promise(setImmediate);
  assert.equal(pilotOptions.filter(o => !o.disabled).length, 3, "status should enable live pilots");
  assert.equal($("status").textContent, "READY · CLICK THE SKY TO PLACE THE LANDER");

  // Browser MPC: place far left, pad 3, fly.
  $("canvas").onclick({ clientX: 250, clientY: 100 });
  assert.match($("position").textContent, /^x 200 · y 6\d\d$/);
  $("pad-2").onclick();
  const fly = async () => { let now = 0; for (let i = 0; i < 600 && vm.runInContext("lab.playing", context); i++) { now += 100; vm.runInContext(`tick(${now})`, context); await new Promise(setImmediate); } };
  await $("launch").onclick();
  assert.equal($("launch").textContent, "Pause");
  await fly();
  assert.equal($("status").textContent, "LANDED", $("log").textContent);
  assert.equal($("score").textContent, "200");
  assert.equal($("lab-laya").hidden, true);

  // Fixed model crashes at ×0.4, adaptive lands.
  $("fault").value = "0.4"; $("fault").onchange();
  $("adaptive").checked = false; $("adaptive").onchange();
  await $("launch").onclick(); await fly();
  assert.equal($("status").textContent, "CRASHED", "fixed model at ×0.4 should crash");
  $("adaptive").checked = true; $("adaptive").onchange();
  await $("launch").onclick(); await fly();
  assert.equal($("status").textContent, "LANDED", "adaptive model at ×0.4 should land: " + $("log").textContent);
  assert.equal($("estimate").textContent, "40%");

  // Live server pilot: decisions come from /reset + /step and the Laya card appears.
  $("pilot").value = "mpc-assisted"; $("pilot").onchange();
  await $("launch").onclick(); await fly();
  assert.ok(calls.includes("/reset") && calls.includes("/step"), calls.join(","));
  assert.equal($("status").textContent, "LANDED", $("log").textContent);
  assert.equal($("lab-laya").hidden, false);
  assert.equal($("lab-shield").textContent, "Laya matched the MPC request.");
  assert.match($("badge").textContent, /LANDED/);

  // Recorded tab: manifest, run, flight, scrub, play.
  context.location.hash = "#recorded"; listeners.hashchange();
  await new Promise(setImmediate); await new Promise(setImmediate); await new Promise(setImmediate);
  assert.equal($("lab-page").hidden, true);
  assert.equal($("run").children.length, 1);
  assert.equal($("flight").children[0].textContent, "seed 3000 · pad 02 · landed");
  assert.equal($("rec-frame").textContent, "DECISION 1 / 2");
  assert.equal($("rec-shield").textContent, "Laya matched the MPC request.");
  $("rec-timeline").value = "1"; $("rec-timeline").oninput();
  assert.match($("rec-shield").textContent, /SHIELD OVERRIDE/);
  assert.equal($("rec-shield").className, "shield warning");
  $("rec-play").onclick();
  vm.runInContext("tick(10000); tick(10100)", context);
  assert.equal($("rec-status").textContent, "LANDED");
  assert.equal($("rec-agreements").textContent, "2 / 2");
  console.log("page: browser MPC, fault toggle, live server pilot, recorded playback: passed");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
