/* BOOKER Fantasy: H2H 8-category league (FG%, FT%, 3PM, REB, AST, STL, BLK, PTS).
 *
 * FX (pure engine, unit-testable in node):
 *   totals(p)            season totals = per-game line x projected games
 *   zscores(P, opts)     per-category z on season totals over the draftable pool; FG%/FT% use
 *                        volume-weighted impact (makes - league% x attempts)
 *   auction(P, z, ...)   dollars = 1 + (money left - open slots) x (z - z_repl)+ / sum over the
 *                        top `open slots` players -- recomputed on the undrafted pool after every
 *                        pick, which is what makes the values dynamic (inflation built in)
 *   weeks(games)         Monday-Sunday fantasy weeks from the real schedule, team games per week
 *   weekLine / matchup   expected weekly category lines and P(win) per category (normal approx
 *                        with per-game overdispersion); simulate() draws from the same model
 * UI: Setup -> Draft (auction or snake) -> Season (matchups, standings, rosters, free agents,
 * trades). State lives in localStorage; export/import as JSON.
 */
(function () {
  "use strict";
  const CATS = ["fg", "ft", "tpm", "reb", "ast", "stl", "blk", "pts"];
  const CAT_LABEL = { fg: "FG%", ft: "FT%", tpm: "3PM", reb: "REB", ast: "AST", stl: "STL", blk: "BLK", pts: "PTS" };
  // per-game variance / mean (overdispersion vs Poisson) for counting cats; pct cats use binomial x OD
  const DISP = { tpm: 1.4, reb: 2.4, ast: 2.2, stl: 1.1, blk: 1.3, pts: 7.0 };
  const PCT_OD = 1.3;

  /* ------------------------------------------------------------------ engine */
  const FX = {
    CATS, CAT_LABEL,
    totals(p) {
      const g = p.g;
      return { g, fgm: p.fgm * g, fga: p.fga * g, ftm: p.ftm * g, fta: p.fta * g, tpm: p.tpm * g,
               reb: p.reb * g, ast: p.ast * g, stl: p.stl * g, blk: p.blk * g, pts: p.pts * g };
    },
    zscores(P, poolSize) {
      const T = P.map((p) => FX.totals(p));
      let pool = P.map((_, i) => i).sort((a, b) => T[b].pts - T[a].pts).slice(0, poolSize);
      let Z = null;
      for (let it = 0; it < 3; it++) {
        const sum = (f) => pool.reduce((s, i) => s + f(T[i]), 0);
        const lgFG = sum((t) => t.fgm) / Math.max(1, sum((t) => t.fga));
        const lgFT = sum((t) => t.ftm) / Math.max(1, sum((t) => t.fta));
        const raw = T.map((t) => ({ fg: t.fgm - lgFG * t.fga, ft: t.ftm - lgFT * t.fta, tpm: t.tpm, reb: t.reb,
                                    ast: t.ast, stl: t.stl, blk: t.blk, pts: t.pts }));
        const mu = {}, sd = {};
        CATS.forEach((c) => {
          const v = pool.map((i) => raw[i][c]);
          mu[c] = v.reduce((a, b) => a + b, 0) / v.length;
          sd[c] = Math.sqrt(v.reduce((a, b) => a + (b - mu[c]) ** 2, 0) / Math.max(1, v.length - 1)) || 1;
        });
        Z = raw.map((r) => { const z = {}; CATS.forEach((c) => (z[c] = (r[c] - mu[c]) / sd[c])); return z; });
        const tot = Z.map((z) => CATS.reduce((s, c) => s + z[c], 0));
        pool = P.map((_, i) => i).sort((a, b) => tot[b] - tot[a]).slice(0, poolSize);
      }
      return Z;
    },
    zsum(z, w) { return CATS.reduce((s, c) => s + (w ? w[c] : 1) * z[c], 0); },
    /* dollars for the AVAILABLE players: `avail` = indices, `slots` open roster spots league-wide,
       `money` league-wide dollars left. Returns {i: dollars}. */
    auction(Z, avail, slots, money, w) {
      const out = {};
      if (!avail.length || slots <= 0) { avail.forEach((i) => (out[i] = 0)); return out; }
      const sc = avail.map((i) => [i, FX.zsum(Z[i], w)]).sort((a, b) => b[1] - a[1]);
      const k = Math.min(slots, sc.length);
      const repl = sc[Math.min(k, sc.length - 1)][1];
      let tot = 0;
      for (let j = 0; j < k; j++) tot += Math.max(0, sc[j][1] - repl);
      const pot = Math.max(0, money - slots);
      sc.forEach(([i, s], j) => {
        const v = Math.max(0, s - repl);
        out[i] = (j < k ? 1 : 0) + (tot > 0 ? pot * v / tot : 0);
      });
      return out;
    },
    weeks(games, season) {
      const g = games.filter((x) => x.season === season && x.type !== "P");
      if (!g.length) return [];
      const d0 = new Date(g.reduce((m, x) => (x.date < m ? x.date : m), g[0].date) + "T12:00:00Z");
      const mon = new Date(d0); mon.setUTCDate(d0.getUTCDate() - ((d0.getUTCDay() + 6) % 7));
      const W = [];
      g.forEach((x) => {
        const d = new Date(x.date + "T12:00:00Z");
        const k = Math.floor((d - mon) / (7 * 864e5));
        if (!W[k]) {
          const s = new Date(mon); s.setUTCDate(mon.getUTCDate() + 7 * k);
          const e = new Date(s); e.setUTCDate(s.getUTCDate() + 6);
          W[k] = { start: s.toISOString().slice(0, 10), end: e.toISOString().slice(0, 10), games: {}, dates: {} };
        }
        [x.home, x.away].forEach((t) => { W[k].games[t] = (W[k].games[t] || 0) + 1; (W[k].dates[t] = W[k].dates[t] || []).push(x.date); });
      });
      return W.filter(Boolean).map((w, i) => ({ ...w, n: i + 1 }));
    },
    /* Weekly team line. opts.actual = {pid: [[date, min, fgm, fga, ftm, fta, tpm, reb, ast, stl, blk, pts]]},
       opts.last = last date with real results: games on/before it count as REAL (fixed, no
       variance); later games in the week are PROJECTED (players listed out in opts.out get 0). */
    weekLine(players, week, opts) {
      const o = opts || {};
      const L = { fgm: 0, fga: 0, ftm: 0, fta: 0, tpm: 0, reb: 0, ast: 0, stl: 0, blk: 0, pts: 0, gp: 0,
                  rgp: 0, pfga: 0, pfta: 0, v: { tpm: 0, reb: 0, ast: 0, stl: 0, blk: 0, pts: 0 } };
      const K = ["fgm", "fga", "ftm", "fta", "tpm", "reb", "ast", "stl", "blk", "pts"];
      players.forEach((p) => {
        let rem = week.games[p.t] || 0;
        if (o.last && week.dates) {
          rem = (week.dates[p.t] || []).filter((d) => d > o.last).length;
          (o.actual && o.actual[p.pid] || []).forEach((r) => {
            if (r[0] < week.start || r[0] > week.end || r[0] > o.last) return;
            K.forEach((k, j) => (L[k] += r[j + 2])); L.gp += 1; L.rgp += 1;
          });
        }
        const gp = (o.out && o.out[p.pid]) ? 0 : rem * Math.min(1, p.g / 82);
        L.gp += gp;
        K.forEach((k) => (L[k] += p[k] * gp));
        L.pfga += p.fga * gp; L.pfta += p.fta * gp;
        Object.keys(DISP).forEach((k) => (L.v[k] += DISP[k] * p[k] * gp));
      });
      return L;
    },
    catDist(L, c) {          // mean, sd of a team's weekly category value
      if (c === "fg" || c === "ft") {
        const m = c === "fg" ? L.fgm : L.ftm, a = c === "fg" ? L.fga : L.fta;
        const p = a > 0 ? m / a : 0;
        const pa = L.pfga == null ? a : (c === "fg" ? L.pfga : L.pfta);     // only projected shots are uncertain
        return [p, a > 0 ? Math.sqrt(PCT_OD * p * (1 - p) * Math.max(pa, 0) / (a * a)) + 1e-6 : 0.05];
      }
      return [L[c], Math.sqrt(Math.max(L.v[c], 1e-6))];
    },
    ncdf(x) { const t = 1 / (1 + 0.2316419 * Math.abs(x)); const d = 0.3989423 * Math.exp(-x * x / 2);
              const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
              return x > 0 ? 1 - p : p; },
    matchup(LA, LB) {
      const r = {};
      CATS.forEach((c) => {
        const [ma, sa] = FX.catDist(LA, c), [mb, sb] = FX.catDist(LB, c);
        const s = Math.sqrt(sa * sa + sb * sb) || 1e-6;
        r[c] = { a: ma, b: mb, pA: FX.ncdf((ma - mb) / s) };
      });
      return r;
    },
    gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng();
                 return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); },
    simulate(LA, LB, rng) {
      const res = { a: {}, b: {}, wA: 0, wB: 0, t: 0 };
      CATS.forEach((c) => {
        const [ma, sa] = FX.catDist(LA, c), [mb, sb] = FX.catDist(LB, c);
        const dec = c === "fg" || c === "ft" ? 3 : 0;
        const xa = +Math.max(0, ma + sa * FX.gauss(rng)).toFixed(dec), xb = +Math.max(0, mb + sb * FX.gauss(rng)).toFixed(dec);
        res.a[c] = xa; res.b[c] = xb;
        if (xa > xb) res.wA++; else if (xb > xa) res.wB++; else res.t++;
      });
      return res;
    },
    roundRobin(n, nWeeks) {      // circle method; odd n -> bye (-1)
      const ids = Array.from({ length: n }, (_, i) => i);
      if (n % 2) ids.push(-1);
      const m = ids.length, rounds = [];
      let arr = ids.slice();
      for (let r = 0; r < m - 1; r++) {
        const pairs = [];
        for (let i = 0; i < m / 2; i++) {
          const a = arr[i], b = arr[m - 1 - i];
          if (a !== -1 && b !== -1) pairs.push(r % 2 ? [b, a] : [a, b]);
        }
        rounds.push(pairs);
        arr = [arr[0], arr[m - 1], ...arr.slice(1, m - 1)];
      }
      return Array.from({ length: nWeeks }, (_, w) => rounds[w % rounds.length]);
    },
    mulberry(seed) { return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; },
  };
  if (typeof module !== "undefined") module.exports = FX;
  if (typeof document === "undefined") return;     // node: engine only
  window.FX = FX;

  /* ------------------------------------------------------------------ state */
  const KEY = "booker_fantasy_v1";
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  let S = null, P = [], Z = [], byPid = {}, WEEKS = [], ACT = {}, LAST = null, INJ = {};

  function defaults() {
    return { v: 1, phase: "setup", teams: 10, budget: 200, roster: 13, mode: "auction", my: 0,
             names: Array.from({ length: 12 }, (_, i) => (i === 0 ? "My Team" : `Team ${i + 1}`)),
             punt: [], picks: [], rosters: null, schedule: null, results: {}, week: 1,
             playoffTeams: 4, playoffWeeks: 3, log: [], seed: 7 };
  }
  function load() { try { const s = JSON.parse(localStorage.getItem(KEY)); return s && s.v === 1 ? s : defaults(); } catch (e) { return defaults(); } }
  function save() { localStorage.setItem(KEY, JSON.stringify(S)); }

  function init() {
    const D = window.BOOKER;
    if (!D || !D.fantasy) return false;
    P = D.fantasy.players.slice();
    P.forEach((p, i) => (byPid[p.pid] = i));
    WEEKS = FX.weeks(D.games || [], D.fantasy.season);
    ACT = D.fantasy.actual || {}; LAST = D.fantasy.lastActual || null; INJ = D.fantasy.injuries || {};
    S = load();
    recomputeZ();
    return true;
  }
  function recomputeZ() { Z = FX.zscores(P, S.teams * S.roster); }

  /* ------------------------------------------------------------------ derived */
  const nT = () => S.teams;
  function rosters() {
    if (S.rosters) return S.rosters;
    const R = Array.from({ length: nT() }, () => []);
    S.picks.forEach((k) => R[k.team].push(k.pid));
    return R;
  }
  function spent(t) { return S.picks.filter((k) => k.team === t).reduce((s, k) => s + (k.price || 0), 0); }
  function owned() { const o = {}; rosters().forEach((r, t) => r.forEach((pid) => (o[pid] = t))); return o; }
  function weights(team) {
    const w = {}; CATS.forEach((c) => (w[c] = S.punt.includes(c) ? 0 : 1));
    if (team == null) return w;
    const R = rosters()[team];
    if (!R.length) return w;
    // category need: compare this team's per-player z in each category to its overall per-player z;
    // weak categories get up to +40%, strong ones down to -40% (punts stay 0)
    const zs = {}; CATS.forEach((c) => (zs[c] = R.reduce((s, pid) => s + Z[byPid[pid]][c], 0) / R.length));
    const avg = CATS.filter((c) => w[c] > 0).reduce((s, c) => s + zs[c], 0) / Math.max(1, CATS.filter((c) => w[c] > 0).length);
    CATS.forEach((c) => { if (w[c] > 0) w[c] = Math.min(1.4, Math.max(0.6, 1 - 0.35 * (zs[c] - avg))); });
    return w;
  }
  function values() {
    const own = owned();
    const avail = P.map((_, i) => i).filter((i) => own[P[i].pid] == null);
    const slots = nT() * S.roster - S.picks.length;
    const money = nT() * S.budget - S.picks.reduce((s, k) => s + (k.price || 0), 0);
    const base = FX.auction(Z, P.map((_, i) => i), nT() * S.roster, nT() * S.budget, weights(null));
    const live = FX.auction(Z, avail, slots, money, weights(null));
    const fit = FX.auction(Z, avail, slots, money, weights(S.my));
    return { base, live, fit, avail, slots, money };
  }
  function maxBid(t) {
    const open = S.roster - rosters()[t].length;
    return open <= 0 ? 0 : S.budget - spent(t) - (open - 1);
  }
  function snakeTeam() {
    const k = S.picks.length, r = Math.floor(k / nT()), i = k % nT();
    return r % 2 ? nT() - 1 - i : i;
  }

  /* ------------------------------------------------------------------ render */
  const fmt = (v, d = 1) => (v == null || isNaN(v) ? "-" : (+v).toFixed(d));
  const zcell = (z) => `<td class="${z > 0.5 ? "pos" : z < -0.5 ? "neg" : ""}">${fmt(z, 1)}</td>`;
  let board = { sort: "live", q: "", hideTaken: true, nominee: null };
  let tradeSel = { a: 0, b: 1, A: [], B: [] };

  function render() {
    const root = $("#fx-root");
    if (!root) return;
    if (!P.length && !init()) { root.innerHTML = `<p class="note">Fantasy projections are not available in this build.</p>`; return; }
    const tabs = [["setup", "League setup"], ["draft", "Draft"], ["season", "Season"], ["teams", "Rosters & moves"], ["data", "Save / load"]];
    const cur = S.view || (S.phase === "setup" ? "setup" : S.phase === "draft" ? "draft" : "season");
    root.innerHTML = `<div class="controls fx-nav">${tabs.map(([k, t]) =>
      `<button class="link-btn${k === cur ? " on" : ""}" data-fx-view="${k}">${t}</button>`).join(" | ")}</div><div id="fx-body"></div>`;
    ({ setup: renderSetup, draft: renderDraft, season: renderSeason, teams: renderTeams, data: renderData })[cur]();
  }

  function renderSetup() {
    const locked = S.picks.length > 0;
    $("#fx-body").innerHTML = `
      <p class="note">Head-to-head categories: ${CATS.map((c) => CAT_LABEL[c]).join(", ")}. Projections are BOOKER's
        2026-27 per-game lines (Marcel-weighted three-season rates, regressed by stat, aged; projected minutes and games
        from the minutes model). Rookies without NBA stats carry a position-average placeholder line (marked R).</p>
      <div class="controls">
        <label>Teams <select id="fx-teams" ${locked ? "disabled" : ""}>${[4, 5, 6, 7, 8, 9, 10, 11, 12].map((n) => `<option ${n === S.teams ? "selected" : ""}>${n}</option>`).join("")}</select></label>
        <label>Budget $ <input id="fx-budget" type="number" min="50" max="1000" value="${S.budget}" ${locked ? "disabled" : ""} style="width:70px"></label>
        <label>Roster size <input id="fx-roster" type="number" min="8" max="18" value="${S.roster}" ${locked ? "disabled" : ""} style="width:60px"></label>
        <label>Draft <select id="fx-mode" ${locked ? "disabled" : ""}><option value="auction" ${S.mode === "auction" ? "selected" : ""}>Auction</option><option value="snake" ${S.mode === "snake" ? "selected" : ""}>Snake</option></select></label>
        <label>Playoff teams <select id="fx-po">${[2, 4, 6, 8].filter((n) => n <= S.teams).map((n) => `<option ${n === S.playoffTeams ? "selected" : ""}>${n}</option>`).join("")}</select></label>
      </div>
      <h3 class="sub-head">Teams</h3>
      <div class="controls fx-names">${Array.from({ length: S.teams }, (_, i) =>
        `<label><input type="radio" name="fx-my" value="${i}" ${i === S.my ? "checked" : ""}> <input class="fx-name" data-i="${i}" value="${esc(S.names[i])}" style="width:130px"></label>`).join("")}</div>
      <p class="note muted">The selected radio button is your team (used for the "Fit $" column and your matchups).</p>
      <h3 class="sub-head">Punt categories (your valuations ignore them)</h3>
      <div class="controls">${CATS.map((c) => `<label><input type="checkbox" class="fx-punt" value="${c}" ${S.punt.includes(c) ? "checked" : ""}> ${CAT_LABEL[c]}</label>`).join("")}</div>
      <div class="controls"><button class="btn" id="fx-start">${locked ? "Back to draft" : "Start draft"}</button>
        ${locked ? "" : `<span class="muted">${nT() * S.roster} roster spots, $${nT() * S.budget} total.</span>`}</div>`;
    const set = (k, v) => { S[k] = v; if (k === "teams" || k === "roster") { S.playoffTeams = Math.min(S.playoffTeams, S.teams); recomputeZ(); } save(); render(); };
    if (!locked) {
      $("#fx-teams").onchange = (e) => set("teams", +e.target.value);
      $("#fx-budget").onchange = (e) => set("budget", Math.max(50, +e.target.value || 200));
      $("#fx-roster").onchange = (e) => set("roster", Math.min(18, Math.max(8, +e.target.value || 13)));
      $("#fx-mode").onchange = (e) => set("mode", e.target.value);
    }
    $("#fx-po").onchange = (e) => set("playoffTeams", +e.target.value);
    document.querySelectorAll(".fx-name").forEach((el) => (el.onchange = () => { S.names[+el.dataset.i] = el.value || `Team ${+el.dataset.i + 1}`; save(); }));
    document.querySelectorAll("input[name=fx-my]").forEach((el) => (el.onchange = () => { S.my = +el.value; save(); }));
    document.querySelectorAll(".fx-punt").forEach((el) => (el.onchange = () => {
      S.punt = Array.from(document.querySelectorAll(".fx-punt:checked")).map((x) => x.value); save(); }));
    $("#fx-start").onclick = () => { S.phase = S.phase === "setup" ? "draft" : S.phase; S.view = "draft"; save(); render(); };
  }

  function playerRow(i, V, own, extra) {
    const p = P[i], z = Z[i], t = own[p.pid];
    return `<tr class="${t != null ? "muted" : ""}" data-i="${i}">
      <td class="col-text"><b>${esc(p.n)}</b>${p.rk ? ' <span class="muted">R</span>' : ""}${injTag(p.pid)}<br><small class="muted">${p.t} ${esc(p.pos)} ${p.age || ""}${soFar(p.pid)}</small></td>
      <td>${fmt(p.g, 0)}</td><td>${fmt(p.mpg, 1)}</td>
      <td>${fmt(100 * p.fgm / Math.max(p.fga, 1e-6), 1)}</td><td>${fmt(100 * p.ftm / Math.max(p.fta, 1e-6), 1)}</td>
      <td>${fmt(p.tpm)}</td><td>${fmt(p.reb)}</td><td>${fmt(p.ast)}</td><td>${fmt(p.stl)}</td><td>${fmt(p.blk)}</td><td>${fmt(p.pts)}</td>
      ${CATS.map((c) => zcell(z[c])).join("")}
      <td><b>${fmt(FX.zsum(z, weights(null)), 2)}</b></td>
      <td>$${fmt(V.base[i], 0)}</td><td>${t != null ? "-" : "$" + fmt(V.live[i], 0)}</td><td>${t != null ? esc(S.names[t]) : "$" + fmt(V.fit[i], 0)}</td>${extra || ""}</tr>`;
  }
  function injTag(pid) {
    const v = INJ[pid]; if (!v) return "";
    const d = String(v[0]).toLowerCase(), k = d === "out" ? "OUT" : d === "doubtful" ? "D" : d === "questionable" ? "Q" : d.slice(0, 3).toUpperCase();
    return ` <span class="neg" title="${esc(v[1])} (${esc(v[2])})">${k}</span>`;
  }
  function soFar(pid) {
    const g = ACT[pid]; if (!g || !g.length) return "";
    const s = (j) => g.reduce((a, r) => a + r[j], 0) / g.length;
    return ` | 26-27: ${g.length} GP, ${s(11).toFixed(1)} pts ${s(7).toFixed(1)} reb ${s(8).toFixed(1)} ast`;
  }
  const HEAD = `<th class="col-text">Player</th><th>G</th><th>MPG</th><th>FG%</th><th>FT%</th><th>3PM</th><th>REB</th><th>AST</th><th>STL</th><th>BLK</th><th>PTS</th>` +
    CATS.map((c) => `<th>z${CAT_LABEL[c]}</th>`).join("") + `<th data-k="z">Z</th><th data-k="base">Base $</th><th data-k="live">Live $</th><th data-k="fit">Fit $</th>`;

  function renderDraft() {
    if (S.phase === "setup") { S.phase = "draft"; save(); }
    const V = values(), own = owned(), R = rosters();
    const done = S.picks.length >= nT() * S.roster;
    let idx = P.map((_, i) => i).filter((i) => (!board.hideTaken || own[P[i].pid] == null) &&
      (!board.q || P[i].n.toLowerCase().includes(board.q) || P[i].t.toLowerCase() === board.q));
    const key = { live: (i) => (own[P[i].pid] != null ? -1 : V.live[i]), fit: (i) => (own[P[i].pid] != null ? -1 : V.fit[i]),
                  base: (i) => V.base[i], z: (i) => FX.zsum(Z[i], weights(null)) }[board.sort] || ((i) => V.live[i]);
    idx.sort((a, b) => key(b) - key(a));
    idx = idx.slice(0, 250);
    const infl = (() => { const rem = V.avail.reduce((s, i) => s + V.base[i], 0); return rem > 0 ? (V.money / rem) : 1; })();
    const onClock = S.mode === "snake" && !done ? snakeTeam() : null;
    const nom = board.nominee != null ? P[board.nominee] : null;
    $("#fx-body").innerHTML = `
      <p class="note">${S.mode === "auction" ? `Live $ = value from the money and roster spots still left (inflation ${fmt(infl, 2)}x vs pre-draft).
        Fit $ = your team's value, with categories you already have covered weighted down and your weak ones up${S.punt.length ? `; punting ${S.punt.map((c) => CAT_LABEL[c]).join(", ")}` : ""}.
        Click a player to nominate, then record the winning team and price.`
        : `Snake draft. Rankings use Live $ (scarcity-adjusted z); Fit $ weights your category needs. Click a player to draft him to the team on the clock.`}</p>
      <div class="fx-teamstrip">${R.map((r, t) => `<div class="fx-team${t === S.my ? " me" : ""}${t === onClock ? " clock" : ""}"><b>${esc(S.names[t])}</b>
        <small>${S.mode === "auction" ? `$${S.budget - spent(t)} left, max bid $${maxBid(t)}` : ""} ${r.length}/${S.roster}</small></div>`).join("")}</div>
      ${done ? `<div class="controls"><b>Draft complete.</b> <button class="btn" id="fx-to-season">Start season</button></div>` : `
      <div class="controls fx-nom">${nom ? `<b>${esc(nom.n)}</b> <span class="muted">${nom.t}</span>
        ${S.mode === "auction" ? `<label>Winner <select id="fx-win">${R.map((r, t) => `<option value="${t}" ${maxBid(t) <= 0 ? "disabled" : ""}>${esc(S.names[t])} (max $${maxBid(t)})</option>`).join("")}</select></label>
          <label>Price $ <input id="fx-price" type="number" min="1" value="${Math.max(1, Math.round(V.live[board.nominee]))}" style="width:60px"></label>
          <button class="btn" id="fx-sold">Sold</button>`
          : `<button class="btn" id="fx-sold">Draft to ${esc(S.names[onClock])}</button>`}
        <button class="link-btn" id="fx-cancel">cancel</button>` : `<span class="muted">${S.mode === "snake" ? `On the clock: <b>${esc(S.names[onClock])}</b> (pick ${S.picks.length + 1}). ` : ""}Select a player below.</span>`}
        <button class="link-btn" id="fx-undo" ${S.picks.length ? "" : "disabled"}>undo last pick</button></div>`}
      <div class="controls"><label>Search <input id="fx-q" value="${esc(board.q)}" placeholder="name or team" style="width:160px"></label>
        <label><input type="checkbox" id="fx-hide" ${board.hideTaken ? "checked" : ""}> hide drafted</label>
        <span class="muted">sort: click Z / Base $ / Live $ / Fit $</span></div>
      <div class="table-scroll"><table class="grid fx-board"><thead><tr>${HEAD}</tr></thead>
        <tbody>${idx.map((i) => playerRow(i, V, own)).join("")}</tbody></table></div>`;
    const go = () => renderDraft();
    $("#fx-q").oninput = (e) => { board.q = e.target.value.trim().toLowerCase(); clearTimeout(go.t); go.t = setTimeout(() => { go(); const q = $("#fx-q"); q.focus(); q.setSelectionRange(q.value.length, q.value.length); }, 250); };
    $("#fx-hide").onchange = (e) => { board.hideTaken = e.target.checked; go(); };
    document.querySelectorAll(".fx-board th[data-k]").forEach((th) => (th.onclick = () => { board.sort = th.dataset.k; go(); }));
    document.querySelectorAll(".fx-board tbody tr").forEach((tr) => (tr.onclick = () => {
      const i = +tr.dataset.i; if (own[P[i].pid] != null || done) return; board.nominee = i; go(); window.scrollTo({ top: $("#fx-body").offsetTop - 60 }); }));
    if ($("#fx-cancel")) $("#fx-cancel").onclick = () => { board.nominee = null; go(); };
    if ($("#fx-undo")) $("#fx-undo").onclick = () => { S.picks.pop(); S.rosters = null; save(); go(); };
    if ($("#fx-to-season")) $("#fx-to-season").onclick = startSeason;
    if ($("#fx-sold")) $("#fx-sold").onclick = () => {
      const i = board.nominee;
      if (S.mode === "auction") {
        const t = +$("#fx-win").value, price = Math.max(1, Math.round(+$("#fx-price").value || 1));
        if (price > maxBid(t)) { alert(`${S.names[t]} can bid at most $${maxBid(t)}.`); return; }
        S.picks.push({ pid: P[i].pid, team: t, price });
      } else {
        S.picks.push({ pid: P[i].pid, team: onClock, price: 0 });
      }
      board.nominee = null; S.rosters = null; save(); go();
    };
  }

  /* ------------------------------------------------------------------ season */
  function startSeason() {
    S.rosters = rosters().map((r) => r.slice());
    // the league starts in the current NBA week (weeks already finished before the draft are not played)
    const today = new Date().toISOString().slice(0, 10);
    const cur = WEEKS.findIndex((w) => w.end >= today);
    S.startWeek = cur < 0 ? 1 : cur + 1;
    const regWeeks = Math.max(1, WEEKS.length - S.playoffWeeks - S.startWeek + 1);
    S.schedule = FX.roundRobin(nT(), regWeeks);
    S.results = {}; S.week = 1; S.phase = "season"; S.view = "season";
    save(); render();
  }
  const teamPlayers = (t) => S.rosters[t].map((pid) => P[byPid[pid]]).filter(Boolean);
  const WK = (k) => WEEKS[(S.startWeek || 1) - 1 + k - 1];          // league week k -> NBA week
  function wkState(w) {
    if (!w || !LAST) return "future";
    if (w.end <= LAST) return "final";
    return w.start <= LAST ? "live" : "future";
  }
  function outMap(w) {        // listed OUT: no games for the rest of this week and next week
    const o = {};
    if (!w || !LAST) return o;
    const horizon = new Date(LAST + "T12:00:00Z"); horizon.setUTCDate(horizon.getUTCDate() + 7);
    if (w.start > horizon.toISOString().slice(0, 10)) return o;
    Object.entries(INJ).forEach(([pid, v]) => { if (String(v[0]).toLowerCase() === "out") o[+pid] = true; });
    return o;
  }
  const line = (t, w) => FX.weekLine(teamPlayers(t), w, { actual: ACT, last: LAST, out: outMap(w) });
  function realResult(LA, LB) {             // a finished week: compare the real totals
    const r = { a: {}, b: {}, wA: 0, wB: 0, t: 0, real: true };
    CATS.forEach((c) => {
      const [xa] = FX.catDist(LA, c), [xb] = FX.catDist(LB, c);
      const d = c === "fg" || c === "ft" ? 3 : 0;
      r.a[c] = +xa.toFixed(d); r.b[c] = +xb.toFixed(d);
      if (r.a[c] > r.b[c]) r.wA++; else if (r.b[c] > r.a[c]) r.wB++; else r.t++;
    });
    return r;
  }
  function finalizeReal() {                 // score finished weeks from real box scores, once
    let changed = false;
    (S.schedule || []).forEach((pairs, j) => {
      const k = j + 1, w = WK(k);
      if (wkState(w) !== "final") return;
      if (S.results[k] && S.results[k].every((g) => g.r.real)) return;
      S.results[k] = pairs.map(([A, B]) => ({ A, B, r: realResult(line(A, w), line(B, w)) }));
      changed = true;
    });
    if (changed) save();
  }
  function standings() {
    const st = Array.from({ length: nT() }, (_, t) => ({ t, w: 0, l: 0, d: 0, mw: 0 }));
    Object.values(S.results).forEach((games) => games.forEach((g) => {
      st[g.A].w += g.r.wA; st[g.A].l += g.r.wB; st[g.A].d += g.r.t;
      st[g.B].w += g.r.wB; st[g.B].l += g.r.wA; st[g.B].d += g.r.t;
      if (g.r.wA > g.r.wB) st[g.A].mw++; else if (g.r.wB > g.r.wA) st[g.B].mw++;
    }));
    return st.sort((a, b) => (b.w + 0.5 * b.d) / Math.max(1, b.w + b.l + b.d) - (a.w + 0.5 * a.d) / Math.max(1, a.w + a.l + a.d) || b.w - a.w);
  }
  function simWeek(k) {
    const wk = WK(k), pairs = S.schedule[k - 1] || [];
    if (wkState(wk) === "final") return;                  // real results, never simulated
    const rng = FX.mulberry(S.seed * 1000 + k);
    S.results[k] = pairs.map(([A, B]) => ({ A, B, r: FX.simulate(line(A, wk), line(B, wk), rng) }));
  }
  function renderSeason() {
    if (!S.schedule) { $("#fx-body").innerHTML = `<p class="note">Finish the draft to start the season.</p>`; return; }
    finalizeReal();
    const reg = S.schedule.length, k = Math.min(Math.max(1, S.week), reg), wk = WK(k);
    const state = wkState(wk);
    const pairs = S.schedule[k - 1] || [], played = S.results[k];
    const st = standings();
    const mm = pairs.map(([A, B], j) => {
      const LA = line(A, wk), LB = line(B, wk);
      const m = FX.matchup(LA, LB), r = played ? played[j].r : null;
      const exp = CATS.reduce((s, c) => s + m[c].pA, 0);
      const cell = (c, side) => {
        const v = r ? r[side][c] : m[c][side];
        const d = c === "fg" || c === "ft" ? 3 : 1;
        const win = r ? (side === "a" ? r.a[c] > r.b[c] : r.b[c] > r.a[c]) : (side === "a" ? m[c].pA > 0.5 : m[c].pA < 0.5);
        return `<td class="${win ? "pos" : ""}">${fmt(v, d)}</td>`;
      };
      const tag = r ? (r.real ? `final (real) ${r.wA}-${r.wB}-${r.t}` : `simulated ${r.wA}-${r.wB}-${r.t}`)
        : state === "live" ? `LIVE: real through ${LAST} + projected rest; expected ${fmt(exp, 1)}-${fmt(8 - exp, 1)} cats; games ${LA.rgp}+${fmt(LA.gp - LA.rgp, 0)} vs ${LB.rgp}+${fmt(LB.gp - LB.rgp, 0)}`
        : `projected ${fmt(exp, 1)}-${fmt(8 - exp, 1)} cats; games ${fmt(LA.gp, 0)} vs ${fmt(LB.gp, 0)}`;
      return `<h3 class="sub-head">${esc(S.names[A])} vs ${esc(S.names[B])} <span class="muted">${tag}</span></h3>
        <div class="table-scroll"><table class="grid"><thead><tr><th class="col-text">Team</th>${CATS.map((c) => `<th>${CAT_LABEL[c]}</th>`).join("")}</tr></thead><tbody>
        <tr><td class="col-text">${esc(S.names[A])}</td>${CATS.map((c) => cell(c, "a")).join("")}</tr>
        <tr><td class="col-text">${esc(S.names[B])}</td>${CATS.map((c) => cell(c, "b")).join("")}</tr>
        ${r ? "" : `<tr class="muted"><td class="col-text">P(${esc(S.names[A])} wins cat)</td>${CATS.map((c) => `<td>${fmt(100 * m[c].pA, 0)}%</td>`).join("")}</tr>`}
        </tbody></table></div>`;
    }).join("");
    const bye = Array.from({ length: nT() }, (_, t) => t).filter((t) => !pairs.some((p) => p.includes(t)));
    const regDone = Object.keys(S.results).length >= reg;
    $("#fx-body").innerHTML = `
      <div class="controls"><button class="link-btn" id="fx-prev">previous</button>
        <b>Week ${k} of ${reg}</b> <span class="muted">${wk ? `${wk.start} to ${wk.end}` : ""}</span>
        <button class="link-btn" id="fx-next">next</button>
        <button class="btn" id="fx-sim" ${played || state === "final" ? "disabled" : ""}>Simulate week</button>
        <button class="link-btn" id="fx-simall">simulate rest of regular season</button>
        <button class="link-btn" id="fx-unsim" ${played && !played[0].r.real ? "" : "disabled"}>clear simulation</button></div>
      <p class="note">${LAST ? `Real box scores through <b>${LAST}</b>: finished weeks are scored from real stats (with the rosters at the time they finished);
        the live week adds real games so far to projections for the rest. ` : ""}Projections count each player's remaining team games times his
        projected availability; players listed OUT on the latest injury report get no games this week or next.
        Simulated weeks draw from the same distributions (category sd from per-game overdispersion).${bye.length ? ` Bye: ${bye.map((t) => esc(S.names[t])).join(", ")}.` : ""}</p>
      ${mm}
      <h3 class="sub-head">Standings (category wins)</h3>
      <div class="table-scroll"><table class="grid"><thead><tr><th>#</th><th class="col-text">Team</th><th>W</th><th>L</th><th>T</th><th>Pct</th><th>Matchups won</th></tr></thead><tbody>
      ${st.map((r, i) => `<tr class="${r.t === S.my ? "me" : ""}"><td>${i + 1}${i < S.playoffTeams ? "*" : ""}</td><td class="col-text">${esc(S.names[r.t])}</td><td>${r.w}</td><td>${r.l}</td><td>${r.d}</td>
        <td>${fmt((r.w + 0.5 * r.d) / Math.max(1, r.w + r.l + r.d), 3)}</td><td>${r.mw}</td></tr>`).join("")}</tbody></table></div>
      <p class="note muted">* playoff position (${S.playoffTeams} teams, final ${S.playoffWeeks} weeks). League weeks start ${WK(1) ? WK(1).start : ""}.${regDone ? " Regular season complete." : ""}</p>
      ${regDone ? renderPlayoffs(st) : ""}`;
    $("#fx-prev").onclick = () => { S.week = Math.max(1, k - 1); save(); renderSeason(); };
    $("#fx-next").onclick = () => { S.week = Math.min(reg, k + 1); save(); renderSeason(); };
    $("#fx-sim").onclick = () => { simWeek(k); S.week = Math.min(reg, k + 1); save(); renderSeason(); };
    $("#fx-unsim").onclick = () => { delete S.results[k]; save(); renderSeason(); };
    $("#fx-simall").onclick = () => { for (let w = 1; w <= reg; w++) if (!S.results[w]) simWeek(w); save(); renderSeason(); };
  }
  function renderPlayoffs(st) {
    let seeds = st.slice(0, S.playoffTeams).map((r) => r.t);
    const out = [];
    let w = S.schedule.length + 1;
    while (seeds.length > 1 && WK(w)) {
      const wk = WK(w), rng = FX.mulberry(S.seed * 7777 + w), nxt = [];
      for (let i = 0; i < seeds.length / 2; i++) {
        const A = seeds[i], B = seeds[seeds.length - 1 - i];
        const LA = line(A, wk), LB = line(B, wk);
        const r = wkState(wk) === "final" ? realResult(LA, LB) : FX.simulate(LA, LB, rng);
        const win = r.wA >= r.wB ? A : B;
        out.push(`${wk.start}: ${esc(S.names[A])} ${r.wA}-${r.wB}-${r.t} ${esc(S.names[B])}${r.real ? " (real)" : ""}`);
        nxt.push(win);
      }
      seeds = nxt; w++;
    }
    return `<h3 class="sub-head">Playoffs</h3><p class="note">${out.join("<br>")}${seeds.length === 1 ? `<br><b>Champion: ${esc(S.names[seeds[0]])}</b>` : ""}</p>`;
  }

  /* ------------------------------------------------------------------ rosters, FA, trades */
  function renderTeams() {
    if (!S.rosters && !S.picks.length) { $("#fx-body").innerHTML = `<p class="note">Draft players first.</p>`; return; }
    const R = S.rosters || rosters(), V = values(), own = owned();
    const tz = (t) => { const z = {}; CATS.forEach((c) => (z[c] = R[t].reduce((s, pid) => s + Z[byPid[pid]][c], 0))); return z; };
    const T = R.map((_, t) => tz(t));
    const rank = (c, t) => 1 + T.filter((z) => z[c] > T[t][c]).length;
    const fa = V.avail.slice().sort((a, b) => V.fit[b] - V.fit[a]).slice(0, 40);
    const ta = tradeSel;
    const side = (t, sel, id) => `<select id="${id}" multiple size="8" style="min-width:220px">${R[t].map((pid) => {
      const p = P[byPid[pid]]; return `<option value="${pid}" ${sel.includes(pid) ? "selected" : ""}>${esc(p.n)} (${fmt(FX.zsum(Z[byPid[pid]], weights(null)), 1)})</option>`; }).join("")}</select>`;
    const delta = (() => {
      if (!ta.A.length && !ta.B.length) return "";
      const d = (pids) => { const z = {}; CATS.forEach((c) => (z[c] = pids.reduce((s, pid) => s + Z[byPid[pid]][c], 0))); return z; };
      const out = d(ta.A), inn = d(ta.B);
      return `<div class="table-scroll"><table class="grid"><thead><tr><th class="col-text">Change in z</th>${CATS.map((c) => `<th>${CAT_LABEL[c]}</th>`).join("")}<th>Total</th></tr></thead><tbody>
        <tr><td class="col-text">${esc(S.names[ta.a])}</td>${CATS.map((c) => zcell(inn[c] - out[c])).join("")}<td><b>${fmt(FX.zsum(inn, weights(ta.a)) - FX.zsum(out, weights(ta.a)), 2)}</b></td></tr>
        <tr><td class="col-text">${esc(S.names[ta.b])}</td>${CATS.map((c) => zcell(out[c] - inn[c])).join("")}<td><b>${fmt(FX.zsum(out, weights(ta.b)) - FX.zsum(inn, weights(ta.b)), 2)}</b></td></tr>
        </tbody></table></div><p class="note muted">Totals use each team's need weights (and your punts for your team).</p>`;
    })();
    $("#fx-body").innerHTML = `
      <h3 class="sub-head">Category strength (sum of player z; rank in league)</h3>
      <div class="table-scroll"><table class="grid"><thead><tr><th class="col-text">Team</th>${CATS.map((c) => `<th>${CAT_LABEL[c]}</th>`).join("")}<th>Players</th></tr></thead><tbody>
      ${R.map((r, t) => `<tr class="${t === S.my ? "me" : ""}"><td class="col-text">${esc(S.names[t])}</td>${CATS.map((c) => `<td>${fmt(T[t][c], 1)} <small class="muted">#${rank(c, t)}</small></td>`).join("")}<td>${r.length}</td></tr>`).join("")}
      </tbody></table></div>
      <div class="controls"><label>Roster <select id="fx-rt">${R.map((_, t) => `<option value="${t}" ${t === (S.rt ?? S.my) ? "selected" : ""}>${esc(S.names[t])}</option>`).join("")}</select></label></div>
      <div class="table-scroll"><table class="grid"><thead><tr>${HEAD}<th></th></tr></thead><tbody>
      ${R[S.rt ?? S.my].map((pid) => playerRow(byPid[pid], V, own, S.rosters ? `<td><button class="link-btn fx-drop" data-pid="${pid}">drop</button></td>` : "<td></td>")).join("")}</tbody></table></div>
      ${S.rosters ? `<h3 class="sub-head">Free agents (best Fit $ for ${esc(S.names[S.rt ?? S.my])})</h3>
      <div class="table-scroll"><table class="grid"><thead><tr>${HEAD}<th></th></tr></thead><tbody>
      ${fa.map((i) => playerRow(i, V, own, `<td><button class="link-btn fx-add" data-pid="${P[i].pid}">add</button></td>`)).join("")}</tbody></table></div>
      <h3 class="sub-head">Trade</h3>
      <div class="controls"><label>Team A <select id="fx-ta">${R.map((_, t) => `<option value="${t}" ${t === ta.a ? "selected" : ""}>${esc(S.names[t])}</option>`).join("")}</select></label>
        <label>Team B <select id="fx-tb">${R.map((_, t) => `<option value="${t}" ${t === ta.b ? "selected" : ""}>${esc(S.names[t])}</option>`).join("")}</select></label></div>
      <div class="controls"><div><small class="muted">${esc(S.names[ta.a])} sends</small><br>${side(ta.a, ta.A, "fx-sa")}</div>
        <div><small class="muted">${esc(S.names[ta.b])} sends</small><br>${side(ta.b, ta.B, "fx-sb")}</div></div>
      ${delta}
      <div class="controls"><button class="btn" id="fx-trade" ${ta.A.length || ta.B.length ? "" : "disabled"}>Execute trade</button></div>` : ""}`;
    const re = () => renderTeams();
    $("#fx-rt").onchange = (e) => { S.rt = +e.target.value; save(); re(); };
    document.querySelectorAll(".fx-drop").forEach((b) => (b.onclick = () => {
      const t = S.rt ?? S.my; S.rosters[t] = S.rosters[t].filter((x) => x !== +b.dataset.pid);
      S.log.push({ type: "drop", team: t, pid: +b.dataset.pid, week: S.week }); save(); re(); }));
    document.querySelectorAll(".fx-add").forEach((b) => (b.onclick = () => {
      const t = S.rt ?? S.my;
      if (S.rosters[t].length >= S.roster) { alert(`Roster is full (${S.roster}). Drop a player first.`); return; }
      S.rosters[t].push(+b.dataset.pid); S.log.push({ type: "add", team: t, pid: +b.dataset.pid, week: S.week }); save(); re(); }));
    if ($("#fx-ta")) {
      $("#fx-ta").onchange = (e) => { ta.a = +e.target.value; ta.A = []; re(); };
      $("#fx-tb").onchange = (e) => { ta.b = +e.target.value; ta.B = []; re(); };
      $("#fx-sa").onchange = (e) => { ta.A = Array.from(e.target.selectedOptions).map((o) => +o.value); re(); };
      $("#fx-sb").onchange = (e) => { ta.B = Array.from(e.target.selectedOptions).map((o) => +o.value); re(); };
      $("#fx-trade").onclick = () => {
        if (ta.a === ta.b) { alert("Pick two different teams."); return; }
        const nA = S.rosters[ta.a].length - ta.A.length + ta.B.length, nB = S.rosters[ta.b].length - ta.B.length + ta.A.length;
        if (nA > S.roster || nB > S.roster) { alert("A roster would exceed the limit; drop players first."); return; }
        S.rosters[ta.a] = S.rosters[ta.a].filter((x) => !ta.A.includes(x)).concat(ta.B);
        S.rosters[ta.b] = S.rosters[ta.b].filter((x) => !ta.B.includes(x)).concat(ta.A);
        S.log.push({ type: "trade", a: ta.a, b: ta.b, A: ta.A, B: ta.B, week: S.week });
        ta.A = []; ta.B = []; save(); re();
      };
    }
  }

  function renderData() {
    $("#fx-body").innerHTML = `
      <p class="note">Your league is saved in this browser automatically. Copy the text below to back it up or move it
        to another device; paste a saved league and press Load to restore it.</p>
      <textarea id="fx-json" rows="10" style="width:100%;font-family:var(--mono);font-size:12px">${esc(JSON.stringify(S))}</textarea>
      <div class="controls"><button class="btn" id="fx-load">Load</button>
        <button class="link-btn" id="fx-reset">reset league (delete everything)</button></div>
      <h3 class="sub-head">Transactions</h3>
      <p class="note">${S.log.length ? S.log.map((x) => x.type === "trade"
        ? `Week ${x.week}: trade ${esc(S.names[x.a])} sends ${x.A.map((pid) => esc(P[byPid[pid]].n)).join(", ") || "nothing"}; ${esc(S.names[x.b])} sends ${x.B.map((pid) => esc(P[byPid[pid]].n)).join(", ") || "nothing"}`
        : `Week ${x.week}: ${esc(S.names[x.team])} ${x.type}s ${esc(P[byPid[x.pid]].n)}`).join("<br>") : "None yet."}</p>`;
    $("#fx-load").onclick = () => {
      try { const s = JSON.parse($("#fx-json").value); if (s.v !== 1) throw new Error("bad version"); S = s; recomputeZ(); save(); render(); }
      catch (e) { alert("Could not read that league: " + e.message); }
    };
    $("#fx-reset").onclick = () => { if (confirm("Delete this league and start over?")) { S = defaults(); recomputeZ(); save(); render(); } };
  }

  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-fx-view]"); if (!b) return;
    S.view = b.dataset.fxView; save(); render();
  });
  window.renderFantasy = render;
})();
