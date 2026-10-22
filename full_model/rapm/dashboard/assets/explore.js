/* BOOKER Explore: analytics search, chart builder (incl. actual vs expected), progression /
 * regression candidates, player props.
 * Reads window.BOOKER (players, xva, movers, props); opens player cards via window.openPlayer.
 *
 * Every metric is a function of one player-season. Searches and chart axes can also read a
 * metric from a neighbouring season ("@-1" = the season before, "@+1" = the season after), which
 * is how "last season vs next season's projection" charts and "rose 2+ points" searches work.
 * State lives in the URL hash (#explore=...) so any search or chart can be shared.
 */
(function () {
  "use strict";
  const D = window.BOOKER || {};
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v) => v != null && typeof v === "number" && isFinite(v);
  const fmt = (v, d = 2) => (num(v) ? (+v).toFixed(d) : "—");
  const sgn = (v, d = 2) => (num(v) ? (v > 0 ? "+" : "") + (+v).toFixed(d) : "—");
  const lab = (s) => `${s - 1}-${String(s).slice(2)}`;
  const PLAYERS = D.players || [];
  const LIVE = 2027;                                     // season in progress (real games so far)
  let view = "search", chart = null;

  /* ================================================================ data join */
  const PL = {};                                          // "pid:season" -> site player row
  PLAYERS.forEach((p) => (PL[p.pid + ":" + p.season] = p));
  const XV = {};                                          // "pid:season" -> expected-vs-actual row
  const XC = (D.xva && D.xva.cols) || [];
  ((D.xva && D.xva.rows) || []).forEach((r) => {
    const o = {}; XC.forEach((c, i) => (o[c] = r[i]));
    XV[o.PLAYER_ID != null ? o.PLAYER_ID + ":" + o.season : o.pid + ":" + o.season] = o;
  });
  const MOV = {}; ((D.movers || {}).rows || []).forEach((m) => (MOV[m.pid + ":" + m.season] = m));
  const NAME = {}, NAME_S = {};               // canonical name = latest real-season spelling
  PLAYERS.forEach((p) => { if (!p.predictive && (NAME_S[p.pid] == null || p.season > NAME_S[p.pid])) { NAME[p.pid] = p.player; NAME_S[p.pid] = p.season; } });
  PLAYERS.forEach((p) => { if (!NAME[p.pid]) NAME[p.pid] = p.player; });
  Object.values(XV).forEach((x) => { const pid = x.PLAYER_ID != null ? x.PLAYER_ID : x.pid; if (!NAME[pid] && x.player) NAME[pid] = x.player; });
  const SEASONS = Array.from(new Set(PLAYERS.map((p) => p.season).concat(Object.values(XV).map((x) => x.season)))).sort((a, b) => a - b);
  const PROJ_FROM = Math.min(...PLAYERS.filter((p) => p.predictive).map((p) => p.season).concat([9999]));
  const LAST_DONE = Math.max(...SEASONS.filter((s) => s < PROJ_FROM));

  // context for one player-season (null if the player has no row of any kind that season)
  function ctx(pid, season) {
    const k = pid + ":" + season, p = PL[k], x = XV[k];
    if (!p && !x) return null;
    return { pid, season, p: p || {}, x: x || {}, proj: !!(p && p.predictive) };
  }
  // age: only forecast rows carry it -> derive from the nearest row of the player that has one
  const AGE = {};
  PLAYERS.forEach((p) => { if (p.age != null) (AGE[p.pid] = AGE[p.pid] || []).push([p.season, p.age]); });
  Object.values(XV).forEach((x) => { const pid = x.PLAYER_ID != null ? x.PLAYER_ID : x.pid; if (x.age != null) (AGE[pid] = AGE[pid] || []).push([x.season, x.age]); });
  function ageOf(c) {
    if (c.p.age != null) return c.p.age;
    if (c.x.age != null) return c.x.age;
    const a = AGE[c.pid]; if (!a) return null;
    const best = a.reduce((m, r) => (Math.abs(r[0] - c.season) < Math.abs(m[0] - c.season) ? r : m));
    return best[1] - (best[0] - c.season);
  }

  /* ================================================================ metric registry */
  // [id, label, group, kind, getter(ctx)]; kind: num | pct (0-1, shown x100) | money ($M) | int
  const actualOnly = (f) => (c) => (c.proj ? null : f(c));
  const xv = (k) => (c) => (num(c.x[k]) ? c.x[k] : null);
  const minAct = (c) => (num(c.x.min) ? c.x.min : c.proj ? null : c.p.min);
  const ME = [
    ["booker", "BOOKER (net /100)", "Impact", "num", (c) => c.p.bookerScore],
    ["off", "BOOKER offense /100", "Impact", "num", (c) => c.p.bookerOff],
    ["def", "BOOKER defense /100", "Impact", "num", (c) => c.p.bookerDef],
    ["waa", "Wins above average", "Impact", "num", (c) => c.p.waa],
    ["waa_off", "WAA offense", "Impact", "num", (c) => c.p.waaOff],
    ["waa_def", "WAA defense", "Impact", "num", (c) => c.p.waaDef],
    ["rank", "BOOKER rank", "Impact", "int", (c) => c.p.rank],
    ["real_pm", "Real on-court +/- /100", "Impact", "num", (c) => c.p.real_pm],
    ["unc", "Rating uncertainty (sd)", "Impact", "num", (c) => (num(c.p.sdOff) && num(c.p.sdDef) ? Math.hypot(c.p.sdOff, c.p.sdDef) : null)],
    ["grade", "Grade (0-100)", "Impact", "int", (c) => c.p.grade],
    ["grade_off", "Offensive grade", "Impact", "int", (c) => c.p.gradeOff],
    ["grade_def", "Defensive grade", "Impact", "int", (c) => c.p.gradeDef],
    ["proj_next", "PROJ: BOOKER next season", "Impact", "num", (c) => c.p.projNext],
    ["opp_q", "Opponent quality faced", "Context", "num", (c) => c.p.opp_quality],
    ["tm_q", "Teammate quality", "Context", "num", (c) => c.p.tm_quality],
    ["age", "Age", "Context", "int", ageOf],
    ["min", "Minutes (projected for future seasons)", "Context", "int", (c) => c.p.min != null ? c.p.min : c.x.min],
    ["height", "Height (in)", "Context", "int", (c) => c.p.heightIn],
    ["value", "True value ($M/yr)", "Contract", "money", (c) => c.p.trueValue],
    ["salary", "Market AAV ($M/yr)", "Contract", "money", (c) => c.p.marketAav2026],
    ["surplus", "Contract surplus ($M/yr)", "Contract", "money", (c) => c.p.surplus],
  ];
  // box score: actual (season totals / per game), expected (walk-forward, made before the season)
  const BOX = [["g", "Games", "int"], ["mpg", "Minutes per game", "num"], ["pts", "Points per game", "num"], ["reb", "Rebounds per game", "num"],
    ["ast", "Assists per game", "num"], ["stl", "Steals per game", "num"], ["blk", "Blocks per game", "num"], ["tpm", "3PM per game", "num"],
    ["tov", "Turnovers per game", "num"], ["ts_pct", "True shooting %", "pct"], ["fg3_pct", "3P%", "pct"]];
  BOX.forEach(([k, l, kind]) => {
    if (XC.includes(k)) ME.push([k, l, "Box score", kind, xv(k)]);
  });
  if (XC.includes("pts") && XC.includes("mpg")) ME.push(["pts36", "Points per 36", "Box score", "num", (c) => (num(c.x.pts) && c.x.mpg > 0 ? (36 * c.x.pts) / c.x.mpg : null)]);
  if (XC.includes("min")) ME.push(["min_act", "Minutes played (actual / so far)", "Box score", "int", minAct]);
  // expected + over-expected. Actual minutes, BOOKER and WAA for projection seasons don't exist yet.
  const EXP = BOX.concat([["min", "Minutes", "int"], ["booker", "BOOKER", "num"], ["waa", "WAA", "num"]]);
  EXP.forEach(([k, l, kind]) => {
    if (!XC.includes("e_" + k)) return;
    const act = k === "booker" ? actualOnly((c) => c.p.bookerScore) : k === "waa" ? actualOnly((c) => c.p.waa) : k === "min" ? minAct : xv(k);
    ME.push(["x" + k, `Expected ${l === l.toUpperCase() ? l : l.replace(" per game", "").toLowerCase() + (l.includes("per game") ? " per game" : "")}`, "Expected", kind, xv("e_" + k)]);
    ME.push([k + "_oe", `${l.replace(" per game", "")} over expected`, "Expected", kind, (c) => { const a = act(c), e = c.x["e_" + k]; return num(a) && num(e) ? a - e : null; }]);
  });
  // movers (progression / regression model, latest season only)
  const mv = (k) => (c) => { const m = MOV[c.pid + ":" + c.season]; return m && num(m[k]) ? m[k] : null; };
  if (D.movers) ME.push(
    ["mv_exp", "Movers: expected change next season", "Movers", "num", mv("exp")],
    ["mv_raw", "Movers: projected change (raw)", "Movers", "num", mv("raw")],
    ["mv_p", "Movers: P(improve)", "Movers", "pct", mv("p")],
    ["mv_form", "Second-half form (h2 - season)", "Movers", "num", mv("form")],
    ["mv_onoff", "On/off gap (real +/- - rating)", "Movers", "num", mv("onoff")],
    ["mv_luck3", "3P luck (raw - true 3P%)", "Movers", "pct", mv("luck3")]);
  // live season so far (lives on the projection-season row)
  const lv = (f) => (c) => { const L = c.p.live; return L ? f(L) : null; };
  ME.push(
    ["live_gp", `${lab(LIVE)} so far: games`, "Live", "int", lv((L) => L.gp)],
    ["live_pts", `${lab(LIVE)} so far: points per game`, "Live", "num", lv((L) => L.pts)],
    ["live_gr", `${lab(LIVE)} so far: avg game rating`, "Live", "num", lv((L) => L.gameRating)],
    ["live_now", `${lab(LIVE)} in-season rating`, "Live", "num", lv((L) => L.ratingNow)],
    ["live_dr", `${lab(LIVE)} change since opener`, "Live", "num", lv((L) => (num(L.ratingNow) && num(L.ratingOpen) ? L.ratingNow - L.ratingOpen : null))]);
  // skills (value + percentile)
  const skillKeys = {};
  PLAYERS.forEach((p) => p.skills && Object.entries(p.skills).forEach(([k, v]) => (skillKeys[k] = skillKeys[k] || [v.label || k, v.fmt])));
  Object.entries(skillKeys).forEach(([k, [l, f]]) => ME.push(["sk_" + k, l, "Skills", f === "pct" ? "pct" : "num", (c) => (c.p.skills && c.p.skills[k] && num(c.p.skills[k].val) ? c.p.skills[k].val : null)]));
  Object.entries(skillKeys).forEach(([k, [l]]) => ME.push(["pc_" + k, `${l} (percentile)`, "Skill percentiles", "int", (c) => (c.p.skills && c.p.skills[k] && num(c.p.skills[k].pct) ? c.p.skills[k].pct : null)]));
  const M = Object.fromEntries(ME.map((m) => [m[0], { id: m[0], label: m[1], group: m[2], kind: m[3], get: m[4] }]));
  const GROUPS = Array.from(new Set(ME.map((m) => m[2])));

  const OFFS = [[0, "same season"], [-1, "season before"], [-2, "2 seasons before"], [1, "season after"]];
  const offTag = (o) => (o ? `@${o > 0 ? "+" : ""}${o}` : "");
  const offLab = (o) => (o === 0 ? "" : o === -1 ? " (prev season)" : o === 1 ? " (next season)" : ` (${o > 0 ? "+" : ""}${o} seasons)`);
  function val(id, pid, season, off = 0) {
    const m = M[id]; if (!m) return null;
    const c = ctx(pid, season + off); if (!c) return null;
    const v = m.get(c);
    if (!num(v)) return null;
    return m.kind === "money" ? v / 1e6 : v;
  }
  function show(id, v, signed) {
    const m = M[id]; if (!num(v)) return "—";
    const k = m ? m.kind : "num";
    if (k === "pct") return (signed ? sgn : fmt)(100 * v, 1) + (id.startsWith("mv_luck") || id.endsWith("_oe") ? " pp" : "%");
    if (k === "int") return signed ? sgn(v, 0) : fmt(v, 0);
    if (k === "money") return (signed || v < 0 ? sgn(v, 1) : fmt(v, 1)) + "M";
    return (signed ? sgn : fmt)(v, Math.abs(v) >= 100 ? 0 : 2);
  }
  const mlabel = (id, off) => (M[id] ? M[id].label : id) + offLab(off || 0);
  function metricOptions(sel) {
    return GROUPS.map((g) => `<optgroup label="${esc(g)}">${ME.filter((m) => m[2] === g)
      .map((m) => `<option value="${m[0]}" ${m[0] === sel ? "selected" : ""}>${esc(m[1])}</option>`).join("")}</optgroup>`).join("");
  }
  const offOptions = (sel) => OFFS.map(([o, l]) => `<option value="${o}" ${o === sel ? "selected" : ""}>${l}</option>`).join("");

  // every player-season in a set of seasons (union of site rows and box-score rows)
  function population(seasons) {
    const S = new Set(seasons), seen = new Set(), out = [];
    const add = (pid, s) => { const k = pid + ":" + s; if (!S.has(s) || seen.has(k)) return; seen.add(k); out.push(ctx(pid, s)); };
    PLAYERS.forEach((p) => add(p.pid, p.season));
    Object.values(XV).forEach((x) => add(x.PLAYER_ID != null ? x.PLAYER_ID : x.pid, x.season));
    return out;
  }
  const teamOf = (c) => c.p.team || c.x.team || "";
  const nameOf = (c) => NAME[c.pid] || c.p.player || c.x.player || String(c.pid);

  /* ================================================================ filters + text query */
  // filter = {m: metric id, o: offset, op: ">=" | "<=" | ">" | "<" | "=" | "!=", v: number}
  const OPS = [">=", "<=", ">", "<", "=", "!="];
  function pass(f, c) {
    const v = val(f.m, c.pid, c.season, f.o || 0);
    if (!num(v)) return false;
    const t = M[f.m] && M[f.m].kind === "pct" && Math.abs(f.v) > 1 ? f.v / 100 : f.v;   // "ts_pct>=60" means 60%
    return f.op === ">=" ? v >= t : f.op === "<=" ? v <= t : f.op === ">" ? v > t : f.op === "<" ? v < t
      : f.op === "=" ? Math.abs(v - t) < 1e-9 : Math.abs(v - t) >= 1e-9;
  }
  function seasonsOf(spec) {           // "2026" | "2019-2026" | "all"
    if (spec === "all") return SEASONS.filter((s) => s <= LAST_DONE);      // completed seasons only
    const m = String(spec).match(/^(\d{4})(?:-(\d{4}))?$/);
    if (!m) return [LAST_DONE];
    const a = +m[1], b = m[2] ? +m[2] : a;
    return SEASONS.filter((s) => s >= Math.min(a, b) && s <= Math.max(a, b));
  }
  function serialize(q) {
    const parts = [`season:${q.season}`];
    if (q.team) parts.push(`team:${q.team}`);
    if (q.name) parts.push(`name:${q.name.includes(" ") ? '"' + q.name + '"' : q.name}`);
    q.filters.forEach((f) => parts.push(`${f.m}${offTag(f.o)}${f.op}${f.v}`));
    if (q.sort) parts.push(`sort:${q.desc ? "-" : ""}${q.sort.m}${offTag(q.sort.o)}`);
    parts.push(`cols:${q.cols.map((c) => c.m + offTag(c.o)).join(",")}`);
    return parts.join(" ");
  }
  function parseRef(s) {               // "booker@-1" -> {m, o}
    const m = s.match(/^([a-z0-9_]+)(?:@([+-]?\d))?$/i);
    if (!m || !M[m[1].toLowerCase()]) return null;
    return { m: m[1].toLowerCase(), o: m[2] ? +m[2] : 0 };
  }
  function parse(text, base) {
    const q = { season: base.season, team: "", name: "", filters: [], sort: null, desc: true, cols: base.cols.slice() };
    const errs = [];
    const toks = (text.match(/[^\s"]+(?:"[^"]*")?|"[^"]*"/g) || []).map((t) => t.replace(/"/g, ""));
    toks.forEach((t) => {
      let m;
      if ((m = t.match(/^season:(.+)$/i))) q.season = m[1].toLowerCase();
      else if ((m = t.match(/^team:(.*)$/i))) q.team = m[1].toUpperCase();
      else if ((m = t.match(/^name:(.*)$/i))) q.name = m[1];
      else if ((m = t.match(/^sort:(-?)(.+)$/i))) { const r = parseRef(m[2]); if (r) { q.sort = r; q.desc = m[1] === "-"; } else errs.push(t); }
      else if ((m = t.match(/^cols:(.+)$/i))) { const cs = m[1].split(",").map(parseRef).filter(Boolean); if (cs.length) q.cols = cs; }
      else if ((m = t.match(/^([a-z0-9_]+(?:@[+-]?\d)?)(>=|<=|!=|>|<|=)(-?[\d.]+)$/i))) {
        const r = parseRef(m[1]); if (r) q.filters.push({ m: r.m, o: r.o, op: m[2], v: +m[3] }); else errs.push(t);
      } else if (t.trim()) q.name = (q.name ? q.name + " " : "") + t;     // bare words = name search
    });
    return { q, errs };
  }

  /* ================================================================ URL state */
  const PRESETS_SEARCH = [
    ["Beat their points projection", `season:${LAST_DONE} min>=1000 pts_oe>=3 sort:-pts_oe cols:pts,xpts,pts_oe,booker,booker_oe,age`],
    ["Fell short of their points projection", `season:${LAST_DONE} min>=1000 pts_oe<=-3 sort:pts_oe cols:pts,xpts,pts_oe,booker,booker_oe,age`],
    ["Beat their WAA projection", `season:${LAST_DONE} min>=1000 sort:-waa_oe cols:waa,xwaa,waa_oe,min_act,xmin,booker,age`],
    ["Biggest risers last season", `season:${LAST_DONE} min>=1500 min@-1>=1500 sort:-booker cols:booker@-1,booker,age,off,def`],
    ["Projected risers next season", `season:${PROJ_FROM} min>=1200 min@-1>=1200 sort:-booker cols:booker@-1,booker,waa@-1,waa,age`],
    ["Young and already good", `season:${LAST_DONE} age<=23 booker>=1 min>=1000 sort:-booker cols:age,booker,off,def,waa,proj_next`],
    ["Two-way players", `season:${LAST_DONE} pc_offense>=75 pc_defense>=75 min>=1000 sort:-booker cols:booker,off,def,grade,waa`],
    ["Most underpaid", `season:${LAST_DONE} min>=1000 sort:-surplus cols:value,salary,surplus,booker,age`],
    ["3P luck: due to cool off", `season:${LAST_DONE} min>=1000 mv_luck3>=0.03 sort:-mv_luck3 cols:mv_luck3,sk_three_pct,fg3_pct,mv_exp,booker`],
    ["Hot starts this season", `season:${LIVE} live_gp>=3 sort:-live_dr cols:live_gp,live_pts,live_gr,live_dr,booker,xpts`],
  ];
  const sq = parse(`season:${LAST_DONE} min>=1000 sort:-booker cols:age,min,booker,off,def,waa,grade`, { season: String(LAST_DONE), cols: [] }).q;
  const st = { base: String(LAST_DONE), minMin: 1000, team: "", x: { m: "off", o: 0 }, y: { m: "def", o: 0 }, c: { m: "age", o: 0 },
               q: "", labels: true, trend: true, ident: false, useSearch: false };
  function saveHash() {
    const s = view === "search" ? { v: view, q: serialize(sq) } : view === "chart" ? { v: view, s: st } : { v: view };
    try { history.replaceState(null, "", "#explore=" + encodeURIComponent(JSON.stringify(s))); } catch (e) { /* file:// */ }
  }
  function loadHash() {
    const h = location.hash.match(/^#explore=(.+)$/); if (!h) return false;
    try {
      const s = JSON.parse(decodeURIComponent(h[1]));
      if (s.v) view = s.v;
      if (s.q) Object.assign(sq, parse(s.q, sq).q);
      if (s.s) Object.assign(st, s.s);
      return true;
    } catch (e) { return false; }
  }

  /* ================================================================ shell */
  function nav() {
    const tabs = [["search", "Search"], ["chart", "Chart builder"], ["movers", "Progression / regression"], ["props", "Player props"]];
    return `<div class="controls fx-nav">${tabs.map(([k, t]) => `<button class="link-btn${k === view ? " on" : ""}" data-ex="${k}">${t}</button>`).join(" | ")}</div><div id="ex-body"></div>`;
  }
  function render() {
    const root = $("#ex-root"); if (!root) return;
    root.innerHTML = nav();
    $$("[data-ex]", root).forEach((b) => (b.onclick = () => { view = b.dataset.ex; render(); }));
    if (chart) { chart.destroy(); chart = null; }
    ({ search: renderSearch, chart: renderChart, movers: renderMovers, props: renderProps })[view]();
    saveHash();
  }
  const openP = (pid) => window.openPlayer && window.openPlayer(+pid);

  /* ================================================================ search */
  function runSearch(q) {
    const seasons = seasonsOf(q.season);
    const nm = (q.name || "").toLowerCase();
    let rows = population(seasons).filter((c) => (!q.team || teamOf(c) === q.team) && (!nm || nameOf(c).toLowerCase().includes(nm))
      && q.filters.every((f) => pass(f, c)));
    const s = q.sort || (q.filters[0] ? { m: q.filters[0].m, o: q.filters[0].o || 0 } : { m: "booker", o: 0 });
    const desc = q.sort ? q.desc : true;
    const key = (c) => val(s.m, c.pid, c.season, s.o);
    rows = rows.map((c) => ({ c, k: key(c) })).sort((a, b) => (num(a.k) ? (num(b.k) ? (desc ? b.k - a.k : a.k - b.k) : -1) : num(b.k) ? 1 : 0)).map((r) => r.c);
    return { rows, s, desc, seasons };
  }
  function renderSearch() {
    const teams = Array.from(new Set(PLAYERS.map((p) => p.team).filter((t) => t && t.length === 3))).sort();
    const sOpts = [["all", "All completed seasons"]].concat(SEASONS.slice().reverse().map((s) => [String(s), lab(s) + (s >= PROJ_FROM ? " (projected)" : s === LIVE ? " (live)" : "")]));
    const isRange = !sOpts.some(([v]) => v === sq.season);
    $("#ex-body").innerHTML = `
      <p class="note">Find players by any combination of ratings, box-score stats, expectations, skills and contract numbers, for one season, a range, or all seasons.
        Type a query or build one with the controls; both stay in sync. <b>Expected</b> values are what the models projected <i>before</i> the season, using only earlier seasons
        (walk-forward), so "over expected" is a fair surprise measure. Seasons from ${lab(PROJ_FROM)} on are projections.</p>
      <div class="controls" style="align-items:center">
        <input id="sq-text" type="search" style="flex:1;min-width:280px" value="${esc(serialize(sq))}"
          placeholder="e.g. season:2026 age<=24 booker>=2 min>=1000 sort:-waa cols:booker,waa,pts">
        <button class="link-btn" id="sq-run">Run</button> <button class="link-btn" id="sq-help-b">Syntax</button>
      </div>
      <div class="note" id="sq-help" hidden>
        <b>Filters</b> <code>metric&gt;=value</code> with <code>&gt;= &lt;= &gt; &lt; = !=</code>; add <code>@-1</code> to read the season before (<code>booker@-1&gt;=2</code>) or <code>@+1</code> for the season after.
        Percentages accept either form (<code>ts_pct&gt;=0.6</code> or <code>ts_pct&gt;=60</code>). Money is in $M.
        <b>Scope</b> <code>season:2026</code>, <code>season:2019-2026</code>, <code>season:all</code> (completed seasons), <code>team:OKC</code>, <code>name:"jalen"</code> (bare words also search names).
        <b>Output</b> <code>sort:-waa</code> (minus = descending), <code>cols:booker,waa@-1,pts_oe</code>.
        <b>Metric codes</b>: ${ME.map((m) => `<code title="${esc(m[1])}">${m[0]}</code>`).join(" ")}</div>
      <div class="controls">
        <label>Season <select id="sq-season">${(isRange ? [[sq.season, sq.season]] : []).concat(sOpts).map(([v, l]) => `<option value="${v}" ${v === sq.season ? "selected" : ""}>${l}</option>`).join("")}</select></label>
        <label>Team <select id="sq-team"><option value="">All</option>${teams.map((t) => `<option ${t === sq.team ? "selected" : ""}>${t}</option>`).join("")}</select></label>
        <label>Name <input id="sq-name" type="text" value="${esc(sq.name)}" style="width:130px"></label>
        <span class="presets">Quick searches: <select id="sq-pre"><option value="">choose...</option>${PRESETS_SEARCH.map(([l], i) => `<option value="${i}">${esc(l)}</option>`).join("")}</select></span>
      </div>
      <div id="sq-filters"></div>
      <div class="controls"><button class="link-btn" id="sq-add">+ filter</button>
        <span class="muted" style="font-size:12px">Columns:</span> <span id="sq-cols"></span> <button class="link-btn" id="sq-addc">+ column</button></div>
      <p class="note" id="sq-count"></p>
      <div class="table-scroll"><table class="grid" id="sq-table"><thead></thead><tbody></tbody></table></div>
      <div class="controls"><button class="link-btn" id="sq-more" hidden>show more</button> <button class="link-btn" id="sq-csv">Download CSV</button>
        <button class="link-btn" id="sq-plot">Plot the first two columns</button></div>`;
    const refresh = () => { renderSearch(); saveHash(); };
    $("#sq-text").onkeydown = (e) => { if (e.key === "Enter") apply(); };
    $("#sq-text").addEventListener("search", () => { if (!$("#sq-text").value) apply(); });
    $("#sq-run").onclick = apply;
    function apply() {
      const { q, errs } = parse($("#sq-text").value, sq);
      Object.assign(sq, q); renderSearch(); saveHash();
      if (errs.length) $("#sq-count").insertAdjacentHTML("afterbegin", `<span class="neg">Not understood: ${esc(errs.join(" "))}. </span>`);
    }
    $("#sq-help-b").onclick = () => ($("#sq-help").hidden = !$("#sq-help").hidden);
    $("#sq-season").onchange = (e) => { sq.season = e.target.value; refresh(); };
    $("#sq-team").onchange = (e) => { sq.team = e.target.value; refresh(); };
    $("#sq-name").onchange = (e) => { sq.name = e.target.value.trim(); refresh(); };
    $("#sq-pre").onchange = (e) => { if (e.target.value === "") return; Object.assign(sq, parse(PRESETS_SEARCH[+e.target.value][1], { season: sq.season, cols: sq.cols }).q, { team: "", name: "" }); refresh(); };
    // filter rows
    $("#sq-filters").innerHTML = sq.filters.map((f, i) => `<div class="controls" data-fi="${i}" style="margin-bottom:4px">
        <select class="f-m" style="max-width:260px">${metricOptions(f.m)}</select>
        <select class="f-o">${offOptions(f.o || 0)}</select>
        <select class="f-op">${OPS.map((o) => `<option ${o === f.op ? "selected" : ""}>${o}</option>`).join("")}</select>
        <input class="f-v" type="number" step="any" value="${f.v}" style="width:80px">
        <button class="link-btn f-x">remove</button></div>`).join("");
    $$("[data-fi]").forEach((row) => {
      const f = sq.filters[+row.dataset.fi];
      $(".f-m", row).onchange = (e) => { f.m = e.target.value; refresh(); };
      $(".f-o", row).onchange = (e) => { f.o = +e.target.value; refresh(); };
      $(".f-op", row).onchange = (e) => { f.op = e.target.value; refresh(); };
      $(".f-v", row).onchange = (e) => { f.v = +e.target.value; refresh(); };
      $(".f-x", row).onclick = () => { sq.filters.splice(+row.dataset.fi, 1); refresh(); };
    });
    $("#sq-add").onclick = () => { sq.filters.push({ m: "booker", o: 0, op: ">=", v: 0 }); refresh(); };
    // columns
    $("#sq-cols").innerHTML = sq.cols.map((c, i) => `<span class="sq-col" style="white-space:nowrap;margin-right:8px;font-size:12px">${esc(mlabel(c.m, c.o))}
        <button class="link-btn" data-cx="${i}" title="remove">x</button></span>`).join("");
    $$("[data-cx]").forEach((b) => (b.onclick = () => { sq.cols.splice(+b.dataset.cx, 1); refresh(); }));
    $("#sq-addc").onclick = () => {
      const box = document.createElement("span");
      box.innerHTML = `<select id="sq-nc" style="max-width:240px">${metricOptions("")}</select> <select id="sq-nco">${offOptions(0)}</select> <button class="link-btn" id="sq-ncb">add</button>`;
      $("#sq-addc").replaceWith(box);
      $("#sq-ncb").onclick = () => { sq.cols.push({ m: $("#sq-nc").value, o: +$("#sq-nco").value }); refresh(); };
    };
    // results
    const R = runSearch(sq);
    const multi = R.seasons.length > 1;
    let shown = 100;
    const cols = sq.cols.slice();
    // filter metrics not in the column list are shown too (you want to see what you filtered on)
    sq.filters.forEach((f) => { if (!cols.some((c) => c.m === f.m && (c.o || 0) === (f.o || 0))) cols.push({ m: f.m, o: f.o || 0 }); });
    const isSort = (c) => R.s.m === c.m && (R.s.o || 0) === (c.o || 0);
    $("#sq-table thead").innerHTML = `<tr><th>#</th><th class="col-text">Player</th><th>Team</th>${multi ? "<th>Season</th>" : ""}${cols.map((c, i) =>
      `<th data-sc="${i}" title="${esc(mlabel(c.m, c.o))}" style="cursor:pointer">${esc(short(c))}${isSort(c) ? `<span class="arrow">${R.desc ? "▼" : "▲"}</span>` : ""}</th>`).join("")}</tr>`;
    const body = () => R.rows.slice(0, shown).map((c, i) => `<tr class="clickable" data-pid="${c.pid}"><td>${i + 1}</td>
        <td class="col-text"><b>${esc(nameOf(c))}</b>${c.proj ? ' <small class="muted">proj.</small>' : ""}</td><td>${teamOf(c)}</td>${multi ? `<td>${lab(c.season)}</td>` : ""}
        ${cols.map((k) => { const v = val(k.m, c.pid, c.season, k.o); const oe = k.m.endsWith("_oe") || k.m === "live_dr";
          return `<td class="${oe && num(v) ? (v > 0 ? "pos" : "neg") : ""}">${show(k.m, v, oe)}</td>`; }).join("")}</tr>`).join("");
    const paint = () => {
      $("#sq-table tbody").innerHTML = body();
      $$("#sq-table tbody tr").forEach((tr) => (tr.onclick = () => openP(tr.dataset.pid)));
      $("#sq-more").hidden = R.rows.length <= shown;
    };
    paint();
    $("#sq-more").onclick = () => { shown += 200; paint(); };
    $$("[data-sc]").forEach((th) => (th.onclick = () => {
      const c = cols[+th.dataset.sc];
      sq.desc = isSort(c) ? !R.desc : true; sq.sort = { m: c.m, o: c.o || 0 }; refresh();
    }));
    $("#sq-count").innerHTML = `<b>${R.rows.length.toLocaleString()}</b> player-season${R.rows.length === 1 ? "" : "s"} in ${R.seasons.length > 1 ? lab(R.seasons[0]) + " to " + lab(R.seasons[R.seasons.length - 1]) : lab(R.seasons[0])}. Click a row to open the player.`;
    $("#sq-csv").onclick = () => {
      const head = ["player", "pid", "team", "season"].concat(cols.map((c) => c.m + offTag(c.o)));
      const lines = [head.join(",")].concat(R.rows.map((c) => [`"${nameOf(c).replace(/"/g, '""')}"`, c.pid, teamOf(c), c.season]
        .concat(cols.map((k) => { const v = val(k.m, c.pid, c.season, k.o); return num(v) ? +v.toFixed(4) : ""; })).join(",")));
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
      a.download = `booker_search_${String(sq.season).replace(/[^0-9a-z-]/gi, "")}.csv`; a.click();
    };
    $("#sq-plot").onclick = () => {
      const [a, b] = cols.length >= 2 ? cols : [{ m: "off", o: 0 }, { m: "def", o: 0 }];
      Object.assign(st, { x: { m: a.m, o: a.o || 0 }, y: { m: b.m, o: b.o || 0 }, base: R.seasons.length > 1 ? String(sq.season) : String(R.seasons[0]),
                          team: sq.team, useSearch: true, minMin: 0, ident: sameUnit(a.m, b.m) });
      view = "chart"; render();
    };
  }
  const SHORT = { booker: "BOOKER", off: "OFF", def: "DEF", waa: "WAA", waa_off: "WAA off", waa_def: "WAA def", min: "MIN", min_act: "MIN played",
    age: "Age", grade: "Grade", proj_next: "PROJ next", value: "Value $M", salary: "AAV $M", surplus: "Surplus $M", real_pm: "On-court +/-",
    g: "G", mpg: "MPG", pts: "PTS", reb: "REB", ast: "AST", stl: "STL", blk: "BLK", tpm: "3PM", tov: "TOV", ts_pct: "TS%", fg3_pct: "3P%", pts36: "PTS/36" };
  function short(c) {
    const m = M[c.m], f = fam(c.m);
    let s = SHORT[c.m];
    if (!s && m && m.group === "Expected") s = c.m.endsWith("_oe") ? `${SHORT[f] || f} OE` : `x${SHORT[f] || f}`;
    if (!s) s = m ? m.label : c.m;
    return s + (c.o ? ` (${c.o > 0 ? "+" : ""}${c.o})` : "");
  }

  /* ================================================================ chart builder */
  const fam = (id) => id.replace(/^x(?=[a-z])/, "").replace(/_oe$/, "").replace(/^min_act$/, "min");
  const sameUnit = (a, b) => !!M[a] && !!M[b] && (fam(a) === fam(b) || (M[a].group === "Expected" && fam(a) === b) || (M[b].group === "Expected" && fam(b) === a));
  const CHART_PRESETS = [
    ["pts", "Points vs expected", { x: { m: "xpts", o: 0 }, y: { m: "pts", o: 0 }, c: { m: "age", o: 0 }, base: String(LAST_DONE), ident: true, minMin: 1000 }],
    ["waa", "WAA vs expected", { x: { m: "xwaa", o: 0 }, y: { m: "waa", o: 0 }, c: { m: "age", o: 0 }, base: String(LAST_DONE), ident: true, minMin: 1000 }],
    ["minx", "Minutes vs expected", { x: { m: "xmin", o: 0 }, y: { m: "min_act", o: 0 }, c: { m: "age", o: 0 }, base: String(LAST_DONE), ident: true, minMin: 0 }],
    ["bkx", "BOOKER vs expected", { x: { m: "xbooker", o: 0 }, y: { m: "booker", o: 0 }, c: { m: "age", o: 0 }, base: String(LAST_DONE), ident: true, minMin: 1000 }],
    ["next", "Last season vs next season (projected)", { x: { m: "booker", o: -1 }, y: { m: "booker", o: 0 }, c: { m: "age", o: 0 }, base: String(PROJ_FROM), ident: true, minMin: 1000 }],
    ["nextw", "WAA last season vs projected", { x: { m: "waa", o: -1 }, y: { m: "waa", o: 0 }, c: { m: "age", o: 0 }, base: String(PROJ_FROM), ident: true, minMin: 1000 }],
    ["nextp", "Points last season vs projected", { x: { m: "pts", o: -1 }, y: { m: "xpts", o: 0 }, c: { m: "age", o: 0 }, base: String(PROJ_FROM), ident: true, minMin: 500 }],
    ["yoy", "BOOKER year over year", { x: { m: "booker", o: -1 }, y: { m: "booker", o: 0 }, c: { m: "age", o: 0 }, base: String(LAST_DONE), ident: true, minMin: 1000 }],
    ["od", "Offense vs defense", { x: { m: "off", o: 0 }, y: { m: "def", o: 0 }, c: { m: "age", o: 0 }, base: String(LAST_DONE), ident: false, minMin: 1000 }],
    ["age", "Age curve", { x: { m: "age", o: 0 }, y: { m: "booker", o: 0 }, c: { m: "min", o: 0 }, base: "all", ident: false, minMin: 1000 }],
    ["luck", "3P luck vs expected change", { x: { m: "mv_luck3", o: 0 }, y: { m: "mv_exp", o: 0 }, c: { m: "age", o: 0 }, base: String(LAST_DONE), ident: false, minMin: 1000 }],
    ["value", "Value vs salary", { x: { m: "salary", o: 0 }, y: { m: "value", o: 0 }, c: { m: "booker", o: 0 }, base: String(LAST_DONE), ident: true, minMin: 1000 }],
  ].filter(([, , p]) => M[p.x.m] && M[p.y.m]);
  function renderChart() {
    const teams = Array.from(new Set(PLAYERS.map((p) => p.team).filter((t) => t && t.length === 3))).sort();
    const sOpts = [["all", "All completed seasons (pooled)"]].concat(SEASONS.slice().reverse().map((s) => [String(s), lab(s) + (s >= PROJ_FROM ? " (projected)" : s === LIVE ? " (live)" : "")]));
    if (!sOpts.some(([v]) => v === st.base)) sOpts.unshift([st.base, st.base]);
    const axis = (k, nm) => `<label class="ex-ax">${nm} <span class="ex-pair"><select id="ex-${k}" style="max-width:250px">${k === "c" ? `<option value="">None</option>` : ""}${metricOptions(st[k] ? st[k].m : "")}</select>
      <select id="ex-${k}o">${offOptions(st[k] ? st[k].o || 0 : 0)}</select></span></label>`;
    $("#ex-body").innerHTML = `
      <p class="note">Plot any metric against any other, for one season or pooled across seasons. Each axis can read a neighbouring season, so
        "last season vs next season's projection" is X = BOOKER (season before), Y = BOOKER, season ${lab(PROJ_FROM)}. <b>Expected</b> metrics are pre-season,
        walk-forward projections. The dashed diagonal is y = x (on target); labelled players are the biggest misses. Click a point to open the player.</p>
      <div class="controls"><span class="presets">Presets: ${CHART_PRESETS.map(([k, l]) => `<button class="link-btn" data-pre="${k}">${esc(l)}</button>`).join("")}</span></div>
      <div class="controls">
        <label>Season <select id="ex-season">${sOpts.map(([v, l]) => `<option value="${v}" ${v === st.base ? "selected" : ""}>${l}</option>`).join("")}</select></label>
        <label>Min minutes <select id="ex-min">${[0, 250, 500, 1000, 1500, 2000].map((m) => `<option ${m === st.minMin ? "selected" : ""}>${m}</option>`).join("")}</select></label>
        <label>Team <select id="ex-team"><option value="">All</option>${teams.map((t) => `<option ${t === st.team ? "selected" : ""}>${t}</option>`).join("")}</select></label>
        <label>Highlight <input id="ex-q" type="text" value="${esc(st.q)}" placeholder="player name" style="width:130px"></label>
      </div>
      <div class="controls">${axis("x", "X")} ${axis("y", "Y")} ${axis("c", "Color")}</div>
      <div class="controls">
        <label><input type="checkbox" id="ex-lab" ${st.labels ? "checked" : ""}> label biggest misses</label>
        <label><input type="checkbox" id="ex-tr" ${st.trend ? "checked" : ""}> trend line</label>
        <label><input type="checkbox" id="ex-id" ${st.ident ? "checked" : ""}> y = x line</label>
        <label title="Only players matching the filters on the Search tab"><input type="checkbox" id="ex-us" ${st.useSearch ? "checked" : ""}> apply Search filters (${sq.filters.length})</label>
        <button class="link-btn" id="ex-swap">swap axes</button>
      </div>
      <div class="panel" style="height:540px;padding:8px"><canvas id="ex-canvas"></canvas></div>
      <p class="note" id="ex-stats"></p>
      <div class="panel-row"><div><h3 class="sub-head" id="ex-h-up"></h3><div class="table-scroll"><table class="grid" id="ex-up"><thead></thead><tbody></tbody></table></div></div>
        <div><h3 class="sub-head" id="ex-h-dn"></h3><div class="table-scroll"><table class="grid" id="ex-dn"><thead></thead><tbody></tbody></table></div></div></div>`;
    const re = () => { renderChart(); saveHash(); };
    $("#ex-season").onchange = (e) => { st.base = e.target.value; re(); };
    $("#ex-min").onchange = (e) => { st.minMin = +e.target.value; re(); };
    $("#ex-team").onchange = (e) => { st.team = e.target.value; re(); };
    $("#ex-q").onchange = (e) => { st.q = e.target.value.trim(); re(); };
    ["x", "y", "c"].forEach((k) => {
      $(`#ex-${k}`).onchange = (e) => { st[k] = e.target.value ? { m: e.target.value, o: st[k] ? st[k].o || 0 : 0 } : null;
        if (k !== "c") st.ident = sameUnit(st.x.m, st.y.m); re(); };
      $(`#ex-${k}o`).onchange = (e) => { if (st[k]) st[k].o = +e.target.value; re(); };
    });
    $("#ex-lab").onchange = (e) => { st.labels = e.target.checked; re(); };
    $("#ex-tr").onchange = (e) => { st.trend = e.target.checked; re(); };
    $("#ex-id").onchange = (e) => { st.ident = e.target.checked; re(); };
    $("#ex-us").onchange = (e) => { st.useSearch = e.target.checked; re(); };
    $("#ex-swap").onclick = () => { [st.x, st.y] = [st.y, st.x]; re(); };
    $$("[data-pre]").forEach((b) => (b.onclick = () => {
      const P = CHART_PRESETS.find((p) => p[0] === b.dataset.pre)[2];
      Object.assign(st, JSON.parse(JSON.stringify(P)), { team: "", useSearch: false });
      if (b.dataset.pre === "luck" && D.movers && D.movers.rows.length) st.base = String(D.movers.rows[0].season);
      re();
    }));
    draw();
  }
  function draw() {
    const seasons = seasonsOf(st.base);
    const multi = seasons.length > 1;
    const q = st.q.toLowerCase();
    const minOf = (c) => { const v = c.p.min != null ? c.p.min : c.x.min; return num(v) ? v : 0; };
    const rows = population(seasons)
      .filter((c) => minOf(c) >= st.minMin && (!st.team || teamOf(c) === st.team) && (!st.useSearch || sq.filters.every((f) => pass(f, c))))
      .map((c) => ({ c, x: val(st.x.m, c.pid, c.season, st.x.o), y: val(st.y.m, c.pid, c.season, st.y.o), k: st.c ? val(st.c.m, c.pid, c.season, st.c.o) : null }))
      .filter((r) => num(r.x) && num(r.y));
    const n = rows.length;
    const cv = rows.map((r) => r.k).filter(num), cl = Math.min(...cv), ch = Math.max(...cv);
    const col = (v) => { if (!num(v)) return "rgba(111,106,92,.55)"; const t = ch > cl ? (v - cl) / (ch - cl) : 0.5;
      const a = [59, 110, 168], b = [181, 70, 60]; return `rgba(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(",")},.72)`; };
    const mx = rows.reduce((s, r) => s + r.x, 0) / Math.max(n, 1), my = rows.reduce((s, r) => s + r.y, 0) / Math.max(n, 1);
    const sxy = rows.reduce((s, r) => s + (r.x - mx) * (r.y - my), 0), sxx = rows.reduce((s, r) => s + (r.x - mx) ** 2, 0), syy = rows.reduce((s, r) => s + (r.y - my) ** 2, 0);
    const slope = sxx > 0 ? sxy / sxx : 0, r = sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
    // residual: vs y = x when that line is on (actual minus expected), else vs the trend line
    const res = rows.map((rw) => (st.ident ? rw.y - rw.x : rw.y - (my + slope * (rw.x - mx))));
    const ord = rows.map((_, i) => i).sort((a, b) => Math.abs(res[b]) - Math.abs(res[a]));
    const lbl = new Set(st.labels ? ord.slice(0, multi ? 12 : 10) : []);
    if (st.labels) ["x", "y"].forEach((k) => { const s = rows.map((_, i) => i).sort((a, b) => rows[b][k] - rows[a][k]); s.slice(0, 2).forEach((i) => lbl.add(i)); });
    const hit = (rw) => q && nameOf(rw.c).toLowerCase().includes(q);
    rows.forEach((rw, i) => { if (hit(rw)) lbl.add(i); });
    const xs = rows.map((rw) => rw.x), ys = rows.map((rw) => rw.y);
    const lo = Math.min(...xs), hi = Math.max(...xs);
    const ds = [{ type: "scatter", data: rows.map((rw) => ({ x: rw.x, y: rw.y })), pointRadius: rows.map((rw) => (hit(rw) ? 7 : n > 1500 ? 2.5 : 4)),
                  pointBackgroundColor: rows.map((rw) => (hit(rw) ? "#141310" : col(rw.k))), pointBorderWidth: 0 }];
    if (st.trend && n > 2) ds.push({ type: "line", data: [{ x: lo, y: my + slope * (lo - mx) }, { x: hi, y: my + slope * (hi - mx) }],
                                      borderColor: "rgba(31,92,67,.75)", pointRadius: 0, borderWidth: 1.5 });
    if (st.ident && n > 0) { const a = Math.min(lo, ...ys), b = Math.max(hi, ...ys);
      ds.push({ type: "line", data: [{ x: a, y: a }, { x: b, y: b }], borderColor: "rgba(20,19,16,.45)", borderDash: [5, 4], pointRadius: 0, borderWidth: 1.2 }); }
    const nm = (rw) => nameOf(rw.c) + (multi ? ` ${String(rw.c.season).slice(2)}` : "");
    const labelPlugin = { id: "exlabels", afterDatasetsDraw(c) {
      const g = c.ctx, meta = c.getDatasetMeta(0); g.save(); g.font = "11px Arial, Helvetica, sans-serif"; g.fillStyle = "#141310";
      lbl.forEach((i) => { const pt = meta.data[i]; if (pt) g.fillText(nm(rows[i]), pt.x + 6, pt.y - 5); }); g.restore(); } };
    if (chart) chart.destroy();
    const XL = mlabel(st.x.m, st.x.o), YL = mlabel(st.y.m, st.y.o), CL = st.c ? mlabel(st.c.m, st.c.o) : "";
    chart = new Chart($("#ex-canvas"), {
      data: { datasets: ds }, plugins: [labelPlugin],
      options: { maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false }, tooltip: { filter: (it) => it.datasetIndex === 0, callbacks: { label: (it) => {
          const rw = rows[it.dataIndex];
          return `${nm(rw)} (${teamOf(rw.c)}): ${XL} ${show(st.x.m, rw.x)}, ${YL} ${show(st.y.m, rw.y)}${st.c ? `, ${CL} ${show(st.c.m, rw.k)}` : ""}`; } } } },
        scales: { x: { title: { display: true, text: XL + (M[st.x.m] && M[st.x.m].kind === "pct" ? " (0-1)" : "") } },
                  y: { title: { display: true, text: YL + (M[st.y.m] && M[st.y.m].kind === "pct" ? " (0-1)" : "") } } },
        onClick: (e, els) => { const el = els.find((x) => x.datasetIndex === 0); if (el) openP(rows[el.index].c.pid); } } });
    const mae = n ? rows.reduce((s, rw) => s + Math.abs(rw.y - rw.x), 0) / n : null, bias = n ? my - mx : null;
    $("#ex-stats").innerHTML = n ? `${n.toLocaleString()} player-season${n === 1 ? "" : "s"}${multi ? ` across ${lab(seasons[0])} to ${lab(seasons[seasons.length - 1])}` : ` in ${lab(seasons[0])}`}.
      Correlation r = <b>${fmt(r, 3)}</b> (r&sup2; ${fmt(r * r, 2)}); trend slope ${fmt(slope, 3)}.
      ${st.ident ? `Mean absolute gap |y - x| <b>${show(st.y.m, mae)}</b>; average y - x ${show(st.y.m, bias, true)}.` : ""}
      ${PROJ_FROM <= Math.max(...seasons) ? ` Seasons from ${lab(PROJ_FROM)} are projections.` : ""}`
      + xvNote() : `No players have both values. Expected box-score values start in the first season with three prior seasons of data; actual BOOKER / WAA do not exist for projection seasons.`;
    const tbl = (sel, list) => {
      $(`${sel} thead`).innerHTML = `<tr><th class="col-text">Player</th>${multi ? "<th>Season</th>" : ""}<th>X</th><th>Y</th><th>${st.ident ? "Y - X" : "vs trend"}</th></tr>`;
      $(`${sel} tbody`).innerHTML = list.map((i) => { const rw = rows[i];
        return `<tr class="clickable" data-pid="${rw.c.pid}"><td class="col-text"><b>${esc(nameOf(rw.c))}</b> <small class="muted">${teamOf(rw.c)}</small></td>${multi ? `<td>${lab(rw.c.season)}</td>` : ""}
          <td>${show(st.x.m, rw.x)}</td><td>${show(st.y.m, rw.y)}</td><td class="${res[i] > 0 ? "pos" : "neg"}">${show(st.y.m, res[i], true)}</td></tr>`; }).join("");
      $$(`${sel} tbody tr`).forEach((tr) => (tr.onclick = () => openP(tr.dataset.pid)));
    };
    const byRes = rows.map((_, i) => i).sort((a, b) => res[b] - res[a]);
    $("#ex-h-up").textContent = st.ident ? "Most above the line (Y beat X)" : "Most above the trend";
    $("#ex-h-dn").textContent = st.ident ? "Most below the line (Y fell short of X)" : "Most below the trend";
    tbl("#ex-up", byRes.slice(0, 20)); tbl("#ex-dn", byRes.slice(-20).reverse());
  }

  // walk-forward backtest of the expected metric on the chart (vs "same as last season")
  function xvNote() {
    const P = D.xva && D.xva.validation && D.xva.validation.pooled; if (!P) return "";
    const ex = [st.x, st.y].find((a) => a && M[a.m] && M[a.m].group === "Expected" && a.m[0] === "x");
    const v = ex && P[fam(ex.m)]; if (!v) return "";
    const k = fam(ex.m), pct = M[k] && M[k].kind === "pct", f = (x) => (pct ? fmt(100 * x, 1) + " pp" : fmt(x, Math.abs(x) >= 100 ? 0 : 2));
    return `<br>Backtest of ${esc(M[ex.m].label)}, 2018-19 to 2025-26 (players with 500+ minutes, ${v.n.toLocaleString()} player-seasons):
      mean absolute error <b>${f(v.mae)}</b> vs ${f(v.naive_mae)} for simply repeating last season
      (${v.mae < v.naive_mae ? `${fmt(100 * (1 - v.mae / v.naive_mae), 0)}% better` : `${fmt(100 * (v.mae / v.naive_mae - 1), 0)}% worse`}).
      ${k === "booker" || k === "waa" ? "The expectation is the PROJ forecast, which is built to predict isolated impact and pulls players toward average; the displayed score is half skill-grade and stickier, so last season's number tracks it more closely." : ""}`;
  }

  /* ================================================================ movers */
  const mst = { minMin: 1000, ageLo: 18, ageHi: 40, tier: false, n: 25 };
  function renderMovers() {
    const MV = D.movers;
    if (!MV) { $("#ex-body").innerHTML = `<p class="note">Movers data not available in this build.</p>`; return; }
    const V = MV.validation || {};
    const rows = MV.rows.filter((r) => r.min >= mst.minMin && (r.age == null || (r.age >= mst.ageLo && r.age <= mst.ageHi)));
    const tierOf = (c) => (c < -1 ? 0 : c < 1 ? 1 : c < 3 ? 2 : 3);
    if (mst.tier) {      // rank WITHIN starting tier: projected change minus the tier's mean projected change
      const mean = [0, 1, 2, 3].map((k) => { const g = rows.filter((r) => tierOf(r.cur) === k); return g.reduce((s, r) => s + r.raw, 0) / Math.max(g.length, 1); });
      rows.forEach((r) => (r.key = r.raw - mean[tierOf(r.cur)]));
    } else rows.forEach((r) => (r.key = r.raw));
    const up = rows.slice().sort((a, b) => b.key - a.key).slice(0, mst.n), dn = rows.slice().sort((a, b) => a.key - b.key).slice(0, mst.n);
    const live = (pid) => { const p = PL[pid + ":" + LIVE]; return p && p.live ? p.live : null; };
    const tbl = (list) => `<div class="table-scroll"><table class="grid"><thead><tr><th class="col-text">Player</th><th>Age</th><th>Min</th><th>Rating ${lab(list[0] ? list[0].season : LAST_DONE)}</th>
      <th>PROJ next</th><th>Raw change</th><th>Expected change</th><th>P(improve)</th><th>2nd-half form</th><th>On/off gap</th><th>3P luck</th><th>Live: change since opener</th></tr></thead><tbody>` +
      list.map((r) => { const L = live(r.pid); const dl = L && num(L.ratingNow) && num(L.ratingOpen) ? L.ratingNow - L.ratingOpen : null;
        return `<tr class="clickable ex-row" data-pid="${r.pid}"><td class="col-text"><b>${esc(r.n)}</b> <small class="muted">${r.t}</small></td><td>${fmt(r.age, 0)}</td><td>${r.min}</td>
        <td>${sgn(r.cur)}</td><td>${sgn(r.proj)}</td><td>${sgn(r.raw)}</td><td class="${r.exp > 0 ? "pos" : "neg"}"><b>${sgn(r.exp)}</b></td><td>${fmt(100 * r.p, 0)}%</td>
        <td class="${r.form > 1 ? "pos" : r.form < -1 ? "neg" : ""}">${sgn(r.form, 1)}</td><td class="${r.onoff > 3 ? "pos" : r.onoff < -3 ? "neg" : ""}">${sgn(r.onoff, 1)}</td>
        <td class="${r.luck3 != null && r.luck3 > 0.03 ? "neg" : r.luck3 != null && r.luck3 < -0.03 ? "pos" : ""}">${r.luck3 == null ? "—" : sgn(100 * r.luck3, 1) + " pp"}</td>
        <td>${dl == null ? "—" : `${sgn(dl)} <small class="muted">${L.gp} GP</small>`}</td></tr>`; }).join("") + `</tbody></table></div>`;
    const dec = (V.deciles || []).map((d) => `<tr><td>${d.dec}</td><td>${sgn(d.pred)}</td><td><b>${sgn(d.real)}</b></td><td>${fmt(100 * d.improved, 0)}%</td></tr>`).join("");
    $("#ex-body").innerHTML = `
      <p class="note">Ranked by BOOKER's projected change next season (PROJ minus the current rating; PROJ is trained only on earlier seasons).
        <b>Expected change</b> maps that projection to what actually happened historically (isotonic fit, ${V.seasons ? V.seasons[0] + "-" + V.seasons[V.seasons.length - 1] : ""}):
        raw projected rises run about 2-3x larger than what happened, and the biggest projected declines (mostly elite players) barely materialised,
        so read the expected column. Drivers: second-half form, on-court +/- above the isolated rating,
        and 3P luck (raw minus true 3P% on volume; red = shooting above his true rate). "Within tier" ranks each player against others who started at a similar rating,
        removing plain regression of extreme ratings.</p>
      <div class="controls">
        <label>Min minutes <select id="mv-min">${[500, 1000, 1500, 2000].map((m) => `<option ${m === mst.minMin ? "selected" : ""}>${m}</option>`).join("")}</select></label>
        <label>Age <input id="mv-lo" type="number" value="${mst.ageLo}" style="width:50px"> to <input id="mv-hi" type="number" value="${mst.ageHi}" style="width:50px"></label>
        <label><input type="checkbox" id="mv-tier" ${mst.tier ? "checked" : ""}> rank within starting tier</label>
        <label>Show <select id="mv-n">${[10, 25, 50].map((m) => `<option ${m === mst.n ? "selected" : ""}>${m}</option>`).join("")}</select></label>
      </div>
      <h3 class="sub-head">Progression candidates</h3>${tbl(up)}
      <h3 class="sub-head">Regression candidates</h3>${tbl(dn)}
      <h3 class="sub-head">How good is this? (walk-forward, ${V.n || 0} player-seasons with 1,000+ minutes in both years)</h3>
      <p class="note">Spearman of projected vs realized change: <b>${fmt(V.rho_rating, 3)}</b> (BOOKER rating). Top-25 progression candidates improved
        <b>${fmt(100 * V.top_improved, 0)}%</b> of the time (base rate ${fmt(100 * V.base_improved, 0)}%), average realized change ${sgn(V.top_mean_change)};
        bottom-25 declined ${fmt(100 * V.bottom_declined, 0)}% (${sgn(V.bottom_mean_change)}). Within starting tiers the signal is weaker
        (${(V.by_tier || []).map((t) => `${t.tier}: ${fmt(t.rho, 2)}`).join(", ")}): much of it is regression of extreme ratings.
        3P luck predicts a drop in the <i>rating</i> (${V.luck3_real_delta ? fmt(V.luck3_real_delta.coef_per_10pct, 2) + " per 10 pp, t " + fmt(V.luck3_real_delta.t, 1) : ""})
        but not in pure on-court impact (t ${V.luck3_pure_delta ? fmt(V.luck3_pure_delta.t, 1) : ""}), so it flags shooting-inflated ratings rather than true decline.</p>
      <div class="table-scroll"><table class="grid"><thead><tr><th>Decile of projected change</th><th>Projected</th><th>Realized</th><th>Improved</th></tr></thead><tbody>${dec}</tbody></table></div>`;
    $("#mv-min").onchange = (e) => { mst.minMin = +e.target.value; renderMovers(); };
    $("#mv-lo").onchange = (e) => { mst.ageLo = +e.target.value; renderMovers(); };
    $("#mv-hi").onchange = (e) => { mst.ageHi = +e.target.value; renderMovers(); };
    $("#mv-tier").onchange = (e) => { mst.tier = e.target.checked; renderMovers(); };
    $("#mv-n").onchange = (e) => { mst.n = +e.target.value; renderMovers(); };
    $$(".ex-row").forEach((tr) => (tr.onclick = () => openP(tr.dataset.pid)));
  }

  /* ================================================================ props */
  function lgamma(x) {       // Lanczos
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
      12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    x -= 1; let a = c[0]; const t = x + g + 0.5; for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  function nbCdf(y, mu, k) { let s = 0; for (let j = 0; j <= y; j++) s += Math.exp(lgamma(j + k) - lgamma(k) - lgamma(j + 1) + k * Math.log(k / (k + mu)) + j * Math.log(mu / (k + mu))); return Math.min(1, s); }
  function ncdf(x) { const t = 1 / (1 + 0.2316419 * Math.abs(x)); const d = 0.3989423 * Math.exp(-x * x / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return x > 0 ? 1 - p : p; }
  function pOver(line, mu, dist) {
    if (dist[0] === "normal") { const sd = dist[1] * Math.pow(Math.max(mu, 0.3), dist[2]); return 1 - ncdf((line - mu) / sd); }
    return 1 - nbCdf(Math.floor(line), Math.max(mu, 1e-6), dist[1]);
  }
  function quant(q, mu, dist) { for (let y = 0; y < 200; y++) if (1 - pOver(y + 0.5, mu, dist) >= q) return y; return 200; }
  const SL = { pts: "Points", reb: "Rebounds", ast: "Assists", tpm: "3PM", stl: "Steals", blk: "Blocks", tov: "Turnovers", pra: "Pts+Reb+Ast" };
  const pst = { date: null, gid: "", stat: "pts", lines: {} };
  function renderProps() {
    const PR = D.props;
    if (!PR) { $("#ex-body").innerHTML = `<p class="note">Prop projections are not available in this build.</p>`; return; }
    const C = Object.fromEntries(PR.cols.map((c, i) => [c, i]));
    const dates = Array.from(new Set(PR.rows.map((r) => r[C.date]))).sort();
    if (!pst.date || !dates.includes(pst.date)) pst.date = dates[0];
    const day = PR.rows.filter((r) => r[C.date] === pst.date);
    const games = Array.from(new Map(day.map((r) => [r[C.gid], r])).values());
    const V = PR.validation, vs = V.stats[pst.stat] || {}, dist = PR.dist[pst.stat];
    const rows = day.filter((r) => !pst.gid || String(r[C.gid]) === pst.gid).sort((a, b) => b[C[pst.stat]] - a[C[pst.stat]]);
    $("#ex-body").innerHTML = `
      <p class="note">Per-game projections conditional on the player playing (books void DNPs). Rates start from the preseason projection and update
        with every game played; possessions blend projected minutes with minutes so far and adjust for both teams' pace. Enter any line to get
        P(over) / P(under) from the fitted ${dist[0] === "nb" ? "negative-binomial" : "normal"} distribution. No sportsbook lines are stored or shown.
        Players listed OUT on the latest injury report are removed; Q / D are flagged.</p>
      <div class="controls">
        <label>Date <select id="pp-date">${dates.map((d) => `<option ${d === pst.date ? "selected" : ""}>${d}</option>`).join("")}</select></label>
        <label>Game <select id="pp-gid"><option value="">All games</option>${games.map((g) => `<option value="${g[C.gid]}" ${String(g[C.gid]) === pst.gid ? "selected" : ""}>${g[C.team]} vs ${g[C.opp]}</option>`).join("")}</select></label>
        <label>Stat <select id="pp-stat">${Object.entries(SL).map(([k, l]) => `<option value="${k}" ${k === pst.stat ? "selected" : ""}>${l}</option>`).join("")}</select></label>
      </div>
      <div class="table-scroll"><table class="grid"><thead><tr><th class="col-text">Player</th><th>Matchup</th><th>MIN</th><th>${SL[pst.stat]}</th><th>10-90% range</th>
        <th>Line</th><th>P(over)</th><th>P(under)</th><th class="col-text">Pts / Reb / Ast</th></tr></thead><tbody>
      ${rows.map((r) => { const mu = r[C[pst.stat]], key = r[C.gid] + ":" + r[C.pid] + ":" + pst.stat;
        const line = pst.lines[key] != null ? pst.lines[key] : Math.floor(mu) + 0.5; const po = pOver(line, mu, dist);
        return `<tr><td class="col-text"><b class="ex-pl" data-pid="${r[C.pid]}" style="cursor:pointer">${esc(r[C.name])}</b>${r[C.status] ? ` <span class="neg">${esc(r[C.status]).toUpperCase().slice(0, 1)}</span>` : ""}</td>
          <td>${r[C.team]} v ${r[C.opp]}</td><td>${fmt(r[C.min], 1)}</td><td><b>${fmt(mu, 1)}</b></td><td>${quant(0.1, mu, dist)}-${quant(0.9, mu, dist)}</td>
          <td><input class="pp-line" data-k="${key}" type="number" step="0.5" value="${line}" style="width:62px"></td>
          <td class="${po > 0.55 ? "pos" : ""}">${fmt(100 * po, 1)}%</td><td class="${po < 0.45 ? "pos" : ""}">${fmt(100 * (1 - po), 1)}%</td>
          <td class="col-text"><small class="muted">${r[C.pts].toFixed(1)} / ${r[C.reb].toFixed(1)} / ${r[C.ast].toFixed(1)}</small></td></tr>`; }).join("")}
      </tbody></table></div>
      <h3 class="sub-head">Backtest (${lab(V.season)}, ${V.n_test} held-out second-half player-games)</h3>
      <p class="note">${SL[pst.stat]}: mean absolute error <b>${fmt(vs.mae_model, 2)}</b> vs ${fmt(vs.mae_last_season, 2)} for last season's per-game average and
        ${fmt(vs.mae_season_to_date, 2)} for the season-to-date average${V.early && V.early[pst.stat] ? ` (first 20 games: ${fmt(V.early[pst.stat].mae_model, 2)} vs ${fmt(V.early[pst.stat].mae_last_season, 2)} / ${fmt(V.early[pst.stat].mae_season_to_date, 2)})` : ""};
        bias ${sgn(vs.bias, 2)}; 80% range covered ${fmt(100 * vs.cov80, 0)}% of outcomes. At the line nearest the projection, average P(over) ${fmt(100 * vs.p_over_mean, 1)}% vs a real over rate of ${fmt(100 * vs.over_rate, 1)}%
        ${pst.stat === "pts" || pst.stat === "pra" ? " -- for points and PRA, overs near the projection run about 4 points more often than this model says" : ""}.
        Calibration [predicted, actual, n]: ${(vs.calibration || []).map((c) => `[${fmt(100 * c[0], 0)}%, ${fmt(100 * c[1], 0)}%, ${c[2]}]`).join(" ")}.
        Near the projection every prop is close to a coin flip; the model's value is in fair prices for lines set away from it.</p>`;
    $("#pp-date").onchange = (e) => { pst.date = e.target.value; pst.gid = ""; renderProps(); };
    $("#pp-gid").onchange = (e) => { pst.gid = e.target.value; renderProps(); };
    $("#pp-stat").onchange = (e) => { pst.stat = e.target.value; renderProps(); };
    $$(".pp-line").forEach((el) => (el.onchange = () => { pst.lines[el.dataset.k] = +el.value; renderProps(); }));
    $$(".ex-pl").forEach((b) => (b.onclick = () => openP(b.dataset.pid)));
  }

  window.renderExplore = render;
  // deep link: #explore=... opens the Explore tab with that search / chart
  if (loadHash()) window.addEventListener("load", () => { const t = document.querySelector('.tab[data-view="explore"]'); if (t) t.click(); });
})();
