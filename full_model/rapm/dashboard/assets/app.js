/* BOOKER WAA dashboard ---------------------------------------------------- */
(function () {
  "use strict";
  const D = window.BOOKER;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const seasonLabel = (s) => `${s - 1}-${String(s).slice(2)}`;
  const fmt = (v, d = 1) => (v == null || isNaN(v) ? "\u2013" : Number(v).toFixed(d));
  const signed = (v, d = 1) => (v > 0 ? "+" : "") + fmt(v, d);
  const money = (v) => {
    if (v == null || isNaN(v)) return "\u2013";
    const n = Math.round(v), sign = n < 0 ? "-" : "", a = Math.abs(n);
    if (a >= 1e6) return sign + "$" + (a / 1e6).toFixed(2) + "M";
    return sign + "$" + a.toLocaleString();
  };
  const teamName = (ab) => D.teamNames[ab] || ab;
  const MAX_TRADE_ASSETS = 6;
  const latest = Math.max.apply(null, D.seasons);
  // last full (non-projection) season -- the leaderboard's default view
  const _fullSeasons = D.players.filter((p) => !p.predictive).map((p) => p.season);
  const lastFull = _fullSeasons.length ? Math.max.apply(null, _fullSeasons) : latest;
  // Every season dropdown defaults to the last *played* season (not a projection
  // year and not "All"). Reads the select's own numeric options; falls back to the
  // max available season if lastFull isn't among them.
  function defaultSeason(sel) {
    const nums = Array.from(sel.options).map((o) => Number(o.value)).filter((n) => !isNaN(n));
    if (!nums.length) return;
    sel.value = String(nums.includes(lastFull) ? lastFull : Math.max.apply(null, nums));
  }

  /* ---- savant-style percentile color + skill / value rendering --------- */
  // diverging ramp: low percentile = cool blue, mid = grey, high = warm red
  function pctColor(p, alpha) {
    p = Math.max(0, Math.min(100, p));
    const lo = [60, 110, 165], mid = [200, 196, 168], hi = [165, 42, 38];
    const lerp = (a, b, t) => Math.round(a + (b - a) * t);
    let c;
    if (p < 50) { const t = p / 50; c = lo.map((v, i) => lerp(v, mid[i], t)); }
    else { const t = (p - 50) / 50; c = mid.map((v, i) => lerp(v, hi[i], t)); }
    return alpha == null ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
  }
  const hasSkills = (p) => !!(p && p.skills && Object.keys(p.skills).length > 0);
  // true-skill value formatting: "pct" skills store a 0-1 rate (shown x100); others a number
  function fmtSkillVal(s) {
    if (s == null || s.val == null) return "—";
    return s.fmt === "pct" ? (s.val * 100).toFixed(1) + "%" : signed(s.val, 1);
  }
  function fmtSkillRaw(s) {
    if (s == null || s.raw == null) return null;
    return s.fmt === "pct" ? (s.raw * 100).toFixed(1) + "%" : fmt(s.raw, 1);
  }
  function skillTitle(s) {
    const r = fmtSkillRaw(s);
    return `${s.pct}th pctile` + (r != null ? ` · raw ${r}` : "");
  }
  function renderSkillProfile(row) {
    const note = $("#skill-season-note");
    if (!hasSkills(row)) {
      if (note) note.textContent = "";
      $("#skill-bars").innerHTML = `<p class="result-note">No true-skill profile for this player-season (needs 500+ minutes).</p>`;
      return;
    }
    if (note) note.textContent = `${seasonLabelFull(row.season)}`;
    const order = ["offense", "defense", "three_pct", "rim_finish", "efficiency", "shot_making",
                   "free_throw", "self_creation", "off_gravity", "on_gravity", "playmaking",
                   "creation", "foul_draw", "ball_security", "rebounding", "steals", "rim_protect",
                   "discipline", "rim_contest", "perimeter_contest", "rim_deterrence",
                   "make_limiting", "shot_difficulty", "usage"];
    const rows = order.filter((k) => row.skills[k]).map((k) => {
      const s = row.skills[k];
      return `<div class="skill-row" title="${skillTitle(s)}">` +
        `<span class="skill-name">${s.label}</span>` +
        `<span class="skill-track"><span class="skill-fill" style="width:${s.pct}%;background:${pctColor(s.pct)}"></span></span>` +
        `<span class="skill-val">${fmtSkillVal(s)}</span>` +
        `<span class="skill-pct" style="color:${pctColor(s.pct)}">${s.pct}</span>` +
        `</div>`;
    }).join("");
    $("#skill-bars").innerHTML = rows;
  }
  // ---- head-to-head player comparison (full true-skill profile) ----
  const SKILL_ORDER = ["offense", "defense", "three_pct", "rim_finish", "efficiency",
    "shot_making", "free_throw", "self_creation", "off_gravity", "on_gravity", "playmaking",
    "creation", "foul_draw", "ball_security", "rebounding", "steals", "rim_protect",
    "discipline", "rim_contest", "perimeter_contest", "rim_deterrence", "make_limiting",
    "shot_difficulty", "usage"];
  function profileSeasonFor(pid) {
    const ss = D.players.filter((p) => p.pid === pid).sort((a, b) => a.season - b.season);
    if (!ss.length) return null;
    const rev = [...ss].reverse();
    return rev.find((s) => hasSkills(s) && s.skills.playmaking) || rev.find((s) => hasSkills(s)) || ss[ss.length - 1];
  }
  // shared searchable player-name index (compare box + trajectory adder)
  let _cmpByName = null;
  function ensureNameIndex() {
    if (_cmpByName) return _cmpByName;
    _cmpByName = {};
    const seen = {};
    D.players.forEach((p) => {
      if (!p.player) return;
      _cmpByName[p.player.toLowerCase()] = p.pid;
      seen[p.player] = 1;
    });
    const dl = $("#cmp-datalist");
    if (dl) dl.innerHTML = Object.keys(seen).sort().map((nm) => `<option value="${nm}"></option>`).join("");
    return _cmpByName;
  }
  let cmpInit = false, curProfRow = null;
  function setupCompare(rowA) {
    curProfRow = rowA;
    const sel = $("#skill-compare");
    if (!sel) return;
    if (!cmpInit) {
      cmpInit = true;
      ensureNameIndex();  // searchable input + datalist (a 1100-option select is unusable)
      const go = () => {
        const q = sel.value.trim().toLowerCase();
        if (!q || !curProfRow) { renderSkillProfile(curProfRow); return; }
        const bpid = _cmpByName[q];
        if (bpid == null) return;                       // keep typing
        const rowB = profileSeasonFor(bpid);
        rowB ? renderCompareProfile(curProfRow, rowB) : renderSkillProfile(curProfRow);
      };
      sel.addEventListener("change", go);
      sel.addEventListener("search", go);               // clearing the field resets
    }
    sel.value = "";
  }
  function renderCompareProfile(a, b) {
    const note = $("#skill-season-note");
    if (note) note.textContent = `${a.player} (${seasonLabel(a.season)}) vs ${b.player} (${seasonLabel(b.season)}) · true-skill, winner bold`;
    const gradeRow = (a.grade != null && b.grade != null) ?
      `<div class="cmp-row cmp-grade"><span class="cmp-a ${a.grade >= b.grade ? "win" : ""}">${a.gradeLetter} ${a.grade}</span>` +
      `<span class="cmp-lab">GRADE</span><span class="cmp-b ${b.grade >= a.grade ? "win" : ""}">${b.gradeLetter} ${b.grade}</span></div>` : "";
    const rows = SKILL_ORDER.filter((k) => (a.skills && a.skills[k]) || (b.skills && b.skills[k])).map((k) => {
      const sa = a.skills && a.skills[k], sb = b.skills && b.skills[k];
      const lab = (sa || sb).label;
      const pa = sa ? sa.pct : -1, pb = sb ? sb.pct : -1;
      const aw = sa && sb && pa >= pb, bw = sa && sb && pb >= pa;
      return `<div class="cmp-row">` +
        `<span class="cmp-a ${aw ? "win" : ""}" style="color:${sa ? pctColor(pa) : "#bbb"}" title="${sa ? skillTitle(sa) : "—"}">${sa ? fmtSkillVal(sa) : "—"}</span>` +
        `<span class="cmp-lab">${lab}</span>` +
        `<span class="cmp-b ${bw ? "win" : ""}" style="color:${sb ? pctColor(pb) : "#bbb"}" title="${sb ? skillTitle(sb) : "—"}">${sb ? fmtSkillVal(sb) : "—"}</span></div>`;
    }).join("");
    $("#skill-bars").innerHTML =
      `<div class="cmp-head"><span>${a.player}</span><span></span><span>${b.player}</span></div>` + gradeRow + rows;
  }
  function vbar(label, val, max, color, fmtFn) {
    const w = Math.max(0, Math.min(100, (Math.abs(val) / max) * 100));
    return `<div class="vderiv-row"><span class="vderiv-k">${label}</span>` +
      `<span class="vderiv-track"><span class="vderiv-fill" style="width:${w}%;background:${color}"></span></span>` +
      `<span class="vderiv-v">${fmtFn(val)}</span></div>`;
  }
  function renderValueDerivation(row) {
    if (!row) { $("#value-derivation").innerHTML = ""; return; }
    const off = row.waaOff != null ? row.waaOff : 0;
    const def = row.waaDef != null ? row.waaDef : 0;
    const tot = row.waaModel != null ? row.waaModel : row.waa;
    const booker = row.bookerScore;
    const tv = row.trueValue != null ? row.trueValue : row.fairAav2026;
    const mkt = row.marketAav2026;
    const wmax = Math.max(2, Math.abs(off), Math.abs(def), Math.abs(tot));
    let html = `<div class="vderiv-sec">Two-way value (${seasonLabelFull(row.season)})</div>`;
    html += vbar("Offense WAA", off, wmax, pctColor(off >= 0 ? 72 : 28), (v) => signed(v, 1));
    html += vbar("Defense WAA", def, wmax, pctColor(def >= 0 ? 72 : 28), (v) => signed(v, 1));
    html += vbar("Total WAA", tot, wmax, COL.ink, (v) => signed(v, 1));
    if (booker != null) {
      html += `<div class="vderiv-sec">Skill rate &amp; value</div>`;
      html += `<div class="vderiv-line"><span>BOOKER <em>(+/- per 100 poss, avg starting-caliber context, usage → optimum)</em></span><b>${signed(booker, 1)}</b></div>`;
    }
    if (tv != null) {
      html += `<div class="vderiv-line"><span>True Value <em>(skill-based fair AAV)</em></span><b>${money(tv)}</b></div>`;
      if (mkt != null) {
        html += `<div class="vderiv-line"><span>Actual contract</span><b>${money(mkt)}</b></div>`;
        const surp = tv - mkt;
        html += `<div class="vderiv-line"><span>Surplus</span><b class="${surp >= 0 ? "pos" : "neg"}">${money(surp)}</b></div>`;
      }
    }
    if (row.role) {
      const rl = row.role;
      const over = rl.misuse > 1.5, under = rl.misuse < -1.5;
      const verdict = over ? "over-used — worth more in a smaller, more selective role"
        : under ? "under-used — efficient enough to carry more"
        : "used about right";
      html += `<div class="vderiv-sec">Usage fit <em>(skill curve — what he'd be worth used properly)</em></div>`;
      html += `<div class="vderiv-line"><span>Usage now → optimal</span><b>${rl.usage.toFixed(0)}% → ${rl.optUsage.toFixed(0)}%</b></div>`;
      html += `<div class="vderiv-line"><span>TS% at that role</span><b>${(rl.tsNow * 100).toFixed(1)} → ${(rl.tsOpt * 100).toFixed(1)}</b></div>`;
      html += `<div class="vderiv-line"><span class="${over ? "neg" : under ? "pos" : ""}">${verdict}</span>` +
        `<b class="${rl.upside >= 0 ? "pos" : "neg"}">${signed(rl.upside, 1)}/100</b></div>`;
    }
    if (row.clutch) {
      const c = row.clutch, d = 100 * (c.tsC - c.tsN);
      const read = d > 1 ? "rises when it tightens" : d < -5 ? "big efficiency drop under the heaviest defense"
        : "league-typical dip (primary options absorb the hardest shots)";
      html += `<div class="vderiv-sec">Crunch time <em>(last 5 min of a ≤5-pt game, pooled 2018-25)</em></div>`;
      html += `<div class="vderiv-line"><span>Clutch volume</span><b>${c.tsaC} true-shot att</b></div>`;
      html += `<div class="vderiv-line"><span>TS% normal → clutch</span>` +
        `<b>${(100 * c.tsN).toFixed(1)} → ${(100 * c.tsC).toFixed(1)} <span class="${d >= 0 ? "pos" : "neg"}">(${d >= 0 ? "+" : ""}${d.toFixed(1)})</span></b></div>`;
      html += `<div class="vderiv-line"><span>${read}</span><b title="league average clutch TS delta is −1.4; the game itself gets ~5 pts/100 harder">lg −1.4</b></div>`;
    }
    $("#value-derivation").innerHTML = html;
  }
  // skill percentile trajectories across a player's seasons
  const TRAJ_KEYS = [["offense", "Offense", "#7a2820"], ["defense", "Defense", "#2f5d34"],
    ["shot_making", "Shot-Making", "#241c12"], ["efficiency", "True eFG%", "#b8860b"],
    ["three_pct", "True 3P%", "#4a6fa5"], ["playmaking", "Playmaking", "#8a5a44"]];
  function renderSkillTrajectory(seasons) {
    destroy("skilltraj");
    const ws = seasons.filter((s) => hasSkills(s));
    const panel = $("#skill-traj-panel");
    if (ws.length < 2) { if (panel) panel.style.display = "none"; return; }
    if (panel) panel.style.display = "";
    const labels = ws.map((s) => seasonLabelFull(s.season));
    const datasets = TRAJ_KEYS.map(([k, lab, c]) => ({
      label: lab, data: ws.map((s) => (s.skills[k] ? s.skills[k].pct : null)),
      borderColor: c, backgroundColor: c, borderWidth: 2, pointRadius: 2, tension: .25, spanGaps: true,
    }));
    charts.skilltraj = new Chart($("#skill-traj-chart"), {
      type: "line", data: { labels, datasets },
      options: { maintainAspectRatio: false,
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, usePointStyle: true } } },
        scales: { y: { min: 0, max: 100, title: { display: true, text: "percentile vs league" }, grid: { color: COL.grid } },
          x: { grid: { display: false } } } },
    });
  }

  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 12;
  Chart.defaults.color = "#6f6044";
  const COL = { ink: "#241c12", accent: "#7a2820", pos: "#2f5d34", neg: "#7a2820",
               grey: "#8a785a", grid: "#ddcca8", paper: "#f4ecd6" };
  const seasonLabelFull = (s) => `${s - 1}-${String(s).slice(2)}`;

  /* ---- view router ----------------------------------------------------- */
  const charts = {};
  function destroy(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }
  function showView(name) {
    $$(".view").forEach((v) => (v.hidden = v.id !== "view-" + name));
    $$(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.view === name));
    window.scrollTo({ top: 0 });
  }
  $("#tabs").addEventListener("click", (e) => {
    const t = e.target.closest(".tab"); if (!t) return;
    const v = t.dataset.view;
    showView(v);
    if (v === "forecast") renderForecast();
    if (v === "trade") renderTrade();
    if (v === "lineup") renderLineup();
    if (v === "trajectory") renderTrajectory();
    if (v === "gameodds") renderGameOdds();
    if (v === "leaders") renderLeaders();
    if (v === "matchups") renderMatchup();
    if (v === "diagnostics") renderDiagnostics();
    if (v === "method") renderMethod();
  });

  /* ===================================================================== *
   *  LEADERBOARD
   * ===================================================================== */
  // BOOKER cell with a VISIBLE credible band: scale the rating's sd (impact/100) into
  // BOOKER units (+/- per 100) via the shared scale constant, so a low-minute, prior-
  // driven estimate (wide band) reads differently from a rock-solid one. Flags rows whose
  // band exceeds half their value as LOW CONFIDENCE (e.g. a 651-min Gafford vs Jokic).
  let _bkScale = null;
  function bookerScale() {
    if (_bkScale != null) return _bkScale;
    const rs = D.players.filter((p) => p.bookerScore != null && p.bfTot100 != null && Math.abs(p.bfTot100) > 2)
      .map((p) => p.bookerScore / p.bfTot100).sort((a, b) => a - b);
    _bkScale = rs.length ? rs[Math.floor(rs.length / 2)] : 1.0;
    return _bkScale;
  }
  function bookerCell(r) {
    if (r.bookerScore == null) return "—";
    const band = (r.sdOff != null && r.sdDef != null) ? bookerScale() * Math.hypot(r.sdOff, r.sdDef) : null;
    const lo = band != null && band > 0.5 * Math.abs(r.bookerScore);
    const cls = (r.bookerScore >= 0 ? "pos" : "neg") + (lo ? " lo-conf" : "");
    const tip = "BOOKER: predictive +/- per 100 possessions given average starting-caliber teammates & opponents, usage regressed toward optimum."
      + (band != null ? ` ±${band.toFixed(1)} (1σ band, widens with fewer minutes)`
        + (lo ? " — LOW CONFIDENCE, small sample" : "") : "");
    return `<span class="${cls}" title="${tip}">${signed(r.bookerScore, 1)}`
      + (band != null ? `<span class="ci">±${band.toFixed(1)}</span>` : "") + "</span>";
  }
  const LB_COLS = [
    { key: "_waaRank", label: "#", text: true,
      cell: (r) => `<span class="rankcell" title="rank blends value delivered (WAA) with per-minute quality (BOOKER rate), 50/50 by season z-score">${r._waaRank != null ? r._waaRank : "—"}</span>` },
    { key: "player", label: "Player", text: true, cell: (r) => `<span class="namecell">${r.player}</span>` },
    { key: "team", label: "Team", text: true, cell: (r) => `<span class="team-tag">${r.team}</span>` },
    { key: "season", label: "Season", cell: (r) => seasonLabel(r.season) },
    { key: "grade", label: "Grade", text: true,
      cell: (r) => r.grade != null ? `<span class="grade grade-${(r.gradeLetter||"").replace("+","p").replace("-","m")}" title="sticky age-curved team-independent grade">${r.gradeLetter} <small>${r.grade}</small></span>` : "\u2014" },
    { key: "min", label: "Min", cell: (r) => r.min.toLocaleString() },
    { key: "bookerScore", label: "BOOKER", cell: (r) => bookerCell(r) },
    { key: "waaModel", label: "WAA", cell: (r) => waaBar(r) },
    { key: "waaOff", label: "Off", cell: (r) => r.waaOff != null
      ? `<span class="${r.waaOff >= 0 ? "pos" : "neg"}" title="Offensive WAA">${signed(r.waaOff, 1)}</span>` : "\u2014" },
    { key: "waaDef", label: "Def", cell: (r) => r.waaDef != null
      ? `<span class="${r.waaDef >= 0 ? "pos" : "neg"}" title="Defensive WAA">${signed(r.waaDef, 1)}</span>` : "\u2014" },
    { key: "real_pm", label: "Real +/-", cell: (r) => {
        if (r.real_pm == null) return "\u2014";
        const ctx = (r.tm_quality != null && r.opp_quality != null)
          ? ` \u2014 teammates ${signed(r.tm_quality, 1)}, opponents ${signed(r.opp_quality, 1)} (avg on-court rating)` : "";
        return `<span class="${r.real_pm >= 0 ? "pos" : "neg"}" title="Actual on-court net /100 (descriptive; for isolated impact use BOOKER)${ctx}">${signed(r.real_pm, 1)}</span>`;
      } },
    { key: "trueValue", label: "True Value", cell: (r) => (r.trueValue != null ? r.trueValue : r.fairAav2026) != null
      ? `<span title="Skill-based fair AAV: predicted from BOOKER score with the market's age penalty removed.">${money(r.trueValue != null ? r.trueValue : r.fairAav2026)}</span>` : "\u2014" },
    { key: "marketAav2026", label: "Contract", cell: (r) => r.marketAav2026 != null ? money(r.marketAav2026) : "\u2014" },
    { key: "surplus", label: "Surplus", cell: (r) => r.surplus != null
      ? `<span class="${r.surplus >= 0 ? "pos" : "neg"}" title="True Value minus actual contract">${money(r.surplus)}</span>` : "\u2014" },
  ];
  // WAA-column value: ACCUMULATED WAA for actual seasons; WAA per 1500 minutes for
  // FUTURE (projection) seasons, where minutes are speculative so a rate is fair.
  function rv(p) {
    const w = p.waaModel != null ? p.waaModel : (p.waa || 0);
    return (p.predictive && p.min > 0) ? w * 1500 / p.min : w;
  }
  // Rank (#): blend of value delivered and per-minute quality -- 0.5·z(WAA) +
  // 0.5·z(BOOKER rate) within each season. Pure accumulated WAA turns a partial
  // season into a durability contest (compilers over injured stars: Mitchell over
  // Giannis); pure rate ignores availability. The blend keeps Jokic/SGA on top and
  // restores injured stars (Giannis, Embiid, Curry) without hiding minutes.
  (function () {
    const by = {};
    D.players.forEach((p) => { (by[p.season] || (by[p.season] = [])).push(p); });
    Object.values(by).forEach((list) => {
      const zs = (vals) => {
        const m = vals.reduce((a, b) => a + b, 0) / vals.length;
        const sd = Math.sqrt(vals.reduce((a, b) => a + (b - m) * (b - m), 0) / vals.length) || 1;
        return (v) => (v - m) / sd;
      };
      const zw = zs(list.map(rv));
      const zr = zs(list.map((p) => p.bookerScore != null ? p.bookerScore : 0));
      list.forEach((p) => {
        p._rankScore = 0.5 * zw(rv(p)) + 0.5 * zr(p.bookerScore != null ? p.bookerScore : 0);
      });
      list.slice().sort((a, b) => b._rankScore - a._rankScore)
          .forEach((p, i) => { p._waaRank = i + 1; });
    });
  })();
  let maxWaa = Math.max.apply(null, D.players.map(rv));
  function waaBar(r) {
    const v = rv(r);
    const w = Math.max(0, (v / maxWaa) * 100);
    const c = v >= 0 ? "pos" : "neg";
    const per = r.predictive ? " per 1500 min (projection)" : "";
    const ci = (r.bfOff100 != null && r.sdOff != null)
      ? ` — rating/100 Off ${signed(r.bfOff100, 1)}±${(1.96 * r.sdOff).toFixed(1)}, Def ${signed(r.bfDef100, 1)}±${(1.96 * r.sdDef).toFixed(1)}` : "";
    return `<span class="bar-cell" title="WAA${per}${ci}"><span class="bar" style="width:${w}%"></span>` +
           `<span class="${c}">${signed(v, 1)}</span></span>`;
  }
  // BookerFormer offense/defense rating per 100 with a 95% credible interval.
  function bfCell(r) {
    if (r.bfOff100 == null || r.sdOff == null) return "—";
    const ci = (v, sd) => `${signed(v, 1)}<span class="ci">±${(1.96 * sd).toFixed(1)}</span>`;
    return `<span title="BookerFormer Bayesian offense / defense rating per 100 possessions, with 95% credible interval. Intervals narrow as a player logs more minutes.">` +
           `${ci(r.bfOff100, r.sdOff)} / ${ci(r.bfDef100, r.sdDef)}</span>`;
  }
  const lbSort = { key: "_waaRank", dir: 1 };   // default: blended value+rate rank (see #)

  function lbFilter() {
    const season = $("#f-season").value;
    const team = $("#f-team").value;
    const q = $("#f-search").value.trim().toLowerCase();
    const mn = +$("#f-minutes").value;
    let rows = D.players.filter((p) => p.min >= mn);
    // season filter always applies (search is scoped to the selected season; pick
    // "All seasons" to search across every year). Team is optional, so search still
    // works with no team selected.
    if (season !== "all") rows = rows.filter((p) => p.season === +season);
    if (team) rows = rows.filter((p) => p.team === team);
    if (q) rows = rows.filter((p) => (p.player || "").toLowerCase().includes(q));
    rows.sort((a, b) => {
      let A = a[lbSort.key], B = b[lbSort.key];
      if (lbSort.key === "waaModel") { A = rv(a); B = rv(b); }   // WAA (rate for projections)
      if (A == null && B == null) return 0;
      if (A == null) return 1;
      if (B == null) return -1;
      const c = typeof A === "string" ? A.localeCompare(B) : A - B;
      return c * lbSort.dir;
    });
    return rows;
  }
  function renderLB() {
    const rows = lbFilter();
    const thead = $("#lb-table thead");
    thead.innerHTML = "<tr>" + LB_COLS.map((c) => {
      const arr = lbSort.key === c.key ? `<span class="arrow">${lbSort.dir < 0 ? "\u25BC" : "\u25B2"}</span>` : "";
      return `<th class="${c.text ? "col-text" : ""}" data-key="${c.key}">${c.label}${arr}</th>`;
    }).join("") + "</tr>";
    const tb = $("#lb-table tbody");
    tb.innerHTML = rows.slice(0, 400).map((r) =>
      `<tr class="clickable" data-pid="${r.pid}" data-season="${r.season}">` +
      LB_COLS.map((c) => `<td class="${c.text ? "col-text" : ""}">${c.cell(r)}</td>`).join("") +
      "</tr>").join("");
    $("#lb-note").textContent =
      `${rows.length.toLocaleString()} player-seasons` +
      (rows.length > 400 ? " \u00b7 top 400" : "");
  }
  $("#lb-table thead").addEventListener("click", (e) => {
    const th = e.target.closest("th"); if (!th) return;
    const k = th.dataset.key;
    if (lbSort.key === k) lbSort.dir *= -1;
    else {
      lbSort.key = k;
      lbSort.dir = (k === "player" || k === "team" || k === "modelType" || k === "_waaRank") ? 1 : -1;
    }
    renderLB();
  });
  $("#lb-table tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr"); if (!tr) return;
    openPlayer(+tr.dataset.pid);
  });
  ["#f-season", "#f-team", "#f-search"].forEach((s) =>
    $(s).addEventListener("input", renderLB));
  $("#f-minutes").addEventListener("input", (e) => {
    $("#f-minutes-val").textContent = e.target.value; renderLB();
  });

  function initFilters() {
    $("#f-season").innerHTML = `<option value="all">All seasons</option>` +
      D.seasons.slice().reverse().map((s) => `<option value="${s}">${seasonLabel(s)}</option>`).join("");
    $("#f-season").value = String(lastFull);
    const teams = Array.from(new Set(D.players.map((p) => p.team))).sort();
    $("#f-team").innerHTML = `<option value="">All teams</option>` +
      teams.map((t) => `<option value="${t}">${t} \u2014 ${teamName(t)}</option>`).join("");
  }

  /* ===================================================================== *
   *  PLAYER
   * ===================================================================== */
  $("#player-back").addEventListener("click", () => showView("leaderboard"));
  // Bible-grounded one-line scouting read from the DNA style percentiles.
  const AXIS_STRONG = {
    "Shooting": "elite floor-spacer", "Rim / Interior": "interior anchor",
    "Playmaking": "high-level creator", "Ball Dominance": "primary on-ball engine",
    "Perimeter D": "stout point-of-attack defender", "Efficiency": "efficient finisher",
  };
  const AXIS_WEAK = {
    "Shooting": "limited spacing", "Rim / Interior": "little rim protection",
    "Playmaking": "not a shot-creator", "Ball Dominance": "off-ball role",
    "Perimeter D": "targeted on defense", "Efficiency": "inefficient scoring",
  };
  function scoutNote(axes) {
    if (!axes || !axes.length) return "";
    const sorted = axes.slice().sort((a, b) => b.pct - a.pct);
    const strengths = sorted.filter((a) => a.pct >= 78).slice(0, 2)
      .map((a) => AXIS_STRONG[a.axis]).filter(Boolean);
    let s = strengths.length ? strengths.join(" and ") : "balanced profile, no standout dimension";
    const weak = sorted[sorted.length - 1];
    if (weak && weak.pct <= 22 && AXIS_WEAK[weak.axis]) s += `; ${AXIS_WEAK[weak.axis]}`;
    return s.charAt(0).toUpperCase() + s.slice(1) + ".";
  }

  function renderDNA(row) {
    const panel = $("#comps-panel"), box = $("#player-comps");
    if (!box) return;
    const axes = row && row.styleAxes, cs = row && row.comparables;
    if ((!axes || !axes.length) && (!cs || !cs.length)) { if (panel) panel.hidden = true; return; }
    if (panel) panel.hidden = false;
    let html = "";
    const sn = scoutNote(axes);
    if (sn) html += `<div class="scout-note">${sn}</div>`;
    if (axes && axes.length) {
      html += `<div class="fp-wrap">` + axes.map((a) =>
        `<div class="fp-row"><span class="fp-name">${a.axis}</span>` +
        `<span class="fp-track"><span class="fp-fill" style="width:${a.pct}%;background:${pctColor(a.pct)}"></span></span>` +
        `<span class="fp-val">${a.pct}</span></div>`).join("") + `</div>`;
    }
    if (cs && cs.length) {
      html += `<div class="fp-sub">Most similar players</div><div class="comps-wrap">` + cs.map((c) =>
        `<button class="comp-chip" data-pid="${c.pid}" title="style similarity ${(c.sim * 100).toFixed(0)}%">` +
        `${c.player}<small>'${String(c.season).slice(2)} · ${(c.sim * 100).toFixed(0)}%</small></button>`).join("") + `</div>`;
    }
    box.innerHTML = html;
  }
  const compsBox = $("#player-comps");
  if (compsBox) compsBox.addEventListener("click", (e) => {
    const b = e.target.closest(".comp-chip"); if (b) openPlayer(+b.dataset.pid);
  });
  // "Where he'd fit best": rank teams by STYLE COMPLEMENTARITY -- where the rotation most
  // needs what he provides (spacing / shot creation / rim protection / perimeter D) -- plus
  // the usage a proper role would give him. This is meaningful for every player (unlike a
  // title-lift, which needs a positive-impact star); title-lift lives in the trade screen.
  const AX_TO_FIT = {
    Spacing: (ax) => ax["Shooting"],
    "Shot creation": (ax) => (num(ax["Playmaking"]) + num(ax["Ball Dominance"])) / 2,
    "Rim protection": (ax) => ax["Rim / Interior"],
    "Perimeter D": (ax) => ax["Perimeter D"],
  };
  function renderBestFits(latest) {
    const panel = $("#bestfit-panel"), box = $("#bestfit");
    if (!box) return;
    const st = latest && styleByPid()[latest.pid];
    if (!D.trade || !D.trade.teamNet || !latest || latest.predictive || !st) { panel.hidden = true; return; }
    const prov = {};                                   // what he provides (above-average = >0)
    Object.keys(AX_TO_FIT).forEach((d) => { prov[d] = Math.max(0, num(AX_TO_FIT[d](st.ax)) - 50); });
    const rows = Object.keys(D.trade.teamNet).filter((t) => t !== "FA" && t !== latest.team).map((t) => {
      const tf = teamFit(tradeRoster(t));
      if (!tf) return null;
      let score = 0, best = null;
      tf.forEach((d) => {
        const contrib = (prov[d.name] || 0) * Math.max(0, 50 - d.val) / 100;   // he provides × they need
        score += contrib;
        if (!best || contrib > best.c) best = { name: d.name, c: contrib };
      });
      return { team: t, score, fill: best };
    }).filter(Boolean).sort((a, b) => b.score - a.score);
    const role = latest.role;
    const usageNote = role
      ? ` A proper role uses him at ~${Math.round(role.optUsage)}% (now ${Math.round(role.usage)}%${role.misuse > 1.5 ? ", currently over-used" : role.misuse < -1.5 ? ", could handle more" : ""}).`
      : "";
    box.innerHTML =
      `<p class="result-note">Teams whose rotation most needs what he brings (style complementarity) — where his game fits best.${usageNote}</p>` +
      `<div class="bf-list">` + rows.slice(0, 6).map((r, i) =>
        `<div class="bf-row"><span class="bf-rank">${i + 1}</span>` +
        `<span class="bf-team"><span class="team-tag">${r.team}</span> ${teamName(r.team)}</span>` +
        `<span class="bf-fill">fills ${r.fill && r.fill.c > 1 ? r.fill.name : "depth"}</span>` +
        `<span class="bf-delta">fit ${Math.round(r.score)}</span></div>`).join("") + `</div>`;
    panel.hidden = false;
  }

  function openPlayer(pid) {
    const seasons = D.players.filter((p) => p.pid === pid).sort((a, b) => a.season - b.season);
    if (!seasons.length) return;
    const name = seasons[seasons.length - 1].player;
    const teams = Array.from(new Set(seasons.map((s) => s.team)));
    const totWaa = seasons.reduce((a, s) => a + s.waa, 0);
    const peak = seasons.reduce((a, s) => (s.waa > a.waa ? s : a));
    const peak32 = seasons.reduce((a, s) => ((s.waa32 != null ? s.waa32 : -999) > (a.waa32 != null ? a.waa32 : -999) ? s : a));
    const ranks = seasons.map((s) => s.rank).filter((r) => r != null && !isNaN(r));
    const bestRank = ranks.length ? Math.min.apply(null, ranks) : null;
    // grade banner uses the last REAL (non-projection) season -- projection rows carry
    // only an overall grade (no off/def/rank), which showed as "undefined"/"#NaN".
    const realSeasons = seasons.filter((s) => !s.predictive);
    const latest = realSeasons.length ? realSeasons[realSeasons.length - 1] : seasons[seasons.length - 1];
    const gradeHero = (latest.grade != null) ?
      `<div class="grade-hero">` +
        `<div class="gh-main"><span class="gh-letter" style="color:${pctColor(latest.grade)}">${latest.gradeLetter}</span>` +
        `<span class="gh-num">${latest.grade}</span><span class="gh-cap">GRADE \u00b7 ${seasonLabel(latest.season)}</span></div>` +
        `<div class="gh-od"><span class="gh-od-i">Off <b style="color:${pctColor(latest.gradeOff)}">${latest.gradeOff}</b></span>` +
        `<span class="gh-od-i">Def <b style="color:${pctColor(latest.gradeDef)}">${latest.gradeDef}</b></span></div>` +
        (latest.gradeProj ? `<div class="gh-proj"><span class="gh-cap">3-YR OUTLOOK</span> ` +
          latest.gradeProj.map((p) => `<span class="gh-pj" title="age ${p.age}">${p.letter}<small>${p.grade}</small></span>`).join("<span class='gh-arr'>\u2192</span>") + `</div>` : "") +
      `</div>` : "";
    $("#player-head").innerHTML =
      `<h2>${name}</h2>` +
      `<span class="ph-meta">${teams.join(", ")} \u00b7 ${seasons.length} season${seasons.length > 1 ? "s" : ""}` +
      ` \u00b7 ${seasonLabel(seasons[0].season)}\u2013${seasonLabel(seasons[seasons.length - 1].season)}` +
      `${latest.scarcityPct != null ? ` \u00b7 <span title="profile rarity vs the league (PCA-space density)">rarity ${latest.scarcityPct}th %ile</span>` : ""}</span>` +
      gradeHero +
      `<div class="ph-stat">` +
      `<div class="s"><div class="k">Total WAA</div><div class="v">${signed(totWaa, 1)}</div></div>` +
      `<div class="s"><div class="k">Peak WAA@32</div><div class="v">${peak32.waa32 != null ? signed(peak32.waa32, 1) : "\u2014"}</div></div>` +
      `<div class="s"><div class="k">Peak season</div><div class="v">${signed(peak.waa, 1)}</div></div>` +
      `<div class="s"><div class="k">Best rank</div><div class="v">${bestRank != null ? "#" + bestRank : "—"}</div></div></div>`;

    // profile the most recent season with a true-skill breakdown; PBP rate skills run
    // through 2024-25, shooting through the latest season; fall back to the latest.
    const rev = [...seasons].reverse();
    const profSeason = rev.find((s) => hasSkills(s) && s.skills.playmaking) ||
                       rev.find((s) => hasSkills(s)) || seasons[seasons.length - 1];
    renderSkillProfile(profSeason);
    setupCompare(profSeason);
    renderValueDerivation(profSeason);
    renderBestFits(latest);
    renderSkillTrajectory(seasons);
    renderDNA(latest);

    const tcols = [
      ["season", "Season", (r) => seasonLabel(r.season), true],
      ["team", "Team", (r) => `<span class="team-tag">${r.team}</span>`, true],
      ["rankModel", "Rank", (r) => "#" + (r.rankModel != null ? r.rankModel : r.rank)],
      ["grade", "Grade", (r) => r.grade != null
        ? `<span title="Sticky, age-curved, team-independent grade">${r.gradeLetter} <small>${r.grade}</small></span>` : "\u2014", true],
      ["min", "Min", (r) => r.min.toLocaleString()],
      ["bookerScore", "BOOKER", (r) => r.bookerScore != null
        ? `<span class="${r.bookerScore >= 0 ? "pos" : "neg"}" title="BOOKER: predictive +/- per 100 possessions, avg starting-caliber context, usage regressed toward optimum">${signed(r.bookerScore, 1)}</span>` : "\u2014"],
      ["waaOff", "Off WAA", (r) => r.waaOff != null ? signed(r.waaOff, 1) : "\u2014"],
      ["waaDef", "Def WAA", (r) => r.waaDef != null ? signed(r.waaDef, 1) : "\u2014"],
      ["real_pm", "Real +/-", (r) => r.real_pm != null
        ? `<span class="${r.real_pm >= 0 ? "pos" : "neg"}" title="Actual on-court net rating per 100 possessions (descriptive; for isolated impact use BOOKER)">${signed(r.real_pm, 1)}</span>` : "\u2014"],
      ["tm_quality", "Tm Q", (r) => r.tm_quality != null
        ? `<span title="Avg teammate rating on court with him (impact/100) -- supporting cast">${signed(r.tm_quality, 1)}</span>` : "\u2014"],
      ["opp_quality", "Opp Q", (r) => r.opp_quality != null
        ? `<span title="Avg opponent rating on court against him (impact/100) -- competition faced">${signed(r.opp_quality, 1)}</span>` : "\u2014"],
      ["waaModel", "WAA", (r) => `<span class="${(r.waaModel != null ? r.waaModel : r.waa) >= 0 ? "pos" : "neg"}">${signed(r.waaModel != null ? r.waaModel : r.waa, 1)}</span>`],
      ["waa32", "WAA@32", (r) => r.waa32 != null
        ? `<span class="${r.waa32 >= 0 ? "pos" : "neg"}" title="Bayesian predictive WAA at 32 mpg (82 games); low-minute seasons shrink toward prior">${signed(r.waa32, 1)}</span>`
        : "\u2014"],
      ["bfRating", "BookerFormer O/D \u00b195%", (r) => bfCell(r)],
      ["trueValue", "True Value", (r) => (r.trueValue != null ? r.trueValue : r.fairAav2026) != null
        ? money(r.trueValue != null ? r.trueValue : r.fairAav2026) : "\u2014"],
      ["marketAav2026", "Contract", (r) => r.marketAav2026 != null ? money(r.marketAav2026) : "\u2014"],
      ["surplus", "Surplus", (r) => r.surplus != null ? money(r.surplus) : "\u2014"],
    ];
    $("#player-table thead").innerHTML = "<tr>" +
      tcols.map((c) => `<th class="${c[3] ? "col-text" : ""}">${c[1]}</th>`).join("") + "</tr>";
    $("#player-table tbody").innerHTML = seasons.map((r) =>
      "<tr>" + tcols.map((c) => `<td class="${c[3] ? "col-text" : ""}">${c[2](r)}</td>`).join("") + "</tr>").join("");

    destroy("player");
    const ctx = $("#player-chart");
    charts.player = new Chart(ctx, {
      type: "bar",
      data: {
        labels: seasons.map((s) => seasonLabel(s.season)),
        datasets: [
          { type: "bar", label: "WAA wins (actual min)", yAxisID: "y",
            data: seasons.map((s) => s.waa),
            backgroundColor: seasons.map((s) => s.waa >= 0 ? "rgba(47,93,52,.78)" : "rgba(122,40,32,.78)"),
            borderRadius: 2, order: 3 },
          { type: "bar", label: "WAA@32 (Bayesian predictive)", yAxisID: "y",
            data: seasons.map((s) => s.waa32 != null ? s.waa32 : null),
            backgroundColor: seasons.map((s) => (s.waa32 != null && s.waa32 >= 0) ? "rgba(36,28,18,.35)" : "rgba(122,40,32,.35)"),
            borderRadius: 2, order: 2 },
          { type: "line", label: "WAA / 100 poss", yAxisID: "y2",
            data: seasons.map((s) => s.waa100), borderColor: COL.ink, backgroundColor: COL.ink,
            borderWidth: 2, pointRadius: 3, tension: .25, order: 1 },
        ],
      },
      options: {
        maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, usePointStyle: true } } },
        scales: {
          y: { position: "left", title: { display: true, text: "WAA wins" }, grid: { color: COL.grid } },
          y2: { position: "right", title: { display: true, text: "WAA / 100" }, grid: { drawOnChartArea: false } },
          x: { grid: { display: false } },
        },
      },
    });
    showView("player");
  }

  /* ===================================================================== *
   *  WIN FORECAST
   * ===================================================================== */
  let forecastReady = false;
  function renderForecast() {
    const pooled = D.metrics.find((m) => m.label === "Pooled");
    const foldYrs = D.metrics.filter((m) => m.season != null);
    const oosSpan = foldYrs.length
      ? `${foldYrs[0].label.split("-")[0]}\u2013${seasonLabel(foldYrs[foldYrs.length - 1].season)}`
      : "out-of-sample";
    $("#forecast-cards").innerHTML = [
      ["Wins RMSE", fmt(pooled.winsRmse, 1), `out-of-sample, ${oosSpan}`],
      ["Wins R\u00b2", fmt(pooled.winsR2, 2), `${D.teamPred.length} team-seasons`],
      ["vs predict-41", fmt(D.baselines.predict41, 1), "naive baseline RMSE"],
      ["vs old box model", fmt(D.baselines.oldBox, 1), "retrodictive RMSE"],
    ].map((c) => `<div class="card"><div class="k">${c[0]}</div><div class="v">${c[1]}</div><div class="d">${c[2]}</div></div>`).join("");

    // team season filter
    const fcSel = $("#f-fc-season");
    if (!fcSel.dataset.init) {
      const yrs = Array.from(new Set(D.teamPred.map((t) => t.season))).sort((a, b) => b - a);
      fcSel.innerHTML = `<option value="all">All seasons</option>` +
        yrs.map((s) => `<option value="${s}">${seasonLabel(s)}</option>`).join("");
      fcSel.addEventListener("change", renderTeamTable);
      defaultSeason(fcSel);
      fcSel.dataset.init = "1";
    }
    renderTeamTable();

    // scatter
    destroy("scatter");
    const pts = D.teamPred.map((t) => ({ x: t.actualWins, y: t.predWins, s: t.season, team: t.team }));
    charts.scatter = new Chart($("#scatter-chart"), {
      type: "scatter",
      data: { datasets: [
        { label: "team-season", data: pts,
          pointBackgroundColor: "rgba(122,40,32,.55)", pointBorderColor: COL.paper, pointRadius: 4, pointHoverRadius: 6 },
        { type: "line", label: "perfect", data: [{ x: 12, y: 12 }, { x: 74, y: 74 }],
          borderColor: COL.neg, borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0, fill: false },
      ]},
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => c.raw.team ?
            `${c.raw.team} ${seasonLabel(c.raw.s)}: actual ${c.raw.x}, pred ${fmt(c.raw.y,1)}` : "" } },
        },
        scales: {
          x: { min: 12, max: 74, title: { display: true, text: "Actual wins" }, grid: { color: COL.grid } },
          y: { min: 12, max: 74, title: { display: true, text: "Predicted wins" }, grid: { color: COL.grid } },
        },
      },
    });

    // rmse by year
    destroy("rmse");
    const folds = D.metrics.filter((m) => m.season != null);
    charts.rmse = new Chart($("#rmse-chart"), {
      type: "bar",
      data: {
        labels: folds.map((f) => f.label),
        datasets: [{ label: "WAA out-of-sample", data: folds.map((f) => f.winsRmse),
          backgroundColor: "rgba(122,40,32,.78)", borderRadius: 0 }],
      },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          annotation: false,
          tooltip: { callbacks: { label: (c) => `RMSE ${fmt(c.raw,1)}` } },
        },
        scales: { y: { beginAtZero: true, title: { display: true, text: "Wins RMSE" }, grid: { color: COL.grid } },
                  x: { grid: { display: false } } },
      },
    });
    // baseline reference lines drawn via a tiny plugin
    drawBaselines();

    renderPreseason();
    renderTimeline();
  }

  /* ---- preseason projection table -------------------------------------- */
  const preSort = { key: "projWins", dir: -1 };
  function renderPreseason() {
    if (!D.preseason || !D.preseason.length) return;
    const sel = $("#f-pre-season");
    if (!sel.dataset.init) {
      const yrs = Array.from(new Set(D.preseason.map((p) => p.season))).sort((a, b) => b - a);
      sel.innerHTML = yrs.map((s) => `<option value="${s}">${seasonLabel(s)}</option>`).join("");
      sel.addEventListener("change", drawPreseason);
      defaultSeason(sel);
      sel.dataset.init = "1";
    }
    drawPreseason();
  }
  const PRE_COLS = [
    ["team", "Team", (r) => `<span class="team-tag">${r.team}</span> ${teamName(r.team)}`, true],
    ["predNet", "Pred Net", (r) => signed(r.predNet, 1)],
    ["projWins", "Proj W", (r) => `<b title="schedule-sim mean \u2014 converts net to wins through the real schedule (beats the linear map out-of-sample)">${fmt(pw(r), 1)}</b>`],
    ["range", "Sim range (10\u201390%)", (r) => `${r.p10}\u2013${r.p90}`, true],
    ["pPlayoff", "Playoff", (r) => pctCell(r.pPlayoff)],
    ["pChamp", "Champ", (r) => r.pChamp == null ? "\u2014" : pctCell(r.pChamp)],
    ["actualWins", "Actual W", (r) => r.actualWins == null ? "\u2014" : winCmp(r)],
  ];
  function pw(r) { return r.simMean != null ? r.simMean : r.projWins; }
  function winCmp(r) {
    const err = pw(r) - r.actualWins;
    const cls = Math.abs(err) <= 5 ? "pos" : "neg";
    return `${r.actualWins} <span class="${cls}" style="font-size:11px">(${signed(err, 1)})</span>`;
  }
  function drawPreseason() {
    const yr = +$("#f-pre-season").value;
    let rows = D.preseason.filter((p) => p.season === yr);
    rows.sort((a, b) => {
      if (preSort.key === "range") return (a.p50 - b.p50) * preSort.dir;
      if (preSort.key === "projWins") return (pw(a) - pw(b)) * preSort.dir;
      const A = a[preSort.key], B = b[preSort.key];
      const c = typeof A === "string" ? A.localeCompare(B) : A - B; return c * preSort.dir;
    });
    $("#preseason-table thead").innerHTML = "<tr>" + PRE_COLS.map((c) => {
      const arr = preSort.key === c[0] ? `<span class="arrow">${preSort.dir < 0 ? "\u25BC" : "\u25B2"}</span>` : "";
      return `<th class="${c[3] ? "col-text" : ""}" data-key="${c[0]}">${c[1]}${arr}</th>`;
    }).join("") + "</tr>";
    $("#preseason-table tbody").innerHTML = rows.map((r) =>
      "<tr>" + PRE_COLS.map((c) => `<td class="${c[3] ? "col-text" : ""}">${c[2](r)}</td>`).join("") + "</tr>").join("");
  }
  $("#preseason-table thead").addEventListener("click", (e) => {
    const th = e.target.closest("th"); if (!th) return;
    const k = th.dataset.key;
    if (preSort.key === k) preSort.dir *= -1;
    else { preSort.key = k; preSort.dir = (k === "team" ? 1 : -1); }
    drawPreseason();
  });

  /* ---- in-season trajectory chart -------------------------------------- */
  function renderTimeline() {
    if (!D.timeline || !D.timeline.length) return;
    const sel = $("#f-tl-season");
    if (!sel.dataset.init) {
      const yrs = Array.from(new Set(D.timeline.map((t) => t.season))).sort((a, b) => b - a);
      sel.innerHTML = yrs.map((s) => `<option value="${s}">${seasonLabel(s)}</option>`).join("");
      sel.addEventListener("change", drawTimeline);
      defaultSeason(sel);
      sel.dataset.init = "1";
    }
    drawTimeline();
  }
  function drawTimeline() {
    const yr = +$("#f-tl-season").value;
    const rows = D.timeline.filter((t) => t.season === yr);
    const teams = Array.from(new Set(rows.map((r) => r.team)));
    // colour each line by its final projected wins (sepia->oxblood ramp)
    const finals = {};
    teams.forEach((t) => {
      const last = rows.filter((r) => r.team === t).sort((a, b) => a.frac - b.frac).slice(-1)[0];
      finals[t] = last ? last.projFinal : 41;
    });
    const lo = Math.min(...Object.values(finals)), hi = Math.max(...Object.values(finals));
    const ramp = (v) => {
      const f = hi > lo ? (v - lo) / (hi - lo) : 0.5;
      const r = Math.round(150 + f * 60), g = Math.round(120 - f * 70), b = Math.round(70 - f * 35);
      return `rgba(${r},${g},${b},.8)`;
    };
    const datasets = teams.map((t) => {
      const d = rows.filter((r) => r.team === t).sort((a, b) => a.frac - b.frac);
      return { label: t, data: d.map((r) => ({ x: Math.round(r.frac * 100), y: r.projFinal })),
        borderColor: ramp(finals[t]), backgroundColor: ramp(finals[t]),
        borderWidth: 1.4, pointRadius: 0, pointHoverRadius: 4, tension: 0.25 };
    });
    destroy("timeline");
    charts.timeline = new Chart($("#timeline-chart"), {
      type: "line",
      data: { datasets },
      options: {
        maintainAspectRatio: false, interaction: { mode: "nearest", intersect: false },
        plugins: { legend: { display: false },
          tooltip: { callbacks: { title: (c) => `${c[0].raw.x}% of season`,
            label: (c) => `${c.dataset.label}: proj ${fmt(c.raw.y, 1)} W` } } },
        scales: {
          x: { type: "linear", min: 0, max: 100, title: { display: true, text: "Season progress (%)" }, grid: { color: COL.grid } },
          y: { title: { display: true, text: "Projected final wins" }, grid: { color: COL.grid } },
        },
      },
    });
    const fin = D.timeline.filter((t) => t.season === yr && t.actualWins != null);
    $("#tl-note").textContent = fin.length
      ? "Lines converge toward each team's realized win total as the season is played out."
      : "Live trajectory \u2014 actual final wins not yet known.";
  }

  /* ===================================================================== *
   *  GAME ODDS
   * ===================================================================== */
  const goSort = { key: "date", dir: 1 };
  function renderGameOdds() {
    const gm = D.gameMetrics || [];
    const pooled = gm.find((m) => m.label === "Pooled") || {};
    const hasMkt = pooled.marketLogloss != null;
    $("#go-lede").textContent = hasMkt
      ? "Pre-tip win probabilities vs the closing moneyline (vig removed)."
      : "Pre-tip win probabilities vs outcomes; market lines unavailable for recent seasons.";
    const cards = [
      ["Model log-loss", fmt(pooled.modelLogloss, 4), "lower is better"],
      ["Model accuracy", pooled.modelAcc != null ? (pooled.modelAcc * 100).toFixed(1) + "%" : "\u2014", "straight-up picks"],
    ];
    if (hasMkt) {
      cards.push(["Market log-loss", fmt(pooled.marketLogloss, 4), `closing line (n=${pooled.marketGames})`]);
      cards.push(["Flat-bet ROI", pooled.roi != null ? signed(pooled.roi * 100, 1) + "%" : "\u2014",
        `value side, ${pooled.nBets} bets`]);
    } else {
      cards.push(["Games scored", String(pooled.games || 0), "regular season"]);
      cards.push(["Brier score", fmt(pooled.modelBrier, 4), "lower is better"]);
    }
    $("#go-cards").innerHTML = cards.map((c) =>
      `<div class="card"><div class="k">${c[0]}</div><div class="v">${c[1]}</div><div class="d">${c[2]}</div></div>`).join("");

    drawCalibration();
    drawGoSeason();
    initGoTable();
  }
  function drawCalibration() {
    const cal = D.calibration || [];
    destroy("calib");
    charts.calib = new Chart($("#calib-chart"), {
      type: "scatter",
      data: { datasets: [
        { label: "BOOKER", data: cal.map((b) => ({ x: b.predMean, y: b.empirical, n: b.count })),
          showLine: true, borderColor: COL.accent, backgroundColor: COL.accent,
          pointRadius: 4, borderWidth: 2, tension: 0.2 },
        { label: "perfect", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], type: "line",
          borderColor: COL.grey, borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0 },
      ]},
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: (c) => c.raw.n
            ? `pred ${(c.raw.x*100).toFixed(0)}% \u2192 actual ${(c.raw.y*100).toFixed(0)}% (n=${c.raw.n})` : "" } } },
        scales: {
          x: { min: 0, max: 1, title: { display: true, text: "Predicted home win prob." }, grid: { color: COL.grid } },
          y: { min: 0, max: 1, title: { display: true, text: "Observed home win rate" }, grid: { color: COL.grid } },
        },
      },
    });
  }
  function drawGoSeason() {
    const folds = (D.gameMetrics || []).filter((m) => m.season != null);
    destroy("goSeason");
    const ds = [{ label: "BOOKER", data: folds.map((f) => f.modelLogloss),
      backgroundColor: "rgba(122,40,32,.8)" }];
    if (folds.some((f) => f.marketLogloss != null)) {
      ds.push({ label: "Market", data: folds.map((f) => f.marketLogloss),
        backgroundColor: "rgba(138,120,90,.75)" });
    }
    charts.goSeason = new Chart($("#go-season-chart"), {
      type: "bar",
      data: { labels: folds.map((f) => f.label), datasets: ds },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: ds.length > 1, labels: { boxWidth: 12 } },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.raw, 4)}` } } },
        scales: { y: { title: { display: true, text: "Log-loss" }, grid: { color: COL.grid } },
                  x: { grid: { display: false } } },
      },
    });
  }
  const GO_COLS = [
    ["date", "Date", (r) => r.date, true],
    ["away", "Away", (r) => `<span class="team-tag">${r.away}</span>`, true],
    ["home", "Home", (r) => `<span class="team-tag">${r.home}</span>`, true],
    ["modelPHome", "BOOKER P(home)", (r) => pctCell(r.modelPHome)],
    ["marketPHome", "Market P(home)", (r) => r.marketPHome == null ? "\u2014" : pctCell(r.marketPHome)],
    ["edge", "Edge", (r) => r.marketPHome == null ? "\u2014" : edgeCell(r)],
    ["homeWin", "Result", (r) => r.homeWin ? `<span class="pos">Home W</span>` : `<span class="neg">Away W</span>`, true],
  ];
  function edgeCell(r) {
    const e = r.modelPHome - r.marketPHome;
    return `<span class="${e >= 0 ? "pos" : "neg"}">${signed(e * 100, 1)}</span>`;
  }
  function initGoTable() {
    const inp = $("#f-go-search");
    if (!inp.dataset.init) { inp.addEventListener("input", drawGoTable); inp.dataset.init = "1"; }
    drawGoTable();
  }
  function drawGoTable() {
    const q = $("#f-go-search").value.trim().toUpperCase();
    let rows = (D.recentGames || []).slice();
    if (q) rows = rows.filter((r) => r.home.includes(q) || r.away.includes(q));
    rows.sort((a, b) => {
      let A = a[goSort.key], B = b[goSort.key];
      if (goSort.key === "edge") { A = (a.modelPHome - (a.marketPHome||0)); B = (b.modelPHome - (b.marketPHome||0)); }
      const c = typeof A === "string" ? A.localeCompare(B) : A - B; return c * goSort.dir;
    });
    rows = rows.slice(0, 400);
    $("#go-table thead").innerHTML = "<tr>" + GO_COLS.map((c) => {
      const arr = goSort.key === c[0] ? `<span class="arrow">${goSort.dir < 0 ? "\u25BC" : "\u25B2"}</span>` : "";
      return `<th class="${c[3] ? "col-text" : ""}" data-key="${c[0]}">${c[1]}${arr}</th>`;
    }).join("") + "</tr>";
    $("#go-table tbody").innerHTML = rows.map((r) =>
      "<tr>" + GO_COLS.map((c) => `<td class="${c[3] ? "col-text" : ""}">${c[2](r)}</td>`).join("") + "</tr>").join("");
  }
  $("#go-table thead").addEventListener("click", (e) => {
    const th = e.target.closest("th"); if (!th) return;
    const k = th.dataset.key;
    if (goSort.key === k) goSort.dir *= -1;
    else { goSort.key = k; goSort.dir = (k === "date" || k === "home" || k === "away" || k === "homeWin") ? 1 : -1; }
    drawGoTable();
  });
  function pctCell(p) {
    const w = Math.max(0, Math.min(100, p * 100));
    return `<span class="prob-bar"><span class="bar-cell"><span class="bar" style="width:${w}%"></span>` +
           `<span>${(p * 100).toFixed(0)}%</span></span></span>`;
  }
  function drawBaselines() {
    const ch = charts.rmse; if (!ch) return;
    const refs = [
      { v: D.baselines.predict41, c: COL.neg, t: "predict-41 (12.0)" },
      { v: D.baselines.oldBox, c: COL.grey, t: "old box, retrodictive (9.0)" },
    ];
    ch.options.plugins.tooltip.enabled = true;
    if (!ch.$refs) {
      ch.$refs = {
        id: "refs",
        afterDraw(c) {
          const { ctx, chartArea: a, scales: { y } } = c;
          refs.forEach((r) => {
            const yy = y.getPixelForValue(r.v);
            ctx.save();
            ctx.strokeStyle = r.c; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(a.left, yy); ctx.lineTo(a.right, yy); ctx.stroke();
            ctx.setLineDash([]); ctx.fillStyle = r.c; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
            ctx.fillText(r.t, a.right - 4, yy - 4); ctx.restore();
          });
        },
      };
    }
    Chart.register(ch.$refs); ch.update();
  }

  const TEAM_COLS = [
    ["season", "Season", (r) => seasonLabel(r.season), true],
    ["team", "Team", (r) => `<span class="team-tag">${r.team}</span> ${teamName(r.team)}`, true],
    ["actualWins", "Actual W", (r) => r.actualWins],
    ["predWins", "Pred W", (r) => fmt(r.predWins, 1)],
    ["winErr", "Error", (r) => `<span class="${Math.abs(r.winErr) <= 5 ? "pos" : "neg"}">${signed(r.winErr, 1)}</span>`],
    ["actualNet", "Actual Net", (r) => signed(r.actualNet, 1)],
    ["predNet", "Pred Net", (r) => signed(r.predNet, 1)],
  ];
  const teamSort = { key: "actualWins", dir: -1 };
  function renderTeamTable() {
    const sel = $("#f-fc-season").value;
    let rows = D.teamPred.slice();
    if (sel !== "all") rows = rows.filter((r) => r.season === +sel);
    rows.sort((a, b) => {
      const A = a[teamSort.key], B = b[teamSort.key];
      const c = typeof A === "string" ? A.localeCompare(B) : A - B; return c * teamSort.dir;
    });
    $("#team-table thead").innerHTML = "<tr>" + TEAM_COLS.map((c) => {
      const arr = teamSort.key === c[0] ? `<span class="arrow">${teamSort.dir < 0 ? "\u25BC" : "\u25B2"}</span>` : "";
      return `<th class="${c[3] ? "col-text" : ""}" data-key="${c[0]}">${c[1]}${arr}</th>`;
    }).join("") + "</tr>";
    $("#team-table tbody").innerHTML = rows.map((r) =>
      "<tr>" + TEAM_COLS.map((c) => `<td class="${c[3] ? "col-text" : ""}">${c[2](r)}</td>`).join("") + "</tr>").join("");
  }
  $("#team-table thead").addEventListener("click", (e) => {
    const th = e.target.closest("th"); if (!th) return;
    const k = th.dataset.key;
    if (teamSort.key === k) teamSort.dir *= -1;
    else { teamSort.key = k; teamSort.dir = (k === "team" ? 1 : -1); }
    renderTeamTable();
  });

  /* ===================================================================== *
   *  TRADE MACHINE
   * ===================================================================== */
  let tradeReady = false;

  function selectedTradePids(id) {
    const sel = $(id);
    return Array.from(sel.selectedOptions).slice(0, MAX_TRADE_ASSETS).map((o) => +o.value);
  }

  function limitTradeSelection(e) {
    const sel = e.target;
    if (sel.selectedOptions.length > MAX_TRADE_ASSETS) {
      const keep = Array.from(sel.selectedOptions).slice(0, MAX_TRADE_ASSETS);
      Array.from(sel.options).forEach((o) => { o.selected = keep.includes(o); });
    }
  }

  function salaryMatch(outSal, inSal) {
    const out = outSal.reduce((a, b) => a + b, 0);
    const inn = inSal.reduce((a, b) => a + b, 0);
    if (out <= 0 && inn <= 0) return { ok: true, note: "no salary" };
    if (out <= inn) return { ok: true, note: "incoming covers" };
    const need = 1.25 * out + 250000;
    const ok = inn >= need - 1;
    return { ok, note: ok ? "125%+$250k matched" : `need ${money(need - inn)} more` };
  }

  // ---- Bible-grounded roster fit (descriptive; win delta stays talent-based) ----
  // Maps each player to their latest real-season DNA style axes, then rolls a
  // roster up on the four dimensions that decide half-court games per the
  // Basketball Bible: spacing, rim protection, shot creation, perimeter defense.
  let _styleByPid = null;
  function styleByPid() {
    if (_styleByPid) return _styleByPid;
    _styleByPid = {};
    (D.players || []).forEach((p) => {
      if (!p.styleAxes || p.predictive) return;
      const cur = _styleByPid[p.pid];
      if (!cur || p.season > cur.season) {
        const ax = {};
        p.styleAxes.forEach((a) => { ax[a.axis] = a.pct; });
        _styleByPid[p.pid] = { season: p.season, ax };
      }
    });
    return _styleByPid;
  }
  const FIT_DIMS = [
    ["Spacing", (ax) => ax["Shooting"], "shooting that pulls help out of the paint"],
    ["Rim protection", (ax) => ax["Rim / Interior"], "interior size &amp; rim presence — “protect the rim first”"],
    ["Shot creation", (ax) => (num(ax["Playmaking"]) + num(ax["Ball Dominance"])) / 2, "on-ball creation to beat a set defense"],
    ["Perimeter D", (ax) => ax["Perimeter D"], "on-ball perimeter defense — “stop the ball”"],
  ];
  function num(x) { return typeof x === "number" ? x : 0; }
  function teamFit(roster) {
    const S = styleByPid();
    const rows = roster.map((p) => ({ mn: p.minutes, ax: (S[p.pid] || {}).ax })).filter((r) => r.ax);
    const totMin = rows.reduce((a, r) => a + r.mn, 0);
    if (!totMin || rows.length < 3) return null;
    return FIT_DIMS.map(([name, f, desc]) => ({
      name, desc, val: rows.reduce((a, r) => a + num(f(r.ax)) * r.mn, 0) / totMin,
    }));
  }
  function rosterAfter(team, out, inc) {
    const outIds = new Set(out.map((p) => p.pid));
    return tradeRoster(team).filter((p) => !outIds.has(p.pid)).concat(inc);
  }

  // ---- championship-odds model for the trade screen ----
  // Saturating fit of title odds vs team net, calibrated to the season's bracket-sim
  // champ%s, so Δtitle-odds respects diminishing returns: mid-tier contenders gain the
  // most per net point, favorites near the ceiling gain little, lottery teams ~nothing.
  // (A runaway softmax would say a star lifts the best team +40 pts -- it doesn't.)
  let _champFit = null;
  function champCurve() {
    if (_champFit) return _champFit;
    const base = {};
    (D.preseason || []).forEach((r) => {
      if (r.season === D.trade.season && r.pChamp != null) base[r.team] = r.pChamp; });
    const pts = Object.keys(D.trade.teamNet).map((t) => ({ net: D.trade.teamNet[t], ch: base[t] || 0 }));
    const Lmax = Math.max(0.6, Math.max.apply(null, pts.map((p) => p.ch)) * 1.15);
    const xs = [], ys = [];
    pts.forEach((p) => { const f = p.ch / Lmax; if (f > 0.004 && f < 0.98) { xs.push(p.net); ys.push(Math.log(f / (1 - f))); } });
    let b = 0.3, a = -3;
    if (xs.length > 3) {
      const n = xs.length, mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
      let sxy = 0, sxx = 0;
      for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
      b = sxx > 0 ? sxy / sxx : 0.3; a = my - b * mx;
    }
    _champFit = { fn: (net) => Lmax / (1 + Math.exp(-(a + b * net))), base };
    return _champFit;
  }
  function champAfter(team, newNet) {          // title odds after a team's net changes
    const cc = champCurve();
    const b0 = cc.base[team] || 0;
    return Math.max(0, Math.min(0.99, b0 + cc.fn(newNet) - cc.fn(D.trade.teamNet[team])));
  }
  // free agents = rated players NOT on any roster this season (LeBron-style signings)
  let _faPool = null;
  function faPool() {
    if (_faPool) return _faPool;
    const rostered = new Set((D.trade.players || []).map((p) => p.pid));
    const best = {};
    (D.players || []).forEach((p) => {
      if (p.predictive || rostered.has(p.pid) || (p.min || 0) < 500) return;
      const cur = best[p.pid];
      if (!cur || p.season > cur.season) {
        const imp = p.bfTot100 != null ? p.bfTot100 : (p.bfOff100 || 0) + (p.bfDef100 || 0);
        best[p.pid] = {
          pid: p.pid, player: p.player, team: "FA", season: p.season,
          minutes: Math.min(2200, Math.max(1000, p.min || 1500)),
          impactTotal: imp, waaTotal: p.waa || 0, marketAav2026: p.marketAav2026 || p.fairAav2026 || 0,
          gradeOff: p.gradeOff, gradeDef: p.gradeDef, age: p.age,
        };
      }
    });
    // only players active in the last full season(s) are real free agents (drop retirees)
    _faPool = Object.values(best).filter((f) => f.season >= lastFull - 1)
      .sort((a, b) => b.impactTotal - a.impactTotal);
    return _faPool;
  }

  // ---- Lineup Lab: build a rotation, get projected net -> wins + title odds ----
  let LL = null, llReady = false;
  function llAllPlayers() {
    const m = {};
    (D.trade.players || []).forEach((p) => { m[p.pid] = { pid: p.pid, player: p.player, impactTotal: p.impactTotal }; });
    faPool().forEach((p) => { if (!m[p.pid]) m[p.pid] = { pid: p.pid, player: p.player, impactTotal: p.impactTotal }; });
    return m;
  }
  function renderLineup() {
    if (!D.trade || !D.trade.teamNet) { $("#ll-lede").textContent = "Lineup data unavailable."; return; }
    $("#ll-lede").textContent = `${seasonLabel(D.trade.season)} · build a rotation and see projected wins & championship odds. Add any player (or a free agent) for what-ifs.`;
    const teams = Object.keys(D.trade.teamNet).sort();
    const sel = $("#ll-team");
    if (!llReady) {
      sel.innerHTML = teams.map((t) => `<option value="${t}">${t} — ${teamName(t)}</option>`).join("");
      sel.value = teams.includes("OKC") ? "OKC" : teams[0];
      sel.addEventListener("change", () => llLoadTeam(sel.value));
      $("#ll-reset").addEventListener("click", () => llLoadTeam(sel.value));
      const all = llAllPlayers();
      $("#ll-datalist").innerHTML = Object.values(all).map((p) => `<option value="${p.player}"></option>`).join("");
      $("#ll-add").addEventListener("change", () => {
        const nm = $("#ll-add").value.trim().toLowerCase();
        const hit = Object.values(all).find((p) => p.player.toLowerCase() === nm);
        if (hit && LL && !LL.roster.some((r) => r.pid === hit.pid)) {
          LL.roster.push({ pid: hit.pid, player: hit.player, impactTotal: hit.impactTotal, minutes: 1500 });
          $("#ll-add").value = ""; llRenderRoster(); llRecompute();
        }
      });
      $("#ll-roster").addEventListener("input", (e) => {
        const inp = e.target.closest("input[data-pid]"); if (!inp || !LL) return;
        const r = LL.roster.find((x) => x.pid === +inp.dataset.pid);
        if (r) { r.minutes = Math.max(0, +inp.value || 0); llRecompute(); }
      });
      $("#ll-roster").addEventListener("click", (e) => {
        const b = e.target.closest("button[data-pid]"); if (!b || !LL) return;
        LL.roster = LL.roster.filter((x) => x.pid !== +b.dataset.pid);
        llRenderRoster(); llRecompute();
      });
      llReady = true;
    }
    llLoadTeam(sel.value);
  }
  function llLoadTeam(team) {
    LL = { team, roster: tradeRoster(team).filter((p) => p.minutes > 0)
      .map((p) => ({ pid: p.pid, player: p.player, impactTotal: p.impactTotal, minutes: Math.round(p.minutes) })) };
    llRenderRoster(); llRecompute();
  }
  function llRenderRoster() {
    const tot = LL.roster.reduce((a, r) => a + r.minutes, 0);
    $("#ll-roster").innerHTML = `<div class="ll-tot">${tot.toLocaleString()} total min · ${LL.roster.length} players</div>`
      + LL.roster.slice().sort((a, b) => b.minutes - a.minutes).map((r) =>
        `<div class="ll-row"><span class="ll-name">${r.player}</span>`
        + `<span class="ll-imp ${r.impactTotal >= 0 ? "pos" : "neg"}" title="impact / 100 poss">${signed(r.impactTotal, 1)}</span>`
        + `<input type="number" min="0" step="50" value="${r.minutes}" data-pid="${r.pid}" class="ll-min" />`
        + `<button data-pid="${r.pid}" class="ll-x" title="remove">✕</button></div>`).join("");
  }
  function llRecompute() {
    const rost = LL.roster.filter((r) => r.minutes > 0);
    const tot = rost.reduce((a, r) => a + r.minutes, 0);
    const out = $("#ll-out");
    if (tot <= 0 || rost.length < 5) { out.innerHTML = `<p class="result-note">Add at least 5 players with minutes.</p>`; return; }
    const net = rost.reduce((a, r) => a + r.impactTotal * (r.minutes / (tot / 5)), 0);
    const wins = Math.max(5, Math.min(82, D.trade.k * net + D.trade.c));
    const champ = champAfter(LL.team, net);
    const bNet = D.trade.teamNet[LL.team];
    const bWins = (D.trade.teamSimWins && D.trade.teamSimWins[LL.team]) || (D.trade.k * bNet + D.trade.c);
    const bChamp = champCurve().base[LL.team] || 0;
    const dW = wins - bWins, dC = (champ - bChamp) * 100;
    const fit = teamFit(rost);
    out.innerHTML =
      `<div class="ll-hero">`
      + `<div class="ll-hcell"><div class="ll-k">Proj net</div><div class="ll-v">${signed(net, 1)}</div></div>`
      + `<div class="ll-hcell"><div class="ll-k">Proj wins</div><div class="ll-v">${fmt(wins, 1)}<span class="ll-d ${dW >= 0 ? "pos" : "neg"}">${signed(dW, 1)}</span></div></div>`
      + `<div class="ll-hcell"><div class="ll-k">Championship</div><div class="ll-v">${(champ * 100).toFixed(1)}%<span class="ll-d ${dC >= 0 ? "pos" : "neg"}">${dC >= 0 ? "+" : ""}${dC.toFixed(1)}</span></div></div>`
      + `</div>`
      + `<p class="result-note">vs ${LL.team}'s actual rotation — ${fmt(bWins, 0)} wins, ${(bChamp * 100).toFixed(1)}% title. Δ shown in green/red.</p>`
      + (fit ? `<div class="ll-fit"><div class="fp-sub">Rotation balance</div>` + fit.map((d) =>
        `<div class="fp-row"><span class="fp-name">${d.name}</span>`
        + `<span class="fp-track"><span class="fp-fill" style="width:${Math.round(d.val)}%;background:${pctColor(d.val)}"></span></span>`
        + `<span class="fp-val">${Math.round(d.val)}</span></div>`).join("") + `</div>` : "");
  }

  /* ---- TRAJECTORY: career BOOKER curve + projections ------------------- */
  const TJ = { players: [], ready: false };
  const TJ_COLORS = ["#141310", "#7a2820", "#3d5a80", "#6b8e23", "#8b5e83", "#b8860b"];
  const AGE_PEAK_TJ = 27.0, AGE_QUAD_TJ = -0.03;

  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function tjCareer(pid) {
    // realized seasons only, with the in-season evidence glide: the rating shown at
    // fraction t of season s is prior + (posterior - prior)·t -- the Bayesian update
    // path as games accumulate (prior = last season's rating). Also collects the raw
    // per-game on-court nets (scatter) and a 95% band from each season's rating sd.
    const ss = D.players.filter((p) => p.pid === pid && !p.predictive && p.bookerScore != null)
      .sort((a, b) => a.season - b.season);
    if (!ss.length) return null;
    const pts = [], hi = [], lo = [], dots = [];
    let prev = null;
    ss.forEach((p) => {
      const v = p.bookerScore;
      const start = prev == null ? v : prev;
      const sd = (p.sdOff != null) ? Math.hypot(p.sdOff, p.sdDef) : 1.5;
      for (let t = 0; t <= 1.0001; t += 0.25) {
        const x = p.season - 1 + t;
        const y = start + (v - start) * t;
        pts.push({ x, y: +y.toFixed(2) });
        hi.push({ x, y: +(y + 1.96 * sd).toFixed(2) });
        lo.push({ x, y: +(y - 1.96 * sd).toFixed(2) });
      }
      if (p.gameNets && p.gameNets.length) {
        // anchor the cloud to the exact value the LINE plots (bookerScore), so the
        // dots' mean sits on the curve regardless of score-definition drift.
        const n = p.gameNets.length;
        const mean = p.gameNets.reduce((a, b) => a + b, 0) / n;
        p.gameNets.forEach((g, i) => {
          dots.push({ x: p.season - 1 + (i + 0.5) / n, y: +(g - mean + v).toFixed(1) });
        });
      }
      prev = v;
    });
    const last = ss[ss.length - 1];
    return { pts, hi, lo, dots, last, name: last.player, seasons: ss };
  }

  function tjProjection(career, years, optimal) {
    // forward path along the aging curve from the last real rating; ages from the
    // grade projection when present. "optimal usage" adds the remaining un-credited
    // role upside (BOOKER already carries 25% of it).
    const last = career.last;
    let v = last.bookerScore;
    if (optimal && last.role && last.role.upside) v += 0.75 * last.role.upside;
    let age = (last.gradeProj && last.gradeProj[0] && last.gradeProj[0].age != null)
      ? last.gradeProj[0].age - 1 : null;
    const sd0 = (last.sdOff != null) ? Math.hypot(last.sdOff, last.sdDef) : 1.5;
    const out = [{ x: last.season, y: +v.toFixed(2), lo: v, hi: v }];
    for (let k = 1; k <= years; k++) {
      if (age != null) {
        const a1 = age + k, a0 = age + k - 1;
        v += AGE_QUAD_TJ * ((a1 - AGE_PEAK_TJ) ** 2 - (a0 - AGE_PEAK_TJ) ** 2);
      }
      const sd = 1.96 * sd0 * (1 + 0.25 * k);      // 95% band, widening with horizon
      out.push({ x: last.season + k, y: +v.toFixed(2), lo: +(v - sd).toFixed(2), hi: +(v + sd).toFixed(2) });
    }
    return out;
  }

  function renderTrajectory() {
    if (!TJ.ready) {
      TJ.ready = true;
      ensureNameIndex();
      $("#tj-add").addEventListener("change", () => {
        const q = $("#tj-add").value.trim().toLowerCase();
        const pid = _cmpByName[q];
        if (pid == null) return;
        if (!TJ.players.some((p) => p.pid === pid)) TJ.players.push({ pid });
        $("#tj-add").value = "";
        drawTrajectory();
      });
      ["#tj-proj", "#tj-opt", "#tj-band"].forEach((id) =>
        $(id).addEventListener("change", drawTrajectory));
      $("#tj-chips").addEventListener("click", (e) => {
        const b = e.target.closest("[data-rm]");
        if (b) { TJ.players = TJ.players.filter((p) => p.pid !== +b.dataset.rm); drawTrajectory(); }
      });
      if (!TJ.players.length) {
        // seed with the current #1 so the page never opens empty
        const best = D.players.filter((p) => !p.predictive && p._waaRank === 1)
          .sort((a, b) => b.season - a.season)[0];
        if (best) TJ.players.push({ pid: best.pid });
      }
    }
    drawTrajectory();
  }

  function drawTrajectory() {
    const years = +$("#tj-proj").value;
    const optimal = +$("#tj-opt").value === 1;
    const band = +$("#tj-band").value === 1;
    const careers = TJ.players.map((p) => tjCareer(p.pid)).filter(Boolean);
    $("#tj-chips").innerHTML = careers.map((c, i) =>
      `<button class="comp-chip" data-rm="${c.last.pid}" style="border-color:${TJ_COLORS[i % TJ_COLORS.length]}">` +
      `${c.name}<small>✕</small></button>`).join("");
    destroy("tj");
    if (!careers.length) {
      $("#tj-note").textContent = "Type a name above to add a player.";
      return;
    }
    const ds = [];
    careers.forEach((c, i) => {
      const col = TJ_COLORS[i % TJ_COLORS.length];
      // raw per-game on-court net (the noisy observable the model filters)
      if (c.dots.length) {
        ds.push({ label: "_dots", type: "scatter", data: c.dots,
                  pointBackgroundColor: hexA(col, 0.18), pointBorderColor: "transparent",
                  pointRadius: 1.8, pointHitRadius: 0 });
      }
      // 95% band on the realized rating (every player)
      if (band) {
        ds.push({ label: "_hi", data: c.hi, borderColor: "transparent", pointRadius: 0, fill: false });
        ds.push({ label: "_lo", data: c.lo, borderColor: "transparent", pointRadius: 0,
                  fill: "-1", backgroundColor: hexA(col, 0.09) });
      }
      ds.push({ label: c.name, data: c.pts, borderColor: col, backgroundColor: col,
                borderWidth: 2.2, pointRadius: 0, pointHitRadius: 6, tension: 0.25 });
      if (years > 0) {
        const proj = tjProjection(c, years, optimal);
        if (band) {
          ds.push({ label: "_hi", data: proj.map((p) => ({ x: p.x, y: p.hi })),
                    borderColor: "transparent", pointRadius: 0, fill: false });
          ds.push({ label: "_lo", data: proj.map((p) => ({ x: p.x, y: p.lo })),
                    borderColor: "transparent", pointRadius: 0, fill: "-1",
                    backgroundColor: hexA(col, 0.07) });
        }
        ds.push({ label: `${c.name} (proj)`, data: proj.map((p) => ({ x: p.x, y: p.y })),
                  borderColor: col, borderDash: [6, 5], borderWidth: 1.8, pointRadius: 2.5,
                  pointBackgroundColor: col, fill: false });
      }
    });
    charts.tj = new Chart($("#tj-chart"), {
      type: "line",
      data: { datasets: ds },
      options: {
        maintainAspectRatio: false, animation: false, parsing: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { labels: { filter: (it) => !it.text.startsWith("_"),
                              font: { family: "ui-monospace, Menlo, monospace", size: 11 } } },
          tooltip: { callbacks: {
            title: (its) => its.length ? seasonLabel(Math.ceil(its[0].parsed.x)) : "",
            label: (it) => it.dataset.label.startsWith("_") ? null
              : `${it.dataset.label}: ${signed(it.parsed.y, 1)} / 100 poss`,
          } },
        },
        scales: {
          x: { type: "linear", ticks: { stepSize: 1, callback: (v) => Number.isInteger(v) ? seasonLabel(v) : "",
               font: { family: "ui-monospace, Menlo, monospace", size: 10 } }, grid: { color: "rgba(20,19,16,.07)" } },
          y: { title: { display: true, text: "BOOKER (+/- per 100 poss)" },
               grid: { color: "rgba(20,19,16,.07)" } },
        },
      },
    });
    $("#tj-note").textContent = `Dots = game-by-game impact: each game's on-court net minus what the other nine players + home court predict (his leave-one-out residual), anchored to the season rating. Solid = rating as each season's evidence accumulates. Dashed = ${years}-year projection along the aging curve`
      + (optimal ? " at OPTIMAL usage (adds the un-credited role upside)" : " in the current role")
      + (band ? "; shaded = 95% credible band per player, widening with horizon." : ".");
    // decision read
    const vp = $("#tj-verdict-panel");
    if (years > 0 && careers.length >= 1) {
      vp.hidden = false;
      const reads = careers.map((c, i) => {
        const proj = tjProjection(c, years, optimal);
        const avg = proj.slice(1).reduce((a, p) => a + p.y, 0) / years;
        const dir = proj[years].y - proj[0].y;
        const trend = dir > 0.4 ? "ascending" : dir < -0.4 ? "declining" : "flat";
        return { name: c.name, avg, trend, now: proj[0].y, color: TJ_COLORS[i % TJ_COLORS.length] };
      }).sort((a, b) => b.avg - a.avg);
      $("#tj-verdict").innerHTML = reads.map((r, i) =>
        `<div class="tj-verdict-row"><span class="tj-rank">${i + 1}</span>` +
        `<span class="tj-name" style="border-left:3px solid ${r.color};padding-left:8px">${r.name}</span>` +
        `<span>now ${signed(r.now, 1)}</span><span>avg next ${years}y <b>${signed(r.avg, 1)}</b></span>` +
        `<span class="${r.trend === "ascending" ? "pos" : r.trend === "declining" ? "neg" : ""}">${r.trend}</span></div>`).join("") +
        `<p class="result-note">Ranked by projected average BOOKER over the window — the "who do you want for the next ${years} years" read. Toggle role scenario / horizon above to stress-test the decision.</p>`;
    } else vp.hidden = true;
  }

  function renderTrade() {
    const T = D.trade;
    if (!T || !T.players) {
      $("#trade-lede").textContent = "Trade data unavailable — re-run export_dashboard_data.py.";
      return;
    }
    const cap = T.capRules || {};
    $("#trade-lede").textContent = `${seasonLabel(T.season)} · trade two teams, or set Team B to “Free agents” to sign one · verdict = championship-odds swing`;
    if (cap.cap) {
      $("#trade-cap-hint").textContent =
        `cap ${money(cap.cap)} · tax ${money(cap.tax)} · MLE ${money(cap.mle)}`;
    }

    const teams = Object.keys(T.teamNet).sort();
    const selA = $("#trade-team-a"), selB = $("#trade-team-b");
    if (!tradeReady) {
      selA.innerHTML = teams.map((t) => `<option value="${t}">${t} \u2014 ${teamName(t)}</option>`).join("");
      selB.innerHTML = `<option value="FA">\u2726 Free agents (sign)</option>`
        + teams.map((t) => `<option value="${t}">${t} \u2014 ${teamName(t)}</option>`).join("");
      selA.value = teams.includes("LAL") ? "LAL" : teams[0];
      selB.value = teams.includes("NOP") ? "NOP" : teams[1] || teams[0];
      selA.addEventListener("change", fillTradePlayers);
      selB.addEventListener("change", fillTradePlayers);
      $("#trade-players-a").addEventListener("change", limitTradeSelection);
      $("#trade-players-b").addEventListener("change", limitTradeSelection);
      $("#trade-run").addEventListener("click", runTrade);
      tradeReady = true;
    }
    fillTradePlayers();
    drawTradeBaseline();
  }

  function tradeRoster(team) {
    if (team === "FA") return faPool();
    return (D.trade.players || [])
      .filter((p) => p.team === team)
      .sort((a, b) => b.minutes - a.minutes);
  }

  function fillTradePlayers() {
    const ta = $("#trade-team-a").value, tb = $("#trade-team-b").value;
    const ra = tradeRoster(ta), rb = tradeRoster(tb);
    const mk = (list, id) => {
      $(id).innerHTML = list.map((p) =>
        `<option value="${p.pid}">${p.player} (${p.minutes} min, ${signed(p.waaTotal, 1)} WAA, ${money(p.marketAav2026)})</option>`).join("");
    };
    mk(ra, "#trade-players-a");
    mk(rb, "#trade-players-b");
  }

  function netContribOnTeam(p, team, minutesOverride) {
    const tm = D.trade.teamMinutes[team] || 1;
    const mn = minutesOverride != null ? minutesOverride : p.minutes;
    const pres = mn / (tm / 5.0);
    return p.impactTotal * pres;
  }

  function playerSalary2026(p) {
    return p.marketAav2026 || p.fairAav2026 || 0;
  }

  function runTrade() {
    const T = D.trade;
    const ta = $("#trade-team-a").value, tb = $("#trade-team-b").value;
    if (ta === tb) return;
    const pidsA = selectedTradePids("#trade-players-a");
    const pidsB = selectedTradePids("#trade-players-b");
    if (!pidsA.length && !pidsB.length) return;

    const findP = (pool, id) => pool.find((p) => p.pid === id) || T.players.find((p) => p.pid === id);
    const outA = pidsA.map((id) => findP(tradeRoster(ta), id)).filter(Boolean);
    const outB = pidsB.map((id) => findP(tradeRoster(tb), id)).filter(Boolean);
    const inA = outB, inB = outA;

    let newNetA = T.teamNet[ta];
    let newNetB = T.teamNet[tb];
    outA.forEach((p) => { newNetA -= netContribOnTeam(p, ta); });
    outB.forEach((p) => { newNetB -= netContribOnTeam(p, tb); });
    inA.forEach((p) => { newNetA += netContribOnTeam(p, ta, p.minutes); });
    inB.forEach((p) => { newNetB += netContribOnTeam(p, tb, p.minutes); });

    const dA = T.k * (newNetA - T.teamNet[ta]);
    const dB = T.k * (newNetB - T.teamNet[tb]);
    const wA0 = T.teamSimWins[ta] || T.teamWins[ta] || 0;
    const wB0 = T.teamSimWins[tb] || T.teamWins[tb] || 0;
    const faB = tb === "FA";
    const chA0 = champCurve().base[ta] || 0, chB0 = champCurve().base[tb] || 0;
    const chA1 = champAfter(ta, newNetA);
    const chB1 = faB ? chB0 : champAfter(tb, newNetB);

    const salInA = inA.map(playerSalary2026);
    const matchA = salaryMatch(outA.map(playerSalary2026), salInA);
    const matchB = salaryMatch(outB.map(playerSalary2026), inB.map(playerSalary2026));

    // biggest rotation-balance shift after the deal (spacing / rim / creation / perim D)
    const fitTag = (team, out, gets) => {
      if (team === "FA") return "";
      const bf = teamFit(tradeRoster(team)), af = teamFit(rosterAfter(team, out, gets));
      if (!bf || !af) return "";
      let best = null;
      bf.forEach((b, i) => { const d = af[i].val - b.val; if (!best || Math.abs(d) > Math.abs(best.d)) best = { name: b.name, d }; });
      if (!best || Math.abs(best.d) < 1) return `<div class="sb-fit">fit \u2248 unchanged</div>`;
      return `<div class="sb-fit ${best.d >= 0 ? "pos" : "neg"}">${best.d >= 0 ? "\u25b2 +" : "\u25bc "}${Math.round(best.d)} ${best.name}</div>`;
    };
    const teamCard = (team, w0, dW, ch0, ch1, out, gets) => {
      const dch = (ch1 - ch0) * 100;
      return `<div class="sb-card">
        <div class="sb-team"><span class="team-tag">${team}</span> ${teamName(team)}</div>
        <div class="sb-hero">
          <div class="sb-hlbl">Championship odds</div>
          <div class="sb-hval">${(ch0 * 100).toFixed(1)}<span class="sb-to">\u2192</span><b class="${dch >= 0 ? "pos" : "neg"}">${(ch1 * 100).toFixed(1)}%</b></div>
          <div class="sb-hdelta ${dch >= 0 ? "pos" : "neg"}">${dch >= 0 ? "+" : ""}${dch.toFixed(1)} title pts</div>
        </div>
        <div class="sb-row"><span>Projected wins</span><b>${fmt(w0, 1)} \u2192 ${fmt(w0 + dW, 1)} <span class="${dW >= 0 ? "pos" : "neg"}">(${signed(dW, 1)})</span></b></div>
        ${fitTag(team, out, gets)}
        <div class="sb-moves">gets <b>${gets.map((p) => p.player).join(", ") || "\u2014"}</b></div>
        ${out.length ? `<div class="sb-moves out">sends ${out.map((p) => p.player).join(", ")}</div>` : ""}
      </div>`;
    };
    // verdict by championship-odds swing (the hero metric); tie-break on wins
    let verdict, vcls, legal;
    if (faB) {
      const cost = salInA.reduce((a, b) => a + b, 0);
      verdict = `${ta} signs ${inA.map((p) => p.player).join(", ") || "\u2014"}`;
      vcls = "win"; legal = true;
      $("#trade-cards").dataset.faCost = money(cost);
    } else {
      const swA = chA1 - chA0, swB = chB1 - chB0;
      if (Math.abs(swA - swB) < 0.005 && Math.abs(dA - dB) < 1.0) { verdict = "Fair deal \u2014 both sides roughly even"; vcls = "even"; }
      else if (swA > swB || (Math.abs(swA - swB) < 0.005 && dA >= dB)) verdict = `${ta} wins this deal`, vcls = "win";
      else verdict = `${tb} wins this deal`, vcls = "win";
      legal = matchA.ok && matchB.ok;
    }
    const legalHtml = faB
      ? `<span class="sb-legal ok">signs for ${$("#trade-cards").dataset.faCost}/yr</span>`
      : `<span class="sb-legal ${legal ? "ok" : "bad"}">${legal ? "\u2713 salary-legal" : "\u2717 salary mismatch"}</span>`;
    $("#trade-cards").innerHTML = `<div class="scoreboard">
      <div class="sb-verdict ${vcls}"><span>${verdict}</span>${legalHtml}</div>
      <div class="sb-grid ${faB ? "solo" : ""}">${teamCard(ta, wA0, dA, chA0, chA1, outA, inA)}${faB ? "" : teamCard(tb, wB0, dB, chB0, chB1, outB, inB)}</div>
    </div>`;

    // --- trade impact breakdown: skill profile, timeline (age), depth (minutes) ---
    const sideAgg = (list) => {
      const mn = list.reduce((a, p) => a + p.minutes, 0);
      const w = (f) => mn ? list.reduce((a, p) => a + (f(p) || 0) * p.minutes, 0) / mn : null;
      return { n: list.length, min: mn, off: w((p) => p.gradeOff), def: w((p) => p.gradeDef), age: w((p) => p.age) };
    };
    const dCell = (ov, nv, dec, invert) => {
      if (ov == null && nv == null) return "—";
      const d = (nv || 0) - (ov || 0);
      const cls = (invert ? -d : d) >= 0 ? "pos" : "neg";
      return `${ov == null ? "—" : ov.toFixed(dec)} → ${nv == null ? "—" : nv.toFixed(dec)} <span class="${cls}">(${d >= 0 ? "+" : ""}${d.toFixed(dec)})</span>`;
    };
    const impactRow = (team, out, inc) => {
      const o = sideAgg(out), i = sideAgg(inc);
      return `<div class="ti-team"><div class="ti-name"><span class="team-tag">${team}</span> sends ${out.length}, gets ${inc.length}</div>` +
        `<div class="ti-grid">` +
        `<div><span class="ti-k">Offense skill</span>${dCell(o.off, i.off, 0)}</div>` +
        `<div><span class="ti-k">Defense skill</span>${dCell(o.def, i.def, 0)}</div>` +
        `<div><span class="ti-k">Avg age</span>${dCell(o.age, i.age, 1, true)}</div>` +
        `<div><span class="ti-k">Rotation min in/out</span>${dCell(o.min, i.min, 0)}</div>` +
        `</div></div>`;
    };
    let imp = document.getElementById("trade-impact");
    if (!imp) { imp = document.createElement("div"); imp.id = "trade-impact"; imp.className = "panel"; $("#trade-cards").insertAdjacentElement("afterend", imp); }
    imp.innerHTML = `<h3 class="panel-title">Trade impact — skill, timeline & depth</h3>` +
      `<p class="result-note">Minutes-weighted grade of pieces leaving vs arriving (skill-profile change), average age (timeline fit), and rotation minutes (depth). Higher off/def grade is better; younger is greener.</p>` +
      impactRow(ta, outA, inA) + impactRow(tb, outB, inB);

    // --- roster fit / balance (Bible-grounded, descriptive) ---
    const fitRow = (team, out, inc) => {
      const before = teamFit(tradeRoster(team)), after = teamFit(rosterAfter(team, out, inc));
      if (!before || !after) {
        return `<div class="ti-team"><div class="ti-name"><span class="team-tag">${team}</span> roster fit — insufficient style data</div></div>`;
      }
      const cells = before.map((b, i) => {
        const a = after[i], d = a.val - b.val, cls = d >= 0 ? "pos" : "neg";
        const gap = a.val < 34 ? ` <span class="fit-gap" title="${b.desc}">thin</span>` : "";
        return `<div><span class="ti-k">${b.name}${gap}</span>${Math.round(b.val)} → ${Math.round(a.val)} ` +
          `<span class="${cls}">(${d >= 0 ? "+" : ""}${Math.round(d)})</span></div>`;
      }).join("");
      return `<div class="ti-team"><div class="ti-name"><span class="team-tag">${team}</span> rotation balance (percentile)</div><div class="ti-grid">${cells}</div></div>`;
    };
    let fit = document.getElementById("trade-fit");
    if (!fit) { fit = document.createElement("div"); fit.id = "trade-fit"; fit.className = "panel"; imp.insertAdjacentElement("afterend", fit); }
    fit.innerHTML = `<h3 class="panel-title">Roster fit &amp; balance</h3>` +
      `<p class="result-note">How each rotation's profile shifts after the trade on the four dimensions that decide half-court games — spacing, rim protection, shot creation, perimeter defense (Basketball Bible). Minutes-weighted percentiles; “thin” flags a bottom-third dimension. Descriptive roster construction — the win projection above stays talent-based.</p>` +
      fitRow(ta, outA, inA) + fitRow(tb, outB, inB);

    const namesA = outA.map((p) => p.player).join(", ") || "(none)";
    const namesB = outB.map((p) => p.player).join(", ") || "(none)";
    $("#trade-lede").textContent = `${ta} sends ${namesA} for ${namesB}.`;

    const contractRows = [...outA, ...inA, ...outB, ...inB]
      .filter((p, i, arr) => arr.findIndex((x) => x.pid === p.pid) === i);
    $("#trade-contract-table thead").innerHTML =
      "<tr><th class=\"col-text\">Player</th><th>Team</th><th>Age</th><th>WAA</th>" +
      "<th>True Value</th><th>Market AAV</th><th>Surplus</th><th>Tier</th></tr>";
    $("#trade-contract-table tbody").innerHTML = contractRows.map((p) => {
      const tier = p.waaTotal >= 6 ? "max" : p.waaTotal >= 3.5 ? "star" : p.waaTotal >= 1.5 ? "starter" : "MLE band";
      return `<tr><td class="col-text namecell">${p.player}</td><td><span class="team-tag">${p.team}</span></td>` +
        `<td>${p.age || "\u2014"}</td><td>${signed(p.waaTotal, 1)}</td>` +
        `<td>${money(p.fairAav2026)}</td><td>${money(p.marketAav2026)}${p.isKnownDeal ? "*" : ""}</td>` +
        `<td class="${p.surplus >= 0 ? "pos" : "neg"}">${money(p.surplus)}</td>` +
        `<td>${tier}</td></tr>`;
    }).join("");

    const rows = [
      { team: ta, before: wA0, after: wA0 + dA, netB: T.teamNet[ta], netA: newNetA },
      { team: tb, before: wB0, after: wB0 + dB, netB: T.teamNet[tb], netA: newNetB },
    ];
    $("#trade-table thead").innerHTML =
      "<tr><th>Team</th><th>Proj wins before</th><th>Proj wins after</th><th>Delta</th><th>Net before</th><th>Net after</th></tr>";
    $("#trade-table tbody").innerHTML = rows.map((r) =>
      `<tr><td><span class="team-tag">${r.team}</span> ${teamName(r.team)}</td>` +
      `<td>${fmt(r.before, 1)}</td><td><b>${fmt(r.after, 1)}</b></td>` +
      `<td class="${r.after - r.before >= 0 ? "pos" : "neg"}">${signed(r.after - r.before, 1)}</td>` +
      `<td>${signed(r.netB, 1)}</td><td>${signed(r.netA, 1)}</td></tr>`).join("");
  }

  function drawTradeBaseline() {
    const T = D.trade;
    if (!T) return;
    const pre = (D.preseason || []).filter((p) => p.season === T.season)
      .sort((a, b) => b.projWins - a.projWins);
    if (!pre.length) return;
    $("#trade-table thead").innerHTML =
      "<tr><th>Team</th><th>Preseason proj W</th><th>Pred net</th><th>Playoff %</th></tr>";
    $("#trade-table tbody").innerHTML = pre.map((r) =>
      `<tr><td><span class="team-tag">${r.team}</span> ${teamName(r.team)}</td>` +
      `<td><b>${fmt(r.projWins, 1)}</b></td><td>${signed(r.predNet, 1)}</td>` +
      `<td>${(r.pPlayoff * 100).toFixed(0)}%</td></tr>`).join("");
  }

  /* ===================================================================== *
   *  STAT LEADERS
   * ===================================================================== */
  // one-table true-skill leaders board (key, short header). Values + hover-raw are
  // formatted from each skill's stored `fmt` via fmtSkillVal / skillTitle.
  const LEADER_COLS = [
    ["grade", "Grade"],
    ["bookerScore", "BOOKER"],
    ["offense", "Off"],
    ["defense", "Def"],
    ["three_pct", "3P%"],
    ["rim_finish", "Rim%"],
    ["efficiency", "eFG%"],
    ["shot_making", "ShotMk"],
    ["self_creation", "SelfCr"],
    ["playmaking", "AST"],
    ["creation", "Creat"],
    ["foul_draw", "FTDraw"],
    ["free_throw", "FT%"],
    ["ball_security", "BallSec"],
    ["off_gravity", "OffGrav"],
    ["on_gravity", "OnGrav"],
    ["rebounding", "REB"],
    ["steals", "STL"],
    ["rim_protect", "BLK"],
    ["discipline", "Disc"],
    ["rim_contest", "RimCon"],
    ["perimeter_contest", "PerimCon"],
  ];
  let leadersInit = false;
  const ldSort = { key: "grade", dir: -1 };
  function renderLeaders() {
    if (!leadersInit) {
      leadersInit = true;
      const seasons = Array.from(new Set(D.players.filter((p) => hasSkills(p))
        .map((p) => p.season))).sort((a, b) => b - a);
      $("#ld-season").innerHTML = seasons.map((s) => `<option value="${s}">${seasonLabelFull(s)}</option>`).join("");
      defaultSeason($("#ld-season"));
      $("#ld-season").addEventListener("change", drawLeaders);
      $("#ld-min").addEventListener("change", drawLeaders);
      $("#ld-table thead").addEventListener("click", (e) => {
        const th = e.target.closest("th"); if (!th || !th.dataset.key) return;
        const k = th.dataset.key;
        if (ldSort.key === k) ldSort.dir *= -1; else { ldSort.key = k; ldSort.dir = -1; }
        drawLeaders();
      });
    }
    drawLeaders();
  }
  function ldVal(p, key) {
    if (key === "grade") return p.grade;
    if (key === "bookerScore") return p.bookerScore;
    if (key === "min") return p.min;
    if (key === "player") return p.player;
    return p.skills && p.skills[key] ? p.skills[key].val : null;
  }
  function drawLeaders() {
    const season = +$("#ld-season").value, mn = +$("#ld-min").value;
    const pool = D.players.filter((p) => p.season === season && hasSkills(p) && p.min >= mn);
    pool.sort((a, b) => {
      const A = ldVal(a, ldSort.key), B = ldVal(b, ldSort.key);
      if (A == null) return 1; if (B == null) return -1;
      if (typeof A === "string") return A.localeCompare(B) * (ldSort.dir < 0 ? -1 : 1);
      return (A - B) * ldSort.dir;
    });
    const arrow = (k) => ldSort.key === k ? `<span class="arrow">${ldSort.dir < 0 ? "▼" : "▲"}</span>` : "";
    $("#ld-table thead").innerHTML = "<tr>" +
      `<th>#</th><th class="col-text" data-key="player">Player</th><th>Team</th>` +
      `<th data-key="min">Min${arrow("min")}</th>` +
      LEADER_COLS.map(([k, sh]) => `<th data-key="${k}" title="${k}">${sh}${arrow(k)}</th>`).join("") + "</tr>";
    $("#ld-table tbody").innerHTML = pool.slice(0, 200).map((p, i) => {
      const cells = LEADER_COLS.map(([k]) => {
        if (k === "grade") {
          return p.grade != null
            ? `<td style="background:${pctColor(p.grade, 0.5)}" title="sticky age-curved team-independent grade"><b>${p.gradeLetter}</b> <small>${p.grade}</small></td>`
            : "<td>—</td>";
        }
        if (k === "bookerScore") {
          return p.bookerScore != null ? `<td><b>${signed(p.bookerScore, 1)}</b></td>` : "<td>—</td>";
        }
        const s = p.skills[k];
        if (!s) return "<td>—</td>";
        return `<td style="background:${pctColor(s.pct, 0.5)}" title="${skillTitle(s)}">${fmtSkillVal(s)}</td>`;
      }).join("");
      return `<tr class="clickable" data-pid="${p.pid}" data-season="${p.season}">` +
        `<td>${i + 1}</td><td class="col-text">${p.player}</td>` +
        `<td><span class="team-tag">${p.team}</span></td><td>${p.min.toLocaleString()}</td>${cells}</tr>`;
    }).join("");
    $$("#ld-table tbody tr.clickable").forEach((tr) =>
      tr.addEventListener("click", () => openPlayer(+tr.dataset.pid)));
  }

  /* ===================================================================== *
   *  MATCHUPS  (descriptive scouting view; not a win-prob predictor)
   * ===================================================================== */
  let muInit = false;
  const muSeasons = () => Array.from(new Set(D.players.filter(hasSkills).map((p) => p.season))).sort((a, b) => b - a);
  const muTeams = (s) => Array.from(new Set(D.players.filter((p) => p.season === s && hasSkills(p)).map((p) => p.team)))
    .filter((t) => t && t !== "?").sort();
  function teamProfile(season, team) {
    const roster = D.players.filter((p) => p.season === season && p.team === team && p.min >= 200 && hasSkills(p));
    if (!roster.length) return null;
    const wpct = {}, keys = new Set();
    roster.forEach((p) => Object.keys(p.skills).forEach((k) => keys.add(k)));
    keys.forEach((k) => {
      let num = 0, den = 0;
      roster.forEach((p) => { const s = p.skills[k]; if (s) { num += s.pct * p.min; den += p.min; } });
      if (den) wpct[k] = num / den;
    });
    const wh = (list) => { const t = list.reduce((a, p) => a + p.min, 0); return t ? list.reduce((a, p) => a + (p.heightIn || 0) * p.min, 0) / t : null; };
    const byH = roster.filter((p) => p.heightIn).sort((a, b) => b.heightIn - a.heightIn);
    const n = Math.max(3, Math.round(byH.length * 0.4));
    let gn = 0, gd = 0;
    roster.forEach((p) => { if (p.grade != null) { gn += p.grade * p.min; gd += p.min; } });
    return { wpct, front: wh(byH.slice(0, n)), back: wh(byH.slice(-n)), grade: gd ? gn / gd : null };
  }
  const avgKeys = (prof, ks) => { const v = ks.map((k) => prof.wpct[k]).filter((x) => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const ftIn = (inch) => inch == null ? "—" : `${Math.floor(inch / 12)}'${Math.round(inch % 12)}"`;
  const muPct = (v) => v == null ? "—" : `<span class="mu-pct" style="color:${pctColor(v)}">${Math.round(v)}</span>`;
  function muSection(title, sub, rows) {
    let h = `<div class="mu-sec">${title}<small>${sub}</small></div>`;
    rows.forEach(([lab, lv, rv, f]) => {
      const fmt = f || muPct, lWin = lv != null && rv != null && lv >= rv, rWin = lv != null && rv != null && rv > lv;
      h += `<div class="mu-row"><span class="mu-a ${lWin ? "mu-win" : ""}">${fmt(lv)}</span>` +
        `<span class="mu-lab">${lab}</span><span class="mu-b ${rWin ? "mu-win" : ""}">${fmt(rv)}</span></div>`;
    });
    return h;
  }
  function renderMatchup() {
    const seasons = muSeasons();
    if (!seasons.length) { $("#mu-body").innerHTML = "<p class='result-note'>No skill data available.</p>"; return; }
    if (!muInit) {
      muInit = true;
      $("#mu-season").innerHTML = seasons.map((s) => `<option value="${s}">${seasonLabelFull(s)}</option>`).join("");
      defaultSeason($("#mu-season"));
      $("#mu-season").addEventListener("change", () => { fillMuTeams(); drawMatchup(); });
      $("#mu-team-a").addEventListener("change", drawMatchup);
      $("#mu-team-b").addEventListener("change", drawMatchup);
      fillMuTeams();
    }
    drawMatchup();
  }
  function fillMuTeams() {
    const teams = muTeams(+$("#mu-season").value), a = $("#mu-team-a"), b = $("#mu-team-b");
    const opt = teams.map((t) => `<option value="${t}">${t} — ${teamName(t)}</option>`).join("");
    a.innerHTML = opt; b.innerHTML = opt;
    a.value = teams.includes("OKC") ? "OKC" : teams[0];
    b.value = teams.includes("SAS") ? "SAS" : (teams[1] || teams[0]);
  }
  function drawMatchup() {
    const s = +$("#mu-season").value, ta = $("#mu-team-a").value, tb = $("#mu-team-b").value;
    const A = teamProfile(s, ta), B = teamProfile(s, tb);
    if (!A || !B) { $("#mu-body").innerHTML = "<p class='result-note'>Pick two teams with rotation data.</p>"; return; }
    const RIM_ATT = ["rim_finish", "on_gravity"], RIM_PROT = ["rim_protect", "rim_contest"],
      SHOOT = ["three_pct", "off_gravity"], PERIM_D = ["perimeter_contest", "steals"], CREATE = ["playmaking", "creation"];
    let h = `<div class="mu-head"><div class="mu-t">${ta}<small>${teamName(ta)}</small>` +
      `<b style="color:${pctColor(A.grade)}">${A.grade != null ? Math.round(A.grade) : "—"}</b><em>team grade</em></div>` +
      `<div class="mu-vs">vs</div><div class="mu-t">${tb}<small>${teamName(tb)}</small>` +
      `<b style="color:${pctColor(B.grade)}">${B.grade != null ? Math.round(B.grade) : "—"}</b><em>team grade</em></div></div>`;
    h += muSection("Size", `${ta} — ${tb} · minutes-weighted rotation height`, [
      ["Frontcourt height", A.front, B.front, ftIn], ["Backcourt height", A.back, B.back, ftIn]]);
    h += muSection(`When ${ta} attacks`, `${ta} offense — ${tb} defense · percentile vs league`, [
      ["Rim", avgKeys(A, RIM_ATT), avgKeys(B, RIM_PROT)],
      ["Outside shooting", avgKeys(A, SHOOT), avgKeys(B, PERIM_D)],
      ["Playmaking / creation", avgKeys(A, CREATE), avgKeys(B, PERIM_D)]]);
    h += muSection(`When ${tb} attacks`, `${ta} defense — ${tb} offense · percentile vs league`, [
      ["Rim", avgKeys(A, RIM_PROT), avgKeys(B, RIM_ATT)],
      ["Outside shooting", avgKeys(A, PERIM_D), avgKeys(B, SHOOT)],
      ["Playmaking / creation", avgKeys(A, PERIM_D), avgKeys(B, CREATE)]]);
    $("#mu-body").innerHTML = h;
  }

  /* ===================================================================== *
   *  DIAGNOSTICS
   * ===================================================================== */
  let diagReady = false;
  function renderDiagnostics() {
    const d = D.diagnostics;
    if (!d) { $("#diag-intro").textContent = "Diagnostics unavailable."; return; }
    $("#diag-intro").innerHTML = `${d.model} · 95% credible intervals (±1.96·SD).`;

    // backtest table
    const bt = d.backtest;
    $("#diag-backtest thead").innerHTML =
      "<tr><th class='col-text'>Model</th><th>Net RMSE</th><th>Wins R²</th><th>90% cov.</th></tr>";
    $("#diag-backtest tbody").innerHTML = bt.rows.map((r) =>
      `<tr><td class="col-text">${r.model}</td><td>${fmt(r.netRmse, 2)}</td>` +
      `<td>${fmt(r.winsR2, 3)}</td><td>${r.cov90 != null ? fmt(r.cov90, 3) : "—"}</td></tr>`).join("");
    $("#diag-backtest-note").textContent = bt.note;

    // True Value equation
    const tv = d.trueValue;
    $("#diag-tv-eq").innerHTML =
      `monotonic GBM (BOOKER&uarr;, age&darr;), uncapped tail · n=${tv.n} signings · value read at age ${tv.refAge}`;

    // credible intervals table (top players)
    $("#diag-intervals thead").innerHTML =
      "<tr><th class='col-text'>Player</th><th>BOOKER</th><th>Off ±95%</th><th>Def ±95%</th></tr>";
    $("#diag-intervals tbody").innerHTML = d.intervals.map((p) => {
      const ci = (v, sd) => `${signed(v, 1)}<span class="ci">±${(1.96 * sd).toFixed(1)}</span>`;
      return `<tr><td class="col-text">${p.player}</td><td>${signed(p.booker, 1)}</td>` +
        `<td>${ci(p.off, p.offSd)}</td><td>${ci(p.def, p.defSd)}</td></tr>`;
    }).join("");

    if (diagReady) return;   // charts only need building once
    diagReady = true;

    // SD-vs-minutes bars
    const sd = d.sdByMinutes;
    new Chart($("#diag-sd-chart"), {
      type: "bar",
      data: { labels: sd.map((b) => b.bin),
        datasets: [
          { label: "Off SD", data: sd.map((b) => b.sdOff), backgroundColor: "rgba(122,40,32,.75)" },
          { label: "Def SD", data: sd.map((b) => b.sdDef), backgroundColor: "rgba(47,93,52,.7)" },
        ] },
      options: { maintainAspectRatio: false,
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, usePointStyle: true } } },
        scales: { x: { title: { display: true, text: "minutes played" }, grid: { display: false } },
          y: { title: { display: true, text: "posterior SD (pts/100)" }, grid: { color: COL.grid } } } },
    });

    // FA scatter: BOOKER (prior season) vs signed AAV, with the fitted monotonic
    // True Value curve (young ref age) and an older-age curve (the age penalty).
    const pts = d.faScatter.map((f) => ({ x: f.booker, y: f.aav / 1e6 }));
    const young = tv.curveYoung.map((c) => ({ x: c.booker, y: c.aav / 1e6 }));
    const old = tv.curveOld.map((c) => ({ x: c.booker, y: c.aav / 1e6 }));
    new Chart($("#diag-fa-chart"), {
      type: "scatter",
      data: { datasets: [
        { label: "FA signings", data: pts, backgroundColor: "rgba(36,28,18,.4)", pointRadius: 3 },
        { type: "line", label: `True Value (age ${tv.refAge})`, data: young, borderColor: COL.accent,
          borderWidth: 2.5, pointRadius: 0, fill: false, tension: .2 },
        { type: "line", label: "same skill, age 33", data: old, borderColor: COL.grey,
          borderWidth: 2, borderDash: [5, 4], pointRadius: 0, fill: false, tension: .2 },
      ] },
      options: { maintainAspectRatio: false,
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, usePointStyle: true } } },
        scales: { x: { title: { display: true, text: "BOOKER score (season before FA)" }, grid: { color: COL.grid } },
          y: { title: { display: true, text: "value ($M, 2026)" }, grid: { color: COL.grid } } } },
    });
  }

  /* ===================================================================== *
   *  METHODOLOGY
   * ===================================================================== */
  function renderMethod() {
    $("#method-body").innerHTML = `
      <h2>How BOOKER WAA works</h2>
      <p>BOOKER estimates each player's two-way impact <em>in the context of who else is on the
      floor</em>, then rolls those impacts up to a team's expected point differential and converts
      that to wins. Every number on this site comes from the pipeline below.</p>
      <div class="pipe">
        <span class="step">Play-by-play stints</span><span class="arr">&rarr;</span>
        <span class="step">BookerFormer Bayesian RAPM</span><span class="arr">&rarr;</span>
        <span class="step">Player WAA + uncertainty</span><span class="arr">&rarr;</span>
        <span class="step">Team net rating</span><span class="arr">&rarr;</span>
        <span class="step">Wins</span>
      </div>
      <h3>1. On-court stints</h3>
      <p>From public play-by-play (2015-2026) we reconstruct the exact 10 players on the court for
      every stint, with its point margin and duration &mdash; about 33,000 stints per season. The
      2025-26 season is pulled live from the NBA Stats API (play-by-play plus boxscore starters,
      with substitutions walked forward to recover every lineup).</p>
      <h3>2. Box-prior blended RAPM</h3>
      <p>A ridge regression (RAPM) attributes each stint's per-100 margin to the players on the floor,
      adjusting for teammates and opponents. Rather than shrinking toward zero, we shrink toward each
      player's box-score prior (Basketball-Reference BPM) &mdash; the RPM/PIPM-style approach that keeps
      low-minute estimates stable.</p>
      <h3>3. BookerFormer — Bayesian transformer attribution (offense / defense)</h3>
      <p>Each 5v5 stint is modeled as a <strong>matchup of two sets</strong> — five offensive vs five defensive
      players — predicting the offense's points/100. <strong>BookerFormer</strong> gives every player two scalar
      <em>variational random effects</em> (offense and defense) with priors centered on the box-score (BPM) split;
      each effect carries a mean <em>and</em> a posterior variance, so a player's rating comes with a calibrated
      credible interval that <strong>narrows as he logs minutes</strong>. A Set-Transformer with BayesFormer-style
      Monte-Carlo dropout sits on top as an optional synergy / matchup layer. Per-player offense/defense ratings
      (with 95% intervals) appear on the player page and in the leaderboard WAA tooltip.</p>
      <p style="color:var(--muted-2);font-size:12.5px">Out-of-sample backtests: the additive Bayesian model beats
      the prior ridge RAPM on team net-rating and wins prediction (wins R&sup2; 0.61 vs 0.55) with ~90% interval
      coverage. The transformer layer is kept opt-in because, at current stint-data scale (~33k stints/season),
      its extra capacity did not improve out-of-sample accuracy.</p>
      <h3>4. Betting lines (SBR + local Pinnacle / oddsData)</h3>
      <p>Closing moneylines come from the Sportsbook Review archive (2015&ndash;2022) and local
      CSV feeds: <code>oddsData.csv</code> (2008&ndash;2023, partial last season) and
      <code>nba_main_lines.csv</code> Pinnacle snapshots (2025&ndash;26). Vig is removed before
      comparing BOOKER win probabilities to the market on the Game Odds tab.</p>
      <h3>5. WAA, net rating, wins</h3>
      <p><strong>WAA wins</strong> is a player's wins contributed above an average player, given his
      minutes and role. Summing a roster's WAA gives the team's predicted net rating, which a learned
      linear map turns into wins (a team of league-average players sits at ~41 wins).</p>
      <h3>6. Preseason forecast</h3>
      <p>Before a season starts we know each roster but none of its results, so player value comes from
      <em>prior seasons only</em> (3-year window, recency-weighted, aged to the target year). Rolling the
      roster up gives a predicted net rating and win total; a Monte-Carlo of the full schedule yields the
      win-total range and playoff odds shown on the Win Forecast tab. Pooled out-of-sample error is
      <strong>${fmt(D.metrics.find(m=>m.label==="Pooled").winsRmse,1)} wins RMSE</strong>, beating both a
      predict-41 baseline (${fmt(D.baselines.predict41,1)}) and the older box-only model's
      <em>retrodictive</em> fit (${fmt(D.baselines.oldBox,1)}).</p>
      <h3>7. In-season updating</h3>
      <p>As games are played we rebuild impacts from prior seasons <em>plus the current season's stints to
      date</em>; the live possessions naturally outweigh the prior as they accumulate. Banked wins are added
      to the expected wins over the remaining schedule, so each team's projected final win total updates
      through the year.</p>
      <h3>8. Per-game odds vs the market</h3>
      <p>For every game we form a home win probability from pre-tip impacts (home court folded into a
      calibrated logistic on the net-rating gap) and compare it to the closing moneyline with vig removed.
      The Game Odds tab reports log-loss, Brier score, accuracy and flat-bet ROI &mdash; an honest accounting
      against a very efficient market.</p>
      <h3>9. Trade machine, contracts &amp; forward seasons</h3>
      <p>The Trade Machine supports up to six players per side. Wins come from
      re-aggregating fit-adjusted offensive/defensive impacts on a cloned schedule (2026&ndash;27 reused for
      2027&ndash;28). Contract fair value uses local salary files (2000&ndash;2025 CSV + 2026 BBR export)
      and FA signings inflated to <strong>2026 cap dollars</strong>, with position-specific big-man premia.</p>
      <p style="color:var(--muted-2);font-size:12.5px">Data: NBA Stats API &amp; shufinskiy/nba_data play-by-play; Basketball-Reference box priors; historical closing moneylines (Sportsbook Review archive). Generated ${D.generated}.</p>`;
  }

  /* ---- boot ------------------------------------------------------------ */
  initFilters();
  renderLB();
  renderMethod();
  $("#foot-note").textContent =
    `BOOKER WAA \u00b7 ${D.players.length.toLocaleString()} player-seasons \u00b7 ` +
    `${seasonLabel(D.seasons[0])} to ${seasonLabel(latest)} \u00b7 built ${D.generated}`;
})();
