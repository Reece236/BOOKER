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
    if (note) note.textContent = `${seasonLabelFull(row.season)}${row.archetype ? " · " + row.archetype : ""} · true-skill percentile vs league (hover a row for the raw stat)`;
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
  let cmpInit = false, curProfRow = null;
  function setupCompare(rowA) {
    curProfRow = rowA;
    const sel = $("#skill-compare");
    if (!sel) return;
    if (!cmpInit) {
      cmpInit = true;
      const seen = {};
      D.players.forEach((p) => { if (!seen[p.pid]) seen[p.pid] = p.player; });
      sel.insertAdjacentHTML("beforeend", Object.entries(seen)
        .sort((a, b) => a[1].localeCompare(b[1]))
        .map(([pid, nm]) => `<option value="${pid}">${nm}</option>`).join(""));
      sel.addEventListener("change", () => {
        const bpid = +sel.value;
        if (!bpid || !curProfRow) { renderSkillProfile(curProfRow); return; }
        const rowB = profileSeasonFor(bpid);
        rowB ? renderCompareProfile(curProfRow, rowB) : renderSkillProfile(curProfRow);
      });
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
      html += `<div class="vderiv-line"><span>BOOKER score <em>(WAA / 3000 poss, no aging)</em></span><b>${signed(booker, 1)}</b></div>`;
    }
    if (tv != null) {
      html += `<div class="vderiv-line"><span>True Value <em>(skill-based fair AAV)</em></span><b>${money(tv)}</b></div>`;
      if (mkt != null) {
        html += `<div class="vderiv-line"><span>Actual contract</span><b>${money(mkt)}</b></div>`;
        const surp = tv - mkt;
        html += `<div class="vderiv-line"><span>Surplus</span><b class="${surp >= 0 ? "pos" : "neg"}">${money(surp)}</b></div>`;
      }
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
    if (v === "gameodds") renderGameOdds();
    if (v === "leaders") renderLeaders();
    if (v === "matchups") renderMatchup();
    if (v === "diagnostics") renderDiagnostics();
    if (v === "method") renderMethod();
  });

  /* ===================================================================== *
   *  LEADERBOARD
   * ===================================================================== */
  const LB_COLS = [
    { key: "rankModel", label: "#", text: true,
      cell: (r) => `<span class="rankcell">${r.rankModel != null ? r.rankModel : r.rank}</span>` },
    { key: "player", label: "Player", text: true, cell: (r) => `<span class="namecell">${r.player}</span>` },
    { key: "team", label: "Team", text: true, cell: (r) => `<span class="team-tag">${r.team}</span>` },
    { key: "season", label: "Season", cell: (r) => seasonLabel(r.season) },
    { key: "grade", label: "Grade", text: true,
      cell: (r) => r.grade != null ? `<span class="grade grade-${(r.gradeLetter||"").replace("+","p").replace("-","m")}" title="sticky age-curved team-independent grade">${r.gradeLetter} <small>${r.grade}</small></span>` : "\u2014" },
    { key: "min", label: "Min", cell: (r) => r.min.toLocaleString() },
    { key: "bookerScore", label: "BOOKER", cell: (r) => r.bookerScore != null
      ? `<span class="${r.bookerScore >= 0 ? "pos" : "neg"}" title="BOOKER score: forward-looking predictive WAA per 3000 possessions (pure skill, no aging).">${signed(r.bookerScore, 1)}</span>`
      : "\u2014" },
    { key: "waaModel", label: "WAA", cell: (r) => waaBar(r) },
    { key: "waaOff", label: "Off", cell: (r) => r.waaOff != null
      ? `<span class="${r.waaOff >= 0 ? "pos" : "neg"}" title="Offensive WAA">${signed(r.waaOff, 1)}</span>` : "\u2014" },
    { key: "waaDef", label: "Def", cell: (r) => r.waaDef != null
      ? `<span class="${r.waaDef >= 0 ? "pos" : "neg"}" title="Defensive WAA">${signed(r.waaDef, 1)}</span>` : "\u2014" },
    { key: "real_pm", label: "Real +/-", cell: (r) => r.real_pm != null
      ? `<span class="${r.real_pm >= 0 ? "pos" : "neg"}" title="Actual on-court net rating /100 possessions (descriptive; for isolated impact use BOOKER)">${signed(r.real_pm, 1)}</span>` : "\u2014" },
    { key: "tm_quality", label: "Tm Q", cell: (r) => r.tm_quality != null
      ? `<span title="Avg teammate rating on court with him (impact/100) -- supporting cast">${signed(r.tm_quality, 1)}</span>` : "\u2014" },
    { key: "opp_quality", label: "Opp Q", cell: (r) => r.opp_quality != null
      ? `<span title="Avg opponent rating on court against him (impact/100) -- competition faced">${signed(r.opp_quality, 1)}</span>` : "\u2014" },
    { key: "waaEnhanced", label: "Enh WAA", cell: (r) => r.waaEnhanced != null
      ? `<span class="${r.waaEnhanced >= 0 ? "pos" : "neg"}" title="Prior teammate-fit ridge model (enhanced), shown for comparison. Click to sort and compare orderings.">${signed(r.waaEnhanced, 1)}</span>`
      : "\u2014" },
    { key: "trueValue", label: "True Value", cell: (r) => (r.trueValue != null ? r.trueValue : r.fairAav2026) != null
      ? `<span title="Skill-based fair AAV: predicted from BOOKER score with the market's age penalty removed.">${money(r.trueValue != null ? r.trueValue : r.fairAav2026)}</span>` : "\u2014" },
    { key: "marketAav2026", label: "Contract", cell: (r) => r.marketAav2026 != null ? money(r.marketAav2026) : "\u2014" },
    { key: "surplus", label: "Surplus", cell: (r) => r.surplus != null
      ? `<span class="${r.surplus >= 0 ? "pos" : "neg"}" title="True Value minus actual contract">${money(r.surplus)}</span>` : "\u2014" },
  ];
  let maxWaa = Math.max.apply(null, D.players.map((p) => p.waaModel != null ? p.waaModel : p.waa));
  function waaBar(r) {
    const v = r.waaModel != null ? r.waaModel : r.waa;
    const w = Math.max(0, (v / maxWaa) * 100);
    const c = v >= 0 ? "pos" : "neg";
    const tip = (r.bfOff100 != null && r.sdOff != null)
      ? ` title="BookerFormer rating/100 — Off ${signed(r.bfOff100, 1)}±${(1.96 * r.sdOff).toFixed(1)}, ` +
        `Def ${signed(r.bfDef100, 1)}±${(1.96 * r.sdDef).toFixed(1)} (95% CI)"`
      : "";
    return `<span class="bar-cell"${tip}><span class="bar" style="width:${w}%"></span>` +
           `<span class="${c}">${signed(v, 1)}</span></span>`;
  }
  // BookerFormer offense/defense rating per 100 with a 95% credible interval.
  function bfCell(r) {
    if (r.bfOff100 == null || r.sdOff == null) return "—";
    const ci = (v, sd) => `${signed(v, 1)}<span class="ci">±${(1.96 * sd).toFixed(1)}</span>`;
    return `<span title="BookerFormer Bayesian offense / defense rating per 100 possessions, with 95% credible interval. Intervals narrow as a player logs more minutes.">` +
           `${ci(r.bfOff100, r.sdOff)} / ${ci(r.bfDef100, r.sdDef)}</span>`;
  }
  const lbSort = { key: "bookerScore", dir: -1 };

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
      if (lbSort.key === "waaModel") {
        A = a.waaModel != null ? a.waaModel : a.waa;
        B = b.waaModel != null ? b.waaModel : b.waa;
      }
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
      (rows.length > 400 ? " (showing top 400)" : "") +
      ` \u00b7 ranked by BOOKER (predictive WAA / 3000 poss, skill); WAA = hybrid skills+impact` +
      ` \u00b7 search is scoped to the selected season (choose \u201cAll seasons\u201d to search every year)` +
      ` \u00b7 True Value = skill-based AAV from BOOKER, age penalty removed; surplus = True Value \u2212 contract`;
  }
  $("#lb-table thead").addEventListener("click", (e) => {
    const th = e.target.closest("th"); if (!th) return;
    const k = th.dataset.key;
    if (lbSort.key === k) lbSort.dir *= -1;
    else {
      lbSort.key = k;
      lbSort.dir = (k === "player" || k === "team" || k === "modelType") ? 1 : -1;
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
  function openPlayer(pid) {
    const seasons = D.players.filter((p) => p.pid === pid).sort((a, b) => a.season - b.season);
    if (!seasons.length) return;
    const name = seasons[seasons.length - 1].player;
    const teams = Array.from(new Set(seasons.map((s) => s.team)));
    const totWaa = seasons.reduce((a, s) => a + s.waa, 0);
    const peak = seasons.reduce((a, s) => (s.waa > a.waa ? s : a));
    const peak32 = seasons.reduce((a, s) => ((s.waa32 != null ? s.waa32 : -999) > (a.waa32 != null ? a.waa32 : -999) ? s : a));
    const bestRank = Math.min.apply(null, seasons.map((s) => s.rank));
    const latest = seasons[seasons.length - 1];
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
      ` \u00b7 ${seasonLabel(seasons[0].season)}\u2013${seasonLabel(seasons[seasons.length - 1].season)}</span>` +
      gradeHero +
      `<div class="ph-stat">` +
      `<div class="s"><div class="k">Total WAA</div><div class="v">${signed(totWaa, 1)}</div></div>` +
      `<div class="s"><div class="k">Peak WAA@32</div><div class="v">${peak32.waa32 != null ? signed(peak32.waa32, 1) : "\u2014"}</div></div>` +
      `<div class="s"><div class="k">Peak season</div><div class="v">${signed(peak.waa, 1)}</div></div>` +
      `<div class="s"><div class="k">Best rank</div><div class="v">#${bestRank}</div></div></div>`;

    // profile the most recent season with a true-skill breakdown; PBP rate skills run
    // through 2024-25, shooting through the latest season; fall back to the latest.
    const rev = [...seasons].reverse();
    const profSeason = rev.find((s) => hasSkills(s) && s.skills.playmaking) ||
                       rev.find((s) => hasSkills(s)) || seasons[seasons.length - 1];
    renderSkillProfile(profSeason);
    setupCompare(profSeason);
    renderValueDerivation(profSeason);
    renderSkillTrajectory(seasons);

    const tcols = [
      ["season", "Season", (r) => seasonLabel(r.season), true],
      ["team", "Team", (r) => `<span class="team-tag">${r.team}</span>`, true],
      ["rankModel", "Rank", (r) => "#" + (r.rankModel != null ? r.rankModel : r.rank)],
      ["grade", "Grade", (r) => r.grade != null
        ? `<span title="Sticky, age-curved, team-independent grade">${r.gradeLetter} <small>${r.grade}</small></span>` : "\u2014", true],
      ["min", "Min", (r) => r.min.toLocaleString()],
      ["bookerScore", "BOOKER", (r) => r.bookerScore != null
        ? `<span class="${r.bookerScore >= 0 ? "pos" : "neg"}" title="Predictive WAA per 3000 possessions (skill, no aging)">${signed(r.bookerScore, 1)}</span>` : "\u2014"],
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
      sel.dataset.init = "1";
    }
    drawPreseason();
  }
  const PRE_COLS = [
    ["team", "Team", (r) => `<span class="team-tag">${r.team}</span> ${teamName(r.team)}`, true],
    ["predNet", "Pred Net", (r) => signed(r.predNet, 1)],
    ["projWins", "Proj W", (r) => `<b>${fmt(r.projWins, 1)}</b>`],
    ["range", "Sim range (10\u201390%)", (r) => `${r.p10}\u2013${r.p90}`, true],
    ["pPlayoff", "Playoff", (r) => pctCell(r.pPlayoff)],
    ["actualWins", "Actual W", (r) => r.actualWins == null ? "\u2014" : winCmp(r)],
  ];
  function winCmp(r) {
    const err = r.projWins - r.actualWins;
    const cls = Math.abs(err) <= 5 ? "pos" : "neg";
    return `${r.actualWins} <span class="${cls}" style="font-size:11px">(${signed(err, 1)})</span>`;
  }
  function drawPreseason() {
    const yr = +$("#f-pre-season").value;
    let rows = D.preseason.filter((p) => p.season === yr);
    rows.sort((a, b) => {
      if (preSort.key === "range") return (a.p50 - b.p50) * preSort.dir;
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
      ? "BOOKER's pre-tip win probabilities use only player impacts known before each game, then meet the closing moneyline (vig removed). Beating the market is hard \u2014 here is how close we get."
      : "BOOKER's pre-tip win probabilities use only player impacts known before each game. Market lines were unavailable for the latest seasons; skill is shown against outcomes.";
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

  function renderTrade() {
    const T = D.trade;
    if (!T || !T.players) {
      $("#trade-lede").textContent = "Trade data unavailable — re-run export_dashboard_data.py.";
      return;
    }
    const cap = T.capRules || {};
    $("#trade-lede").textContent =
      `Build a package trade for ${seasonLabel(T.season)} (up to ${MAX_TRADE_ASSETS} players per side). ` +
      `True Value uses local salary history + FA signings inflated to 2026 cap dollars.`;
    if (cap.cap) {
      $("#trade-cap-hint").textContent =
        `cap ${money(cap.cap)} · tax ${money(cap.tax)} · MLE ${money(cap.mle)}`;
    }

    const teams = Object.keys(T.teamNet).sort();
    const selA = $("#trade-team-a"), selB = $("#trade-team-b");
    if (!tradeReady) {
      selA.innerHTML = teams.map((t) => `<option value="${t}">${t} \u2014 ${teamName(t)}</option>`).join("");
      selB.innerHTML = teams.map((t) => `<option value="${t}">${t} \u2014 ${teamName(t)}</option>`).join("");
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

    const outA = pidsA.map((id) => T.players.find((p) => p.pid === id)).filter(Boolean);
    const outB = pidsB.map((id) => T.players.find((p) => p.pid === id)).filter(Boolean);
    const inA = outB, inB = outA;

    let newNetA = T.teamNet[ta];
    let newNetB = T.teamNet[tb];
    outA.forEach((p) => { newNetA -= netContribOnTeam(p, ta); });
    outB.forEach((p) => { newNetB -= netContribOnTeam(p, tb); });
    inA.forEach((p) => { newNetA += netContribOnTeam(p, ta, p.minutes); });
    inB.forEach((p) => { newNetB += netContribOnTeam(p, tb, p.minutes); });

    const dA = T.k * (newNetA - T.teamNet[ta]);
    const dB = T.k * (newNetB - T.teamNet[tb]);
    const wA0 = T.teamSimWins[ta] || T.teamWins[ta];
    const wB0 = T.teamSimWins[tb] || T.teamWins[tb];

    const salOutA = outA.map(playerSalary2026);
    const salInA = inA.map(playerSalary2026);
    const salOutB = outB.map(playerSalary2026);
    const salInB = inB.map(playerSalary2026);
    const matchA = salaryMatch(salOutA, salInA);
    const matchB = salaryMatch(salOutB, salInB);

    $("#trade-cards").innerHTML = [
      [`${ta} wins`, `${fmt(wA0, 1)} \u2192 ${fmt(wA0 + dA, 1)}`, `${signed(dA, 1)} wins`],
      [`${tb} wins`, `${fmt(wB0, 1)} \u2192 ${fmt(wB0 + dB, 1)}`, `${signed(dB, 1)} wins`],
      [`${ta} salary`, `${money(salOutA.reduce((a, b) => a + b, 0))} out`, `<span class="${matchA.ok ? "salary-ok" : "salary-bad"}">${matchA.note}</span>`],
      [`${tb} salary`, `${money(salOutB.reduce((a, b) => a + b, 0))} out`, `<span class="${matchB.ok ? "salary-ok" : "salary-bad"}">${matchB.note}</span>`],
    ].map((c) => `<div class="card"><div class="k">${c[0]}</div><div class="v">${c[1]}</div><div class="d">${c[2]}</div></div>`).join("");

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

    const namesA = outA.map((p) => p.player).join(", ") || "(none)";
    const namesB = outB.map((p) => p.player).join(", ") || "(none)";
    $("#trade-lede").textContent =
      `${ta} sends ${namesA} for ${namesB}. ` +
      `Fair value from local salary history + FA signings (2026 cap dollars).`;

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
    $("#diag-intro").innerHTML =
      `${d.model}. Each player carries a posterior <em>mean</em> and <em>standard deviation</em> ` +
      `on their offensive and defensive rating; the intervals below are 95% credible (±1.96·SD).`;

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
