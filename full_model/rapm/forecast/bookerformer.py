"""
BookerFormer: a Bayesian Set-Transformer for player-skill (RAPM successor).

Motivation
----------
Ridge RAPM and the PyMC matchup model are *linear, additive*: a possession's
outcome is the sum of independent per-player coefficients (offense +1, defense
-1) shrunk toward a box-score prior. That can't represent nonlinear synergy /
redundancy (two ball-dominant guards; spacing x rim-protection complements) or
matchup-specific effects, and a single regularized number hides confidence.

Design
------
A possession is modeled as a *matchup of two sets* -- 5 offensive players vs 5
defensive players -- predicting the offense's points/100. Two ingredients:

  1. Per-player scalar variational random effects (a_off, a_def). These ARE the
     player rating and its uncertainty. Each has a mean and a log-variance and a
     Gaussian prior centered at the DARKO-decayed box-score O/D prior. Added
     directly to the prediction, their posterior variance behaves like a proper
     Bayesian estimate -- ~1 / (possessions/sigma^2 + 1/tau^2) -- so it lives on a
     points/100 scale and *narrows as a player accumulates minutes*. With the
     transformer disabled this is exactly informative-prior Bayesian RAPM.

  2. A Set-Transformer correction (Lee et al. 2019 SAB/PMA) over the 10 player
     embeddings, permutation-invariant within a side and offense/defense-aware via
     a learned side encoding. This learns nonlinear synergy / matchup effects on
     top of the additive base. It carries BayesFormer-style structured MC dropout
     (Sankararaman et al., arXiv:2206.00826) over embeddings, attention, and FFN --
     kept active at inference so T stochastic passes give predictive uncertainty.

Per-player ratings come out by leave-one-out marginal contribution (replace the
player with a league-average player, measure the change in predicted points),
which preserves RAPM's "value vs replacement" semantics for a nonlinear body. The
additive base contributes the player's own random effect exactly; the transformer
contributes their average synergy adjustment. Running the LOO under the MC
ensemble (sampling the random effects + dropout) yields impact_off/impact_def plus
sd_off/sd_def -- the latter dominated by the calibrated random-effect posterior.

Public API mirrors forecast.bayesian_matchup.fit_bayesian:

    post_df, model = fit_bookerformer(train_seasons, target_season, data)

post_df columns: PLAYER_ID, NAME, impact_off, impact_def, impact_total,
sd_off, sd_def -- exactly what build_bayesian_ratings.py / build_bookerformer_
ratings.py consume.

Attention default
-----------------
`use_attention` defaults to False. Out-of-sample backtests (bookerformer_backtest.py)
show the additive Bayesian model BEATS ridge RAPM on team net / wins prediction
(wins R2 0.61 vs 0.55) with well-calibrated 90% intervals, whereas turning the
transformer ON *hurts* OOS team prediction (wins R2 ~0.40) and does not improve
stint RMSE -- per-stint outcomes are dominated by irreducible noise (~47 pts/100),
so the extra capacity overfits at NBA stint-data scale (~14k stints/season). The
transformer synergy/matchup layer is therefore opt-in (use_attention=True) and the
production ratings use the additive mode. Revisit if much more data is pooled.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import player_impacts as pi
from . import enhanced_impacts as ei

# ---------------------------------------------------------------------------
# Hyper-parameters (shared scale constants come from player_impacts)
# ---------------------------------------------------------------------------
LEAGUE = pi.LEAGUE_PPP100          # 108.0 points per 100 possessions baseline
DECAY = pi.DECAY                   # 0.70 DARKO year-over-year decay
PRIOR_BASE = pi.PRIOR_BASE         # -1.0 replacement-ish prior for unknowns

EMB_DIM = 24                       # transformer-correction embedding / model width
N_HEADS = 4
N_BLOCKS = 2
FFN_MULT = 2
DROPOUT = 0.10                     # BayesFormer dropout rate (~5-10% recommended)

# Per-player scalar random effects: Gaussian prior N(box_prior, TAU_RATING^2).
# TAU_RATING is the prior std of a player's O/D rating in points/100. 2.0 keeps
# low-minute players shrunk to their (informative) box prior so the leaderboard
# stays sane; looser priors let small-sample noise inflate role players.
TAU_RATING = 2.0
# Proper ELBO scaling: the data term is a possession-WEIGHTED MEAN NLL, so the full
# KL must enter divided by the total training weight (KL_BETA=1.0). With that
# normalization, each player's effective likelihood weight equals their possession
# count, giving the textbook posterior-variance shrinkage with minutes. Raise
# KL_BETA to widen all intervals if calibration is over-confident.
KL_BETA = 1.0
WEIGHT_DECAY = 1e-4                # Gaussian prior on transformer/embedding weights

Y_CLIP = 175.0                     # clip centered targets; tiny stints are noisy
LR = 5e-3
BATCH = int(os.environ.get("BF_BATCH", 4096))
VAL_FRAC = 0.10                    # chronologically-latest slice held out for stopping
MAX_CTX = 48                       # leave-one-out contexts sampled per player/side
SEED = 17

# Training-length knobs are env-overridable so an overnight run can crank iterations
# (e.g. BF_EPOCHS=600 BF_PATIENCE=50 BF_MC=200) without changing interactive defaults.
MAX_EPOCHS = int(os.environ.get("BF_EPOCHS", 60))
PATIENCE = int(os.environ.get("BF_PATIENCE", 8))
MC_SAMPLES = int(os.environ.get("BF_MC", 40))  # T stochastic passes for the predictive dist
# Offensive prior nudge from the shot-quality skill composite, in points/100 per SD.
# Moderate (1.0): skilled creators start ~1 pt/100 higher per SD of skill, then the
# RAPM stint likelihood pulls back toward results. Set BF_SKILL_NUDGE=0 to disable.
SKILL_NUDGE = float(os.environ.get("BF_SKILL_NUDGE", 1.0))
# Defensive prior nudge from the shot-based defensive value (deterrence + make-
# limiting), points/100 per SD. Lifts rim deterrents (Wemby) and reins in vets whose
# RAPM defense isn't shot-backed (CP3). BF_DEF_NUDGE=0 disables.
DEF_NUDGE = float(os.environ.get("BF_DEF_NUDGE", 1.0))
# Low-usage offensive prior tightening (de-collinearization): a player at USAGE_REF+
# usage keeps the full TAU_RATING offensive prior; lower usage shrinks tau_off toward
# TAU_RATING*TAU_OFF_MIN so RAPM can't inflate a connective vet's offense.
USAGE_REF = 22.0
TAU_OFF_MIN = float(os.environ.get("BF_TAU_OFF_MIN", 0.45))


def _device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
@dataclass
class FormerData:
    """Tensors + bookkeeping the model and the LOO extractor need."""
    pids: list                     # ordered player ids -> parameter rows
    pid_to_idx: dict
    names: dict                    # pid -> name
    prior_off: np.ndarray          # [P] box offensive prior (pts/100, +good)
    prior_def: np.ndarray          # [P] box defensive prior (pts/100, +good D)
    # training observations (one matchup = offense set vs defense set)
    off_idx: np.ndarray            # [N, 5] parameter rows of the offense
    def_idx: np.ndarray            # [N, 5] parameter rows of the defense
    y: np.ndarray                  # [N] centered target (offense pts/100 - LEAGUE)
    w: np.ndarray                  # [N] sample weight (POSS * decay)
    order: np.ndarray              # [N] chronological-ish order key for val split
    tau_off: np.ndarray = None     # [P] per-player offensive prior std (usage-tightened)
    tau_def: np.ndarray = None     # [P] per-player defensive prior std
    off_ctx: dict = field(default_factory=dict)   # pid -> list[(O[5], D[5])]
    def_ctx: dict = field(default_factory=dict)


def _lineup_rows(lineup, pid_to_idx):
    """Map a list of player ids to parameter rows, or None if not exactly 5 known."""
    rows = [pid_to_idx[p] for p in lineup if p in pid_to_idx]
    return rows if len(rows) == 5 else None


_USAGE = {}


def _usage_map(data, train_seasons, target_season, pids):
    """{pid: decayed usage%} over the training seasons, from the box master dataset
    (joined by name). Used to tighten the offensive prior for low-usage players."""
    master = pi.ROOT / "full_model" / "nba_master_dataset_with_archetypes.csv"
    if "df" not in _USAGE:
        if master.exists():
            m = pd.read_csv(master)
            m["nm"] = m.playerName.map(pi.norm_name)
            m["ss"] = pd.to_numeric(m.season, errors="coerce")
            _USAGE["df"] = {(nm, int(ss)): float(u) for nm, ss, u in
                            zip(m.nm, m.ss, pd.to_numeric(m.usagePercent, errors="coerce"))
                            if pd.notna(ss) and pd.notna(u)}
        else:
            _USAGE["df"] = {}
    ub = _USAGE["df"]
    # pid -> name
    nm_of = {}
    for s in data.seasons:
        pl = data.PLAYERS.get(s)
        if pl is not None:
            for pid, nm in zip(pl.PLAYER_ID, pl.NAME):
                nm_of[int(pid)] = pi.norm_name(nm)
    acc, wsum = {}, {}
    for s in train_seasons:
        w = DECAY ** (target_season - 1 - s)
        for p in pids:
            u = ub.get((nm_of.get(int(p)), s))
            if u is not None:
                acc[p] = acc.get(p, 0.0) + w * u
                wsum[p] = wsum.get(p, 0.0) + w
    return {p: acc[p] / wsum[p] for p in acc if wsum[p] > 0}


def prepare_data(data, train_seasons, target_season) -> FormerData:
    """Build the player universe, box-score O/D priors, and matchup observations."""
    rng = np.random.default_rng(SEED)

    pid_set = set()
    for s in train_seasons:
        d = data.STINTS[s]
        for hl, al in zip(d.home, d.away):
            pid_set.update(hl)
            pid_set.update(al)
    pids = sorted(pid_set)
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    names = {}
    for s in data.seasons:
        pl = data.PLAYERS.get(s)
        if pl is None:
            continue
        for pid, nm in zip(pl.PLAYER_ID, pl.NAME):
            if int(pid) in pid_to_idx:
                names[int(pid)] = nm

    # box-score priors: decayed BPM total, split into O/D via box shares
    prior_total, _ = pi._decayed_priors(data, train_seasons, target_season)
    shares = ei.box_off_def_shares(data, target_season, pids)
    prior_off = np.array([prior_total.get(p, PRIOR_BASE) * shares.get(p, 0.5)
                          for p in pids], dtype=np.float32)
    prior_def = np.array([prior_total.get(p, PRIOR_BASE) * (1.0 - shares.get(p, 0.5))
                          for p in pids], dtype=np.float32)
    # blend shot-quality skill into the OFFENSIVE prior: shot-making over expected +
    # self-creation load (decayed z-composite, points/100). Lifts skilled high-usage
    # creators; the stint/RAPM likelihood still pulls the posterior toward results.
    if SKILL_NUDGE:
        try:
            from . import shot_quality as sq
            nudge = sq.skill_prior_nudge(train_seasons, target_season, pids,
                                         nudge_pts=SKILL_NUDGE)
            if nudge:
                prior_off = prior_off + np.array(
                    [nudge.get(int(p), 0.0) for p in pids], dtype=np.float32)
        except Exception as exc:
            print(f"  (skill nudge skipped: {exc})")
    # blend shot-based DEFENSE (deterrence + make-limiting, RAPM-adjusted) into the
    # defensive prior: lifts rim deterrents, reins in lineup-inferred vet defense.
    if DEF_NUDGE:
        try:
            from . import shot_defense as sdf
            dn = sdf.def_prior_nudge(train_seasons, target_season, pids, nudge_pts=DEF_NUDGE)
            if dn:
                prior_def = prior_def + np.array(
                    [dn.get(int(p), 0.0) for p in pids], dtype=np.float32)
        except Exception as exc:
            print(f"  (def nudge skipped: {exc})")

    # per-player offensive prior std: tighten for low-usage players so RAPM can't
    # inflate a connective vet's offense (de-collinearization). Usage from box data.
    usage = _usage_map(data, train_seasons, target_season, pids)
    tau_off = np.array([TAU_RATING * max(TAU_OFF_MIN,
                        min(1.0, (usage.get(p, USAGE_REF) / USAGE_REF) ** 0.5))
                        for p in pids], dtype=np.float32)
    tau_def = np.full(len(pids), TAU_RATING, dtype=np.float32)

    off_rows, def_rows, ys, ws, orders = [], [], [], [], []
    off_ctx = {p: [] for p in pids}
    def_ctx = {p: [] for p in pids}
    for s in train_seasons:
        d = data.STINTS[s]
        dw = DECAY ** (target_season - 1 - s)
        if "Y_OFF_HOME" not in d.columns or "Y_DEF_HOME" not in d.columns:
            raise ValueError(f"stints_{s}.csv missing Y_OFF_HOME/Y_DEF_HOME; "
                             "run stint_off_def.enrich_season first")
        gid = d.GAME_ID.to_numpy()
        for hl, al, poss, yo, yd, g in zip(d.home, d.away, d.POSS,
                                           d.Y_OFF_HOME, d.Y_DEF_HOME, gid):
            ho = _lineup_rows(hl, pid_to_idx)
            aw = _lineup_rows(al, pid_to_idx)
            if ho is None or aw is None:
                continue
            wt = float(poss) * float(dw)
            order_key = s * 10_000_000 + int(g) % 10_000_000
            # home offense vs away defense
            off_rows.append(ho); def_rows.append(aw)
            ys.append(float(yo) - LEAGUE); ws.append(wt); orders.append(order_key)
            # away offense vs home defense
            off_rows.append(aw); def_rows.append(ho)
            ys.append(float(yd) - LEAGUE); ws.append(wt); orders.append(order_key)
            # leave-one-out contexts (store the parameter-row lineups)
            for p in hl:
                off_ctx[p].append((ho, aw))
                def_ctx[p].append((aw, ho))   # home player defends in obs B
            for p in al:
                off_ctx[p].append((aw, ho))
                def_ctx[p].append((ho, aw))

    fd = FormerData(
        pids=pids, pid_to_idx=pid_to_idx, names=names,
        prior_off=prior_off, prior_def=prior_def,
        off_idx=np.array(off_rows, dtype=np.int64),
        def_idx=np.array(def_rows, dtype=np.int64),
        y=np.clip(np.array(ys, dtype=np.float32), -Y_CLIP, Y_CLIP),
        w=np.array(ws, dtype=np.float32),
        order=np.array(orders, dtype=np.int64),
        tau_off=tau_off, tau_def=tau_def,
        off_ctx=off_ctx, def_ctx=def_ctx,
    )
    for ctx in (fd.off_ctx, fd.def_ctx):
        for p, lst in ctx.items():
            if len(lst) > MAX_CTX:
                sel = rng.choice(len(lst), size=MAX_CTX, replace=False)
                ctx[p] = [lst[i] for i in sel]
    return fd


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SAB(nn.Module):
    """Set Attention Block: multihead self-attention + FFN, with MC dropout."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * FFN_MULT), nn.GELU(),
            nn.Linear(dim * FFN_MULT, dim),
        )
        self.drop = dropout

    def forward(self, x, mc):
        # attention-probability dropout lives inside nn.MultiheadAttention (active
        # when training=mc); we additionally drop FFN input/output per BayesFormer.
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        h = self.ff(F.dropout(x, self.drop, training=mc))
        x = self.ln2(x + F.dropout(h, self.drop, training=mc))
        return x


class PMA(nn.Module):
    """Pooling by Multihead Attention: one learned seed pools a set to a vector."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        q = self.seed.expand(x.size(0), 1, -1)
        a, _ = self.attn(q, x, x, need_weights=False)
        return self.ln(a).squeeze(1)


class BookerFormer(nn.Module):
    """Offense points/100 (centered) for a 5v5 matchup, with per-player uncertainty.

    Prediction = sum(a_off over offense) - sum(a_def over defense) + transformer
    synergy correction, where a_off/a_def are per-player scalar variational random
    effects with box-prior means.
    """

    def __init__(self, n_players, prior_off, prior_def, use_attention=True,
                 tau_off=None, tau_def=None):
        super().__init__()
        self.P = n_players
        self.use_attention = use_attention
        d = EMB_DIM
        # per-player prior std (points/100). Tightening tau_off for low-usage players
        # keeps RAPM from pulling a connective vet's OFFENSE far from his modest box
        # prior (de-collinearization); default is the global TAU_RATING.
        to = (np.full(n_players, TAU_RATING, dtype=np.float32) if tau_off is None
              else np.asarray(tau_off, dtype=np.float32))
        td = (np.full(n_players, TAU_RATING, dtype=np.float32) if tau_def is None
              else np.asarray(tau_def, dtype=np.float32))
        self.register_buffer("tau_off2", torch.tensor(to ** 2))
        self.register_buffer("tau_def2", torch.tensor(td ** 2))

        # per-player scalar random effects (the rating + its uncertainty)
        self.a_off_mean = nn.Parameter(torch.tensor(prior_off).clone())
        self.a_def_mean = nn.Parameter(torch.tensor(prior_def).clone())
        self.a_off_logvar = nn.Parameter(torch.log(torch.tensor(to ** 2)))   # init at prior var
        self.a_def_logvar = nn.Parameter(torch.log(torch.tensor(td ** 2)))
        self.register_buffer("prior_off", torch.tensor(prior_off))
        self.register_buffer("prior_def", torch.tensor(prior_def))

        # global aleatoric noise (per-100 single-stint outcome variance)
        self.log_sigma2 = nn.Parameter(torch.tensor(float(np.log(45.0 ** 2))))

        if use_attention:
            # player embeddings (point estimates; dropout supplies their uncertainty)
            self.emb = nn.Parameter(torch.randn(n_players, d) * 0.1)
            self.side_off = nn.Parameter(torch.randn(d) * 0.1)
            self.side_def = nn.Parameter(torch.randn(d) * 0.1)
            self.eff_proj = nn.Linear(1, d)        # inject scalar skill into the token
            self.blocks = nn.ModuleList(
                [SAB(d, N_HEADS, DROPOUT) for _ in range(N_BLOCKS)])
            self.pool = PMA(d, N_HEADS, DROPOUT)
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.zeros_(self.head[-1].weight)   # correction == 0 at init
            nn.init.zeros_(self.head[-1].bias)

    # -- sampling of the variational random effects -------------------------
    def sample_effects(self, sample):
        if sample:
            ao = self.a_off_mean + torch.exp(0.5 * self.a_off_logvar) * torch.randn_like(self.a_off_mean)
            ad = self.a_def_mean + torch.exp(0.5 * self.a_def_logvar) * torch.randn_like(self.a_def_mean)
            return ao, ad
        return self.a_off_mean, self.a_def_mean

    def kl(self):
        """KL(q(a) || N(box_prior, tau^2)) summed over offense + defense effects,
        with a per-player prior variance tau^2."""
        def _kl(mean, logvar, prior, tau2):
            var = torch.exp(logvar)
            return 0.5 * ((var + (mean - prior) ** 2) / tau2 - 1.0 - logvar + torch.log(tau2))
        return (_kl(self.a_off_mean, self.a_off_logvar, self.prior_off, self.tau_off2).sum()
                + _kl(self.a_def_mean, self.a_def_logvar, self.prior_def, self.tau_def2).sum())

    # -- forward over gathered effects/embeddings ---------------------------
    def forward(self, off_a, def_a, off_e, def_e, mc):
        """off_a/def_a: [B,5] sampled scalar effects; off_e/def_e: [B,5,d] embeddings
        (ignored when use_attention is False). Returns (mu, log_sigma2)."""
        base = off_a.sum(1) - def_a.sum(1)
        logs2 = self.log_sigma2.expand(base.shape)
        if not self.use_attention:
            return base, logs2

        ot = off_e + self.side_off + self.eff_proj(off_a.unsqueeze(-1))
        dt = def_e + self.side_def + self.eff_proj(def_a.unsqueeze(-1))
        x = torch.cat([ot, dt], dim=1)               # [B,10,d]
        x = F.dropout(x, DROPOUT, training=mc)
        for blk in self.blocks:
            x = blk(x, mc)
        corr = self.head(self.pool(x[:, :5])).squeeze(-1)   # pool offense tokens
        return base + corr, logs2

    def param_groups(self):
        """Random effects + embeddings get no weight decay; transformer weights do."""
        no_wd = {"a_off_mean", "a_def_mean", "a_off_logvar", "a_def_logvar",
                 "log_sigma2", "emb"}
        g_wd, g_no = [], []
        for name, p in self.named_parameters():
            (g_no if name in no_wd else g_wd).append(p)
        return [{"params": g_no, "weight_decay": 0.0},
                {"params": g_wd, "weight_decay": WEIGHT_DECAY}]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(fd: FormerData, use_attention=True, verbose=False):
    torch.manual_seed(SEED)
    dev = _device()
    model = BookerFormer(len(fd.pids), fd.prior_off, fd.prior_def,
                         use_attention=use_attention,
                         tau_off=fd.tau_off, tau_def=fd.tau_def).to(dev)
    opt = torch.optim.Adam(model.param_groups(), lr=LR)

    off_idx = torch.from_numpy(fd.off_idx)
    def_idx = torch.from_numpy(fd.def_idx)
    y = torch.from_numpy(fd.y)
    w = torch.from_numpy(fd.w)

    cut = np.quantile(fd.order, 1.0 - VAL_FRAC)
    val_mask = fd.order >= cut
    tr = np.where(~val_mask)[0]
    va = np.where(val_mask)[0]
    w_total = float(w[torch.from_numpy(tr)].sum())   # for ELBO KL normalization

    def batch_loss(ao, ad, b_idx, mc):
        oi, di = off_idx[b_idx].to(dev), def_idx[b_idx].to(dev)
        off_e = model.emb[oi] if use_attention else None
        def_e = model.emb[di] if use_attention else None
        mu, logs2 = model(ao[oi], ad[di], off_e, def_e, mc=mc)
        yt, wt = y[b_idx].to(dev), w[b_idx].to(dev)
        nll = 0.5 * (logs2 + (yt - mu) ** 2 * torch.exp(-logs2))
        return (wt * nll).sum() / wt.sum().clamp_min(1e-6)

    best_val, best_state, bad, best_epoch = float("inf"), None, 0, 0
    history = []   # per-epoch convergence diagnostics
    rng = np.random.default_rng(SEED)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = rng.permutation(tr)
        ep_loss, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            b_idx = torch.from_numpy(perm[i:i + BATCH])
            ao, ad = model.sample_effects(sample=True)
            opt.zero_grad()
            loss = batch_loss(ao, ad, b_idx, mc=True) + KL_BETA * model.kl() / w_total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach()); nb += 1

        model.eval()
        with torch.no_grad():
            ao, ad = model.sample_effects(sample=False)
            vloss = batch_loss(ao, ad, torch.from_numpy(va), mc=False).item()
            kl = float(model.kl().detach() / w_total)
        history.append({"epoch": epoch, "train_loss": ep_loss / max(nb, 1),
                        "val_nll": vloss, "kl": kl})
        if verbose:
            print(f"  epoch {epoch:3d}  val_nll={vloss:.4f}  kl={kl:.4f}")
        if vloss < best_val - 1e-4:
            best_val, bad, best_epoch = vloss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model._history = history
    model._best_epoch = best_epoch
    return model, best_val


# ---------------------------------------------------------------------------
# Leave-one-out per-player ratings + uncertainty (MC ensemble)
# ---------------------------------------------------------------------------
def _build_loo_rows(ctx, pid_to_idx):
    """Flatten every player's contexts into batched arrays.

    Returns O[M,5], D[M,5] (param rows), owner[M] (removed player's row), and
    who[M] (player index the row contributes to)."""
    O_list, D_list, owner, who = [], [], [], []
    for pidx, pid in enumerate(sorted(pid_to_idx, key=lambda p: pid_to_idx[p])):
        lst = ctx.get(pid)
        if not lst:
            continue
        row = pid_to_idx[pid]
        for (Orow, Drow) in lst:
            O_list.append(Orow); D_list.append(Drow)
            owner.append(row); who.append(pidx)
    if not O_list:
        return None
    return (np.array(O_list, dtype=np.int64), np.array(D_list, dtype=np.int64),
            np.array(owner, dtype=np.int64), np.array(who, dtype=np.int64))


def extract_ratings(model: BookerFormer, fd: FormerData):
    """Per-player offense/defense rating + uncertainty.

    impact_off/impact_def: deterministic leave-one-out marginal value vs a league-
    average player (posterior-mean effects, dropout off) -- in additive mode this is
    exactly the player's random-effect mean; with attention it adds their average
    synergy/matchup adjustment.

    sd_off/sd_def: the scalar random-effect posterior std (exp(0.5*logvar)). This is
    the calibrated, minutes-and-identifiability-aware uncertainty; sourcing it here
    (rather than from the MC dropout spread) avoids the transformer's near-constant
    dropout-variance floor washing out the minutes signal."""
    dev = _device()
    model.eval()
    P = model.P
    avg_row = P

    # posterior-mean effects + zero (league-average) replacement row
    with torch.no_grad():
        ao = torch.cat([model.a_off_mean.detach(), torch.zeros(1)])
        ad = torch.cat([model.a_def_mean.detach(), torch.zeros(1)])
        emb = (torch.cat([model.emb.detach(), torch.zeros(1, EMB_DIM)], 0)
               if model.use_attention else None)
        sd_off_vec = torch.exp(0.5 * model.a_off_logvar.detach()).cpu().numpy()
        sd_def_vec = torch.exp(0.5 * model.a_def_logvar.detach()).cpu().numpy()

    def predict(O, D):
        Ot, Dt = torch.from_numpy(O).to(dev), torch.from_numpy(D).to(dev)
        oe = emb[Ot] if model.use_attention else None
        de = emb[Dt] if model.use_attention else None
        with torch.no_grad():
            mu, _ = model(ao[Ot], ad[Dt], oe, de, mc=False)   # deterministic
        return mu.cpu().numpy()

    def loo_side(ctx, slot_is_offense):
        out = np.full(P, np.nan)
        built = _build_loo_rows(ctx, fd.pid_to_idx)
        if built is None:
            return out
        O, D, owner, who = built
        Or, Dr = O.copy(), D.copy()
        if slot_is_offense:
            Or[O == owner[:, None]] = avg_row
        else:
            Dr[D == owner[:, None]] = avg_row
        delta = (predict(O, D) - predict(Or, Dr)) if slot_is_offense \
            else (predict(Or, Dr) - predict(O, D))
        counts = np.bincount(who, minlength=P).astype(np.float64)
        agg = np.zeros(P, dtype=np.float64)
        np.add.at(agg, who, delta)
        nz = counts > 0
        out[nz] = agg[nz] / counts[nz]
        return out

    impact_off = loo_side(fd.off_ctx, slot_is_offense=True)
    impact_def = loo_side(fd.def_ctx, slot_is_offense=False)

    rows = []
    for idx, pid in enumerate(fd.pids):
        if np.isnan(impact_off[idx]) and np.isnan(impact_def[idx]):
            continue
        io = float(np.nan_to_num(impact_off[idx]))
        idf = float(np.nan_to_num(impact_def[idx]))
        rows.append({
            "PLAYER_ID": int(pid),
            "NAME": fd.names.get(int(pid), str(pid)),
            "impact_off": io, "impact_def": idf, "impact_total": io + idf,
            "sd_off": float(sd_off_vec[idx]), "sd_def": float(sd_def_vec[idx]),
        })
    return pd.DataFrame(rows)


def _ensemble(model, mc_samples):
    """Pre-sample T (a_off, a_def, emb) draws, each extended with a league-average
    replacement row (effect 0, zero embedding)."""
    P = model.P
    torch.manual_seed(SEED + 1)
    draws = []
    with torch.no_grad():
        emb_ext = (torch.cat([model.emb.detach(), torch.zeros(1, EMB_DIM)], 0)
                   if model.use_attention else None)
        for _ in range(mc_samples):
            ao, ad = model.sample_effects(sample=True)
            ao = torch.cat([ao.detach(), torch.zeros(1)])
            ad = torch.cat([ad.detach(), torch.zeros(1)])
            draws.append((ao, ad, emb_ext))
    return draws, P


def predict_offense_points(model: BookerFormer, fd: FormerData,
                           off_lineups, def_lineups, mc_samples=MC_SAMPLES):
    """Predict offense points/100 for (offense pids, defense pids) matchups.

    Unknown players map to the league-average row. Returns (mu, sd, keep_mask):
    mu is the predictive mean and sd the TOTAL predictive std (epistemic spread of
    the mean across the ensemble + mean aleatoric std), in un-centered points/100;
    keep_mask flags rows with exactly 5+5 mappable slots."""
    dev = _device()
    model.eval()
    draws, P = _ensemble(model, mc_samples)
    avg_row = P

    def to_rows(lineup):
        rows = [fd.pid_to_idx.get(int(p), avg_row) for p in lineup]
        return rows if len(rows) == 5 else None

    O_list, D_list, keep = [], [], []
    for ol, dl in zip(off_lineups, def_lineups):
        orow, drow = to_rows(ol), to_rows(dl)
        ok = orow is not None and drow is not None
        keep.append(ok)
        if ok:
            O_list.append(orow); D_list.append(drow)
    keep = np.array(keep, dtype=bool)
    if not O_list:
        return np.array([]), np.array([]), keep
    O = np.array(O_list, dtype=np.int64)
    D = np.array(D_list, dtype=np.int64)
    Ot, Dt = torch.from_numpy(O).to(dev), torch.from_numpy(D).to(dev)

    preds = np.empty((len(O), mc_samples), dtype=np.float64)
    alea_var = np.empty((len(O), mc_samples), dtype=np.float64)
    with torch.no_grad():
        for t, (ao, ad, emb) in enumerate(draws):
            oe = emb[Ot] if model.use_attention else None
            de = emb[Dt] if model.use_attention else None
            mu, logs2 = model(ao[Ot], ad[Dt], oe, de, mc=True)
            preds[:, t] = mu.cpu().numpy() + LEAGUE
            alea_var[:, t] = torch.exp(logs2).cpu().numpy()
    epi_var = preds.var(axis=1, ddof=1) if mc_samples > 1 else np.zeros(len(O))
    total_sd = np.sqrt(epi_var + alea_var.mean(axis=1))
    return preds.mean(axis=1), total_sd, keep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fit_bookerformer(train_seasons, target_season, data, use_attention=False,
                     mc_samples=MC_SAMPLES, verbose=False):
    """Fit BookerFormer on `train_seasons` and return (post_df, model).

    post_df columns: PLAYER_ID, NAME, impact_off, impact_def, impact_total,
    sd_off, sd_def -- matching forecast.bayesian_matchup.fit_bayesian so the
    existing build_*_ratings runners can consume it unchanged."""
    fd = prepare_data(data, train_seasons, target_season)
    if verbose:
        print(f"  prepared {len(fd.y)} matchup obs over {len(fd.pids)} players")
    model, val = train_model(fd, use_attention=use_attention, verbose=verbose)
    if verbose:
        print(f"  trained (val_nll={val:.4f}); extracting LOO ratings...")
    post = extract_ratings(model, fd)
    model._fd = fd
    return post, model
