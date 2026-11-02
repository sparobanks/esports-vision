"""Prediction models on top of the event data.

1. Pre-match team ratings (Elo), learned only from results, used for
   pre-match prices and as an input to the in-play model.
2. In-play win probability: the chance the home team wins, updated after
   every event from the score difference, time left, possession and the
   pre-match rating gap. Checked for calibration, because a price is only
   as good as its probability.
3. Irregular play: flags matches where a team plays far below what its
   rating predicts for a stretch of the game. Flags are for a person to
   review (for example an integrity analyst), never an automatic decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .sim import GAME_SECONDS, Match


# ------------------------------------------------------------------ Elo
@dataclass
class Elo:
    k: float = 24.0
    home_adv: float = 30.0

    def run(self, matches: list[Match], fixtures: list[tuple[str, str]]):
        """Return each match's pre-match home win probability and rating gap, updating as results come in."""
        r: dict[str, float] = {}
        pre, gaps = [], []
        for m, (h, a) in zip(matches, fixtures):
            rh, ra = r.get(h, 1500.0), r.get(a, 1500.0)
            gap = rh + self.home_adv - ra
            p = 1 / (1 + 10 ** (-gap / 400))
            pre.append(p)
            gaps.append(gap)
            res = m.home_win
            r[h] = rh + self.k * (res - p)
            r[a] = ra - self.k * (res - p)
        self.ratings = r
        return np.array(pre), np.array(gaps)


def fair_odds(p: float) -> float:
    """Decimal odds with no bookmaker margin."""
    return round(1 / max(min(p, 0.999), 0.001), 2)


# ------------------------------------------------------------------ in-play model
def states(m: Match, pre_p: float, every: float = 20.0):
    """Game states sampled every `every` seconds of play: (features, label)."""
    rows = []
    evs = m.events
    i = 0
    h = a = 0
    t = 0.0
    while t < GAME_SECONDS:
        while i < len(evs) and evs[i].t <= t:
            h, a = evs[i].home, evs[i].away
            i += 1
        nxt = evs[i] if i < len(evs) else None
        poss = 0 if nxt is None else (1 if nxt.team == "home" else -1)
        rows.append(features(h - a, GAME_SECONDS - t, poss, pre_p))
        t += every
    return rows


def features(diff: float, remaining: float, poss: int, pre_p: float) -> list[float]:
    left = max(remaining, 0.0) / 60.0
    pre = math.log(pre_p / (1 - pre_p))
    return [diff, diff / math.sqrt(left + 0.5), poss, poss / math.sqrt(left + 0.5), pre, pre * left / 20.0]


FEATURE_NAMES = ["score_diff", "score_diff_scaled", "possession", "possession_scaled", "prematch", "prematch_x_time"]


class WinModel:
    def __init__(self):
        self.lr = LogisticRegression(max_iter=2000, C=1.0)

    def fit(self, X, y):
        self.lr.fit(np.asarray(X), np.asarray(y))
        return self

    def prob(self, X) -> np.ndarray:
        return self.lr.predict_proba(np.atleast_2d(X))[:, 1]


def ece(y, p, bins=10) -> float:
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(err)


def scores(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-4, 1 - 1e-4)
    return dict(log_loss=round(log_loss(y, p), 4), brier=round(brier_score_loss(y, p), 4), ece=round(ece(y, p), 4))


# ------------------------------------------------------------------ irregular play
def possessions(m: Match):
    """(time, team, points, turnover) for each possession."""
    out = []
    seen = set()
    for e in m.events:
        key = (e.possession_start, e.team)
        if key in seen:
            out[-1][2] += e.points
            continue
        seen.add(key)
        out.append([e.t, e.team, e.points, e.kind == "turnover"])
    return out


OUTCOMES = ["empty", "scored", "turnover"]


class IrregularDetector:
    """Sequential test (CUSUM) on each team's possessions.

    For every possession we compare two explanations: the team is playing at
    the level its rating predicts, or it is deliberately underperforming
    (turnovers about twice as likely, scoring clearly less likely). The
    evidence for the second is added up possession by possession, resetting
    when play looks normal. The highest total in the match is its score.
    """

    def __init__(self, slump_turnover_odds: float = 2.2, slump_score_logit: float = -0.75):
        self.to_odds, self.sc_shift = slump_turnover_odds, slump_score_logit

    def fit(self, matches, gaps):
        X, y = [], []
        for m, g in zip(matches, gaps):
            for t, team, p, to in possessions(m):
                X.append([(g if team == "home" else -g) / 400])
                y.append(2 if to else (1 if p > 0 else 0))
        self.clf = LogisticRegression(max_iter=2000).fit(np.array(X), np.array(y))
        return self

    def _llr(self, edge: float, outcome: int) -> float:
        pn = self.clf.predict_proba([[edge / 400]])[0]           # empty, scored, turnover
        to = pn[2] * self.to_odds / (1 - pn[2] + pn[2] * self.to_odds)
        rest = 1 - to
        sc_share = pn[1] / (pn[0] + pn[1])
        sc_odds = sc_share / (1 - sc_share) * math.exp(self.sc_shift)
        sc = rest * sc_odds / (1 + sc_odds)
        ps = [rest - sc, sc, to]
        return math.log(ps[outcome] / pn[outcome])

    def team_scores(self, m: Match, gap: float) -> dict[str, tuple[float, float | None]]:
        """For each team: highest CUSUM value in the match and when it peaked."""
        cache, out = {}, {}
        for team, sign in (("home", 1), ("away", -1)):
            edge = sign * gap
            s, best, when = 0.0, 0.0, None
            for t, tm, p, to in possessions(m):
                if tm != team:
                    continue
                o = 2 if to else (1 if p > 0 else 0)
                key = (round(edge), o)
                if key not in cache:
                    cache[key] = self._llr(edge, o)
                s = max(0.0, s + cache[key])
                if s > best:
                    best, when = s, t
            out[team] = (best, when)
        return out

    def score(self, m: Match, gap: float) -> tuple[float, str, float | None]:
        ts = self.team_scores(m, gap)
        team = max(ts, key=lambda k: ts[k][0])
        return ts[team][0], team, ts[team][1]


def irregular_eval(matches, gaps, det: IrregularDetector, alert_rate: float = 0.03) -> dict:
    s, team, label, right_team = [], [], [], []
    for m, g in zip(matches, gaps):
        sc, tm, _ = det.score(m, g)
        s.append(sc)
        team.append(tm)
        label.append(int(m.irregular_team is not None))
        right_team.append(tm == m.irregular_team)
    s, label = np.array(s), np.array(label)
    thr = np.quantile(s, 1 - alert_rate)
    flag = s >= thr
    return dict(matches=len(s), irregular=int(label.sum()), roc_auc=round(roc_auc_score(label, s), 3),
                alert_rate=alert_rate, flagged=int(flag.sum()),
                precision=round(float(label[flag].mean()), 3) if flag.any() else None,
                recall=round(float(flag[label == 1].mean()), 3),
                right_team_when_caught=round(float(np.mean([rt for rt, f, lb in zip(right_team, flag, label)
                                                            if f and lb])), 3) if (flag & (label == 1)).any() else None)

