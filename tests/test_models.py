import numpy as np

from esv.models import Elo, IrregularDetector, WinModel, fair_odds, features, states
from esv.sim import make_season


def test_elo_learns_team_strength():
    ms, fx, st, _ = make_season(1500, seed=3)
    _, gaps = Elo().run(ms, fx)
    true = np.array([st[h] - st[a] for h, a in fx])
    assert np.corrcoef(gaps[500:], true[500:])[0, 1] > 0.8


def test_win_probability_behaves_sensibly():
    ms, fx, _, _ = make_season(800, seed=4)
    pre, _ = Elo().run(ms, fx)
    X, Y = [], []
    for m, p in zip(ms, pre):
        for r in states(m, p):
            X.append(r)
            Y.append(m.home_win)
    wm = WinModel().fit(X, Y)
    lead = wm.prob([features(8, 120, 0, 0.5)])[0]
    trail = wm.prob([features(-8, 120, 0, 0.5)])[0]
    early = wm.prob([features(8, 1100, 0, 0.5)])[0]
    assert lead > 0.85 and trail < 0.15 and 0.5 < early < lead   # a lead matters more late in the game
    assert fair_odds(0.5) == 2.0


def test_irregular_detector_scores_turnovers_as_suspicious():
    ms, fx, _, _ = make_season(600, seed=5, irregular_rate=0)
    _, gaps = Elo().run(ms, fx)
    det = IrregularDetector().fit(ms, gaps)
    assert det._llr(0, 2) > 0 > det._llr(0, 1)     # turnover adds evidence, a score removes it
