import random

from esv.sim import GAME_SECONDS, make_season, play_match


def test_scores_add_up_and_never_go_down():
    m = play_match("t", 0.2, -0.1, random.Random(3))
    h = a = 0
    for e in m.events:
        assert e.home >= h and e.away >= a
        if e.team == "home":
            h += e.points
        else:
            a += e.points
        assert (e.home, e.away) == (h, a)
    assert m.home_win in (0, 1)        # overtime always produces a winner


def test_irregular_window_is_in_second_half():
    m = play_match("t", 0, 0, random.Random(1), irregular="home")
    s, e = m.irregular_window
    assert GAME_SECONDS / 2 <= s < e <= GAME_SECONDS


def test_stronger_team_wins_more():
    ms, fx, strengths, _ = make_season(600, seed=2)
    wins = [m.home_win if strengths[h] > strengths[a] else 1 - m.home_win for m, (h, a) in zip(ms, fx)]
    assert sum(wins) / len(wins) > 0.6
