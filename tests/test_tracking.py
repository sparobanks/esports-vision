import numpy as np
import cv2

from esv.render import H, HOOPS, W, draw_court
from esv.tracking import ShotDetector, TeamSplitter, detect


def frame_with(players, ball):
    img = np.zeros((H, W, 3), np.uint8)
    draw_court(img)
    for (x, y), col in players:
        cv2.circle(img, (x, y), 13, col, -1)
    cv2.circle(img, ball, 6, (0, 140, 255), -1)
    return img


def test_detects_players_ball_and_teams():
    red, blue = (40, 40, 220), (220, 120, 30)
    pts = [((200, 200), red), ((300, 300), red), ((600, 250), blue), ((700, 400), blue)]
    d = detect(frame_with(pts, (450, 350)))
    found = sorted(map(tuple, np.round(d.players).astype(int)))
    assert len(found) == 4
    for (x, y), _ in pts:
        assert min(np.hypot(x - fx, y - fy) for fx, fy in found) < 3
    assert np.hypot(*(d.ball - [450, 350])) < 3
    ts = TeamSplitter()
    for _ in range(10):
        labels = ts.assign(d.colours)
    order = np.argsort(d.players[:, 0])
    assert labels[order[0]] == labels[order[1]] != labels[order[2]] == labels[order[3]]


def test_shot_is_a_ball_flight_towards_a_hoop():
    sd = ShotDetector()
    players = np.array([[300.0, 300.0]])
    hx, hy = HOOPS["home"]
    shots = []
    for f in range(20):
        ball = np.array([500 + f * (hx - 500) / 20, 300 + f * (hy - 300) / 20])
        s = sd.update(f, ball, players)
        if s:
            shots.append(s)
    assert len(shots) == 1 and shots[0]["team"] == "home"
    sd = ShotDetector()
    held = [sd.update(f, players[0] + [8, -8], players) for f in range(20)]   # ball held: no shot
    assert not any(held)
