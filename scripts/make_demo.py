"""Make an annotated demo GIF: what the pipeline sees, frame by frame.

    python scripts/make_demo.py

Draws tracked players (with IDs and team), the ball, detected shots and
score events, the scoreboard reading and the live win probability.
Writes docs/img/demo.gif.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from esv import hud  # noqa: E402
from esv.models import Elo, WinModel, features, states  # noqa: E402
from esv.render import FPS, STYLES  # noqa: E402
from esv.sim import QUARTER_SECONDS, make_season, play_match  # noqa: E402
from esv.tracking import ShotDetector, TeamSplitter, Tracker, detect  # noqa: E402

SECONDS = 10


def win_model():
    ms, fx, _, _ = make_season(2500, seed=1)
    pre, _ = Elo().run(ms, fx)
    X, Y = [], []
    for m, p in zip(ms[1000:], pre[1000:]):
        for r in states(m, p):
            X.append(r)
            Y.append(m.home_win)
    return WinModel().fit(X, Y)


def main():
    clip = json.loads((ROOT / "reports" / "models.json").read_text())["live_demo"]["clip"]
    d = ROOT / "data" / "clips" / clip
    truth = json.loads((d / "truth.json").read_text())
    det_events = json.loads((d / "detected.json").read_text())["score_events"]
    first = det_events[len(det_events) // 2]["frame"]
    start = max(0, first - 7 * FPS)
    model = hud.DigitNet()
    model.load_state_dict(torch.load(ROOT / "models" / "digitnet.pt"))
    model.eval()
    reader = hud.HudReader(model, STYLES[truth["style"]])
    scores, shots, tracker, teams = hud.ScoreTracker(), ShotDetector(), Tracker(), TeamSplitter()
    wm = win_model()
    seed = int(truth["match_id"].replace("clip", ""))
    rng = random.Random(seed)
    hs, as_ = rng.gauss(0, 0.6), rng.gauss(0, 0.6)
    mc = random.Random(5)
    pre_p = float(np.mean([play_match("mc", hs, as_, mc).home_win for _ in range(300)]))
    cap = cv2.VideoCapture(str(d / "clip.mp4"))
    tmp = Path(tempfile.mkdtemp())
    writer = cv2.VideoWriter(str(tmp / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540))
    banner, banner_until, shot_until = "", -1, -1
    ball_side, score = [], None
    team_col = [(60, 220, 255), (255, 255, 255)]
    f = 0
    while f < start + SECONDS * FPS:
        ok, frame = cap.read()
        if not ok:
            break
        r = reader.read(frame)
        det = detect(frame)
        lab = teams.assign(det.colours)
        ids = tracker.update(det.players)
        shot = shots.update(f, det.ball, det.players)
        evs = scores.update(f, r)
        if score is None and r is not None:
            score = (r["home"], r["away"])
        if det.ball is not None:
            ball_side.append(1 if det.ball[0] > 480 else -1)
        for e in evs:
            score = (e["home"], e["away"])
            banner, banner_until = f"SCORE: {e['team'].upper()} +{e['points']}", f + 2 * FPS
        if shot:
            shot_until = f + FPS
        if f >= start:
            out = frame.copy()
            for (x, y), t, i in zip(det.players, lab, ids):
                cv2.circle(out, (int(x), int(y)), 17, team_col[int(t)], 2)
                cv2.putText(out, str(i), (int(x) + 14, int(y) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if det.ball is not None:
                cv2.circle(out, (int(det.ball[0]), int(det.ball[1])), 10, (0, 255, 255), 2)
            panel_y = 470
            cv2.rectangle(out, (0, panel_y), (960, 540), (25, 25, 25), -1)
            txt = "scoreboard: unreadable" if r is None else \
                f"scoreboard read: {r['home']}-{r['away']}  Q{r['quarter']} {r['clock'] // 60}:{r['clock'] % 60:02d}"
            cv2.putText(out, txt, (14, panel_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            if r is not None and score is not None:
                remaining = (4 - r["quarter"]) * QUARTER_SECONDS + r["clock"]
                recent = ball_side[-FPS:]
                poss = int(np.sign(sum(recent))) if recent else 0
                p = float(wm.prob([features(score[0] - score[1], remaining, poss, pre_p)])[0])
                cv2.putText(out, f"home win chance {p:.0%}", (14, panel_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (230, 230, 230), 1, cv2.LINE_AA)
                cv2.rectangle(out, (240, panel_y + 44), (240 + int(300 * p), panel_y + 60), (80, 180, 80), -1)
                cv2.rectangle(out, (240, panel_y + 44), (540, panel_y + 60), (200, 200, 200), 1)
            if f <= shot_until:
                cv2.putText(out, "SHOT", (600, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
            if f <= banner_until:
                cv2.putText(out, banner, (700, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 120), 2)
            writer.write(out)
        f += 1
    writer.release()
    gif = ROOT / "docs" / "img" / "demo.gif"
    gif.parent.mkdir(parents=True, exist_ok=True)
    pal = tmp / "pal.png"
    vf = "fps=8,scale=560:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp / "demo.mp4"), "-vf", f"{vf},palettegen", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp / "demo.mp4"), "-i", str(pal),
                    "-lavfi", f"{vf}[x];[x][1:v]paletteuse", str(gif)], check=True)
    print("wrote", gif, gif.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
