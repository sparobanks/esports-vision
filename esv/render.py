"""Render a match as broadcast-style video, with exact ground truth.

The picture is a top-down court with ten players (two team colours), a ball
and a scoreboard bar (HUD) showing both scores, the quarter and the clock.
To make it closer to a real stream, each clip gets:
  * video compression (JPEG quality per frame, then MP4 encoding),
  * small changes in brightness and occasional blur,
  * "replay" wipes that cover the scoreboard for about a second,
  * a short delay between the ball going in and the scoreboard updating.

Two scoreboard styles exist. Style "A" is used to train the digit reader;
style "B" (different colours and layout offsets) is kept back to test how
the reader copes with a broadcast look it has never seen.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .sim import QUARTER_SECONDS, Match

W, H, FPS = 960, 540, 25
HUD_H = 56
COURT = (40, HUD_H + 20, W - 40, H - 20)          # x0, y0, x1, y1
HOOPS = {"home": (COURT[2] - 30, (COURT[1] + COURT[3]) // 2),   # home attacks right
         "away": (COURT[0] + 30, (COURT[1] + COURT[3]) // 2)}
SCORE_DELAY = 0.4                                  # seconds from ball in to scoreboard change


@dataclass(frozen=True)
class Style:
    name: str
    bar: tuple
    box: tuple
    digit: tuple
    home_col: tuple
    away_col: tuple
    dx: int = 0                                     # layout offset
    font: int = cv2.FONT_HERSHEY_SIMPLEX


STYLES = {
    "A": Style("A", bar=(30, 30, 30), box=(60, 60, 60), digit=(255, 255, 255),
               home_col=(40, 40, 220), away_col=(220, 120, 30)),
    "B": Style("B", bar=(90, 30, 10), box=(140, 70, 20), digit=(200, 255, 255),
               home_col=(30, 160, 30), away_col=(160, 40, 160), dx=6, font=cv2.FONT_HERSHEY_DUPLEX),
}

# Fixed digit slots (x of slot centre); every slot is 22 px wide, 36 px tall.
SLOT_W, SLOT_H, SLOT_Y = 22, 36, 10
SLOTS = {
    "home": [250, 272, 294],
    "away": [666, 688, 710],
    "quarter": [470],
    "clock": [514, 552, 574],       # m : s s  (colon drawn between the first two slots)
}


def slot_boxes(style: Style):
    """Pixel boxes (x0, y0, x1, y1) of every digit slot, for a given style."""
    out = {}
    for field, xs in SLOTS.items():
        out[field] = [(x + style.dx - SLOT_W // 2, SLOT_Y, x + style.dx + SLOT_W // 2, SLOT_Y + SLOT_H) for x in xs]
    return out


def _digits(value: int, n: int) -> list[str]:
    s = str(value)
    return [" "] * (n - len(s)) + list(s[-n:])


def hud_text(home: int, away: int, quarter: int, clock_s: int) -> dict[str, list[str]]:
    m, s = divmod(max(0, clock_s), 60)
    return {"home": _digits(home, 3), "away": _digits(away, 3), "quarter": [str(min(quarter, 9))],
            "clock": [str(m % 10), str(s // 10), str(s % 10)]}


def draw_hud(img, style: Style, text: dict, labels=("HOME", "AWAY")):
    cv2.rectangle(img, (0, 0), (W, HUD_H), style.bar, -1)
    boxes = slot_boxes(style)
    for field, bxs in boxes.items():
        x0, y0 = bxs[0][0] - 4, bxs[0][1] - 3
        x1, y1 = bxs[-1][2] + 4, bxs[-1][3] + 3
        cv2.rectangle(img, (x0, y0), (x1, y1), style.box, -1)
        for (bx0, by0, bx1, by1), ch in zip(bxs, text[field]):
            if ch == " ":
                continue
            (tw, th), _ = cv2.getTextSize(ch, style.font, 1.0, 2)
            cv2.putText(img, ch, ((bx0 + bx1 - tw) // 2, (by0 + by1 + th) // 2), style.font, 1.0, style.digit, 2, cv2.LINE_AA)
    cx = (boxes["clock"][0][2] + boxes["clock"][1][0]) // 2
    cv2.circle(img, (cx, 22), 2, style.digit, -1)
    cv2.circle(img, (cx, 34), 2, style.digit, -1)
    cv2.putText(img, labels[0], (120 + style.dx, 36), style.font, 0.8, style.home_col, 2, cv2.LINE_AA)
    cv2.putText(img, labels[1], (760 + style.dx, 36), style.font, 0.8, style.away_col, 2, cv2.LINE_AA)
    cv2.putText(img, "Q", (424 + style.dx, 36), style.font, 0.8, style.digit, 2, cv2.LINE_AA)


def draw_court(img):
    x0, y0, x1, y1 = COURT
    img[HUD_H:] = (60, 110, 160)
    cv2.rectangle(img, (x0, y0), (x1, y1), (230, 230, 230), 2)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    cv2.line(img, (cx, y0), (cx, y1), (230, 230, 230), 2)
    cv2.circle(img, (cx, cy), 50, (230, 230, 230), 2)
    for team, (hx, hy) in HOOPS.items():
        side = 1 if hx > cx else -1
        cv2.ellipse(img, (hx, hy), (150, 150), 0, 90 if side > 0 else -90, 270 if side > 0 else 90, (230, 230, 230), 2)
        cv2.circle(img, (hx, hy), 9, (40, 90, 230), 2)


class Choreography:
    """Moves players and the ball so they follow the match events."""

    def __init__(self, match: Match, rng: random.Random):
        self.m, self.rng = match, rng
        x0, y0, x1, y1 = COURT
        self.pos = {f"{t}{i}": np.array([rng.uniform(x0 + 40, x1 - 40), rng.uniform(y0 + 30, y1 - 30)])
                    for t in ("home", "away") for i in range(5)}
        self.ball = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self.holder = "home0"
        self.flight = None   # (t_start, t_end, start, end, kind)
        self.next_pass = 0.0
        # possessions: (start, end, team, final event list)
        poss = []
        evs = match.events
        i = 0
        while i < len(evs):
            e = evs[i]
            group = [e]
            while i + 1 < len(evs) and evs[i + 1].t == e.t and evs[i + 1].team == e.team:
                i += 1
                group.append(evs[i])
            poss.append((e.possession_start, e.t, e.team, group))
            i += 1
        self.poss = poss

    def possession_at(self, t):
        for p in self.poss:
            if p[0] <= t <= p[1] + 0.6:
                return p
        return self.poss[-1]

    def step(self, t: float, dt: float):
        start, end, team, group = self.possession_at(t)
        other = "away" if team == "home" else "home"
        hx, hy = HOOPS[team]
        side = 1 if team == "home" else -1
        x0, y0, x1, y1 = COURT
        # targets: attackers spread in front of their hoop, defenders between them and the hoop
        for i in range(5):
            ang = -1.1 + i * 0.55
            a_target = np.array([hx - side * (110 + 60 * (i % 2)) * math.cos(ang) - side * 20,
                                 hy + 150 * math.sin(ang)])
            a_target += np.array([self.rng.gauss(0, 6), self.rng.gauss(0, 6)])
            d_target = a_target + np.array([side * 35, 0])
            for key, tgt in ((f"{team}{i}", a_target), (f"{other}{i}", d_target)):
                tgt = np.clip(tgt, [x0 + 15, y0 + 15], [x1 - 15, y1 - 15])
                v = tgt - self.pos[key]
                n = np.linalg.norm(v)
                speed = 180 * dt
                self.pos[key] = self.pos[key] + (v if n < speed else v / n * speed)
        shooting = group[0].kind in ("shot2", "shot3", "ft")
        release = end - 0.8
        if self.flight and t >= self.flight[1]:
            self.flight = None
        if self.flight is None and shooting and release <= t < end:
            shooter = self.holder if self.holder.startswith(team) else f"{team}0"
            start_pt = self.pos[shooter].copy()
            if group[0].kind == "ft":
                start_pt = np.array([hx - side * 140, hy])
            self.flight = (t, end, start_pt, np.array([hx, hy], float), "shot")
        if self.flight:
            t0, t1, a, b, _ = self.flight
            f = min(1.0, (t - t0) / max(1e-6, t1 - t0))
            self.ball = a + (b - a) * f
            self.ball[1] -= 40 * math.sin(math.pi * f)      # arc
            return
        if not self.holder.startswith(team):
            self.holder = f"{team}{self.rng.randrange(5)}"
        if t >= self.next_pass and t < release - 0.3:
            self.holder = f"{team}{self.rng.randrange(5)}"
            self.next_pass = t + self.rng.uniform(1.0, 2.2)
        self.ball = self.pos[self.holder] + np.array([8 * side, -8])


def render_clip(match: Match, t0: float, t1: float, out_dir: str | Path, style_name: str = "A",
                seed: int = 0, replay_wipes: int = 3, degrade: str = "normal") -> dict:
    """Write <out_dir>/clip.mp4 and truth.json covering match time t0..t1.

    degrade="hard" simulates a poor stream: heavy compression, a low-resolution
    source scaled up, sensor noise and more blur.
    """
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    style = STYLES[style_name]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    chor = Choreography(match, rng)
    # warm up the choreography to t0 at low resolution
    t = 0.0
    while t < t0:
        chor.step(t, 0.2)
        t += 0.2
    n = int((t1 - t0) * FPS)
    wipes = sorted(rng.sample(range(FPS * 5, n - FPS * 3), replay_wipes)) if n > FPS * 10 else []
    wipe_len = {w: rng.randint(int(0.6 * FPS), int(1.2 * FPS)) for w in wipes}
    quality = rng.randint(35, 75) if degrade == "normal" else rng.randint(8, 16)
    scored = [e for e in match.events if e.points > 0]
    frames, events = [], []
    base = np.zeros((H, W, 3), np.uint8)
    draw_court(base)
    shown = None
    for f in range(n):
        t = t0 + f / FPS
        chor.step(t, 1 / FPS)
        past = [e for e in scored if e.t + SCORE_DELAY <= t]
        home = past[-1].home if past else 0
        away = past[-1].away if past else 0
        q = min(4, int(t // QUARTER_SECONDS) + 1) if t < 4 * QUARTER_SECONDS else 5
        qstart = (q - 1) * QUARTER_SECONDS if q <= 4 else 4 * QUARTER_SECONDS + 60 * (int((t - 1200) // 60))
        qlen = QUARTER_SECONDS if q <= 4 else 60
        clock = max(0, math.ceil(qlen - (t - qstart) - 1e-9))
        img = base.copy()
        for key, p in chor.pos.items():
            col = style.home_col if key.startswith("home") else style.away_col
            cv2.circle(img, (int(p[0]), int(p[1])), 13, col, -1)
            cv2.circle(img, (int(p[0]), int(p[1])), 13, (20, 20, 20), 1)
        bx, by = chor.ball
        cv2.circle(img, (int(bx), int(by)), 6, (0, 140, 255), -1)
        text = hud_text(home, away, q, clock)
        draw_hud(img, style, text)
        hud_visible = True
        for w in wipes:
            if w <= f < w + wipe_len[w]:
                hud_visible = False
                k = (f - w) / wipe_len[w]
                cv2.rectangle(img, (0, 0), (int(W * min(1, 2 * k + 0.3)), H // 3), (10, 10, 10), -1)
                cv2.putText(img, "REPLAY", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        # broadcast imperfections
        alpha = 1 + rng.gauss(0, 0.04)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=rng.gauss(0, 4))
        if rng.random() < (0.08 if degrade == "normal" else 0.3):
            img = cv2.GaussianBlur(img, (3, 3), 0)
        if degrade == "hard":
            small = cv2.resize(img, (int(W * 0.4), int(H * 0.4)), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
            noise = rng_np.normal(0, 8, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        writer.write(img)
        state = (home, away)
        if shown is not None and state != shown:
            team = "home" if home != shown[0] else "away"
            events.append(dict(frame=f, t=round(t, 2), team=team, points=(home - shown[0]) + (away - shown[1]),
                               home=home, away=away))
        shown = state
        frames.append(dict(home=home, away=away, quarter=q, clock=clock, hud_visible=hud_visible,
                           text=text, ball=[round(float(bx), 1), round(float(by), 1)],
                           players={k: [round(float(v[0]), 1), round(float(v[1]), 1)] for k, v in chor.pos.items()}))
    writer.release()
    shots, seen = [], set()
    for e in match.events:
        if e.kind in ("shot2", "shot3", "ft") and t0 + 0.8 <= e.t <= t1 and (e.t, e.team) not in seen:
            seen.add((e.t, e.team))   # two free throws share one ball flight
            shots.append(dict(frame=int(round((e.t - 0.8 - t0) * FPS)), team=e.team, kind=e.kind, made=e.made))
    truth = dict(match_id=match.match_id, t0=t0, t1=t1, fps=FPS, style=style_name, degrade=degrade, jpeg_quality=quality,
                 replay_wipes=[(w, wipe_len[w]) for w in wipes], score_events=events, shot_releases=shots,
                 frames=frames)
    (out / "truth.json").write_text(json.dumps(truth))
    return truth
