"""The live pipeline: video frames in, game events out.

For every frame: read the scoreboard, find players and the ball, keep player
tracks, and check for shots. Score events come from the ScoreTracker, which
only confirms a new score after it has been read the same way for several
frames. Every stage is timed, so we can check it keeps up with live video.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .hud import HudReader, ScoreTracker
from .render import FPS
from .tracking import ShotDetector, TeamSplitter, Tracker, detect


@dataclass
class ClipResult:
    readings: list = field(default_factory=list)       # per frame: dict or None
    score_events: list = field(default_factory=list)
    raw_events: list = field(default_factory=list)     # same reader, no temporal logic (baseline)
    shots: list = field(default_factory=list)
    detections: list = field(default_factory=list)     # per frame: players, team labels, track ids, ball
    timings_ms: dict = field(default_factory=lambda: {"decode": [], "hud": [], "tracking": [], "total": []})


def process_clip(path: str, reader: HudReader, keep_detections: bool = True, on_event=None) -> ClipResult:
    cap = cv2.VideoCapture(path)
    res = ClipResult()
    scores, raw = ScoreTracker(confirm_frames=3), ScoreTracker(confirm_frames=1, validate=False)
    shots, tracker, teams = ShotDetector(), Tracker(), TeamSplitter()
    f = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        reading = reader.read(frame)
        t2 = time.perf_counter()
        det = detect(frame)
        team = teams.assign(det.colours)
        ids = tracker.update(det.players)
        shot = shots.update(f, det.ball, det.players)
        t3 = time.perf_counter()
        for e in scores.update(f, reading):
            e["t"] = round(f / FPS, 2)
            res.score_events.append(e)
            if on_event:
                on_event({"type": "score", **e})
        res.raw_events += raw.update(f, reading)
        if shot:
            res.shots.append(shot)
            if on_event:
                on_event({"type": "shot", **shot})
        res.readings.append(reading)
        if keep_detections:
            res.detections.append(dict(players=det.players.tolist(), team=team.tolist(), ids=ids,
                                       ball=None if det.ball is None else det.ball.tolist()))
        for k, v in (("decode", t1 - t0), ("hud", t2 - t1), ("tracking", t3 - t2), ("total", t3 - t0)):
            res.timings_ms[k].append(1000 * v)
        f += 1
    cap.release()
    return res


def latency_summary(timings_ms: dict) -> dict:
    return {k: {"median": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                "p99": round(float(np.percentile(v, 99)), 2)} for k, v in timings_ms.items() if v}
