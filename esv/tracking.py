"""Finding and following players and the ball, and spotting shots.

No team colours are given to the detector. It learns the court colour from
the picture, treats anything clearly different as a player or the ball, and
groups players into two teams by colour. Players who overlap form one blob,
which is split by size. Tracks are kept from frame to frame with the
Hungarian algorithm, so each player keeps the same ID.

A shot is a ball flight: the ball leaves every player and travels towards a
hoop for several frames in a row. Passes in this footage are quick, so they
do not look like flights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .render import COURT, HOOPS, HUD_H

PLAYER_AREA = 531          # area of one player disc in pixels (radius 13)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass
class Detections:
    players: np.ndarray            # (n, 2) x, y
    colours: np.ndarray            # (n, 3) mean BGR of each player blob
    ball: np.ndarray | None        # (2,) or None


SCALE = 2                       # work at half resolution for speed
SMALL_AREA = PLAYER_AREA / SCALE ** 2
KERNEL_S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def detect(frame: np.ndarray) -> Detections:
    x0, y0, x1, y1 = COURT
    roi = frame[y0:y1, x0:x1]
    small = cv2.resize(roi, None, fx=1 / SCALE, fy=1 / SCALE, interpolation=cv2.INTER_AREA)
    court = np.median(small[::5, ::5].reshape(-1, 3), axis=0)
    diff = cv2.absdiff(small, np.full_like(small, court.astype(np.uint8)))
    dist = diff.astype(np.float32)
    dist = cv2.sqrt(cv2.add(cv2.add(dist[..., 0] ** 2, dist[..., 1] ** 2), dist[..., 2] ** 2))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))
    bm = cv2.inRange(hsv, (11, 200, 190), (22, 255, 255))   # orange ball; the red-orange hoop rings sit below hue 11
    fg = ((dist > 70) & (white == 0) & (bm == 0)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, KERNEL_S)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(fg)
    pts, cols = [], []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < 0.45 * SMALL_AREA or area > 4.5 * SMALL_AREA:
            continue                      # noise, or a big graphic such as a replay wipe
        if area / float(w * h) < 0.35:
            continue                      # thin shapes (lines, text) are not players
        k = max(1, int(round(area / SMALL_AREA)))
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        sub = labels[y:y + h, x:x + w] == i
        ys, xs = np.nonzero(sub)
        ys, xs = ys + y, xs + x
        if k == 1:
            centres, groups = [cents[i]], [np.ones(len(xs), bool)]
        else:   # overlapping players: split the blob into k parts
            data = np.float32(np.stack([xs, ys], 1))
            _, lab, cen = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                     2, cv2.KMEANS_PP_CENTERS)
            centres, groups = list(cen), [lab.ravel() == j for j in range(k)]
        for c, g in zip(centres, groups):
            pts.append([c[0] * SCALE + x0, c[1] * SCALE + y0])
            cols.append(small[ys[g], xs[g]].mean(axis=0))
    ball = None
    bn, _, bstats, bcent = cv2.connectedComponentsWithStats(bm)
    if bn > 1:
        j = 1 + int(np.argmax(bstats[1:, cv2.CC_STAT_AREA]))
        if bstats[j, cv2.CC_STAT_AREA] >= 4:
            ball = np.array([bcent[j][0] * SCALE + x0, bcent[j][1] * SCALE + y0])
    return Detections(np.array(pts, float).reshape(-1, 2), np.array(cols, float).reshape(-1, 3), ball)


class TeamSplitter:
    """Groups player colours into two teams (learned from the first frames)."""

    def __init__(self):
        self.centres = None
        self.seen: list[np.ndarray] = []

    def assign(self, colours: np.ndarray) -> np.ndarray:
        if self.centres is None:
            self.seen.append(colours)
            data = np.float32(np.concatenate(self.seen))
            if len(data) < 30:
                return np.zeros(len(colours), int)
            _, _, cen = cv2.kmeans(data, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
                                   5, cv2.KMEANS_PP_CENTERS)
            self.centres = cen
        if len(colours) == 0:
            return np.zeros(0, int)
        d = np.linalg.norm(colours[:, None, :] - self.centres[None], axis=2)
        return d.argmin(1)


@dataclass
class Tracker:
    max_dist: float = 30.0
    next_id: int = 0
    tracks: dict = field(default_factory=dict)      # id -> position

    def update(self, pts: np.ndarray) -> list[int]:
        ids = [-1] * len(pts)
        if self.tracks and len(pts):
            tids = list(self.tracks)
            prev = np.array([self.tracks[t] for t in tids])
            cost = np.linalg.norm(prev[:, None] - pts[None], axis=2)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] <= self.max_dist:
                    ids[j] = tids[i]
        new_tracks = {}
        for j, p in enumerate(pts):
            if ids[j] == -1:
                ids[j] = self.next_id
                self.next_id += 1
            new_tracks[ids[j]] = p
        self.tracks = new_tracks
        return ids


class ShotDetector:
    """A shot = the ball away from every player, moving towards a hoop, for `min_frames` frames."""

    def __init__(self, min_frames: int = 4, free_dist: float = 20.0, refractory: int = 25):
        self.min_frames, self.free_dist, self.refractory = min_frames, free_dist, refractory
        self.run, self.prev, self.in_flight = 0, None, False
        self.start, self.last_shot = None, -10_000

    def update(self, frame_no: int, ball, players: np.ndarray) -> dict | None:
        if ball is None:
            self.run, self.prev = 0, None
            return None
        free = len(players) == 0 or np.linalg.norm(players - ball, axis=1).min() > self.free_dist
        towards = None
        if self.prev is not None:
            hoop = min(HOOPS.values(), key=lambda h: np.hypot(*(np.array(h) - ball)))
            towards = np.hypot(*(np.array(hoop) - ball)) < np.hypot(*(np.array(hoop) - self.prev)) - 0.5
        self.prev = ball
        if free and towards:
            if self.run == 0:
                self.start = frame_no - 1
            self.run += 1
        else:
            self.run = 0
            self.in_flight = False
        if self.run >= self.min_frames and not self.in_flight and frame_no - self.last_shot > self.refractory:
            self.in_flight = True
            self.last_shot = frame_no
            hoop_team = "home" if ball[0] > (COURT[0] + COURT[2]) / 2 else "away"
            return dict(frame=self.start, detected_at=frame_no, team=hoop_team)
        return None
