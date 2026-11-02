"""Reading the scoreboard: a small PyTorch CNN for digits, plus temporal logic.

Each scoreboard digit sits in a fixed slot for a given broadcast template, so
we crop every slot and classify it as 0-9 or "no digit". Crops are turned to
grey and contrast-stretched, so the reader does not depend on the template's
colours. A frame only counts as read if every slot is confident and the
digits make sense (scores right-aligned, valid clock).

Reading single frames is not enough: compression, blur and wipes cause
misreads. The ScoreTracker only accepts a new score when it has been read the
same way for several frames in a row and is a legal change (scores only go
up, by 1, 2 or 3 at a time, unless the scoreboard was hidden).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torch import nn

from .render import SLOT_H, SLOT_W, Style, slot_boxes

CLASSES = list("0123456789") + [" "]        # " " = no digit (empty slot, or covered by graphics)
IDX = {c: i for i, c in enumerate(CLASSES)}
FIELDS = ["home", "away", "quarter", "clock"]


def prep(crop: np.ndarray) -> np.ndarray:
    """Grey, contrast-stretched, white digit on dark background, 36x22 float."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    g = (g - lo) / max(hi - lo, 12.0)
    g = np.clip(g, 0, 1)
    if g.mean() > 0.5:          # make digits light on dark whatever the template
        g = 1 - g
    if g.shape != (SLOT_H, SLOT_W):
        g = cv2.resize(g, (SLOT_W, SLOT_H))
    return g


def crops(frame: np.ndarray, style: Style) -> list[np.ndarray]:
    out = []
    for f in FIELDS:
        for x0, y0, x1, y1 in slot_boxes(style)[f]:
            out.append(prep(frame[y0:y1, x0:x1]))
    return out


class DigitNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),          # 18x11
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),         # 9x5
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(32 * 9 * 5, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shifts, scale, blur and noise so the reader copes with other broadcasts."""
    h, w = x.shape
    s = rng.uniform(0.85, 1.15)
    M = np.float32([[s, 0, rng.uniform(-2, 2) + (1 - s) * w / 2], [0, s, rng.uniform(-2, 2) + (1 - s) * h / 2]])
    x = cv2.warpAffine(x, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:
        x = cv2.GaussianBlur(x, (3, 3), rng.uniform(0.3, 1.0))
    if rng.random() < 0.5:   # thicker or thinner strokes, like a different font
        k = np.ones((2, 2), np.float32)
        x = cv2.dilate(x, k) if rng.random() < 0.5 else cv2.erode(x, k)
    x = x * rng.uniform(0.7, 1.1) + rng.normal(0, 0.06, x.shape)
    return np.clip(x, 0, 1).astype(np.float32)


@dataclass
class Trained:
    model: DigitNet
    history: list = field(default_factory=list)


def train(X: np.ndarray, y: np.ndarray, epochs: int = 6, seed: int = 0, aug: bool = True) -> Trained:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DigitNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(X))
        total = 0.0
        for i in range(0, len(X), 256):
            idx = order[i:i + 256]
            xb = np.stack([augment(X[j], rng) for j in idx]) if aug else X[idx]
            xb = torch.from_numpy(xb).unsqueeze(1)
            yb = torch.from_numpy(y[idx])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        hist.append(round(total / len(X), 4))
    model.eval()
    return Trained(model, hist)


@torch.no_grad()
def predict(model: DigitNet, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = model(torch.from_numpy(np.asarray(X, np.float32)).unsqueeze(1))
    p = torch.softmax(logits, 1)
    conf, idx = p.max(1)
    return idx.numpy(), conf.numpy()


def decode(chars: list[str]) -> dict | None:
    """Turn the 10 slot readings into numbers, or None if they do not make sense."""
    home, away, quarter, clock = chars[0:3], chars[3:6], chars[6:7], chars[7:10]

    def num(cs):
        s = "".join(cs).lstrip()          # numbers are right-aligned: blanks only on the left
        if not s or not s.isdigit():
            return None
        return int(s)

    h, a, q = num(home), num(away), num(quarter)
    if None in (h, a, q) or " " in clock:
        return None
    clock_s = int(clock[0]) * 60 + int(clock[1]) * 10 + int(clock[2])
    if int(clock[1]) > 5:
        return None
    return {"home": h, "away": a, "quarter": q, "clock": clock_s}


class HudReader:
    def __init__(self, model: DigitNet, style: Style, min_conf: float = 0.6):
        self.model, self.style, self.min_conf = model, style, min_conf

    def read(self, frame: np.ndarray) -> dict | None:
        idx, conf = predict(self.model, np.stack(crops(frame, self.style)))
        if conf.min() < self.min_conf:
            return None
        return decode([CLASSES[i] for i in idx])


class ScoreTracker:
    """Turns noisy per-frame readings into clean score events."""

    def __init__(self, confirm_frames: int = 3, validate: bool = True):
        self.k, self.validate = confirm_frames, validate
        self.score: tuple[int, int] | None = None
        self.candidate, self.count = None, 0
        self.unreadable_run = 0

    def update(self, frame_no: int, reading: dict | None) -> list[dict]:
        if reading is None:
            self.unreadable_run += 1
            return []
        s = (reading["home"], reading["away"])
        if self.score is None:
            self.score = s
            return []
        if s == self.score:
            self.candidate, self.count = None, 0
            self.unreadable_run = 0
            return []
        if not self.validate:
            ev = self._events(frame_no, frame_no, s)
            self.score = s
            return ev
        dh, da = s[0] - self.score[0], s[1] - self.score[1]
        hidden = self.unreadable_run > 0
        legal = dh >= 0 and da >= 0 and (dh + da) > 0 and (
            (dh <= 3 and da == 0) or (da <= 3 and dh == 0) or (hidden and dh <= 6 and da <= 6))
        if not legal:
            self.candidate, self.count = None, 0
            return []
        if s == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = s, 1
        if self.count >= self.k:
            ev = self._events(frame_no - self.k + 1, frame_no, s)
            self.score = s
            self.candidate, self.count = None, 0
            self.unreadable_run = 0
            return ev
        return []

    def _events(self, first_seen, detected_at, s):
        """first_seen: first frame showing the new score; detected_at: frame when we were sure."""
        out = []
        for i, team in enumerate(("home", "away")):
            d = s[i] - self.score[i]
            if d != 0:
                out.append(dict(frame=first_seen, detected_at=detected_at, team=team, points=d, home=s[0], away=s[1]))
        return out
