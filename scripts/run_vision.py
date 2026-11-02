"""Train the scoreboard reader and test the whole vision pipeline.

    python scripts/make_clips.py      # first, render the clips
    python scripts/run_vision.py

Writes models/digitnet.pt and reports/vision.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from esv import hud  # noqa: E402
from esv.pipeline import latency_summary, process_clip  # noqa: E402
from esv.render import FPS, STYLES, W, slot_boxes  # noqa: E402

CLIPS = ROOT / "data" / "clips"
OUT = ROOT / "reports"
MODEL = ROOT / "models" / "digitnet.pt"


def occluded_slots(truth: dict, f: int, style) -> list[bool]:
    """Which digit slots a replay wipe covers (at least half) in frame f."""
    width = 0
    for w, n in truth["replay_wipes"]:
        if w <= f < w + n:
            width = int(W * min(1, 2 * (f - w) / n + 0.3))
    boxes = [b for fld in hud.FIELDS for b in slot_boxes(style)[fld]]
    return [(b[0] + b[2]) / 2 < width for b in boxes]


def labels_for(truth: dict, f: int, style) -> list[int]:
    t = truth["frames"][f]["text"]
    chars = [c for fld in hud.FIELDS for c in t[fld]]
    occ = occluded_slots(truth, f, style)
    return [hud.IDX[" "] if o else hud.IDX[c] for c, o in zip(chars, occ)]


def dataset(names: list[str], step: int):
    X, y, occ = [], [], []
    for name in names:
        truth = json.loads((CLIPS / name / "truth.json").read_text())
        style = STYLES[truth["style"]]
        cap = cv2.VideoCapture(str(CLIPS / name / "clip.mp4"))
        f = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f % step == 0:
                X += hud.crops(frame, style)
                y += labels_for(truth, f, style)
                occ += occluded_slots(truth, f, style)
            f += 1
    return np.array(X, np.float32), np.array(y, np.int64), np.array(occ, bool)


def match_events(truth: dict, detected: list[dict], tol: int = FPS) -> dict:
    """Match detected score events to true ones (same team and points, close in time)."""
    wipes = truth["replay_wipes"]

    def window_end(f):
        for w, n in wipes:
            if w <= f < w + n + tol:
                return w + n + tol
        return f + tol

    true = truth["score_events"]
    used, tp, lat, pts_ok = set(), 0, [], 0
    for te in true:
        best = None
        for j, de in enumerate(detected):
            if j in used or de["team"] != te["team"]:
                continue
            if te["frame"] - tol <= de["frame"] <= window_end(te["frame"]):
                best = j
                break
        if best is not None:
            used.add(best)
            tp += 1
            pts_ok += detected[best]["points"] == te["points"]
            lat.append((detected[best].get("detected_at", detected[best]["frame"]) - te["frame"]) * 1000 / FPS)
    return dict(true=len(true), detected=len(detected), matched=tp, false_events=len(detected) - tp,
                missed=len(true) - tp, points_correct=pts_ok, latencies_ms=lat)


def tracking_metrics(truth: dict, res) -> dict:
    tp = fp = fn = 0
    team_ok = team_n = 0
    switches = 0
    prev_assign: dict[str, int] = {}
    ball_err = []
    for f, (gt, det) in enumerate(zip(truth["frames"], res.detections)):
        g_keys = list(gt["players"])
        g = np.array([gt["players"][k] for k in g_keys])
        d = np.array(det["players"]).reshape(-1, 2)
        if len(d):
            c = np.linalg.norm(g[:, None] - d[None], axis=2)
            r, cc = linear_sum_assignment(c)
            ok = c[r, cc] < 8
            m = int(ok.sum())
            # team labels: learn the mapping from cluster id to real team on the fly (majority)
            for i, j, good in zip(r, cc, ok):
                if not good:
                    continue
                key = g_keys[i]
                tid = det["ids"][j]
                if key in prev_assign and prev_assign[key] != tid:
                    switches += 1
                prev_assign[key] = tid
                team_n += 1
                team_ok += (det["team"][j] == (0 if key.startswith("home") else 1))
        else:
            m = 0
        tp += m
        fn += len(g) - m
        fp += len(d) - m
        if det["ball"] is not None and gt["hud_visible"]:
            ball_err.append(float(np.hypot(det["ball"][0] - gt["ball"][0], det["ball"][1] - gt["ball"][1])))
    team_acc = team_ok / max(team_n, 1)
    team_acc = max(team_acc, 1 - team_acc)   # cluster ids are arbitrary: 0/1 may be swapped
    return dict(player_precision=round(tp / max(tp + fp, 1), 4), player_recall=round(tp / max(tp + fn, 1), 4),
                team_accuracy=round(team_acc, 4), id_switches=switches,
                id_switches_per_1000_player_frames=round(1000 * switches / max(tp, 1), 2),
                ball_median_error_px=round(float(np.median(ball_err)), 2) if ball_err else None)


def shot_metrics(truth: dict, shots: list[dict], tol: int = 15) -> dict:
    true = truth["shot_releases"]
    used, tp, lat = set(), 0, []
    for ts in true:
        for j, s in enumerate(shots):
            if j not in used and abs(s["frame"] - ts["frame"]) <= tol:
                used.add(j)
                tp += 1
                lat.append((s["detected_at"] - ts["frame"]) * 1000 / FPS)
                break
    return dict(true=len(true), detected=len(shots), matched=tp, latencies_ms=lat)


def frame_accuracy(truth: dict, readings: list) -> dict:
    vis = [(gt, r) for gt, r in zip(truth["frames"], readings) if gt["hud_visible"]]
    full = sum(r is not None and r["home"] == gt["home"] and r["away"] == gt["away"] and r["clock"] == gt["clock"]
               for gt, r in vis)
    unread = sum(r is None for _, r in vis)
    wrong_score = sum(r is not None and (r["home"], r["away"]) != (gt["home"], gt["away"]) for gt, r in vis)
    hidden = [(gt, r) for gt, r in zip(truth["frames"], readings) if not gt["hud_visible"]]
    hidden_wrong = sum(r is not None and (r["home"], r["away"]) != (gt["home"], gt["away"]) for gt, r in hidden)
    return dict(visible_frames=len(vis), fully_correct=full, unreadable=unread, wrong_score=wrong_score,
                hidden_frames=len(hidden), hidden_frames_with_wrong_score=hidden_wrong)


def main():
    OUT.mkdir(exist_ok=True)
    MODEL.parent.mkdir(exist_ok=True)
    index = json.loads((CLIPS / "index.json").read_text())
    train_names = [c["name"] for c in index if c["split"] == "train"]
    t = time.time()
    X, y, _ = dataset(train_names, step=3)
    print(f"training crops: {len(X)} ({time.time() - t:.0f}s)")
    t = time.time()
    trained = hud.train(X, y, epochs=6)
    torch.save(trained.model.state_dict(), MODEL)
    print("loss per epoch", trained.history, f"({time.time() - t:.0f}s)")
    model = trained.model
    report = dict(train=dict(clips=train_names, crops=int(len(X)), loss_per_epoch=trained.history,
                             parameters=int(sum(p.numel() for p in model.parameters()))))
    for split in ("testA", "testB", "testC"):
        names = [c["name"] for c in index if c["split"] == split]
        Xs, ys, occ = dataset(names, step=5)
        pred, _ = hud.predict(model, Xs)
        agg = dict(slot_accuracy_visible=round(float((pred[~occ] == ys[~occ]).mean()), 4),
                   covered_slots_read_as_no_digit=round(float((pred[occ] == hud.IDX[" "]).mean()), 4) if occ.any() else None,
                   slots_tested=int(len(ys)))
        ev, raw, shots, tr, fa, timings = [], [], [], [], [], {"decode": [], "hud": [], "tracking": [], "total": []}
        for name in names:
            truth = json.loads((CLIPS / name / "truth.json").read_text())
            reader = hud.HudReader(model, STYLES[truth["style"]])
            res = process_clip(str(CLIPS / name / "clip.mp4"), reader)
            (CLIPS / name / "detected.json").write_text(json.dumps(dict(score_events=res.score_events, shots=res.shots)))
            ev.append(match_events(truth, res.score_events))
            raw.append(match_events(truth, res.raw_events))
            shots.append(shot_metrics(truth, res.shots))
            tr.append(tracking_metrics(truth, res))
            fa.append(frame_accuracy(truth, res.readings))
            for k in timings:
                timings[k] += res.timings_ms[k]
            print(split, name, ev[-1]["matched"], "/", ev[-1]["true"], "false", ev[-1]["false_events"],
                  "| raw false", raw[-1]["false_events"])

        def total(lst, keys):
            return {k: int(sum(d[k] for d in lst)) for k in keys}

        e = total(ev, ["true", "detected", "matched", "false_events", "missed", "points_correct"])
        r = total(raw, ["true", "detected", "matched", "false_events", "missed"])
        lat = [x for d in ev for x in d["latencies_ms"]]
        s = total(shots, ["true", "detected", "matched"])
        slat = [x for d in shots for x in d["latencies_ms"]]
        f = total(fa, ["visible_frames", "fully_correct", "unreadable", "wrong_score", "hidden_frames",
                       "hidden_frames_with_wrong_score"])
        report[split] = dict(
            style=STYLES[json.loads((CLIPS / names[0] / "truth.json").read_text())["style"]].name,
            degrade=json.loads((CLIPS / names[0] / "truth.json").read_text()).get("degrade", "normal"), clips=names, seconds_of_video=round(sum(d["visible_frames"] + d["hidden_frames"]
                                                                     for d in fa) / FPS),
            digit_reader=agg, frames=f,
            score_events=dict(**e, median_latency_ms=round(float(np.median(lat)), 0) if lat else None,
                              p95_latency_ms=round(float(np.percentile(lat, 95)), 0) if lat else None),
            score_events_without_temporal_logic=r,
            shots=dict(**s, median_latency_ms=round(float(np.median(slat)), 0) if slat else None),
            tracking={k: round(float(np.mean([d[k] for d in tr if d[k] is not None])), 4)
                      for k in tr[0] if k != "id_switches"},
            latency_ms_per_frame=latency_summary(timings),
        )
    (OUT / "vision.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "train"}, indent=1))



if __name__ == "__main__":
    main()
