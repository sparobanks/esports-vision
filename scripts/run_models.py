"""Pricing and integrity models, plus the live end-to-end demo.

    python scripts/run_models.py

1. Elo ratings and pre-match probabilities over a simulated season.
2. In-play win probability: calibration on held-out matches, by quarter,
   compared with simpler baselines.
3. Irregular play: how well underperformance can be spotted, for different
   lengths of underperformance.
4. Live demo: the probability driven only by what the vision pipeline read
   from the video, compared with the probability from the true game state.

Writes reports/models.json and charts in reports/.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from esv import hud  # noqa: E402
from esv.models import (Elo, IrregularDetector, WinModel, ece, fair_odds, features, irregular_eval,  # noqa: E402
                        scores, states)
from esv.pipeline import process_clip  # noqa: E402
from esv.render import FPS, STYLES  # noqa: E402
from esv.sim import GAME_SECONDS, QUARTER_SECONDS, make_season, play_match  # noqa: E402

OUT = ROOT / "reports"
CLIPS = ROOT / "data" / "clips"
INK, MUTED, GRID, C1, C2 = "#1d2321", "#5d6763", "#dde2df", "#1f5fa8", "#c2571a"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(exist_ok=True)
    report = {}

    # ---------------------------------------------------------- season, Elo, in-play model
    matches, fixtures, strengths, _ = make_season(4000, seed=1)
    pre, gaps = Elo().run(matches, fixtures)
    y = np.array([m.home_win for m in matches])
    burn, split = 1000, 3000
    true_gap = np.array([strengths[h] - strengths[a] for h, a in fixtures])
    report["prematch"] = dict(
        matches_after_burn_in=len(matches) - burn,
        elo_vs_true_strength_correlation=round(float(np.corrcoef(gaps[burn:], true_gap[burn:])[0, 1]), 3),
        **scores(y[split:], pre[split:]))

    X, Y = [], []
    for i in range(burn, split):
        for r in states(matches[i], pre[i]):
            X.append(r)
            Y.append(y[i])
    Xt, Yt, P0, phase = [], [], [], []
    for i in range(split, len(matches)):
        for k, r in enumerate(states(matches[i], pre[i])):
            Xt.append(r)
            Yt.append(y[i])
            P0.append(pre[i])
            phase.append(min(3, int(k * 20 // QUARTER_SECONDS)))
    Xt, Yt, P0, phase = np.array(Xt), np.array(Yt), np.array(P0), np.array(phase)
    wm = WinModel().fit(X, Y)
    p = wm.prob(Xt)
    score_only = WinModel().fit(np.array(X)[:, :4], Y).prob(Xt[:, :4])
    report["inplay"] = dict(
        train_states=len(X), test_states=len(Xt), test_matches=len(matches) - split,
        model=scores(Yt, p), score_and_clock_only=scores(Yt, score_only), prematch_only=scores(Yt, P0),
        by_quarter={f"Q{q + 1}": scores(Yt[phase == q], p[phase == q]) for q in range(4)},
        example_prices=[dict(state=s, p_home=round(float(wm.prob([features(*v)])[0]), 3),
                             fair_odds_home=fair_odds(float(wm.prob([features(*v)])[0])))
                        for s, v in [("tip-off, even teams", (0, GAME_SECONDS, 1, 0.55)),
                                     ("home +6, 5 minutes left, away ball", (6, 300, -1, 0.55)),
                                     ("home -3, 30 seconds left, home ball", (-3, 30, 1, 0.55))]])

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    style(ax)
    for probs, label, col in [(p, "In-play model", C1), (P0, "Pre-match price only", C2)]:
        bins = np.array_split(np.argsort(probs), 12)
        ax.plot([probs[b].mean() for b in bins], [Yt[b].mean() for b in bins], "-o", ms=4, color=col, label=label)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    ax.set_xlabel("Predicted chance home team wins", color=INK)
    ax.set_ylabel("Share of games home team won", color=INK)
    ax.set_title(f"Calibration on {len(matches) - split} held-out matches\n"
                 f"in-play calibration error {report['inplay']['model']['ece']:.3f}", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- irregular play
    normal = [(m, g) for m, g in zip(matches[burn:], gaps[burn:]) if m.irregular_team is None]
    det = IrregularDetector().fit([m for m, _ in normal], [g for _, g in normal])
    report["irregular"] = []
    for label, secs in [("2.5 to 5 minutes", (150, 300)), ("5 minutes", (300, 300)), ("10 minutes (whole second half)", (600, 600))]:
        ms, fx, _, _ = make_season(3000, seed=11, slump_seconds=secs)
        _, g2 = Elo().run(ms, fx)
        report["irregular"].append(dict(underperformance_length=label, **irregular_eval(ms[1000:], g2[1000:], det)))
        print("irregular", report["irregular"][-1])

    # ---------------------------------------------------------- live demo from video
    vis = json.loads((OUT / "vision.json").read_text())
    model = hud.DigitNet()
    model.load_state_dict(torch.load(ROOT / "models" / "digitnet.pt"))
    model.eval()
    # demo on the test clip with the closest pre-match odds (the most interesting game to price)
    index = json.loads((CLIPS / "index.json").read_text())
    options = []
    for c in index:
        if c["split"] != "testA":
            continue
        tr = json.loads((CLIPS / c["name"] / "truth.json").read_text())
        rng = random.Random(int(tr["match_id"].replace("clip", "")))
        hs, as_ = rng.gauss(0, 0.6), rng.gauss(0, 0.6)
        mc = random.Random(5)
        pp = float(np.mean([play_match("mc", hs, as_, mc).home_win for _ in range(400)]))   # pre-match price
        options.append((abs(pp - 0.5), c["name"], tr, hs, as_, pp))
    _, demo, truth, hs, as_, pre_p = min(options, key=lambda o: o[0])
    rng = random.Random(int(truth["match_id"].replace("clip", "")))
    rng.gauss(0, 0.6), rng.gauss(0, 0.6)
    m = play_match(truth["match_id"], hs, as_, rng)
    res = process_clip(str(CLIPS / demo / "clip.mp4"), hud.HudReader(model, STYLES[truth["style"]]),
                       keep_detections=True)
    cv_p, true_p, ts = [], [], []
    score = None
    ball_side = []
    for f in range(0, len(res.readings)):
        r = res.readings[f]
        ball = res.detections[f]["ball"]
        if ball is not None:
            ball_side.append(1 if ball[0] > 480 else -1)
        if f % FPS:
            continue
        evs = [e for e in res.score_events if e["detected_at"] <= f]
        if evs:
            score = (evs[-1]["home"], evs[-1]["away"])
        elif score is None and r is not None:
            score = (r["home"], r["away"])
        if score is None or r is None:
            continue
        remaining = (4 - r["quarter"]) * QUARTER_SECONDS + r["clock"]
        recent = ball_side[-FPS:]
        poss = int(np.sign(sum(recent))) if recent else 0
        cv_p.append(float(wm.prob([features(score[0] - score[1], remaining, poss, pre_p)])[0]))
        t = truth["t0"] + f / FPS
        done = [e for e in m.events if e.t <= t]
        h, a = (done[-1].home, done[-1].away) if done else (0, 0)
        nxt = next((e for e in m.events if e.t > t), None)
        tposs = 0 if nxt is None else (1 if nxt.team == "home" else -1)
        true_p.append(float(wm.prob([features(h - a, GAME_SECONDS - t, tposs, pre_p)])[0]))
        ts.append(t)
    cv_p, true_p = np.array(cv_p), np.array(true_p)
    report["live_demo"] = dict(clip=demo, seconds=len(ts), prematch_home_win=round(pre_p, 3),
                               mean_abs_difference_cv_vs_true=round(float(np.mean(np.abs(cv_p - true_p))), 4),
                               max_abs_difference=round(float(np.max(np.abs(cv_p - true_p))), 4))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    style(ax)
    ax.plot(np.array(ts) / 60, true_p, color=MUTED, lw=2.5, alpha=0.6, label="From the true game state")
    ax.plot(np.array(ts) / 60, cv_p, color=C1, lw=1.5, label="From the video only")
    for e in res.score_events:
        ax.axvline((truth["t0"] + e["detected_at"] / FPS) / 60, color=C2 if e["team"] == "away" else C1,
                   lw=0.8, alpha=0.35)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Match time (minutes)", color=INK)
    ax.set_ylabel("Chance home team wins", color=INK)
    ax.set_title("Win probability from the video alone stays within "
                 f"{report['live_demo']['mean_abs_difference_cv_vs_true']:.3f} of the true game state on average",
                 loc="left", fontsize=10)
    ax.plot([], [], color=C1, lw=0.8, alpha=0.5, label="Home scored (detected)")
    ax.plot([], [], color=C2, lw=0.8, alpha=0.5, label="Away scored (detected)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "live_probability.png", dpi=150)
    plt.close(fig)

    (OUT / "models.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "irregular"}, indent=1))


if __name__ == "__main__":
    main()
