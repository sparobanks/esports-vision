"""Render the training and test video clips (made-up matches).

    python scripts/make_clips.py            # full set, a few minutes
    python scripts/make_clips.py --quick    # small set for CI
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from esv.render import render_clip  # noqa: E402
from esv.sim import GAME_SECONDS, play_match  # noqa: E402

OUT = ROOT / "data" / "clips"


def plan(quick: bool):
    rng = random.Random(42)
    jobs = []
    n_train, n_test, train_len, test_len = (2, 1, 40, 60) if quick else (5, 3, 120, 240)
    for k in range(n_train):
        jobs.append(("train", f"train{k}", 100 + k, "A", train_len, rng.random()))
    for k in range(n_test):
        u = rng.random()
        jobs.append(("testA", f"testA{k}", 200 + k, "A", test_len, u))
        jobs.append(("testB", f"testB{k}", 200 + k, "B", test_len, u))   # same moments, unseen broadcast style
        jobs.append(("testC", f"testC{k}", 200 + k, "B", test_len, u))   # unseen style and a poor-quality stream
    return jobs


def run(job):
    split, name, match_seed, style, length, u = job
    degrade = "hard" if split == "testC" else "normal"
    rng = random.Random(match_seed)
    m = play_match(f"clip{match_seed}", rng.gauss(0, 0.6), rng.gauss(0, 0.6), rng)
    t0 = round(u * (GAME_SECONDS - length - 10), 1)
    truth = render_clip(m, t0, t0 + length, OUT / name, style, seed=match_seed * 7 + (style == "B") + 2 * (degrade == "hard"),
                        degrade=degrade)
    return name, split, len(truth["frames"]), len(truth["score_events"]), len(truth["shot_releases"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", help="render only one split, e.g. testC")
    a = ap.parse_args()
    jobs = plan(a.quick)
    if a.only:
        jobs = [j for j in jobs if j[0] == a.only]
    with ProcessPoolExecutor(2) as ex:
        results = list(ex.map(run, jobs))
    index = [dict(name=n, split=s, frames=f, score_events=e, shots=sh) for n, s, f, e, sh in results]
    idx_path = OUT / "index.json"
    if a.only and idx_path.exists():
        index = [r for r in json.loads(idx_path.read_text()) if r["split"] != a.only] + index
    idx_path.write_text(json.dumps(index, indent=1))
    for r in index:
        print(r)


if __name__ == "__main__":
    main()
