"""A simple eBasketball match engine.

It produces play-by-play logs that look like a short esports basketball game:
4 quarters of 5 minutes, possessions of 6 to 20 seconds, 2- and 3-point shots,
free throws, turnovers and offensive rebounds. Each team has a hidden
strength, so better teams score more often.

Some matches can be marked as "irregular": for part of the game one team
plays well below its usual level (more misses and turnovers). This is used to
test whether unusual play can be spotted from the data. All data is made up.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

QUARTERS = 4
QUARTER_SECONDS = 300
GAME_SECONDS = QUARTERS * QUARTER_SECONDS


@dataclass
class Event:
    t: float                 # seconds since tip-off
    team: str                # "home" or "away"
    kind: str                # shot2, shot3, ft, turnover
    made: bool = False
    points: int = 0
    home: int = 0            # score after the event
    away: int = 0
    possession_start: float = 0.0

    @property
    def quarter(self) -> int:
        return min(QUARTERS, int(self.t // QUARTER_SECONDS) + 1)

    @property
    def clock(self) -> int:
        """Seconds left in the quarter, as shown on the scoreboard."""
        left = QUARTER_SECONDS - (self.t - (self.quarter - 1) * QUARTER_SECONDS)
        return max(0, math.ceil(left - 1e-9))


@dataclass
class Match:
    match_id: str
    home_strength: float
    away_strength: float
    events: list[Event] = field(default_factory=list)
    irregular_team: str | None = None
    irregular_window: tuple[float, float] | None = None

    @property
    def final(self) -> tuple[int, int]:
        return (self.events[-1].home, self.events[-1].away) if self.events else (0, 0)

    @property
    def home_win(self) -> int | None:
        h, a = self.final
        return None if h == a else int(h > a)


def _logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def play_match(match_id: str, home: float, away: float, rng: random.Random,
               irregular: str | None = None, overtime: bool = True, slump: float = 1.0,
               slump_seconds: tuple[float, float] = (150, 300)) -> Match:
    """Simulate one match. `irregular` is "home" or "away" to make that team
    underperform for a random stretch in the second half."""
    m = Match(match_id, home, away)
    window = None
    if irregular:
        length = rng.uniform(*slump_seconds)
        start = rng.uniform(0.5 * GAME_SECONDS, max(0.5 * GAME_SECONDS, GAME_SECONDS - length))
        window = (start, min(GAME_SECONDS, start + length))
        m.irregular_team, m.irregular_window = irregular, window
    score = {"home": 0, "away": 0}
    t = 0.0
    team = "home" if rng.random() < 0.5 else "away"
    end = GAME_SECONDS
    while True:
        while t < end:
            start_t = t
            t = min(end, t + rng.uniform(6, 20))
            other = "away" if team == "home" else "home"
            edge = (home - away) if team == "home" else (away - home)
            edge += 0.15 if team == "home" else 0.0          # small home advantage
            in_slump = irregular == team and window[0] <= t <= window[1]
            k = slump if in_slump else 0.0          # how strong the underperformance is (1.0 = default)
            p_turnover = 0.10 * (1 + 1.2 * k)
            p_foul = 0.07
            r = rng.random()
            keep_ball = False
            if r < p_turnover:
                m.events.append(Event(t, team, "turnover", home=score["home"], away=score["away"],
                                      possession_start=start_t))
            elif r < p_turnover + p_foul:
                made_total = 0
                for _ in range(2):
                    made = rng.random() < (0.74 - 0.15 * k)
                    made_total += made
                    if made:
                        score[team] += 1
                    m.events.append(Event(t, team, "ft", made, int(made), score["home"], score["away"], start_t))
            else:
                three = rng.random() < 0.36
                base = -0.45 if three else 0.08
                p = _logistic(base + 0.35 * edge - 0.75 * k)
                made = rng.random() < p
                pts = (3 if three else 2) if made else 0
                score[team] += pts
                m.events.append(Event(t, team, "shot3" if three else "shot2", made, pts,
                                      score["home"], score["away"], start_t))
                keep_ball = (not made) and rng.random() < 0.25   # offensive rebound
            if not keep_ball:
                team = other
        if not overtime or score["home"] != score["away"]:
            break
        end += 60   # one-minute overtime periods until there is a winner
    return m


def make_season(n_matches: int = 2000, n_teams: int = 24, irregular_rate: float = 0.05, seed: int = 1,
                slump: float = 1.0, bad_teams: int = 0, bad_team_rate: float = 0.3,
                slump_seconds: tuple[float, float] = (150, 300)):
    """If bad_teams > 0, irregular play comes only from that many teams, each
    underperforming in about `bad_team_rate` of its matches."""
    """A league of teams with hidden strengths, playing many matches."""
    rng = random.Random(seed)
    strengths = {f"T{i:02d}": rng.gauss(0, 0.8) for i in range(n_teams)}
    teams = list(strengths)
    bad = set(rng.sample(teams, bad_teams)) if bad_teams else set()
    matches, fixtures = [], []
    for k in range(n_matches):
        h, a = rng.sample(teams, 2)
        irregular = None
        if bad:
            options = [side for side, t in (("home", h), ("away", a)) if t in bad and rng.random() < bad_team_rate]
            irregular = options[0] if options else None
        elif rng.random() < irregular_rate:
            irregular = rng.choice(["home", "away"])
        m = play_match(f"M{k:05d}", strengths[h], strengths[a], rng, irregular, slump=slump,
                       slump_seconds=slump_seconds)
        matches.append(m)
        fixtures.append((h, a))
    return matches, fixtures, strengths, sorted(bad)
