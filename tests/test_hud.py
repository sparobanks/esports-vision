from esv.hud import ScoreTracker, decode


def r(h, a, clock=100):
    return {"home": h, "away": a, "quarter": 1, "clock": clock}


def test_decode():
    chars = list(" 12") + list("  9") + ["2"] + list("405")
    assert decode(chars) == {"home": 12, "away": 9, "quarter": 2, "clock": 4 * 60 + 5}
    assert decode(list(" 12") + list(" X9") + ["2"] + list("405")) is None
    assert decode(list(" 12") + list("  9") + ["2"] + list("475")) is None   # 75 seconds is not a clock value


def test_one_bad_frame_does_not_create_an_event():
    t = ScoreTracker(confirm_frames=3)
    seq = [r(10, 8)] * 5 + [r(18, 8)] + [r(10, 8)] * 5        # one misread frame
    events = [e for i, x in enumerate(seq) for e in t.update(i, x)]
    assert events == []
    raw = ScoreTracker(confirm_frames=1, validate=False)
    assert len([e for i, x in enumerate(seq) for e in raw.update(i, x)]) == 2   # baseline is fooled twice


def test_real_change_is_confirmed_after_three_frames():
    t = ScoreTracker(confirm_frames=3)
    seq = [r(10, 8)] * 3 + [r(12, 8)] * 4
    events = [e for i, x in enumerate(seq) for e in t.update(i, x)]
    assert len(events) == 1
    e = events[0]
    assert e["team"] == "home" and e["points"] == 2 and e["frame"] == 3 and e["detected_at"] == 5


def test_illegal_jumps_are_rejected_unless_scoreboard_was_hidden():
    t = ScoreTracker()
    seq = [r(10, 8)] * 3 + [r(20, 8)] * 5          # +10 in one go is impossible
    assert [e for i, x in enumerate(seq) for e in t.update(i, x)] == []
    t = ScoreTracker()
    seq = [r(10, 8)] * 3 + [None] * 20 + [r(12, 11)] * 4   # both scored during a replay wipe
    events = [e for i, x in enumerate(seq) for e in t.update(i, x)]
    assert {(e["team"], e["points"]) for e in events} == {("home", 2), ("away", 3)}


def test_numbers_must_be_right_aligned():
    # a covered last digit must not turn "14" into "1"
    assert decode(list(" 1 ") + list("  9") + ["2"] + list("405")) is None
