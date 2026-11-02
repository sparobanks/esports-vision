"""Start the Java live-stats service and send it events from Python."""
import json
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "java"
pytestmark = pytest.mark.skipif(not shutil.which("javac"), reason="needs a JDK")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    out = tmp_path_factory.mktemp("java")
    srcs = [str(p) for p in (ROOT / "src" / "main" / "java").rglob("*.java")]
    subprocess.run(["javac", "-d", str(out)] + srcs, check=True, capture_output=True)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen(["java", "-cp", str(out), "esv.LiveStatsServer", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.25)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(10)


def test_python_events_reach_java(server):
    events = [{"type": "score", "team": "home", "points": 2, "home": 2, "away": 0, "t": 3.0},
              {"type": "shot", "team": "away", "t": 5.0},
              {"type": "score", "team": "away", "points": 3, "home": 2, "away": 3, "t": 6.0}]
    body = "\n".join(json.dumps(e) for e in events).encode()
    req = urllib.request.Request(f"{server}/events", data=body, method="POST")
    with urllib.request.urlopen(req) as r:
        assert r.status == 202
    state = json.loads(urllib.request.urlopen(f"{server}/state").read())
    assert (state["home"], state["away"]) == (2, 3)
    assert state["away_shots"] == 1 and state["lead_changes"] == 1 and state["resyncs"] == 0
