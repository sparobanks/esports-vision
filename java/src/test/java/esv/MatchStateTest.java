package esv;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;

/**
 * Tests with a tiny built-in runner, so they need nothing but the JDK.
 * Run with: java -ea -cp out esv.MatchStateTest
 */
public final class MatchStateTest {
    interface Test { void run() throws Exception; }

    public static void main(String[] args) {
        List<String> failed = new ArrayList<>();
        int passed = 0;
        Object[][] tests = {
            {"parses flat JSON", (Test) MatchStateTest::parsesJson},
            {"rejects nested JSON", (Test) MatchStateTest::rejectsNested},
            {"keeps the score and runs", (Test) MatchStateTest::scoreAndRuns},
            {"counts lead changes", (Test) MatchStateTest::leadChanges},
            {"resyncs to scoreboard totals", (Test) MatchStateTest::resync},
            {"rejects bad events", (Test) MatchStateTest::rejectsBad},
            {"HTTP server round trip", (Test) MatchStateTest::server},
        };
        for (Object[] t : tests) {
            try {
                ((Test) t[1]).run();
                passed++;
                System.out.println("PASS " + t[0]);
            } catch (Throwable e) {
                failed.add(t[0] + ": " + e);
                System.out.println("FAIL " + t[0] + ": " + e);
            }
        }
        System.out.println(passed + " passed, " + failed.size() + " failed");
        if (!failed.isEmpty()) System.exit(1);
    }

    static void check(boolean ok, String msg) { if (!ok) throw new AssertionError(msg); }

    static void parsesJson() {
        var m = Json.parseObject("{\"type\":\"score\",\"team\":\"home\",\"points\":2,\"t\":12.5,\"ok\":true,\"x\":null}");
        check(m.get("type").equals("score"), "type");
        check(((Number) m.get("points")).intValue() == 2, "points");
        check(((Number) m.get("t")).doubleValue() == 12.5, "t");
        check(m.get("ok").equals(Boolean.TRUE) && m.get("x") == null, "literals");
    }

    static void rejectsNested() {
        try {
            Json.parseObject("{\"a\":{\"b\":1}}");
            throw new AssertionError("should have failed");
        } catch (IllegalArgumentException expected) { /* ok */ }
    }

    static MatchState play(String... lines) {
        MatchState s = new MatchState();
        for (String l : lines) s.apply(Json.parseObject(l));
        return s;
    }

    static void scoreAndRuns() {
        MatchState s = play(
            "{\"type\":\"score\",\"team\":\"home\",\"points\":2,\"home\":2,\"away\":0}",
            "{\"type\":\"shot\",\"team\":\"away\"}",
            "{\"type\":\"score\",\"team\":\"home\",\"points\":3,\"home\":5,\"away\":0}",
            "{\"type\":\"score\",\"team\":\"away\",\"points\":2,\"home\":5,\"away\":2}");
        check(s.home() == 5 && s.away() == 2, "score " + s.toJson());
        check(s.currentRun().equals("away 2-0"), "run " + s.currentRun());
        check(s.toJson().contains("\"best_run_home\":5"), "best run");
        check(s.shots()[1] == 1, "shots");
    }

    static void leadChanges() {
        MatchState s = play(
            "{\"type\":\"score\",\"team\":\"home\",\"points\":2}",
            "{\"type\":\"score\",\"team\":\"away\",\"points\":3}",
            "{\"type\":\"score\",\"team\":\"home\",\"points\":1}",
            "{\"type\":\"score\",\"team\":\"home\",\"points\":2}");
        // home leads, away leads (change), tie, home leads again (change)
        check(s.leadChanges() == 2, "lead changes " + s.leadChanges());
    }

    static void resync() {
        MatchState s = play(
            "{\"type\":\"score\",\"team\":\"home\",\"points\":2,\"home\":2,\"away\":0}",
            "{\"type\":\"score\",\"team\":\"away\",\"points\":2,\"home\":5,\"away\":2}");
        check(s.home() == 5 && s.resyncs() == 1, "resync " + s.toJson());
    }

    static void rejectsBad() {
        MatchState s = new MatchState();
        for (String bad : new String[] {
                "{\"type\":\"score\",\"team\":\"referee\",\"points\":2}",
                "{\"type\":\"score\",\"team\":\"home\",\"points\":9}",
                "{\"type\":\"teleport\",\"team\":\"home\"}"}) {
            try {
                s.apply(Json.parseObject(bad));
                throw new AssertionError("accepted " + bad);
            } catch (IllegalArgumentException expected) { /* ok */ }
        }
        check(s.rejected() == 3 && s.home() == 0, "rejected count");
    }

    static void server() throws Exception {
        LiveStatsServer srv = new LiveStatsServer(18091);
        srv.start();
        try {
            HttpClient c = HttpClient.newHttpClient();
            String body = "{\"type\":\"score\",\"team\":\"away\",\"points\":3,\"home\":0,\"away\":3}\n"
                        + "{\"type\":\"shot\",\"team\":\"home\"}\n";
            var post = c.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:18091/events"))
                    .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
            check(post.statusCode() == 202, "post status " + post.statusCode());
            var get = c.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:18091/state")).GET().build(),
                    HttpResponse.BodyHandlers.ofString());
            check(get.body().contains("\"away\":3") && get.body().contains("\"home_shots\":1"), get.body());
            var bad = c.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:18091/events"))
                    .POST(HttpRequest.BodyPublishers.ofString("{\"type\":\"score\"")).build(),
                    HttpResponse.BodyHandlers.ofString());
            check(bad.statusCode() == 400, "bad json status " + bad.statusCode());
        } finally {
            srv.stop();
        }
    }
}
