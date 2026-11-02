package esv;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * A small HTTP service that receives live events from the Python vision
 * pipeline and serves the current match statistics, so Java and .NET systems
 * can use them without touching the computer vision code.
 *
 *   POST /events   one JSON event per line
 *   GET  /state    current statistics as JSON
 *   GET  /health   "ok"
 *
 * Uses only the JDK (no external libraries).
 */
public final class LiveStatsServer {
    private static final int MAX_BODY = 64 * 1024;
    private final MatchState state = new MatchState();
    private final HttpServer server;
    private final ExecutorService pool = Executors.newFixedThreadPool(4);

    public LiveStatsServer(int port) throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        server.setExecutor(pool);
        server.createContext("/health", ex -> send(ex, 200, "text/plain", "ok"));
        server.createContext("/state", ex -> {
            if (!ex.getRequestMethod().equals("GET")) { send(ex, 405, "text/plain", "GET only"); return; }
            send(ex, 200, "application/json", state.toJson());
        });
        server.createContext("/events", this::events);
    }

    private void events(HttpExchange ex) throws IOException {
        if (!ex.getRequestMethod().equals("POST")) { send(ex, 405, "text/plain", "POST only"); return; }
        String body;
        try (InputStream in = ex.getRequestBody()) {
            byte[] data = in.readNBytes(MAX_BODY + 1);
            if (data.length > MAX_BODY) { send(ex, 413, "text/plain", "body too large"); return; }
            body = new String(data, StandardCharsets.UTF_8);
        }
        int applied = 0;
        for (String line : body.split("\n")) {
            if (line.isBlank()) continue;
            try {
                state.apply(Json.parseObject(line));
                applied++;
            } catch (IllegalArgumentException e) {
                send(ex, 400, "application/json",
                     "{\"error\":" + Json.quote(e.getMessage()) + ",\"applied\":" + applied + "}");
                return;
            }
        }
        send(ex, 202, "application/json", state.toJson());
    }

    private static void send(HttpExchange ex, int status, String type, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", type);
        ex.getResponseHeaders().set("X-Content-Type-Options", "nosniff");
        ex.sendResponseHeaders(status, b.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(b); }
    }

    public void start() { server.start(); }
    public void stop() {
        server.stop(0);
        pool.shutdownNow();
    }
    public MatchState state() { return state; }

    public static void main(String[] args) throws IOException {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 8090;
        LiveStatsServer s = new LiveStatsServer(port);
        s.start();
        System.out.println("Live stats server on http://127.0.0.1:" + port);
    }
}
