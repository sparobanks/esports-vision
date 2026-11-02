package esv;

import java.util.Map;

/**
 * Live statistics for one match, built from the events the vision pipeline sends.
 *
 * Events look like:
 *   {"type":"score","team":"home","points":2,"home":34,"away":30,"t":512.4}
 *   {"type":"shot","team":"away","t":515.0}
 *
 * The state checks every score event against its own running total. If they
 * disagree (for example an event was missed while the scoreboard was hidden),
 * it trusts the scoreboard totals and counts a "resync", which is a useful
 * data-quality signal for the trading team.
 */
public final class MatchState {
    private int home, away;
    private int homeShots, awayShots;
    private int homeScoringEvents, awayScoringEvents;
    private String runTeam = null;
    private int runPoints = 0;
    private int bestRunHome = 0, bestRunAway = 0;
    private int leadChanges = 0;
    private int biggestHomeLead = 0, biggestAwayLead = 0;
    private int resyncs = 0;
    private int rejected = 0;
    private double lastT = 0;
    private int events = 0;

    public synchronized void apply(Map<String, Object> e) {
        String type = str(e, "type");
        String team = str(e, "team");
        if (!"home".equals(team) && !"away".equals(team)) {
            rejected++;
            throw new IllegalArgumentException("team must be home or away");
        }
        double t = num(e, "t", lastT);
        switch (type) {
            case "shot" -> {
                if (team.equals("home")) homeShots++; else awayShots++;
            }
            case "score" -> applyScore(team, e);
            default -> {
                rejected++;
                throw new IllegalArgumentException("unknown event type: " + type);
            }
        }
        lastT = Math.max(lastT, t);
        events++;
    }

    private void applyScore(String team, Map<String, Object> e) {
        int pts = (int) num(e, "points", 0);
        if (pts < 1 || pts > 6) {
            rejected++;
            throw new IllegalArgumentException("points must be between 1 and 6");
        }
        int oldDiff = home - away;
        if (team.equals("home")) { home += pts; homeScoringEvents++; }
        else { away += pts; awayScoringEvents++; }
        if (e.containsKey("home") && e.containsKey("away")) {
            int rh = (int) num(e, "home", home), ra = (int) num(e, "away", away);
            if (rh < 0 || ra < 0) {
                rejected++;
                throw new IllegalArgumentException("scores cannot be negative");
            }
            if (rh != home || ra != away) {
                resyncs++;
                home = rh;
                away = ra;
            }
        }
        if (team.equals(runTeam)) runPoints += pts;
        else { runTeam = team; runPoints = pts; }
        if (team.equals("home")) bestRunHome = Math.max(bestRunHome, runPoints);
        else bestRunAway = Math.max(bestRunAway, runPoints);
        int diff = home - away;
        if (oldDiff != 0 && diff != 0 && Integer.signum(oldDiff) != Integer.signum(diff)) leadChanges++;
        else if (oldDiff == 0 && diff != 0 && events > 0 && lastLeaderWasOther(diff)) leadChanges++;
        biggestHomeLead = Math.max(biggestHomeLead, diff);
        biggestAwayLead = Math.max(biggestAwayLead, -diff);
        if (diff != 0) lastLeader = Integer.signum(diff);
    }

    private int lastLeader = 0;

    private boolean lastLeaderWasOther(int diff) {
        return lastLeader != 0 && lastLeader != Integer.signum(diff);
    }

    public synchronized int home() { return home; }
    public synchronized int away() { return away; }
    public synchronized int leadChanges() { return leadChanges; }
    public synchronized int resyncs() { return resyncs; }
    public synchronized int rejected() { return rejected; }
    public synchronized int[] shots() { return new int[] {homeShots, awayShots}; }
    public synchronized String currentRun() { return runTeam == null ? "none" : runTeam + " " + runPoints + "-0"; }

    public synchronized String toJson() {
        return "{"
            + "\"home\":" + home + ",\"away\":" + away
            + ",\"home_shots\":" + homeShots + ",\"away_shots\":" + awayShots
            + ",\"home_scoring_plays\":" + homeScoringEvents + ",\"away_scoring_plays\":" + awayScoringEvents
            + ",\"current_run\":" + Json.quote(currentRun())
            + ",\"best_run_home\":" + bestRunHome + ",\"best_run_away\":" + bestRunAway
            + ",\"lead_changes\":" + leadChanges
            + ",\"biggest_home_lead\":" + biggestHomeLead + ",\"biggest_away_lead\":" + biggestAwayLead
            + ",\"resyncs\":" + resyncs + ",\"rejected\":" + rejected
            + ",\"events\":" + events + ",\"last_t\":" + lastT
            + "}";
    }

    private static String str(Map<String, Object> e, String k) {
        Object v = e.get(k);
        if (!(v instanceof String s)) throw new IllegalArgumentException(k + " is required");
        return s;
    }

    private static double num(Map<String, Object> e, String k, double dflt) {
        Object v = e.get(k);
        if (v == null) return dflt;
        if (v instanceof Number n) return n.doubleValue();
        throw new IllegalArgumentException(k + " must be a number");
    }
}
