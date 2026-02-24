package atri.palaash.lumenforge.ui;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Central application logger that broadcasts structured log entries to
 * registered UI listeners (e.g. the Logs panel) instead of printing to
 * stdout/stderr.
 *
 * <p>Two independent channels:
 * <ul>
 *   <li>{@link Channel#APPLICATION} — startup, EP selection, configuration, errors</li>
 *   <li>{@link Channel#MODEL} — model download, import, conversion, inference</li>
 * </ul>
 */
public final class AppLogger {

    public enum Channel { APPLICATION, MODEL }
    public enum Level { INFO, WARN, ERROR }

    public record LogEntry(LocalDateTime timestamp, Channel channel, Level level, String message) { }

    private static final DateTimeFormatter TS_FORMAT = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");
    private static final List<Consumer<LogEntry>> listeners = new CopyOnWriteArrayList<>();

    private AppLogger() { }

    /* ── Listener management ─────────────────────────────────────── */

    public static void addListener(Consumer<LogEntry> listener) {
        listeners.add(listener);
    }

    public static void removeListener(Consumer<LogEntry> listener) {
        listeners.remove(listener);
    }

    /* ── Convenience logging ─────────────────────────────────────── */

    public static void app(String message) {
        log(Channel.APPLICATION, Level.INFO, message);
    }

    public static void appWarn(String message) {
        log(Channel.APPLICATION, Level.WARN, message);
    }

    public static void appError(String message) {
        log(Channel.APPLICATION, Level.ERROR, message);
    }

    public static void model(String message) {
        log(Channel.MODEL, Level.INFO, message);
    }

    public static void modelWarn(String message) {
        log(Channel.MODEL, Level.WARN, message);
    }

    public static void modelError(String message) {
        log(Channel.MODEL, Level.ERROR, message);
    }

    /* ── Core ────────────────────────────────────────────────────── */

    public static void log(Channel channel, Level level, String message) {
        LogEntry entry = new LogEntry(LocalDateTime.now(), channel, level, message);
        for (Consumer<LogEntry> listener : listeners) {
            try {
                listener.accept(entry);
            } catch (Exception ignored) { }
        }
    }

    /** Format a log entry for display in the Logs panel. */
    public static String format(LogEntry entry) {
        String prefix = switch (entry.level()) {
            case INFO  -> "";
            case WARN  -> "\u26a0 ";
            case ERROR -> "\u274c ";
        };
        return TS_FORMAT.format(entry.timestamp()) + "  " + prefix + entry.message();
    }
}
