package atri.palaash.jforge.runtime;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

public class SessionManager {

    private final int maxCachedSessions;
    private final LinkedHashMap<String, OrtSession> sessionCache;

    public SessionManager(int maxCachedSessions) {
        this.maxCachedSessions = Math.max(1, maxCachedSessions);
        this.sessionCache = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, OrtSession> eldest) {
                if (size() <= SessionManager.this.maxCachedSessions) {
                    return false;
                }
                closeQuietly(eldest.getValue());
                return true;
            }
        };
    }

    public synchronized OrtSession getSession(OrtEnvironment environment,
                                              Path modelPath,
                                              OrtSession.SessionOptions options) throws OrtException {
        String key = modelPath.toAbsolutePath().toString();
        OrtSession existing = sessionCache.get(key);
        if (existing != null) {
            return existing;
        }
        OrtSession created = environment.createSession(modelPath.toString(), options);
        sessionCache.put(key, created);
        return created;
    }

    public synchronized void evictSession(Path modelPath) {
        String key = modelPath.toAbsolutePath().toString();
        OrtSession session = sessionCache.remove(key);
        if (session != null) {
            closeQuietly(session);
        }
    }

    public synchronized void clearCache() {
        sessionCache.values().forEach(this::closeQuietly);
        sessionCache.clear();
    }

    public DeviceInfo getActiveDevice(boolean preferGpu) {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (!preferGpu) {
            return new DeviceInfo("CPU", "Forced by user preference");
        }
        if (os.contains("mac")) {
            return new DeviceInfo("CoreML", "macOS GPU/ANE");
        }
        if (os.contains("win")) {
            return new DeviceInfo("DirectML/CUDA", "Windows GPU stack");
        }
        return new DeviceInfo("CUDA/ROCm", "Linux GPU stack");
    }

    private void closeQuietly(OrtSession session) {
        try {
            session.close();
        } catch (Exception ignored) {
        }
    }

    public record DeviceInfo(String provider, String notes) {}
}
