package atri.palaash.jforge.runtime;

import atri.palaash.jforge.inference.InferenceRequest;

import java.util.Locale;

/**
 * Estimates per-request memory pressure and blocks requests that are likely to exceed
 * configured JVM heap safety limits.
 */
public class MemoryGuard {

    private static final long MB = 1024L * 1024L;
    private static final long GB = 1024L * MB;

    private final boolean enabled;
    private final double maxHeapFraction;
    private final long reserveBytes;

    public MemoryGuard() {
        this.enabled = !Boolean.getBoolean("jforge.guard.disable");
        this.maxHeapFraction = clampFraction(readDoubleProperty("jforge.guard.maxHeapFraction", 0.85d));
        this.reserveBytes = Math.max(128L * MB, readLongProperty("jforge.guard.reserveMb", 256L) * MB);
    }

    public Decision assess(InferenceRequest request) {
        Runtime runtime = Runtime.getRuntime();
        long maxHeapBytes = runtime.maxMemory();
        long usedHeapBytes = Math.max(0L, runtime.totalMemory() - runtime.freeMemory());
        long availableHeapBytes = Math.max(0L, maxHeapBytes - usedHeapBytes - reserveBytes);
        long heapBudgetBytes = Math.max(256L * MB, (long) (maxHeapBytes * maxHeapFraction));

        long estimatedBytes = estimateBytes(request);

        if (!enabled) {
            return Decision.allowed(
                    estimatedBytes,
                    availableHeapBytes,
                    maxHeapBytes,
                    "Memory guard disabled via -Djforge.guard.disable=true");
        }

        if (estimatedBytes > heapBudgetBytes || estimatedBytes > availableHeapBytes) {
            String message = "Memory guard blocked request: estimated " + toMb(estimatedBytes) + " MB, "
                    + "available " + toMb(availableHeapBytes) + " MB, heap max " + toMb(maxHeapBytes) + " MB. "
                    + "Lower resolution/steps or increase JVM heap (for example: -Xmx8g).";
            return Decision.rejected(estimatedBytes, availableHeapBytes, maxHeapBytes, message);
        }

        String summary = "Memory guard estimate " + toMb(estimatedBytes)
                + " MB (available " + toMb(availableHeapBytes) + " MB).";
        return Decision.allowed(estimatedBytes, availableHeapBytes, maxHeapBytes, summary);
    }

    private long estimateBytes(InferenceRequest request) {
        int width = Math.max(64, request.width());
        int height = Math.max(64, request.height());

        // The current engine uses request.batch as denoising-step workload in multiple pipelines.
        int workloadSteps = Math.max(1, request.batch());
        int boundedSteps = Math.min(workloadSteps, 120);

        double pixels = (double) width * (double) height;
        double latentPixels = Math.max(1.0d,
                (double) Math.max(1, width / 8) * (double) Math.max(1, height / 8));

        double pixelWorkspaceBytes = pixels * Float.BYTES * (4.0d + (boundedSteps / 5.0d));
        double latentWorkspaceBytes = latentPixels * Float.BYTES * (16.0d + boundedSteps);

        long modelOverheadBytes = modelOverheadBytes(request.model().id(), request.upscale());
        long ioOverheadBytes = 0L;

        if (request.inputImagePath() != null && !request.inputImagePath().isBlank()) {
            ioOverheadBytes += 128L * MB;
        }
        if (request.maskImagePath() != null && !request.maskImagePath().isBlank()) {
            ioOverheadBytes += 96L * MB;
        }

        long tensorBytes = safeDoubleToLong(pixelWorkspaceBytes + latentWorkspaceBytes);
        long baselineBytes = 160L * MB; // tokenizer buffers, request metadata, temporary ONNX tensors

        return saturatingAdd(modelOverheadBytes, saturatingAdd(ioOverheadBytes, saturatingAdd(tensorBytes, baselineBytes)));
    }

    private long modelOverheadBytes(String modelId, boolean upscale) {
        if (upscale || containsIgnoreCase(modelId, "esrgan") || containsIgnoreCase(modelId, "upscale")) {
            return 900L * MB;
        }
        if (containsIgnoreCase(modelId, "sd3")) {
            return 5L * GB;
        }
        if (containsIgnoreCase(modelId, "sdxl")) {
            return 3200L * MB;
        }
        if (containsIgnoreCase(modelId, "sd_turbo") || containsIgnoreCase(modelId, "sd_v15")
                || containsIgnoreCase(modelId, "stable-diffusion-v15")) {
            return 2200L * MB;
        }
        return 1800L * MB;
    }

    private boolean containsIgnoreCase(String source, String token) {
        if (source == null || token == null) {
            return false;
        }
        return source.toLowerCase(Locale.ROOT).contains(token.toLowerCase(Locale.ROOT));
    }

    private long saturatingAdd(long a, long b) {
        long result = a + b;
        if (((a ^ result) & (b ^ result)) < 0) {
            return Long.MAX_VALUE;
        }
        return result;
    }

    private long safeDoubleToLong(double value) {
        if (Double.isNaN(value) || value <= 0.0d) {
            return 0L;
        }
        if (value >= Long.MAX_VALUE) {
            return Long.MAX_VALUE;
        }
        return (long) value;
    }

    private double clampFraction(double value) {
        if (Double.isNaN(value)) {
            return 0.85d;
        }
        return Math.max(0.10d, Math.min(0.98d, value));
    }

    private double readDoubleProperty(String key, double fallback) {
        String raw = System.getProperty(key);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return Double.parseDouble(raw.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private long readLongProperty(String key, long fallback) {
        String raw = System.getProperty(key);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return Long.parseLong(raw.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private long toMb(long bytes) {
        return Math.max(0L, bytes / MB);
    }

    public record Decision(
            boolean allowed,
            long estimatedBytes,
            long availableHeapBytes,
            long maxHeapBytes,
            String details
    ) {
        public static Decision allowed(long estimatedBytes, long availableHeapBytes, long maxHeapBytes, String details) {
            return new Decision(true, estimatedBytes, availableHeapBytes, maxHeapBytes, details);
        }

        public static Decision rejected(long estimatedBytes, long availableHeapBytes, long maxHeapBytes, String details) {
            return new Decision(false, estimatedBytes, availableHeapBytes, maxHeapBytes, details);
        }
    }
}
