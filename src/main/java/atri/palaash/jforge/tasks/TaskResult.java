package atri.palaash.jforge.tasks;

import atri.palaash.jforge.inference.InferenceResult;

public record TaskResult(
        boolean success,
        String output,
        String details,
        String artifactPath,
        String artifactType,
        long durationMs
) {
    public static TaskResult fromInference(InferenceResult result, long durationMs) {
        return new TaskResult(
                result.success(),
                result.output(),
                result.details(),
                result.artifactPath(),
                result.artifactType(),
                durationMs);
    }

    public static TaskResult fail(String details, long durationMs) {
        return new TaskResult(false, "", details, "", "none", durationMs);
    }
}
