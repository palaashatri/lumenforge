package atri.palaash.jforge.inference;

public record InferenceResult(
        boolean success,
        String output,
        String details,
        String artifactPath,
        String artifactType
) {

    public static InferenceResult ok(String output, String details, String artifactPath, String artifactType) {
        return new InferenceResult(true, output, details, artifactPath, artifactType);
    }

    public static InferenceResult fail(String details) {
        return new InferenceResult(false, "", details, "", "none");
    }
}
