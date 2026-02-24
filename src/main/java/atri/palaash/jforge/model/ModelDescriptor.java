package atri.palaash.jforge.model;

public record ModelDescriptor(
        String id,
        String displayName,
        TaskType taskType,
        String relativePath,
        String sourceUrl,
        String notes
) {
}
