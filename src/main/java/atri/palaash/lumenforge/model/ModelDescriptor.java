package atri.palaash.lumenforge.model;

public record ModelDescriptor(
        String id,
        String displayName,
        TaskType taskType,
        String relativePath,
        String sourceUrl,
        String notes
) {
}
