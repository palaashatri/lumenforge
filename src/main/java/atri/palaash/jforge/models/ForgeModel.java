package atri.palaash.jforge.models;

import java.nio.file.Path;
import java.util.Map;

public record ForgeModel(
        String id,
        Path path,
        ModelType type,
        Map<String, Object> metadata
) {
}
