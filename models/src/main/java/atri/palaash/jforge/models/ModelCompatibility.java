package atri.palaash.jforge.models;

import java.util.List;

public record ModelCompatibility(
        boolean compatible,
        List<String> issues,
        List<String> warnings
) {
}
