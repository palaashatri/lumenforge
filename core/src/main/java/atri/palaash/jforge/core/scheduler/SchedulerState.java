package atri.palaash.jforge.core.scheduler;

import java.util.Collections;
import java.util.Map;

public record SchedulerState(
        int totalSteps,
        float sigma,
        float guidanceScale,
        Map<String, Object> attributes
) {
    public SchedulerState {
        attributes = attributes == null ? Map.of() : Map.copyOf(attributes);
    }

    public Map<String, Object> attributesOrEmpty() {
        return attributes == null ? Collections.emptyMap() : attributes;
    }
}
