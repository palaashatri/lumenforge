package atri.palaash.jforge.core.sampler;

import atri.palaash.jforge.core.StepCallback;
import atri.palaash.jforge.core.scheduler.Scheduler;
import atri.palaash.jforge.core.tensor.FloatTensor;

import java.util.Map;

public record SamplerContext(
        FloatTensor initialLatents,
        Scheduler scheduler,
        int steps,
        float guidanceScale,
        StepCallback callback,
        Map<String, Object> attributes
) {
    public SamplerContext {
        attributes = attributes == null ? Map.of() : Map.copyOf(attributes);
    }
}
