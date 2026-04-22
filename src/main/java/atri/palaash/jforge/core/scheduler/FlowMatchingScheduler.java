package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;

public class FlowMatchingScheduler implements Scheduler {

    @Override
    public FloatTensor step(FloatTensor latents, int step, SchedulerState state) {
        float progress = (step + 1f) / Math.max(1f, state.totalSteps());
        float flowWeight = Math.min(0.6f, 0.2f + progress * 0.4f);
        float guidanceScale = Math.max(1f, state.guidanceScale());
        float guidedWeight = Math.min(0.9f, guidanceScale / 20f);
        FloatTensor flowAdjusted = latents.multiply(1f - (flowWeight * 0.25f));
        FloatTensor guided = latents.multiply(1f + (guidedWeight * 0.15f));
        return flowAdjusted.blend(guided, flowWeight);
    }
}
