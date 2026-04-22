package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;

public class Dpmpp2mScheduler implements Scheduler {

    @Override
    public FloatTensor step(FloatTensor latents, int step, SchedulerState state) {
        float progress = (step + 1f) / Math.max(1f, state.totalSteps());
        float blend = Math.min(0.35f, 0.1f + progress * 0.2f);
        FloatTensor damped = latents.multiply(1f - blend);
        FloatTensor guided = latents.multiply(1f + (state.guidanceScale() * 0.01f));
        return damped.blend(guided, blend);
    }
}
