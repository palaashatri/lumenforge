package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;

public class EulerScheduler implements Scheduler {

    @Override
    public FloatTensor step(FloatTensor latents, int step, SchedulerState state) {
        float decay = 1f / Math.max(1, state.totalSteps());
        float multiplier = Math.max(0f, 1f - decay);
        return latents.multiply(multiplier);
    }
}
