package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;

public class EulerAncestralScheduler implements Scheduler {

    @Override
    public FloatTensor step(FloatTensor latents, int step, SchedulerState state) {
        float sigma = Math.max(0f, state.sigma());
        float perturbation = (float) Math.sin(step * 0.173f) * 0.01f * sigma;
        float[] values = latents.values();
        for (int i = 0; i < values.length; i++) {
            values[i] = values[i] * (1f - 0.015f) + perturbation;
        }
        return FloatTensor.of(latents.shape(), values);
    }
}
