package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;

public class FlowMatchingScheduler implements Scheduler {

    @Override
    public FloatTensor step(FloatTensor latents, int step, SchedulerState state) {
        FloatTensor velocity = (FloatTensor) state.attributesOrEmpty().get("noise_pred");
        if (velocity == null) {
            return latents;
        }

        float sigma = state.sigma();
        float sigmaNext = (float) state.attributesOrEmpty().getOrDefault("sigma_next", 0f);
        float dt = sigmaNext - sigma;

        // Flow matching: x_{t+1} = x_t + dt * velocity
        return latents.add(velocity.multiply(dt));
    }
}
