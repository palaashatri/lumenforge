package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;

public class EulerScheduler implements Scheduler {

    @Override
    public FloatTensor step(FloatTensor latents, int step, SchedulerState state) {
        FloatTensor noisePred = (FloatTensor) state.attributesOrEmpty().get("noise_pred");
        if (noisePred == null) {
            return latents;
        }

        float sigma = state.sigma();
        float sigmaNext = (float) state.attributesOrEmpty().getOrDefault("sigma_next", 0f);
        float dt = sigmaNext - sigma;

        return latents.add(noisePred.multiply(dt));
    }
}
