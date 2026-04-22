package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;
import java.util.Random;

public class EulerAncestralScheduler implements Scheduler {

    @Override
    public FloatTensor step(FloatTensor latents, int step, SchedulerState state) {
        FloatTensor noisePred = (FloatTensor) state.attributesOrEmpty().get("noise_pred");
        if (noisePred == null) {
            return latents;
        }

        float sigma = state.sigma();
        float sigmaNext = (float) state.attributesOrEmpty().getOrDefault("sigma_next", 0f);
        
        // sigma_up = sqrt(sigma_next^2 * (sigma^2 - sigma_next^2) / sigma^2)
        float sigmaUp = (float) Math.sqrt(Math.pow(sigmaNext, 2) * (Math.pow(sigma, 2) - Math.pow(sigmaNext, 2)) / Math.pow(sigma, 2));
        // sigma_down = sqrt(sigma_next^2 - sigma_up^2)
        float sigmaDown = (float) Math.sqrt(Math.pow(sigmaNext, 2) - Math.pow(sigmaUp, 2));
        
        float dt = sigmaDown - sigma;
        
        // Euler: next = latents + dt * noisePred
        FloatTensor next = latents.add(noisePred.multiply(dt));
        
        if (sigmaUp > 0) {
            FloatTensor noise = FloatTensor.random(42 + step, latents.shape());
            next = next.add(noise.multiply(sigmaUp));
        }
        
        return next;
    }
}
