package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;

import java.util.ArrayList;
import java.util.List;

/**
 * DPM-Solver++ (2M) scheduler.
 * Second-order multistep solver.
 */
public class Dpmpp2mScheduler implements Scheduler {

    private final List<FloatTensor> modelOutputs = new ArrayList<>();

    @Override
    public FloatTensor step(FloatTensor latents, int step, SchedulerState state) {
        FloatTensor noisePred = (FloatTensor) state.attributesOrEmpty().get("noise_pred");
        if (noisePred == null) return latents;

        float sigma = state.sigma();
        float sigmaNext = (float) state.attributesOrEmpty().getOrDefault("sigma_next", 0f);

        // Multistep logic: requires previous noise predictions
        modelOutputs.add(noisePred);
        if (modelOutputs.size() > 2) {
            modelOutputs.remove(0);
        }

        if (step == 0 || sigmaNext == 0) {
            // First order step (Euler)
            float dt = sigmaNext - sigma;
            return latents.add(noisePred.multiply(dt));
        } else {
            // Second order step
            float sigmaPrev = (float) state.attributesOrEmpty().getOrDefault("sigma_prev", sigma);
            FloatTensor noisePrev = modelOutputs.get(modelOutputs.size() - 2);
            
            float lambda = (float) -Math.log(sigma);
            float lambdaNext = (float) -Math.log(sigmaNext);
            float lambdaPrev = (float) -Math.log(sigmaPrev);
            
            float h = lambdaNext - lambda;
            float hPrev = lambda - lambdaPrev;
            
            float r = hPrev / h;
            
            // DPM++ 2M update rule
            FloatTensor denoiser = noisePred.multiply(1f + 0.5f / r).subtract(noisePrev.multiply(0.5f / r));
            float dt = sigmaNext - sigma; // simplified for Euler-space sigmas
            
            return latents.add(denoiser.multiply(dt));
        }
    }
}
