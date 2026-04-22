package atri.palaash.jforge.core.sampler;

import atri.palaash.jforge.core.StepCallback;
import atri.palaash.jforge.core.scheduler.SchedulerState;
import atri.palaash.jforge.core.tensor.FloatTensor;

import java.util.HashMap;
import java.util.Map;

public class SchedulerSampler implements Sampler {

    @Override
    public FloatTensor sample(SamplerContext ctx) {
        int total = Math.max(1, ctx.steps());
        FloatTensor latents = ctx.initialLatents();
        long started = System.currentTimeMillis();
        
        // Timesteps are usually handled by the scheduler in sophisticated implementations,
        // but here we follow the existing pattern in GenericOnnxService.
        // We'll pass attributes to the scheduler to help it.
        
        for (int step = 0; step < total; step++) {
            float progress = (step + 1f) / total;
            
            // 1. Predict noise
            // For CFG, we usually duplicate latents or run twice.
            // For simplicity in this unified sampler, we assume the UNet implementation handles CFG 
            // if guidanceScale > 1.0 or if it's passed in attributes.
            
            Map<String, Object> unetAttributes = new HashMap<>(ctx.attributes());
            unetAttributes.put("guidance_scale", ctx.guidanceScale());
            
            // Timestep calculation (placeholder, should be more robust)
            long timestep = (long) (1000.0 * (1.0 - (double) step / total));
            
            FloatTensor noisePred = ctx.unet().predict(latents, timestep, ctx.encoderHiddenStates(), unetAttributes);
            
            // 2. Scheduler step
            Map<String, Object> schedulerAttributes = new HashMap<>(ctx.attributes());
            schedulerAttributes.put("noise_pred", noisePred);
            
            // Provide sigma if needed (e.g. for Euler)
            float sigma = (float) Math.sqrt((1.0 - progress) / Math.max(0.0001f, progress)); // dummy sigma
            schedulerAttributes.put("sigma", sigma);
            
            SchedulerState state = new SchedulerState(
                    total,
                    sigma,
                    ctx.guidanceScale(),
                    schedulerAttributes);
            
            latents = ctx.scheduler().step(latents, step, state);
            
            // 3. Callback
            StepCallback callback = ctx.callback();
            if (callback != null) {
                callback.onStep(step + 1, progress, System.currentTimeMillis() - started);
            }
            
            // Check for cancellation via attributes or thread interruption
            if (Thread.currentThread().isInterrupted()) {
                break;
            }
        }
        return latents;
    }
}
