package atri.palaash.jforge.core.sampler;

import atri.palaash.jforge.core.StepCallback;
import atri.palaash.jforge.core.scheduler.SchedulerState;
import atri.palaash.jforge.core.tensor.FloatTensor;

public class SchedulerSampler implements Sampler {

    @Override
    public FloatTensor sample(SamplerContext ctx) {
        int total = Math.max(1, ctx.steps());
        FloatTensor latents = ctx.initialLatents();
        long started = System.currentTimeMillis();
        for (int step = 0; step < total; step++) {
            float progress = (step + 1f) / total;
            SchedulerState state = new SchedulerState(
                    total,
                    Math.max(0f, 1f - progress),
                    ctx.guidanceScale(),
                    ctx.attributes());
            latents = ctx.scheduler().step(latents, step, state);
            StepCallback callback = ctx.callback();
            if (callback != null) {
                callback.onStep(step + 1, progress, System.currentTimeMillis() - started);
            }
        }
        return latents;
    }
}
