package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;

public class DpmppSdeScheduler implements Scheduler {

    @Override
    public FloatTensor step(FloatTensor latents, int step, SchedulerState state) {
        float sigma = Math.max(0f, state.sigma());
        float temperature = Math.max(0.02f, sigma * 0.05f);
        float[] values = latents.values();
        for (int i = 0; i < values.length; i++) {
            float jitter = (float) Math.cos((step + i) * 0.013f) * temperature;
            values[i] = values[i] * 0.985f + jitter;
        }
        return FloatTensor.of(latents.shape(), values);
    }
}
