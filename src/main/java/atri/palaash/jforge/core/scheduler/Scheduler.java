package atri.palaash.jforge.core.scheduler;

import atri.palaash.jforge.core.tensor.FloatTensor;

public interface Scheduler {
    FloatTensor step(FloatTensor latents, int step, SchedulerState state);
}
