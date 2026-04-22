package atri.palaash.jforge.tasks;

import atri.palaash.jforge.inference.InferenceRequest;
import atri.palaash.jforge.inference.InferenceResult;

public abstract class AbstractInferenceTask implements ForgeTask {

    private final TaskConfig config;

    protected AbstractInferenceTask(TaskConfig config) {
        this.config = config;
    }

    @Override
    public TaskConfig config() {
        return config;
    }

    protected abstract boolean upscale();

    @Override
    public TaskResult run(TaskContext ctx) {
        long started = System.currentTimeMillis();
        try {
            int effectiveSteps = config.steps() > 0 ? config.steps() : Math.max(1, config.batch());
            InferenceRequest request = new InferenceRequest(
                    ctx.model(),
                    config.prompt(),
                    config.negativePrompt(),
                    config.cfg(),
                    config.seed(),
                    effectiveSteps,
                    config.width(),
                    config.height(),
                    config.style(),
                    upscale(),
                    config.inputImagePath(),
                    config.preferGpu(),
                    ctx.progressCallback(),
                    ctx.cancellationFlag(),
                    config.strength(),
                    config.maskImagePath());

            InferenceResult result = ctx.inferenceService().run(request).join();
            return TaskResult.fromInference(result, System.currentTimeMillis() - started);
        } catch (Exception ex) {
            return TaskResult.fail(ex.getMessage(), System.currentTimeMillis() - started);
        }
    }
}
