package atri.palaash.jforge.inference;

import atri.palaash.jforge.model.ModelRegistry;
import atri.palaash.jforge.model.TaskType;
import atri.palaash.jforge.storage.ModelStorage;

import java.util.concurrent.Executor;

public class ServiceFactory {

    private final ModelRegistry registry;
    private final ModelStorage storage;
    private final Executor executor;

    public ServiceFactory(ModelRegistry registry, ModelStorage storage, Executor executor) {
        this.registry = registry;
        this.storage = storage;
        this.executor = executor;
    }

    public InferenceService create(TaskType taskType) {
        if (registry.byTask(taskType).isEmpty()) {
            return request -> java.util.concurrent.CompletableFuture.completedFuture(
                    InferenceResult.fail("No models registered for this task."));
        }
        return new GenericOnnxService(taskType, storage, executor);
    }
}
