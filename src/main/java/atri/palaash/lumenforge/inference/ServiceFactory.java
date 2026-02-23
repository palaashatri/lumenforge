package atri.palaash.lumenforge.inference;

import atri.palaash.lumenforge.model.ModelRegistry;
import atri.palaash.lumenforge.model.TaskType;
import atri.palaash.lumenforge.storage.ModelStorage;

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

    /**
     * Creates an inference service for the given task using the DJL PyTorch backend.
     * Falls back to the ONNX backend if DJL is not available.
     */
    public InferenceService createDjl(TaskType taskType) {
        if (DjlPyTorchService.isAvailable()) {
            return new DjlPyTorchService(storage, executor);
        }
        return create(taskType);
    }

    /**
     * Creates an inference service for the given model, automatically selecting
     * the DJL backend for PyTorch models and ONNX backend for everything else.
     */
    public InferenceService createForModel(String modelId, TaskType taskType) {
        if (modelId != null && modelId.contains("pytorch")) {
            return createDjl(taskType);
        }
        return create(taskType);
    }

    /** Returns {@code true} if the DJL PyTorch runtime is available. */
    public static boolean isDjlAvailable() {
        return DjlPyTorchService.isAvailable();
    }
}
