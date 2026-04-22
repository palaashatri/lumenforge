package atri.palaash.jforge.tasks;

import atri.palaash.jforge.inference.InferenceService;
import atri.palaash.jforge.model.ModelDescriptor;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

public record TaskContext(
        ModelDescriptor model,
        InferenceService inferenceService,
        Consumer<String> progressCallback,
        AtomicBoolean cancellationFlag
) {
}
