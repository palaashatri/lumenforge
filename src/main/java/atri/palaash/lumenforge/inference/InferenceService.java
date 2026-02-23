package atri.palaash.lumenforge.inference;

import java.util.concurrent.CompletableFuture;

public interface InferenceService {
    CompletableFuture<InferenceResult> run(InferenceRequest request);
}
