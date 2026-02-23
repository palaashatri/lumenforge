package atri.palaash.lumenforge.inference;

import atri.palaash.lumenforge.model.ModelDescriptor;
import atri.palaash.lumenforge.model.TaskType;
import atri.palaash.lumenforge.storage.ModelStorage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GenericOnnxServiceTest {

    private Path tempRoot;
    private ExecutorService executor;
    private ModelStorage storage;

    @BeforeEach
    void setUp() throws IOException {
        tempRoot = Files.createTempDirectory("lumenforge-test-");
        executor = Executors.newVirtualThreadPerTaskExecutor();
        storage = new ModelStorage(tempRoot);
    }

    @AfterEach
    void tearDown() throws IOException {
        if (executor != null) {
            executor.close();
        }
        if (tempRoot != null && Files.exists(tempRoot)) {
            try (var walk = Files.walk(tempRoot)) {
                walk.sorted((a, b) -> b.compareTo(a)).forEach(path -> {
                    try {
                        Files.deleteIfExists(path);
                    } catch (IOException ignored) {
                    }
                });
            }
        }
    }

    @Test
    void failsForInvalidOnnxModelBytes() throws Exception {
        ModelDescriptor descriptor = createDummyModel(TaskType.TEXT_TO_IMAGE, "sd15", "models/sd15.onnx");
        GenericOnnxService service = new GenericOnnxService(TaskType.TEXT_TO_IMAGE, storage, executor);

        InferenceResult result = service.run(new InferenceRequest(
            descriptor,
            "Robot dog playing catch in a field",
            "",
            1.0,
            42,
            1,
            512,
            512,
            "None",
            false,
            "",
            false
        )).join();

        assertFalse(result.success());
        assertFalse(result.details().isBlank());
    }

    @Test
    void failsWhenModelIsMissing() {
        ModelDescriptor descriptor = new ModelDescriptor(
                "missing",
                "Missing",
            TaskType.TEXT_TO_IMAGE,
                "models/missing.onnx",
                "https://example.com/missing.onnx",
                "test"
        );
        GenericOnnxService service = new GenericOnnxService(TaskType.TEXT_TO_IMAGE, storage, executor);

        InferenceResult result = service.run(new InferenceRequest(
            descriptor,
            "Calm narration about a sunrise",
            "",
            1.0,
            42,
            1,
            512,
            512,
            "None",
            false,
            "",
            false
        )).join();

        assertFalse(result.success());
        assertTrue(result.details().contains("Model not found locally"));
    }



    private ModelDescriptor createDummyModel(TaskType taskType, String id, String relativePath) throws IOException {
        ModelDescriptor descriptor = new ModelDescriptor(
                id,
                id,
                taskType,
                relativePath,
                "https://example.com/model.onnx",
                "test"
        );
        storage.ensureParentDirectory(descriptor);
        Files.writeString(storage.modelPath(descriptor), "dummy-onnx-bytes");
        return descriptor;
    }
}
