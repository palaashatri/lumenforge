package atri.palaash.jforge.storage;

import atri.palaash.jforge.model.ModelDescriptor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ModelStorage {

    private final Path modelRoot;

    public ModelStorage() {
        this(Paths.get(System.getProperty("user.home"), ".jforge-models"));
    }

    public ModelStorage(Path modelRoot) {
        this.modelRoot = modelRoot;
    }

    public Path root() {
        return modelRoot;
    }

    public Path modelPath(ModelDescriptor descriptor) {
        return modelRoot.resolve(descriptor.relativePath());
    }

    public boolean isAvailable(ModelDescriptor descriptor) {
        Path file = modelPath(descriptor);
        return Files.exists(file) && Files.isRegularFile(file);
    }

    public void ensureParentDirectory(ModelDescriptor descriptor) throws IOException {
        Path target = modelPath(descriptor);
        Files.createDirectories(target.getParent());
    }
}
