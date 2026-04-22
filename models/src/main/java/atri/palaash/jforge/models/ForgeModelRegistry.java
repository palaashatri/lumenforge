package atri.palaash.jforge.models;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import atri.palaash.jforge.model.ModelDescriptor;
import atri.palaash.jforge.model.ModelRegistry;
import atri.palaash.jforge.storage.ModelStorage;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class ForgeModelRegistry {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final ModelRegistry legacyRegistry;
    private final ModelStorage modelStorage;

    public ForgeModelRegistry(ModelRegistry legacyRegistry, ModelStorage modelStorage) {
        this.legacyRegistry = legacyRegistry;
        this.modelStorage = modelStorage;
    }

    public List<ForgeModel> allModels() {
        List<ForgeModel> output = new ArrayList<>();
        for (ModelDescriptor descriptor : legacyRegistry.allModels()) {
            Path modelPath = modelStorage.modelPath(descriptor);
            output.add(new ForgeModel(
                    descriptor.id(),
                    modelPath,
                    inferType(descriptor),
                    buildMetadata(descriptor, modelPath)));
        }
        return output;
    }

    public Optional<ForgeModel> byId(String id) {
        return allModels().stream().filter(model -> model.id().equals(id)).findFirst();
    }

    public ModelCompatibility checkCompatibility(ForgeModel model,
                                                 String scheduler,
                                                 String sampler) {
        List<String> issues = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        if (!Files.exists(model.path())) {
            issues.add("Model file is not available locally: " + model.path());
        }

        if (model.type() == ModelType.SD3_5 && !"flow matching".equalsIgnoreCase(scheduler)) {
            issues.add("SD3.5 models require Flow Matching scheduler.");
        }

        List<String> supportedSchedulers = readStringList(model.metadata(), "supportedSchedulers");
        if (!supportedSchedulers.isEmpty() && !containsIgnoreCase(supportedSchedulers, scheduler)) {
            issues.add("Scheduler '" + scheduler + "' is not supported by this model. Supported: "
                    + String.join(", ", supportedSchedulers));
        }

        List<String> supportedSamplers = readStringList(model.metadata(), "supportedSamplers");
        if (!supportedSamplers.isEmpty() && !containsIgnoreCase(supportedSamplers, sampler)) {
            issues.add("Sampler '" + sampler + "' is not supported by this model. Supported: "
                    + String.join(", ", supportedSamplers));
        }

        if (model.type() == ModelType.ESRGAN && !sampler.equalsIgnoreCase("euler")) {
            warnings.add("Upscale models typically ignore samplers; selected sampler will be ignored.");
        }

        Object minVram = getNested(model.metadata(), "requirements", "minVramGB");
        if (minVram instanceof Number minVramGb) {
            long heapGb = Math.max(1L, Runtime.getRuntime().maxMemory() / (1024L * 1024L * 1024L));
            if (heapGb < minVramGb.longValue()) {
                warnings.add("Configured model recommends at least " + minVramGb
                        + " GB memory; current JVM heap is " + heapGb + " GB.");
            }
        }

        return new ModelCompatibility(issues.isEmpty(), issues, warnings);
    }

    private ModelType inferType(ModelDescriptor descriptor) {
        String id = descriptor.id().toLowerCase();
        if (id.contains("sd_v15") || id.contains("stable-diffusion-v15")) {
            return ModelType.SD15;
        }
        if (id.contains("sdxl")) {
            return ModelType.SDXL;
        }
        if (id.contains("sd3") || descriptor.relativePath().contains("transformer/")) {
            return ModelType.SD3_5;
        }
        if (id.contains("esrgan") || id.contains("upscale")) {
            return ModelType.ESRGAN;
        }
        return ModelType.CUSTOM;
    }

    private Map<String, Object> buildMetadata(ModelDescriptor descriptor, Path modelPath) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("modelId", descriptor.id());
        metadata.put("displayName", descriptor.displayName());
        metadata.put("taskType", descriptor.taskType().name());
        metadata.put("sourceUrl", descriptor.sourceUrl());
        metadata.put("notes", descriptor.notes());
        metadata.put("available", Files.exists(modelPath));

        Path modelJson = modelPath.getParent() == null
                ? modelPath.resolve("model.json")
                : modelPath.getParent().resolve("model.json");
        if (Files.exists(modelJson)) {
            try {
                Map<String, Object> parsed = OBJECT_MAPPER.readValue(
                        modelJson.toFile(),
                        new TypeReference<>() {
                        });
                metadata.putAll(parsed);
            } catch (Exception ignored) {
                metadata.put("metadataLoadError", "Unable to parse model.json");
            }
        }

        metadata.putIfAbsent("version", "1.0.0");
        metadata.putIfAbsent("tags", inferTags(descriptor));
        return metadata;
    }

    private List<String> inferTags(ModelDescriptor descriptor) {
        String source = (descriptor.id() + " " + descriptor.displayName() + " " + descriptor.notes()).toLowerCase();
        List<String> tags = new ArrayList<>();

        if (source.contains("anime")) {
            tags.add("anime");
        }
        if (source.contains("real") || source.contains("photographic") || source.contains("photo")) {
            tags.add("realistic");
        }
        if (source.contains("turbo")) {
            tags.add("turbo");
        }
        if (source.contains("upscale") || source.contains("esrgan")) {
            tags.add("upscale");
        }
        if (source.contains("sdxl")) {
            tags.add("sdxl");
        }
        if (source.contains("sd3")) {
            tags.add("sd3.5");
        }

        if (tags.isEmpty()) {
            tags.add("general");
        }
        return tags;
    }

    private List<String> readStringList(Map<String, Object> metadata, String key) {
        Object raw = metadata.get(key);
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> values = new ArrayList<>();
        for (Object item : list) {
            if (item != null) {
                String text = item.toString().trim();
                if (!text.isEmpty()) {
                    values.add(text);
                }
            }
        }
        return values;
    }

    private boolean containsIgnoreCase(List<String> values, String selected) {
        return values.stream().anyMatch(value -> value.equalsIgnoreCase(selected));
    }

    private Object getNested(Map<String, Object> map, String key, String nestedKey) {
        Object nested = map.get(key);
        if (nested instanceof Map<?, ?> nestedMap) {
            return nestedMap.get(nestedKey);
        }
        return null;
    }
}
