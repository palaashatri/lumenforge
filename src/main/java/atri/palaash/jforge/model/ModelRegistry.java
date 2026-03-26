package atri.palaash.jforge.model;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ModelRegistry {

    private final Map<TaskType, List<ModelDescriptor>> byTask;
    private final List<ModelDescriptor> downloadableAssets;

    public ModelRegistry() {
        this.byTask = new EnumMap<>(TaskType.class);
        this.downloadableAssets = new ArrayList<>();
        registerDefaults();
    }

    public List<ModelDescriptor> byTask(TaskType taskType) {
        return new ArrayList<>(byTask.getOrDefault(taskType, List.of()));
    }

    public List<ModelDescriptor> allModels() {
        List<ModelDescriptor> output = new ArrayList<>();
        byTask.values().forEach(output::addAll);
        output.addAll(downloadableAssets);
        return output;
    }

    public synchronized int mergeDownloadableAssets(List<ModelDescriptor> discovered) {
        Set<String> knownIds = new HashSet<>();
        for (ModelDescriptor descriptor : allModels()) {
            knownIds.add(descriptor.id());
        }
        int added = 0;
        for (ModelDescriptor descriptor : discovered) {
            if (knownIds.add(descriptor.id())) {
                downloadableAssets.add(descriptor);
                // Also add to the appropriate task list so it shows in workflow combos
                if (descriptor.taskType() == TaskType.TEXT_TO_IMAGE
                        && (descriptor.relativePath().contains("unet/model.onnx")
                            || descriptor.relativePath().contains("transformer/model.onnx")
                            || descriptor.sourceUrl().startsWith("hf-pytorch://"))) {
                    byTask.computeIfAbsent(TaskType.TEXT_TO_IMAGE, ignored -> new ArrayList<>())
                          .add(descriptor);
                } else if (descriptor.taskType() == TaskType.IMAGE_UPSCALE) {
                    byTask.computeIfAbsent(TaskType.IMAGE_UPSCALE, ignored -> new ArrayList<>())
                          .add(descriptor);
                } else if (descriptor.taskType() == TaskType.TEXT_TO_VIDEO) {
                    byTask.computeIfAbsent(TaskType.TEXT_TO_VIDEO, ignored -> new ArrayList<>())
                          .add(descriptor);
                }
                added++;
            }
        }
        return added;
    }

    private void registerDefaults() {
        List<ModelDescriptor> textToImage = new ArrayList<>();
        textToImage.add(new ModelDescriptor("sd_v15_onnx", "Stable Diffusion v1.5 ONNX", TaskType.TEXT_TO_IMAGE,
            "text-image/stable-diffusion-v15/unet/model.onnx",
            "https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX/resolve/main/unet/model.onnx",
            "Auto-downloads UNet ONNX from onnx-community/stable-diffusion-v1-5-ONNX."));
        textToImage.add(new ModelDescriptor("sdxl_base_onnx", "Stable Diffusion XL Base 1.0 ONNX", TaskType.TEXT_TO_IMAGE,
            "text-image/sdxl-base/unet/model.onnx",
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/model.onnx",
            "Auto-downloads SDXL Base 1.0 ONNX from stabilityai. Dual text encoders, 1024×1024 native."));

        List<ModelDescriptor> imageUpscale = new ArrayList<>();
        imageUpscale.add(new ModelDescriptor("realesrgan", "Real-ESRGAN ONNX", TaskType.IMAGE_UPSCALE,
            "text-image/realesrgan/model.onnx",
            "https://huggingface.co/imgdesignart/realesrgan-x4-onnx/resolve/main/onnx/model.onnx",
            "Auto-downloads Real-ESRGAN ONNX model."));

        List<ModelDescriptor> textToVideo = new ArrayList<>();
        // ali-vilab is the canonical/active org for this model; damo-vilab is deprecated.
        // The converter uses a custom per-component export (text_encoder, unet3d, vae_decoder)
        // since the UNet3D architecture is not in optimum's task registry.
        textToVideo.add(new ModelDescriptor(
            "ali-vilab/text-to-video-ms-1.7b",
            "ModelScope Text-to-Video 1.7B",
            TaskType.TEXT_TO_VIDEO,
            "text-video/ali-vilab/text-to-video-ms-1.7b/unet/model.onnx",
            "hf-pytorch://ali-vilab/text-to-video-ms-1.7b",
            "Downloads PyTorch weights and exports each component (text encoder, UNet3D, VAE) to ONNX."));

        byTask.put(TaskType.TEXT_TO_IMAGE, textToImage);
        byTask.put(TaskType.IMAGE_UPSCALE, imageUpscale);
        byTask.put(TaskType.TEXT_TO_VIDEO, textToVideo);

        /* Pipeline component assets for SD v1.5 (shown in Model Manager, not in workflow combos) */
        downloadableAssets.add(new ModelDescriptor("sd_v15_text_encoder", "SD v1.5 Text Encoder ONNX", TaskType.TEXT_TO_IMAGE,
            "text-image/stable-diffusion-v15/text_encoder/model.onnx",
            "https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX/resolve/main/text_encoder/model.onnx",
            "Pipeline component asset for SD v1.5."));
        downloadableAssets.add(new ModelDescriptor("sd_v15_vae_decoder", "SD v1.5 VAE Decoder ONNX", TaskType.TEXT_TO_IMAGE,
            "text-image/stable-diffusion-v15/vae_decoder/model.onnx",
            "https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX/resolve/main/vae_decoder/model.onnx",
            "Pipeline component asset for SD v1.5."));
        downloadableAssets.add(new ModelDescriptor("sd_v15_safety_checker", "SD v1.5 Safety Checker ONNX", TaskType.TEXT_TO_IMAGE,
            "text-image/stable-diffusion-v15/safety_checker/model.onnx",
            "https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX/resolve/main/safety_checker/model.onnx",
            "Optional safety checker component."));
    }
}
