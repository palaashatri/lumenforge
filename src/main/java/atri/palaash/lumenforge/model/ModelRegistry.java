package atri.palaash.lumenforge.model;

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
        List<ModelDescriptor> taskModels = byTask.computeIfAbsent(TaskType.TEXT_TO_IMAGE, ignored -> new ArrayList<>());
        int added = 0;
        for (ModelDescriptor descriptor : discovered) {
            if (knownIds.add(descriptor.id())) {
                downloadableAssets.add(descriptor);
                if (descriptor.taskType() == TaskType.TEXT_TO_IMAGE
                        && descriptor.relativePath().endsWith("unet/model.onnx")) {
                    taskModels.add(descriptor);
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
        textToImage.add(new ModelDescriptor("sd_turbo_onnx", "SD Turbo ONNX (1\u20134 steps)", TaskType.TEXT_TO_IMAGE,
            "text-image/sd-turbo/unet/model.onnx",
            "https://huggingface.co/onnxruntime/sd-turbo/resolve/main/unet/model.onnx",
            "Distilled SD Turbo model. Generates images in 1\u20134 steps. Auto-downloads from onnxruntime/sd-turbo."));
        textToImage.add(new ModelDescriptor("sdxl_turbo_onnx", "SDXL Turbo ONNX (1\u20134 steps, 512\u00d7512)", TaskType.TEXT_TO_IMAGE,
            "text-image/sdxl-turbo/unet/model.onnx",
            "https://huggingface.co/onnxruntime/sdxl-turbo/resolve/main/unet/model.onnx",
            "SDXL Turbo with dual text encoders. 1\u20134 steps, 512\u00d7512. Warning: ~10 GB download."));
        textToImage.add(new ModelDescriptor("sd_pytorch", "SD v1.5 PyTorch (DJL)", TaskType.TEXT_TO_IMAGE,
            "text-image/sd-pytorch/unet_model.pt",
            "",
            "Stable Diffusion v1.5 via DJL/PyTorch. Requires TorchScript export (see scripts/export_torchscript.py). Build with -Ddjl=true."));

        List<ModelDescriptor> imageUpscale = new ArrayList<>();
        imageUpscale.add(new ModelDescriptor("realesrgan", "Real-ESRGAN ONNX", TaskType.IMAGE_UPSCALE,
            "text-image/realesrgan/model.onnx",
            "https://huggingface.co/imgdesignart/realesrgan-x4-onnx/resolve/main/onnx/model.onnx",
            "Auto-downloads Real-ESRGAN ONNX model."));

        byTask.put(TaskType.TEXT_TO_IMAGE, textToImage);
        byTask.put(TaskType.IMAGE_UPSCALE, imageUpscale);

        List<ModelDescriptor> img2img = new ArrayList<>();
        img2img.add(new ModelDescriptor("sd_v15_img2img", "SD v1.5 Img2Img ONNX", TaskType.IMAGE_TO_IMAGE,
            "text-image/stable-diffusion-v15/unet/model.onnx",
            "https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX/resolve/main/unet/model.onnx",
            "Image-to-Image with SD v1.5. Reuses the same ONNX bundle (requires VAE encoder). Supports inpainting masks."));
        img2img.add(new ModelDescriptor("sd_turbo_img2img", "SD Turbo Img2Img ONNX", TaskType.IMAGE_TO_IMAGE,
            "text-image/sd-turbo/unet/model.onnx",
            "https://huggingface.co/onnxruntime/sd-turbo/resolve/main/unet/model.onnx",
            "Image-to-Image with SD Turbo. 1\u20134 steps, fast iteration."));
        byTask.put(TaskType.IMAGE_TO_IMAGE, img2img);

        downloadableAssets.add(new ModelDescriptor("sd_turbo_unet", "SD Turbo UNet ONNX (Experimental)", TaskType.TEXT_TO_IMAGE,
            "text-image/sd-turbo/unet/model.onnx",
            "https://huggingface.co/onnxruntime/sd-turbo/resolve/main/unet/model.onnx",
            "Optional: may require GPU EP (CoreML/DirectML/CUDA) on some systems."));
        downloadableAssets.add(new ModelDescriptor("sd_turbo_text_encoder", "SD Turbo Text Encoder ONNX", TaskType.TEXT_TO_IMAGE,
            "text-image/sd-turbo/text_encoder/model.onnx",
            "https://huggingface.co/onnxruntime/sd-turbo/resolve/main/text_encoder/model.onnx",
            "Pipeline component asset for SD Turbo."));
        downloadableAssets.add(new ModelDescriptor("sd_turbo_vae_decoder", "SD Turbo VAE Decoder ONNX", TaskType.TEXT_TO_IMAGE,
            "text-image/sd-turbo/vae_decoder/model.onnx",
            "https://huggingface.co/onnxruntime/sd-turbo/resolve/main/vae_decoder/model.onnx",
            "Pipeline component asset for SD Turbo."));
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
