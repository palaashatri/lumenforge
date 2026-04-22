package atri.palaash.jforge.tasks;

public record TaskConfig(
        String modelId,
        String prompt,
        String negativePrompt,
        long seed,
        int batch,
        int width,
        int height,
        int steps,
        float cfg,
        String style,
        String sampler,
        String scheduler,
        boolean preferGpu,
        String inputImagePath,
        String maskImagePath,
        double strength
) {
    public TaskConfig {
        prompt = prompt == null ? "" : prompt;
        negativePrompt = negativePrompt == null ? "" : negativePrompt;
        style = style == null ? "None" : style;
        sampler = sampler == null ? "Euler" : sampler;
        scheduler = scheduler == null ? "Euler" : scheduler;
        inputImagePath = inputImagePath == null ? "" : inputImagePath;
        maskImagePath = maskImagePath == null ? "" : maskImagePath;
    }

    public static TaskConfig defaultTxt2Img(String modelId, String prompt) {
        return new TaskConfig(
                modelId,
                prompt,
                "",
                42L,
                1,
                512,
                512,
                20,
                7.5f,
                "None",
                "Euler",
                "Euler",
                true,
                "",
                "",
                0.75);
    }
}
