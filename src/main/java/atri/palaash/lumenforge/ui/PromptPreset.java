package atri.palaash.lumenforge.ui;

public record PromptPreset(
        String name,
        String prompt,
        String negativePrompt,
        String tags,
        String style
) {
}
