package atri.palaash.jforge.model;

public enum TaskType {
    TEXT_TO_IMAGE("Text → Image"),
    IMAGE_UPSCALE("Image Upscale"),
    IMAGE_TO_IMAGE("Image → Image"),
    TEXT_TO_VIDEO("Text → Video");

    private final String displayName;

    TaskType(String displayName) {
        this.displayName = displayName;
    }

    public String displayName() {
        return displayName;
    }
}
