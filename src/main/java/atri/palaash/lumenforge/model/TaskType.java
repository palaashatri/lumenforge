package atri.palaash.lumenforge.model;

public enum TaskType {
    TEXT_TO_IMAGE("Text → Image"),
    IMAGE_UPSCALE("Image Upscale"),
    IMAGE_TO_IMAGE("Image → Image");

    private final String displayName;

    TaskType(String displayName) {
        this.displayName = displayName;
    }

    public String displayName() {
        return displayName;
    }
}
