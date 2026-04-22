package atri.palaash.jforge.tasks;

public class Img2ImgTask extends AbstractInferenceTask {

    public Img2ImgTask(TaskConfig config) {
        super(config);
    }

    @Override
    public String id() {
        return "img2img";
    }

    @Override
    protected boolean upscale() {
        return false;
    }
}
