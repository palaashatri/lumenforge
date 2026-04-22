package atri.palaash.jforge.tasks;

public class Txt2ImgTask extends AbstractInferenceTask {

    public Txt2ImgTask(TaskConfig config) {
        super(config);
    }

    @Override
    public String id() {
        return "txt2img";
    }

    @Override
    protected boolean upscale() {
        return false;
    }
}
