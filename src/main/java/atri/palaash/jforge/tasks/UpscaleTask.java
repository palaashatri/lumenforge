package atri.palaash.jforge.tasks;

public class UpscaleTask extends AbstractInferenceTask {

    public UpscaleTask(TaskConfig config) {
        super(config);
    }

    @Override
    public String id() {
        return "upscale";
    }

    @Override
    protected boolean upscale() {
        return true;
    }
}
