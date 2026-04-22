package atri.palaash.jforge.tasks;

public class InpaintTask extends AbstractInferenceTask {

    public InpaintTask(TaskConfig config) {
        super(config);
    }

    @Override
    public String id() {
        return "inpaint";
    }

    @Override
    protected boolean upscale() {
        return false;
    }
}
