package atri.palaash.jforge.tasks;

public interface ForgeTask {
    String id();
    TaskConfig config();
    TaskResult run(TaskContext ctx);
}
