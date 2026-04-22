package atri.palaash.jforge.core;

@FunctionalInterface
public interface StepCallback {
    void onStep(int step, float progress, long elapsedMs);
}
