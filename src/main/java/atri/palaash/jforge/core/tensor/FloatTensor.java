package atri.palaash.jforge.core.tensor;

import java.util.Arrays;

public final class FloatTensor {

    private final int[] shape;
    private final float[] values;

    private FloatTensor(int[] shape, float[] values) {
        this.shape = Arrays.copyOf(shape, shape.length);
        this.values = Arrays.copyOf(values, values.length);
    }

    public static FloatTensor of(int[] shape, float[] values) {
        int expectedLength = 1;
        for (int dim : shape) {
            expectedLength *= dim;
        }
        if (values.length != expectedLength) {
            throw new IllegalArgumentException("Tensor value count does not match shape: expected "
                    + expectedLength + " but got " + values.length);
        }
        return new FloatTensor(shape, values);
    }

    public static FloatTensor zeros(int... shape) {
        int length = 1;
        for (int dim : shape) {
            length *= dim;
        }
        return new FloatTensor(shape, new float[length]);
    }

    public int rank() {
        return shape.length;
    }

    public int dimension(int index) {
        return shape[index];
    }

    public int length() {
        return values.length;
    }

    public int[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public float[] values() {
        return Arrays.copyOf(values, values.length);
    }

    public FloatTensor add(FloatTensor other) {
        ensureSameShape(other);
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] + other.values[i];
        }
        return new FloatTensor(shape, out);
    }

    public FloatTensor multiply(float scalar) {
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] * scalar;
        }
        return new FloatTensor(shape, out);
    }

    public FloatTensor blend(FloatTensor other, float otherWeight) {
        ensureSameShape(other);
        float[] out = new float[values.length];
        float thisWeight = 1f - otherWeight;
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] * thisWeight + other.values[i] * otherWeight;
        }
        return new FloatTensor(shape, out);
    }

    private void ensureSameShape(FloatTensor other) {
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException("Tensor shapes do not match");
        }
    }
}
