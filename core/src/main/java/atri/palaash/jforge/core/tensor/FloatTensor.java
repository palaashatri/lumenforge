package atri.palaash.jforge.core.tensor;

import java.util.Arrays;
import java.util.Random;

public final class FloatTensor {

    private final int[] shape;
    private final float[] values;

    private FloatTensor(int[] shape, float[] values) {
        this.shape = Arrays.copyOf(shape, shape.length);
        this.values = values; // Internal constructor assumes values match shape
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
        return new FloatTensor(shape, Arrays.copyOf(values, values.length));
    }

    public static FloatTensor zeros(int... shape) {
        int length = 1;
        for (int dim : shape) {
            length *= dim;
        }
        return new FloatTensor(shape, new float[length]);
    }

    public static FloatTensor random(long seed, int... shape) {
        Random random = new Random(seed);
        int length = 1;
        for (int dim : shape) {
            length *= dim;
        }
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = (float) random.nextGaussian();
        }
        return new FloatTensor(shape, values);
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

    public float get(int... indices) {
        return values[flatIndex(indices)];
    }

    public void set(float value, int... indices) {
        values[flatIndex(indices)] = value;
    }

    private int flatIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Index rank mismatch");
        }
        int index = 0;
        int multiplier = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " out of bounds for dim " + i);
            }
            index += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        return index;
    }

    public FloatTensor add(FloatTensor other) {
        ensureSameShape(other);
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] + other.values[i];
        }
        return new FloatTensor(shape, out);
    }

    public FloatTensor subtract(FloatTensor other) {
        ensureSameShape(other);
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] - other.values[i];
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

    public FloatTensor multiply(FloatTensor other) {
        ensureSameShape(other);
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] * other.values[i];
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

    /**
     * Performs classifier-free guidance: noise_uncond + scale * (noise_cond - noise_uncond)
     */
    public static FloatTensor guidance(FloatTensor uncond, FloatTensor cond, float scale) {
        uncond.ensureSameShape(cond);
        float[] out = new float[uncond.values.length];
        for (int i = 0; i < uncond.values.length; i++) {
            out[i] = uncond.values[i] + scale * (cond.values[i] - uncond.values[i]);
        }
        return new FloatTensor(uncond.shape, out);
    }

    public FloatTensor copy() {
        return new FloatTensor(shape, Arrays.copyOf(values, values.length));
    }

    private void ensureSameShape(FloatTensor other) {
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException("Tensor shapes do not match: " 
                    + Arrays.toString(shape) + " vs " + Arrays.toString(other.shape));
        }
    }
}
