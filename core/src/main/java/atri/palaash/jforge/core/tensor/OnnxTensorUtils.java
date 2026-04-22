package atri.palaash.jforge.core.tensor;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.nio.FloatBuffer;

public final class OnnxTensorUtils {

    public static OnnxTensor toOnnx(FloatTensor tensor, OrtEnvironment env) throws OrtException {
        long[] shape = new long[tensor.rank()];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = tensor.dimension(i);
        }
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(tensor.values()), shape);
    }

    public static FloatTensor fromOnnx(OnnxTensor tensor) throws OrtException {
        long[] shapeLong = tensor.getInfo().getShape();
        int[] shape = new int[shapeLong.length];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = (int) shapeLong[i];
        }
        float[] values = tensor.getFloatBuffer().array();
        return FloatTensor.of(shape, values);
    }
}
