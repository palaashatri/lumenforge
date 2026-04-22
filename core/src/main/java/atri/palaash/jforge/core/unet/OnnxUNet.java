package atri.palaash.jforge.core.unet;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import atri.palaash.jforge.core.tensor.FloatTensor;
import atri.palaash.jforge.core.tensor.OnnxTensorUtils;

import java.util.HashMap;
import java.util.Map;

public class OnnxUNet implements UNet {

    private final OrtEnvironment env;
    private final OrtSession session;

    public OnnxUNet(OrtEnvironment env, OrtSession session) {
        this.env = env;
        this.session = session;
    }

    @Override
    public FloatTensor predict(FloatTensor sample, long timestep, FloatTensor encoderHiddenStates, Map<String, Object> additionalInputs) {
        float guidanceScale = (float) additionalInputs.getOrDefault("guidance_scale", 1.0f);
        try {
            if (guidanceScale > 1.0f && encoderHiddenStates.dimension(0) == 2) {
                // CFG path: Split hidden states, run twice or batch if supported
                // For SD1.5/SDXL we usually run as a batch of 2
                return predictBatch(sample, timestep, encoderHiddenStates, additionalInputs, guidanceScale);
            }

            Map<String, OnnxTensor> inputs = new HashMap<>();
            OnnxTensor sampleT = OnnxTensorUtils.toOnnx(sample, env);
            OnnxTensor tsT = createTimestepTensor(timestep);
            OnnxTensor hiddenT = OnnxTensorUtils.toOnnx(encoderHiddenStates, env);
            
            inputs.put(resolveInputName("sample", 0), sampleT);
            inputs.put(resolveInputName("timestep", 1), tsT);
            inputs.put(resolveInputName("encoder_hidden_states", 2), hiddenT);
            
            for (Map.Entry<String, Object> entry : additionalInputs.entrySet()) {
                if (entry.getValue() instanceof FloatTensor ft) {
                    inputs.put(resolveInputName(entry.getKey(), -1), OnnxTensorUtils.toOnnx(ft, env));
                }
            }

            try (OrtSession.Result result = session.run(inputs)) {
                return OnnxTensorUtils.fromOnnx((OnnxTensor) result.get(0));
            } finally {
                closeAll(inputs.values());
                tsT.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("UNet inference failed", e);
        }
    }

    private FloatTensor predictBatch(FloatTensor sample, long timestep, FloatTensor batchedHidden, Map<String, Object> additionalInputs, float scale) throws Exception {
        // Duplicate latents for batch
        int[] batchShape = sample.shape();
        batchShape[0] = 2;
        float[] batchVals = new float[sample.length() * 2];
        System.arraycopy(sample.values(), 0, batchVals, 0, sample.length());
        System.arraycopy(sample.values(), 0, batchVals, sample.length(), sample.length());
        FloatTensor batchedSample = FloatTensor.of(batchShape, batchVals);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        OnnxTensor sampleT = OnnxTensorUtils.toOnnx(batchedSample, env);
        OnnxTensor tsT = createTimestepTensor(timestep);
        OnnxTensor hiddenT = OnnxTensorUtils.toOnnx(batchedHidden, env);

        inputs.put(resolveInputName("sample", 0), sampleT);
        inputs.put(resolveInputName("timestep", 1), tsT);
        inputs.put(resolveInputName("encoder_hidden_states", 2), hiddenT);

        // Handle SDXL time_ids etc if present in additionalInputs
        for (Map.Entry<String, Object> entry : additionalInputs.entrySet()) {
            if (entry.getValue() instanceof FloatTensor ft) {
                inputs.put(resolveInputName(entry.getKey(), -1), OnnxTensorUtils.toOnnx(ft, env));
            }
        }

        try (OrtSession.Result result = session.run(inputs)) {
            FloatTensor output = OnnxTensorUtils.fromOnnx((OnnxTensor) result.get(0));
            // Split and apply guidance: uncond + scale * (cond - uncond)
            int half = output.length() / 2;
            float[] vals = output.values();
            float[] uncondVals = new float[half];
            float[] condVals = new float[half];
            System.arraycopy(vals, 0, uncondVals, 0, half);
            System.arraycopy(vals, half, condVals, 0, half);
            
            int[] outShape = output.shape();
            outShape[0] = 1;
            FloatTensor uncond = FloatTensor.of(outShape, uncondVals);
            FloatTensor cond = FloatTensor.of(outShape, condVals);
            
            return FloatTensor.guidance(uncond, cond, scale);
        } finally {
            closeAll(inputs.values());
            tsT.close();
        }
    }

    private OnnxTensor createTimestepTensor(long timestep) throws Exception {
        String name = resolveInputName("timestep", 1);
        var info = session.getInputInfo().get(name).getInfo();
        if (info.toString().contains("INT64")) {
            return OnnxTensor.createTensor(env, new long[]{timestep});
        }
        return OnnxTensor.createTensor(env, new float[]{(float) timestep});
    }

    private void closeAll(java.util.Collection<OnnxTensor> tensors) {
        for (OnnxTensor t : tensors) {
            try { t.close(); } catch (Exception ignored) {}
        }
    }

    private String resolveInputName(String defaultName, int index) {
        var names = session.getInputNames();
        if (names.contains(defaultName)) return defaultName;
        
        // Fuzzy matching for SDXL/SD3 specific inputs
        if (defaultName.equals("text_embeds") || defaultName.equals("time_ids") || defaultName.equals("y")) {
            for (String name : names) {
                if (name.contains(defaultName)) return name;
            }
        }

        if (index < 0) return defaultName;

        int i = 0;
        for (String name : names) {
            if (i == index) return name;
            i++;
        }
        return defaultName;
    }
}
