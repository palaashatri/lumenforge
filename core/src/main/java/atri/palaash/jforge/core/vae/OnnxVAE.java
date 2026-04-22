package atri.palaash.jforge.core.vae;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import atri.palaash.jforge.core.tensor.FloatTensor;
import atri.palaash.jforge.core.tensor.OnnxTensorUtils;

import java.util.Map;

public class OnnxVAE implements VAE {

    private final OrtEnvironment env;
    private final OrtSession decoderSession;
    private final OrtSession encoderSession;

    public OnnxVAE(OrtEnvironment env, OrtSession decoderSession, OrtSession encoderSession) {
        this.env = env;
        this.decoderSession = decoderSession;
        this.encoderSession = encoderSession;
    }

    @Override
    public FloatTensor encode(FloatTensor image) {
        if (encoderSession == null) {
            throw new UnsupportedOperationException("VAE Encoder session not provided");
        }
        try {
            OnnxTensor imgT = OnnxTensorUtils.toOnnx(image, env);
            try (OrtSession.Result result = encoderSession.run(Map.of(resolveInputName(encoderSession, "sample", 0), imgT))) {
                return OnnxTensorUtils.fromOnnx((OnnxTensor) result.get(0));
            } finally {
                imgT.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("VAE encoding failed", e);
        }
    }

    @Override
    public FloatTensor decode(FloatTensor latents) {
        try {
            OnnxTensor latT = OnnxTensorUtils.toOnnx(latents, env);
            try (OrtSession.Result result = decoderSession.run(Map.of(resolveInputName(decoderSession, "latent", 0), latT))) {
                return OnnxTensorUtils.fromOnnx((OnnxTensor) result.get(0));
            } finally {
                latT.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("VAE decoding failed", e);
        }
    }

    private String resolveInputName(OrtSession session, String defaultName, int index) {
        var names = session.getInputNames();
        if (names.contains(defaultName)) return defaultName;
        int i = 0;
        for (String name : names) {
            if (i == index) return name;
            i++;
        }
        return defaultName;
    }
}
