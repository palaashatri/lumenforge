package atri.palaash.jforge.core;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import atri.palaash.jforge.core.sampler.Sampler;
import atri.palaash.jforge.core.sampler.SamplerContext;
import atri.palaash.jforge.core.sampler.SchedulerSampler;
import atri.palaash.jforge.core.scheduler.FlowMatchingScheduler;
import atri.palaash.jforge.core.scheduler.Scheduler;
import atri.palaash.jforge.core.tensor.FloatTensor;
import atri.palaash.jforge.core.unet.OnnxUNet;
import atri.palaash.jforge.core.unet.UNet;
import atri.palaash.jforge.core.vae.OnnxVAE;
import atri.palaash.jforge.core.vae.VAE;
import atri.palaash.jforge.inference.InferenceRequest;

import java.util.HashMap;
import java.util.Map;

public class SD3Pipeline {

    private final OrtEnvironment env;
    private final UNet transformer;
    private final VAE vae;
    private final Sampler sampler;

    public SD3Pipeline(OrtEnvironment env, OrtSession transformerSession, OrtSession vaeDecoderSession) {
        this.env = env;
        this.transformer = new OnnxUNet(env, transformerSession);
        this.vae = new OnnxVAE(env, vaeDecoderSession, null);
        this.sampler = new SchedulerSampler();
    }

    public FloatTensor run(InferenceRequest request, FloatTensor textEmbeddings, FloatTensor pooledEmbeddings, Scheduler scheduler) {
        int width = Math.max(512, (request.width() / 8) * 8);
        int height = Math.max(512, (request.height() / 8) * 8);
        int latentW = width / 8;
        int latentH = height / 8;
        int steps = Math.max(1, request.batch());
        float guidanceScale = (float) (request.promptWeight() > 0 ? request.promptWeight() : 7.0);

        // SD3 uses 16-channel latents
        FloatTensor initialLatents = FloatTensor.random(request.seed(), 1, 16, latentH, latentW);
        
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("y", pooledEmbeddings);

        SamplerContext ctx = new SamplerContext(
                initialLatents,
                scheduler,
                transformer,
                textEmbeddings,
                steps,
                guidanceScale,
                request.progressCallback() != null ? (step, progress, elapsed) -> {
                    request.reportProgress("Denoising: " + step + "/" + steps + " steps (Modular SD3)");
                } : null,
                attributes
        );

        FloatTensor latents = sampler.sample(ctx);

        // SD3 VAE scaling
        FloatTensor scaled = latents.multiply(1f / 0.13025f);
        return vae.decode(scaled);
    }
}
