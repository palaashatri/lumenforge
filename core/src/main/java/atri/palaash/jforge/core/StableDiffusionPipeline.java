package atri.palaash.jforge.core;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import atri.palaash.jforge.core.sampler.Sampler;
import atri.palaash.jforge.core.sampler.SamplerContext;
import atri.palaash.jforge.core.sampler.SchedulerSampler;
import atri.palaash.jforge.core.scheduler.EulerScheduler;
import atri.palaash.jforge.core.scheduler.Scheduler;
import atri.palaash.jforge.core.tensor.FloatTensor;
import atri.palaash.jforge.core.unet.OnnxUNet;
import atri.palaash.jforge.core.unet.UNet;
import atri.palaash.jforge.core.vae.OnnxVAE;
import atri.palaash.jforge.core.vae.VAE;
import atri.palaash.jforge.inference.InferenceRequest;

import java.util.HashMap;
import java.util.Map;

public class StableDiffusionPipeline {

    private final OrtEnvironment env;
    private final UNet unet;
    private final VAE vae;
    private final Sampler sampler;

    public StableDiffusionPipeline(OrtEnvironment env, OrtSession unetSession, OrtSession vaeDecoderSession) {
        this.env = env;
        this.unet = new OnnxUNet(env, unetSession);
        this.vae = new OnnxVAE(env, vaeDecoderSession, null);
        this.sampler = new SchedulerSampler();
    }

    public FloatTensor run(InferenceRequest request, FloatTensor textEmbeddings, Scheduler scheduler) {
        int width = Math.max(256, (request.width() / 8) * 8);
        int height = Math.max(256, (request.height() / 8) * 8);
        int latentW = width / 8;
        int latentH = height / 8;
        int steps = Math.max(1, request.batch());
        float guidanceScale = (float) (request.promptWeight() > 0 ? request.promptWeight() : 7.5);

        FloatTensor initialLatents = FloatTensor.random(request.seed(), 1, 4, latentH, latentW);
        
        // CFG is handled inside the UNet implementation or by the sampler.
        // For SD 1.5, we usually pass batched embeddings [2, 77, 768].
        
        SamplerContext ctx = new SamplerContext(
                initialLatents,
                scheduler,
                unet,
                textEmbeddings,
                steps,
                guidanceScale,
                request.progressCallback() != null ? (step, progress, elapsed) -> {
                    request.reportProgress("Denoising: " + step + "/" + steps + " steps");
                } : null,
                new HashMap<>()
        );

        FloatTensor latents = sampler.sample(ctx);

        // Scale and decode
        FloatTensor scaled = latents.multiply(1f / 0.18215f);
        return vae.decode(scaled);
    }
}
