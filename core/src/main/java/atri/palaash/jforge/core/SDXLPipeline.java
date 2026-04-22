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

public class SDXLPipeline {

    private final OrtEnvironment env;
    private final UNet unet;
    private final VAE vae;
    private final Sampler sampler;

    public SDXLPipeline(OrtEnvironment env, OrtSession unetSession, OrtSession vaeDecoderSession) {
        this.env = env;
        this.unet = new OnnxUNet(env, unetSession);
        this.vae = new OnnxVAE(env, vaeDecoderSession, null);
        this.sampler = new SchedulerSampler();
    }

    public FloatTensor run(InferenceRequest request, FloatTensor textEmbeddings, FloatTensor pooledEmbeddings, Scheduler scheduler) {
        int width = Math.max(512, (request.width() / 8) * 8);
        int height = Math.max(512, (request.height() / 8) * 8);
        int latentW = width / 8;
        int latentH = height / 8;
        int steps = Math.max(1, request.batch());
        float guidanceScale = (float) (request.promptWeight() > 0 ? request.promptWeight() : 7.5);

        FloatTensor initialLatents = FloatTensor.random(request.seed(), 1, 4, latentH, latentW);
        
        // SDXL specific inputs
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("text_embeds", pooledEmbeddings);
        
        // time_ids: [original_h, original_w, crop_y, crop_x, target_h, target_w] — batched [2, 6]
        float[] timeIdsVals = {
            (float)height, (float)width, 0, 0, (float)height, (float)width,
            (float)height, (float)width, 0, 0, (float)height, (float)width
        };
        FloatTensor timeIds = FloatTensor.of(new int[]{2, 6}, timeIdsVals);
        attributes.put("time_ids", timeIds);

        SamplerContext ctx = new SamplerContext(
                initialLatents,
                scheduler,
                unet,
                textEmbeddings,
                steps,
                guidanceScale,
                request.progressCallback() != null ? (step, progress, elapsed) -> {
                    request.reportProgress("Denoising: " + step + "/" + steps + " steps (Modular SDXL)");
                } : null,
                attributes
        );

        FloatTensor latents = sampler.sample(ctx);

        // SDXL VAE uses 0.13025 scaling
        FloatTensor scaled = latents.multiply(1f / 0.13025f);
        return vae.decode(scaled);
    }
}
