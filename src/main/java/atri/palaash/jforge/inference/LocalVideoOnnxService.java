package atri.palaash.jforge.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import atri.palaash.jforge.model.TaskType;
import atri.palaash.jforge.storage.ModelStorage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/**
 * Inference service for local text-to-video models exported to ONNX.
 *
 * <p>Expects the following component layout (produced by {@code convert_pytorch_to_onnx.py}):
 * <pre>
 *   &lt;model-root&gt;/
 *     text_encoder/model.onnx   — CLIP text encoder
 *     unet/model.onnx           — Temporal UNet3D (denoiser)
 *     vae_decoder/model.onnx    — VAE decoder
 *     tokenizer/vocab.json      — CLIP tokenizer vocabulary
 *     tokenizer/merges.txt      — CLIP tokenizer BPE merges
 * </pre>
 *
 * <p>Pipeline:
 * <ol>
 *   <li>Tokenize and encode the prompt with the CLIP text encoder</li>
 *   <li>Run a DDIM denoising loop with the UNet3D on 5-D latents
 *       {@code [batch, channels, frames, H, W]}</li>
 *   <li>Decode each frame slice with the VAE decoder</li>
 *   <li>Write PNG frames to a temp directory and stitch to MP4 with ffmpeg</li>
 * </ol>
 */
public class LocalVideoOnnxService implements InferenceService {

    // Default generation parameters for text-to-video-ms-1.7b
    private static final int DEFAULT_FRAMES   = 16;
    private static final int DEFAULT_HEIGHT   = 256;
    private static final int DEFAULT_WIDTH    = 256;
    private static final int LATENT_CHANNELS  = 4;
    private static final float VAE_SCALE      = 0.18215f;
    private static final int   DDIM_STEPS     = 25;
    private static final float GUIDANCE_SCALE = 9.0f;

    private final ModelStorage storage;
    private final Executor executor;

    public LocalVideoOnnxService(ModelStorage storage, Executor executor) {
        this.storage  = storage;
        this.executor = executor;
    }

    @Override
    public CompletableFuture<InferenceResult> run(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            if (!checkFfmpegAvailable()) {
                return InferenceResult.fail(
                        "ffmpeg is not installed or not on PATH. "
                        + "It is required to stitch video frames into an MP4.\n"
                        + "Install with:  brew install ffmpeg");
            }
            try {
                return runPipeline(request);
            } catch (Exception ex) {
                return InferenceResult.fail("Text-to-video inference failed: " + ex.getMessage());
            }
        }, executor);
    }

    // ── Pipeline ──────────────────────────────────────────────────────────────

    private InferenceResult runPipeline(InferenceRequest request) throws Exception {
        // Resolve component paths from the model's root directory
        Path unetOnnxPath = storage.modelPath(request.model());   // points to unet/model.onnx
        Path modelRoot    = unetOnnxPath.getParent().getParent(); // e.g. .../ali-vilab/text-to-video-ms-1.7b

        Path textEncoderPath = modelRoot.resolve("text_encoder/model.onnx");
        Path unetPath        = modelRoot.resolve("unet/model.onnx");
        Path vaeDecoderPath  = modelRoot.resolve("vae_decoder/model.onnx");
        Path vocabPath       = modelRoot.resolve("tokenizer/vocab.json");
        Path mergesPath      = modelRoot.resolve("tokenizer/merges.txt");

        // Validate all components are present before starting
        List<Path> required = List.of(textEncoderPath, unetPath, vaeDecoderPath, vocabPath, mergesPath);
        for (Path p : required) {
            if (!Files.exists(p)) {
                return InferenceResult.fail(
                        "Text-to-video model bundle is incomplete — missing: "
                        + modelRoot.relativize(p)
                        + "\n\nOpen the Model Manager, select 'ModelScope Text-to-Video 1.7B', "
                        + "and click Convert to run the ONNX export first.");
            }
        }

        int numFrames   = DEFAULT_FRAMES;
        int height      = snapTo8(request.height()  > 0 ? Math.min(request.height(),  512) : DEFAULT_HEIGHT);
        int width       = snapTo8(request.width()   > 0 ? Math.min(request.width(),   512) : DEFAULT_WIDTH);
        int latentH     = height / 8;
        int latentW     = width  / 8;
        int steps       = request.batch() > 0 ? Math.min(request.batch(), 50) : DDIM_STEPS;
        float guidance  = request.promptWeight() > 0 ? (float) request.promptWeight() : GUIDANCE_SCALE;

        request.reportProgress("Loading ONNX Runtime environment…");
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() - 1));

            // ── 1. Tokenize & encode ──────────────────────────────────────────
            request.reportProgress("Loading text encoder…");
            try (OrtSession textEncoder = env.createSession(textEncoderPath.toString(), opts)) {
                request.reportProgress("Tokenizing prompt…");
                GenericOnnxService.ClipTokenizer tokenizer =
                        GenericOnnxService.ClipTokenizer.load(vocabPath, mergesPath);
                long[] promptTokens   = tokenizer.encode(request.prompt(), 77);
                long[] negTokens      = tokenizer.encode(
                        request.negativePrompt() != null ? request.negativePrompt() : "", 77);

                request.reportProgress("Encoding prompt with CLIP…");
                float[][][] textEmbed  = runTextEncoder(env, textEncoder, promptTokens);
                float[][][] negEmbed   = runTextEncoder(env, textEncoder, negTokens);

                // CFG batch: [neg, pos] concatenated
                float[][][] cfgEmbed = new float[2][textEmbed[0].length][textEmbed[0][0].length];
                cfgEmbed[0] = negEmbed[0];
                cfgEmbed[1] = textEmbed[0];

                // ── 2. Initialize latents ─────────────────────────────────────
                float[][][][][] latents = randomLatents5D(
                        request.seed(), LATENT_CHANNELS, numFrames, latentH, latentW);

                // ── 3. DDIM denoising loop ────────────────────────────────────
                request.reportProgress("Loading UNet3D…");
                try (OrtSession unet = env.createSession(unetPath.toString(), opts)) {
                    int[] timesteps = ddimTimesteps(steps);
                    float[] alphaCumprod = linearAlphaCumprod(1000);

                    request.reportProgress("Denoising: 0/" + steps + " steps…");
                    long stepStart = System.currentTimeMillis();
                    for (int i = 0; i < timesteps.length; i++) {
                        if (request.isCancelled()) {
                            return InferenceResult.fail("Cancelled by user.");
                        }
                        int t     = timesteps[i];
                        int tPrev = (i + 1 < timesteps.length) ? timesteps[i + 1] : 0;

                        // UNet expects [2*batch, ch, frames, H, W] for CFG
                        float[][][][][] latentInput = duplicateBatch5D(latents);

                        Map<String, OnnxTensor> inputs = new HashMap<>();
                        OnnxTensor sampleT  = OnnxTensor.createTensor(env, latentInput);
                        OnnxTensor tsT      = OnnxTensor.createTensor(env, new long[]{t});
                        OnnxTensor hiddenT  = OnnxTensor.createTensor(env, cfgEmbed);
                        inputs.put(resolveInput(unet, "sample",                0), sampleT);
                        inputs.put(resolveInput(unet, "timestep",              1), tsT);
                        inputs.put(resolveInput(unet, "encoder_hidden_states", 2), hiddenT);

                        float[][][][][] noisePred;
                        try (OrtSession.Result result = unet.run(inputs)) {
                            noisePred = extract5D(result);
                        } finally {
                            sampleT.close(); tsT.close(); hiddenT.close();
                        }

                        if (noisePred == null || noisePred.length < 2) {
                            return InferenceResult.fail("UNet3D produced invalid output.");
                        }

                        // Classifier-free guidance: noise = neg + scale * (pos - neg)
                        float[][][][] guided = cfgGuidance5D(noisePred[0], noisePred[1], guidance);
                        latents = ddimStep5D(latents, guided,
                                alphaCumprod[t], alphaCumprod[tPrev]);

                        long elapsed = System.currentTimeMillis() - stepStart;
                        stepStart = System.currentTimeMillis();
                        long etaSec = (elapsed / 1000) * (steps - i - 1);
                        request.reportProgress("Denoising: " + (i + 1) + "/" + steps
                                + " steps ("  + String.format("%.1f", elapsed / 1000.0) + "s/step"
                                + ", ETA: " + etaSec + "s)");
                    }
                }

                // ── 4. Decode frames with VAE ─────────────────────────────────
                request.reportProgress("Decoding " + numFrames + " frames with VAE decoder…");
                List<BufferedImage> frames = new ArrayList<>();
                try (OrtSession vae = env.createSession(vaeDecoderPath.toString(), opts)) {
                    for (int f = 0; f < numFrames; f++) {
                        if (request.isCancelled()) return InferenceResult.fail("Cancelled by user.");

                        // Slice frame f from 5D latents: [1, ch, H, W]
                        float[][][][] frameLatent = sliceFrame(latents, f);
                        // Apply VAE scaling
                        scaleInPlace(frameLatent, 1f / VAE_SCALE);

                        Map<String, OnnxTensor> vaeIn = new HashMap<>();
                        OnnxTensor latTensor = OnnxTensor.createTensor(env, frameLatent);
                        vaeIn.put(resolveInput(vae, "latent_sample", 0), latTensor);

                        float[][][][] decoded;
                        try (OrtSession.Result vaeResult = vae.run(vaeIn)) {
                            decoded = extract4D(vaeResult);
                        } finally {
                            latTensor.close();
                        }

                        if (decoded == null || decoded.length == 0) {
                            return InferenceResult.fail("VAE decoder returned empty output for frame " + f);
                        }
                        frames.add(tensorToImage(decoded[0]));
                        request.reportProgress("Decoded frame " + (f + 1) + "/" + numFrames);
                    }
                }

                // ── 5. Write PNG frames + stitch to MP4 ──────────────────────
                request.reportProgress("Writing frames to disk…");
                Path tempDir = storage.root().resolve("temp_frames_" + System.currentTimeMillis());
                Files.createDirectories(tempDir);

                int fps = 8;
                try {
                    for (int i = 0; i < frames.size(); i++) {
                        File out = tempDir.resolve(String.format("frame_%04d.png", i)).toFile();
                        ImageIO.write(frames.get(i), "png", out);
                    }

                    request.reportProgress("Stitching frames with ffmpeg…");
                    File outputVideo = stitchFramesToVideo(tempDir, fps);

                    String details = "ModelScope T2V pipeline | frames=" + numFrames
                            + " | steps=" + steps + " | resolution=" + width + "×" + height;
                    return InferenceResult.ok(
                            "Generated video for: \"" + request.prompt() + "\"",
                            details,
                            outputVideo.getAbsolutePath(),
                            "video");
                } finally {
                    // Cleanup temp frames
                    try (var stream = Files.walk(tempDir)) {
                        stream.sorted((a, b) -> b.compareTo(a))
                              .forEach(p -> { try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
                    }
                }
            }
        }
    }

    // ── Math / tensor helpers ─────────────────────────────────────────────────

    /**
     * Initialize random 5-D latents {@code [1, ch, frames, H, W]}.
     */
    private static float[][][][][] randomLatents5D(long seed, int ch,
                                                    int frames, int h, int w) {
        Random rng = new Random(seed);
        float[][][][][] lat = new float[1][ch][frames][h][w];
        for (int c = 0; c < ch; c++)
            for (int f = 0; f < frames; f++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        lat[0][c][f][y][x] = (float) rng.nextGaussian();
        return lat;
    }

    /**
     * Duplicate the batch dimension: {@code [1, ...] → [2, ...]}.
     */
    private static float[][][][][] duplicateBatch5D(float[][][][][] in) {
        int ch = in[0].length, fr = in[0][0].length, h = in[0][0][0].length, w = in[0][0][0][0].length;
        float[][][][][] out = new float[2][ch][fr][h][w];
        for (int b = 0; b < 2; b++)
            for (int c = 0; c < ch; c++)
                for (int f = 0; f < fr; f++)
                    for (int y = 0; y < h; y++)
                        System.arraycopy(in[0][c][f][y], 0, out[b][c][f][y], 0, w);
        return out;
    }

    /**
     * Classifier-free guidance: {@code guided = neg + scale * (pos - neg)}.
     */
    private static float[][][][] cfgGuidance5D(float[][][][] neg, float[][][][] pos, float scale) {
        int ch = neg.length, fr = neg[0].length, h = neg[0][0].length, w = neg[0][0][0].length;
        float[][][][] out = new float[ch][fr][h][w];
        for (int c = 0; c < ch; c++)
            for (int f = 0; f < fr; f++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        out[c][f][y][x] = neg[c][f][y][x] + scale * (pos[c][f][y][x] - neg[c][f][y][x]);
        return out;
    }

    /**
     * DDIM step on 5-D latents.
     *
     * <p>{@code x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1}) * noise}
     */
    private static float[][][][][] ddimStep5D(float[][][][][] latents, float[][][][] noisePred,
                                               float alphaT, float alphaPrev) {
        int b = latents.length, ch = latents[0].length;
        int fr = latents[0][0].length, h = latents[0][0][0].length, w = latents[0][0][0][0].length;
        float sqrtA    = (float) Math.sqrt(alphaT);
        float sqrtOneA = (float) Math.sqrt(1f - alphaT);
        float sqrtAp   = (float) Math.sqrt(alphaPrev);
        float sqrtOneAp = (float) Math.sqrt(1f - alphaPrev);
        float[][][][][] out = new float[b][ch][fr][h][w];
        for (int bi = 0; bi < b; bi++)
            for (int c = 0; c < ch; c++)
                for (int f = 0; f < fr; f++)
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++) {
                            float xt = latents[bi][c][f][y][x];
                            float eps = noisePred[c][f][y][x];
                            // Predicted x0
                            float predX0 = (xt - sqrtOneA * eps) / sqrtA;
                            // DDIM update
                            out[bi][c][f][y][x] = sqrtAp * predX0 + sqrtOneAp * eps;
                        }
        return out;
    }

    /**
     * Slice a single frame from 5-D latents: {@code [1,ch,frames,H,W] → [1,ch,H,W]}.
     */
    private static float[][][][] sliceFrame(float[][][][][] latents, int frame) {
        int ch = latents[0].length, h = latents[0][0][0].length, w = latents[0][0][0][0].length;
        float[][][][] out = new float[1][ch][h][w];
        for (int c = 0; c < ch; c++)
            for (int y = 0; y < h; y++)
                System.arraycopy(latents[0][c][frame][y], 0, out[0][c][y], 0, w);
        return out;
    }

    /**
     * Scale all values in a 4-D tensor in place.
     */
    private static void scaleInPlace(float[][][][] t, float factor) {
        for (float[][][] a : t) for (float[][] b : a) for (float[] c : b)
            for (int i = 0; i < c.length; i++) c[i] *= factor;
    }

    /**
     * Build evenly-spaced DDIM timesteps descending from 999 → 0.
     */
    private static int[] ddimTimesteps(int steps) {
        int[] ts = new int[steps];
        for (int i = 0; i < steps; i++) {
            ts[i] = (int) Math.round(999.0 * (steps - 1 - i) / (steps - 1));
        }
        return ts;
    }

    /**
     * Simple linear alpha-cumprod schedule (approximation).
     */
    private static float[] linearAlphaCumprod(int numSteps) {
        float[] ac = new float[numSteps];
        for (int i = 0; i < numSteps; i++) {
            float beta = 0.00085f + (0.012f - 0.00085f) * i / (numSteps - 1);
            ac[i] = (i == 0) ? 1f - beta : ac[i - 1] * (1f - beta);
        }
        return ac;
    }

    /**
     * Convert a 4-D image tensor {@code [CH, H, W]} (values in [-1, 1]) to a
     * {@link BufferedImage}.
     */
    private static BufferedImage tensorToImage(float[][][] tensor) {
        int ch = tensor.length;
        int h  = tensor[0].length;
        int w  = tensor[0][0].length;
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int r = clamp255(ch > 0 ? tensor[0][y][x] : 0f);
                int g = clamp255(ch > 1 ? tensor[1][y][x] : 0f);
                int b = clamp255(ch > 2 ? tensor[2][y][x] : 0f);
                img.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return img;
    }

    /** Clamp a float in [-1, 1] (or [0, 1]) to [0, 255] int. */
    private static int clamp255(float v) {
        // VAE outputs are typically in [-1, 1]; map to [0, 255]
        int i = (int) ((v * 0.5f + 0.5f) * 255f);
        return Math.max(0, Math.min(255, i));
    }

    private static int snapTo8(int v) { return (v / 8) * 8; }

    // ── ONNX helpers ──────────────────────────────────────────────────────────

    private static float[][][] runTextEncoder(OrtEnvironment env, OrtSession encoder,
                                               long[] tokens) throws OrtException {
        String inputName = encoder.getInputNames().iterator().next();
        try (OnnxTensor t = OnnxTensor.createTensor(env, new long[][]{tokens});
             OrtSession.Result r = encoder.run(Map.of(inputName, t))) {
            Object val = r.get(0).getValue();
            if (val instanceof float[][][] arr) return arr;
            return new float[1][77][768];
        }
    }

    /** Extract the first output of an ORT result as a 5-D float array. */
    @SuppressWarnings("unchecked")
    private static float[][][][][] extract5D(OrtSession.Result result) throws OrtException {
        Object val = result.get(0).getValue();
        if (val instanceof float[][][][][] arr) return arr;
        return null;
    }

    /** Extract the first output of an ORT result as a 4-D float array. */
    private static float[][][][] extract4D(OrtSession.Result result) throws OrtException {
        Object val = result.get(0).getValue();
        if (val instanceof float[][][][] arr) return arr;
        return null;
    }

    /**
     * Resolve the actual ONNX input name: try the expected {@code preferred} name first,
     * then fall back to positional index.
     */
    private static String resolveInput(OrtSession session, String preferred, int fallbackIndex)
            throws OrtException {
        if (session.getInputNames().contains(preferred)) return preferred;
        List<String> names = new ArrayList<>(session.getInputNames());
        if (fallbackIndex < names.size()) return names.get(fallbackIndex);
        return preferred;
    }

    // ── ffmpeg ────────────────────────────────────────────────────────────────

    private boolean checkFfmpegAvailable() {
        try {
            Process p = new ProcessBuilder("ffmpeg", "-version")
                    .redirectErrorStream(true).start();
            p.getInputStream().readAllBytes();
            return p.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }

    private File stitchFramesToVideo(Path frameDir, int fps) throws Exception {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        File outVideo = storage.root().resolve("output")
                .resolve("video_" + timestamp + ".mp4").toFile();
        outVideo.getParentFile().mkdirs();

        ProcessBuilder pb = new ProcessBuilder(
                "ffmpeg", "-y",
                "-framerate", String.valueOf(fps),
                "-i", frameDir.resolve("frame_%04d.png").toString(),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                outVideo.getAbsolutePath()
        );
        pb.redirectErrorStream(true);
        Process process = pb.start();
        process.getInputStream().readAllBytes(); // drain
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new Exception("ffmpeg exited with code " + exitCode);
        }
        return outVideo;
    }
}
