package atri.palaash.lumenforge.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import atri.palaash.lumenforge.model.TaskType;
import atri.palaash.lumenforge.storage.ModelStorage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class GenericOnnxService implements InferenceService {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final Pattern TOKEN_PATTERN = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    /* ── Session & Tokenizer Caches ────────────────────────────────── */
    private static final ConcurrentHashMap<String, OrtSession> SESSION_CACHE = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, ClipTokenizer> TOKENIZER_CACHE = new ConcurrentHashMap<>();
    private static volatile String cachedEpKey = "";

    private final TaskType taskType;
    private final ModelStorage storage;
    private final Executor executor;

    public GenericOnnxService(TaskType taskType, ModelStorage storage, Executor executor) {
        this.taskType = taskType;
        this.storage = storage;
        this.executor = executor;
    }

    /* ── Cache helpers ─────────────────────────────────────────────── */

    /**
     * Return a cached OrtSession for the given model path, or create and
     * cache a new one. Subsequent inference calls skip expensive model
     * loading and graph optimization — the single biggest performance win.
     */
    private OrtSession getOrCreateSession(OrtEnvironment env, Path modelPath,
                                           OrtSession.SessionOptions opts) throws OrtException {
        String key = modelPath.toAbsolutePath().toString();
        OrtSession existing = SESSION_CACHE.get(key);
        if (existing != null) {
            return existing;
        }
        OrtSession session = env.createSession(modelPath.toString(), opts);
        SESSION_CACHE.put(key, session);
        return session;
    }

    /**
     * Get or load a ClipTokenizer — vocab.json and merges.txt are cached
     * in memory so repeated inference calls skip disk I/O and JSON parsing.
     */
    private ClipTokenizer getOrCreateTokenizer(Path vocabPath, Path mergesPath) throws Exception {
        String key = vocabPath.toAbsolutePath() + "|" + mergesPath.toAbsolutePath();
        ClipTokenizer cached = TOKENIZER_CACHE.get(key);
        if (cached != null) {
            return cached;
        }
        ClipTokenizer tokenizer = ClipTokenizer.load(vocabPath, mergesPath);
        TOKENIZER_CACHE.put(key, tokenizer);
        return tokenizer;
    }

    /**
     * Evict all cached ORT sessions (e.g. when the user explicitly wants
     * a fresh start or switches execution providers). Also clears the
     * tokenizer cache.
     */
    public static void clearCache() {
        SESSION_CACHE.forEach((k, session) -> {
            try { session.close(); } catch (Exception ignored) { }
        });
        SESSION_CACHE.clear();
        TOKENIZER_CACHE.clear();
        cachedEpKey = "";
    }

    @Override
    public CompletableFuture<InferenceResult> run(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            if (!storage.isAvailable(request.model())) {
                return InferenceResult.fail("Model not found locally. Open Model Manager from the menu bar and download it first.");
            }

            Path modelPath = storage.modelPath(request.model());

            // Temporarily intercept stderr so ONNX Runtime native warnings
            // appear in the application's Log tab instead of only in the console.
            PrintStream origErr = System.err;
            TeeOutputStream tee = new TeeOutputStream(origErr, request.progressCallback());
            System.setErr(new PrintStream(tee, true));

            try {
                request.reportProgress("Loading ONNX Runtime environment\u2026");
                OrtEnvironment environment = OrtEnvironment.getEnvironment();
                OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
                sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                int cpus = Runtime.getRuntime().availableProcessors();
                sessionOptions.setIntraOpNumThreads(Math.max(1, cpus - 1));
                sessionOptions.setInterOpNumThreads(Math.max(1, Math.min(cpus / 2, 4)));
                ProviderSelection providerSelection = configureExecutionProvider(sessionOptions, request.preferGpu());

                // Invalidate session cache when the execution provider changes.
                String epKey = providerSelection.provider() + "|" + request.preferGpu();
                if (!epKey.equals(cachedEpKey)) {
                    clearCache();
                    cachedEpKey = epKey;
                }
                if ("realesrgan".equals(request.model().id())) {
                    OrtSession session = getOrCreateSession(environment,
                            modelPath, sessionOptions);
                    return runRealEsrgan(environment, session, request, providerSelection.provider());
                }
                if ("sd_v15_onnx".equals(request.model().id())) {
                    return runStableDiffusionV15(environment, sessionOptions, request, providerSelection.provider());
                }
                if ("sd_turbo_onnx".equals(request.model().id())) {
                    return runStableDiffusionTurbo(environment, sessionOptions, request, providerSelection.provider());
                }
                if ("sdxl_turbo_onnx".equals(request.model().id())) {
                    return runSdxlTurbo(environment, sessionOptions, request, providerSelection.provider());
                }
                if ("sdxl_base_onnx".equals(request.model().id())) {
                    return runSdxlBase(environment, sessionOptions, request, providerSelection.provider());
                }
                // DJL/PyTorch models — delegate to the DJL backend
                if (request.model().id().contains("pytorch")) {
                    DjlPyTorchService djl = new DjlPyTorchService(storage, executor);
                    return djl.run(request).join();
                }
                // Img2Img pipelines
                if ("sd_v15_img2img".equals(request.model().id())) {
                    return runImg2Img(environment, sessionOptions, request, providerSelection.provider(), false);
                }
                if ("sd_turbo_img2img".equals(request.model().id())) {
                    return runImg2Img(environment, sessionOptions, request, providerSelection.provider(), true);
                }

                String details = "Model loaded but generation is not implemented for this ONNX pipeline: "
                        + request.model().displayName() + " | task=" + taskType.displayName()
                        + " | EP=" + providerSelection.provider()
                        + providerSelection.noteSuffix();
                return InferenceResult.fail(details);
            } catch (OrtException ex) {
                String message = ex.getMessage() == null ? "Unknown ONNX Runtime error" : ex.getMessage();
                if (message.contains("NhwcConv")) {
                    return InferenceResult.fail(
                            "This ONNX model uses custom ops (e.g., NhwcConv) not available in CPUExecutionProvider. "
                                    + "Choose a CPU-compatible ONNX export or switch to a different model.\nOriginal error: "
                                    + message
                    );
                }
                if (message.contains("weights.pb")) {
                    return InferenceResult.fail(
                            "Model requires external tensor data (weights.pb) that is missing in the same directory as the ONNX file. "
                                    + "Import the complete ONNX bundle.\nOriginal error: " + message
                    );
                }
                return InferenceResult.fail("Failed to load ONNX model on CPU: " + message);
            } finally {
                System.setErr(origErr);
            }
        }, executor);
    }

    private InferenceResult runStableDiffusionV15(OrtEnvironment environment,
                                                   OrtSession.SessionOptions sessionOptions,
                                                   InferenceRequest request,
                                                   String provider) {
        try {
            int width = Math.max(256, (request.width() / 8) * 8);
            int height = Math.max(256, (request.height() / 8) * 8);
            int latentWidth = width / 8;
            int latentHeight = height / 8;
            int steps = Math.max(5, Math.min(request.batch() > 0 ? request.batch() : 15, 50));
            float guidanceScale = 7.5f;

            Path base = storage.root().resolve("text-image").resolve("stable-diffusion-v15");
            Path textEncoderPath = base.resolve("text_encoder/model.onnx");
            Path unetPath = base.resolve("unet/model.onnx");
            Path vaeDecoderPath = base.resolve("vae_decoder/model.onnx");
            Path vocabPath = base.resolve("tokenizer/vocab.json");
            Path mergesPath = base.resolve("tokenizer/merges.txt");
            Path schedulerPath = base.resolve("scheduler/scheduler_config.json");
            List<Path> required = List.of(textEncoderPath, unetPath, vaeDecoderPath, vocabPath, mergesPath, schedulerPath);
            for (Path path : required) {
                if (!java.nio.file.Files.exists(path)) {
                    return InferenceResult.fail("Stable Diffusion bundle is incomplete. Open menu bar → Models → Open Model Manager and redownload Stable Diffusion v1.5 ONNX.");
                }
            }

            ClipTokenizer tokenizer = getOrCreateTokenizer(vocabPath, mergesPath);
            long[] promptTokens = tokenizer.encode(request.prompt(), 77);
            long[] negativeTokens = tokenizer.encode(request.negativePrompt() == null ? "" : request.negativePrompt(), 77);

            request.reportProgress("Loading models (text encoder, UNet, VAE decoder)\u2026");
            OrtSession textEncoder = getOrCreateSession(environment, textEncoderPath, sessionOptions);
            OrtSession unet = getOrCreateSession(environment, unetPath, sessionOptions);
            OrtSession vaeDecoder = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);
            {

                request.reportProgress("Encoding text prompt\u2026");
                float[][][] textEmbeddings = runTextEncoder(environment, textEncoder, promptTokens);
                float[][][] negativeEmbeddings = runTextEncoder(environment, textEncoder, negativeTokens);
                float[][][] encoderHidden = new float[2][textEmbeddings[0].length][textEmbeddings[0][0].length];
                encoderHidden[0] = negativeEmbeddings[0];
                encoderHidden[1] = textEmbeddings[0];

                float[][][][] latents = randomLatents(request.seed(), latentHeight, latentWidth);
                float[] alphaCumprod = loadAlphaCumprod(schedulerPath);
                int[] timesteps = createTimesteps(steps, alphaCumprod.length);

                request.reportProgress("Denoising: 0/" + steps + " steps \u2014 EP: " + provider);
                long stepStartTime = System.currentTimeMillis();
                long firstStepDuration = 0;
                for (int stepIndex = 0; stepIndex < timesteps.length; stepIndex++) {
                    int timestep = timesteps[stepIndex];
                    int prevTimestep = stepIndex == timesteps.length - 1 ? 0 : timesteps[stepIndex + 1];

                    float[][][][] latentModelInput = duplicateBatch(latents);
                    Map<String, OnnxTensor> unetInputs = new HashMap<>();
                    OnnxTensor sampleTensor = OnnxTensor.createTensor(environment, latentModelInput);
                    OnnxTensor timestepTensor = createTimestepTensor(environment, unet, timestep);
                    OnnxTensor hiddenTensor = OnnxTensor.createTensor(environment, encoderHidden);
                    unetInputs.put(resolveInputName(unet, "sample", 0), sampleTensor);
                    unetInputs.put(resolveInputName(unet, "timestep", 1), timestepTensor);
                    unetInputs.put(resolveInputName(unet, "encoder_hidden_states", 2), hiddenTensor);

                    float[][][][] noise;
                    try (OrtSession.Result unetResult = unet.run(unetInputs)) {
                        noise = extractTensor4d(unetResult);
                    } finally {
                        sampleTensor.close();
                        timestepTensor.close();
                        hiddenTensor.close();
                    }

                    if (noise == null || noise.length < 2) {
                        return InferenceResult.fail("UNet output is invalid for Stable Diffusion generation.");
                    }

                    float[][][] guidedNoise = guidance(noise[0], noise[1], guidanceScale);
                    latents = ddimStep(latents, guidedNoise, alphaCumprod[timestep], alphaCumprod[prevTimestep]);

                    long elapsed = System.currentTimeMillis() - stepStartTime;
                    stepStartTime = System.currentTimeMillis();
                    if (stepIndex == 0) { firstStepDuration = elapsed; }
                    int remaining = steps - (stepIndex + 1);
                    long avgMs = (stepIndex == 0) ? firstStepDuration : elapsed;
                    long etaSec = (remaining * avgMs) / 1000;
                    String eta = etaSec > 60
                            ? String.format("%dm %02ds", etaSec / 60, etaSec % 60)
                            : etaSec + "s";
                    request.reportProgress("Denoising: " + (stepIndex + 1) + "/" + steps
                            + " steps (" + String.format("%.1f", elapsed / 1000.0) + "s/step, ETA: " + eta + ")");

                    if (request.isCancelled()) {
                        return InferenceResult.fail("Cancelled by user.");
                    }
                }

                request.reportProgress("Decoding latents with VAE\u2026");
                float[][][][] scaledLatents = scaleLatents(latents, 1f / 0.18215f);
                Map<String, OnnxTensor> vaeInputs = new HashMap<>();
                OnnxTensor latentTensor = OnnxTensor.createTensor(environment, scaledLatents);
                vaeInputs.put(resolveInputName(vaeDecoder, "latent", 0), latentTensor);

                float[][][][] decoded;
                try (OrtSession.Result vaeResult = vaeDecoder.run(vaeInputs)) {
                    decoded = extractTensor4d(vaeResult);
                } finally {
                    latentTensor.close();
                }

                if (decoded == null || decoded.length == 0) {
                    return InferenceResult.fail("VAE decoder output is empty.");
                }

                BufferedImage image = tensorToImage(decoded[0]);
                Path outputPath = writeOutputImage(image, "sd-v15");
                String details = "Stable Diffusion v1.5 pipeline completed | EP=" + provider;
                String output = "Generated image for prompt: \"" + request.prompt() + "\"";
                return InferenceResult.ok(output, details, outputPath.toString(), "image");
            }
        } catch (Exception ex) {
            return InferenceResult.fail("Stable Diffusion v1.5 pipeline failed: " + ex.getMessage());
        }
    }

    /* ================================================================== */
    /*  SD Turbo — distilled 1–4 step pipeline (no classifier-free        */
    /*  guidance, uses Euler-based single-step scheduler).                 */
    /* ================================================================== */

    private InferenceResult runStableDiffusionTurbo(OrtEnvironment environment,
                                                    OrtSession.SessionOptions sessionOptions,
                                                    InferenceRequest request,
                                                    String provider) {
        try {
            int width  = Math.max(256, (request.width()  / 8) * 8);
            int height = Math.max(256, (request.height() / 8) * 8);
            int latentWidth  = width  / 8;
            int latentHeight = height / 8;
            int steps = Math.max(1, Math.min(request.batch() > 0 ? request.batch() : 4, 8));

            Path base = storage.root().resolve("text-image").resolve("sd-turbo");
            Path textEncoderPath = base.resolve("text_encoder/model.onnx");
            Path unetPath        = base.resolve("unet/model.onnx");
            Path vaeDecoderPath  = base.resolve("vae_decoder/model.onnx");

            // SD Turbo shares the same vocabulary as SD 1.x — reuse from SD v1.5 if present,
            // otherwise look for them inside the sd-turbo directory.
            Path vocabPath   = resolveTokenizerFile(base, "tokenizer/vocab.json");
            Path mergesPath  = resolveTokenizerFile(base, "tokenizer/merges.txt");

            for (Path p : List.of(textEncoderPath, unetPath, vaeDecoderPath, vocabPath, mergesPath)) {
                if (!java.nio.file.Files.exists(p)) {
                    return InferenceResult.fail("SD Turbo bundle is incomplete (missing " + p.getFileName()
                            + "). Open Models → Model Manager and download all SD Turbo components. "
                            + "Tokenizer files also need to be present.");
                }
            }

            ClipTokenizer tokenizer = getOrCreateTokenizer(vocabPath, mergesPath);
            long[] promptTokens = tokenizer.encode(request.prompt(), 77);

            request.reportProgress("Loading SD Turbo models (text encoder, UNet, VAE decoder)\u2026");
            OrtSession textEncoder = getOrCreateSession(environment, textEncoderPath, sessionOptions);
            OrtSession unet = getOrCreateSession(environment, unetPath, sessionOptions);
            OrtSession vaeDecoder = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);
            {

                request.reportProgress("Encoding text prompt\u2026");
                float[][][] textEmbeddings = runTextEncoder(environment, textEncoder, promptTokens);

                // SD Turbo does NOT use classifier-free guidance — single batch only.
                float[][][][] latents = randomLatents(request.seed(), latentHeight, latentWidth);

                // Euler-style timestep schedule for turbo distillation.
                int[] timesteps = turboTimesteps(steps);

                request.reportProgress("Denoising: 0/" + steps + " steps (SD Turbo) — EP: " + provider);
                long stepStart = System.currentTimeMillis();
                for (int i = 0; i < timesteps.length; i++) {
                    int t = timesteps[i];

                    OnnxTensor sampleTensor   = OnnxTensor.createTensor(environment, latents);
                    OnnxTensor timestepTensor  = createTimestepTensor(environment, unet, t);
                    OnnxTensor hiddenTensor    = OnnxTensor.createTensor(environment, textEmbeddings);
                    Map<String, OnnxTensor> unetInputs = new HashMap<>();
                    unetInputs.put(resolveInputName(unet, "sample", 0), sampleTensor);
                    unetInputs.put(resolveInputName(unet, "timestep", 1), timestepTensor);
                    unetInputs.put(resolveInputName(unet, "encoder_hidden_states", 2), hiddenTensor);

                    float[][][][] noise;
                    try (OrtSession.Result unetResult = unet.run(unetInputs)) {
                        noise = extractTensor4d(unetResult);
                    } finally {
                        sampleTensor.close();
                        timestepTensor.close();
                        hiddenTensor.close();
                    }

                    if (noise == null || noise.length == 0) {
                        return InferenceResult.fail("SD Turbo UNet produced invalid output.");
                    }

                    // Euler step: x_{t-1} = x_t - sigma * noise_pred
                    float sigma = turboSigma(t);
                    float sigmaPrev = (i + 1 < timesteps.length) ? turboSigma(timesteps[i + 1]) : 0f;
                    latents = eulerStep(latents, noise[0], sigma, sigmaPrev);

                    long elapsed = System.currentTimeMillis() - stepStart;
                    stepStart = System.currentTimeMillis();
                    int remaining = steps - (i + 1);
                    long eta = remaining * elapsed / 1000;
                    request.reportProgress("Denoising: " + (i + 1) + "/" + steps
                            + " steps (" + String.format("%.1f", elapsed / 1000.0) + "s/step, ETA: " + eta + "s)");

                    if (request.isCancelled()) {
                        return InferenceResult.fail("Cancelled by user.");
                    }
                }

                request.reportProgress("Decoding latents with VAE\u2026");
                float[][][][] scaledLatents = scaleLatents(latents, 1f / 0.18215f);
                OnnxTensor latentTensor = OnnxTensor.createTensor(environment, scaledLatents);
                Map<String, OnnxTensor> vaeInputs = new HashMap<>();
                vaeInputs.put(resolveInputName(vaeDecoder, "latent", 0), latentTensor);

                float[][][][] decoded;
                try (OrtSession.Result vaeResult = vaeDecoder.run(vaeInputs)) {
                    decoded = extractTensor4d(vaeResult);
                } finally {
                    latentTensor.close();
                }

                if (decoded == null || decoded.length == 0) {
                    return InferenceResult.fail("VAE decoder output is empty.");
                }

                BufferedImage image = tensorToImage(decoded[0]);
                Path outputPath = writeOutputImage(image, "sd-turbo");
                return InferenceResult.ok(
                        "Generated image for prompt: \"" + request.prompt() + "\"",
                        "SD Turbo pipeline completed (" + steps + " steps) | EP=" + provider,
                        outputPath.toString(), "image");
            }
        } catch (Exception ex) {
            return InferenceResult.fail("SD Turbo pipeline failed: " + ex.getMessage());
        }
    }

    /** Resolve tokenizer file — try sd-turbo dir first, then fall back to sd-v1.5 dir. */
    private Path resolveTokenizerFile(Path turboBase, String relativeName) {
        Path turboPath = turboBase.resolve(relativeName);
        if (java.nio.file.Files.exists(turboPath)) { return turboPath; }
        Path v15Path = storage.root().resolve("text-image").resolve("stable-diffusion-v15").resolve(relativeName);
        if (java.nio.file.Files.exists(v15Path)) { return v15Path; }
        return turboPath; // will trigger missing-file error
    }

    /** Generate evenly-spaced timesteps for turbo distillation (1000 → 0 in `steps` jumps). */
    private static int[] turboTimesteps(int steps) {
        int[] ts = new int[steps];
        for (int i = 0; i < steps; i++) {
            ts[i] = (int) (999.0 * (steps - 1 - i) / Math.max(1, steps - 1));
        }
        if (steps == 1) { ts[0] = 999; }
        return ts;
    }

    /** Approximate sigma from timestep for the turbo scheduler. */
    private static float turboSigma(int timestep) {
        // SD Turbo uses ~linear sigma schedule from sqrt(1-alpha_bar) / sqrt(alpha_bar)
        float t = timestep / 999.0f;
        float alphaBar = (float) Math.exp(-0.5 * t * t * 12.0); // approximation
        return (float) Math.sqrt((1 - alphaBar) / alphaBar);
    }

    /** Euler step: x_{t-1} = x_t + (sigma_prev - sigma) * noise_pred (scaled). */
    private static float[][][][] eulerStep(float[][][][] latents, float[][][] noisePred,
                                           float sigma, float sigmaPrev) {
        int ch = latents[0].length, h = latents[0][0].length, w = latents[0][0][0].length;
        float[][][][] out = new float[1][ch][h][w];
        float dt = sigmaPrev - sigma;
        for (int c = 0; c < ch; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    out[0][c][y][x] = latents[0][c][y][x] + dt * noisePred[c][y][x];
                }
            }
        }
        return out;
    }

    /* ================================================================== */
    /*  SDXL Turbo — dual text encoders (CLIP-L + OpenCLIP-bigG),         */
    /*  1–4 step distilled pipeline, 512×512 native.                      */
    /* ================================================================== */

    private InferenceResult runSdxlTurbo(OrtEnvironment environment,
                                         OrtSession.SessionOptions sessionOptions,
                                         InferenceRequest request,
                                         String provider) {
        try {
            int width  = Math.max(256, (request.width()  / 8) * 8);
            int height = Math.max(256, (request.height() / 8) * 8);
            int latentW = width  / 8;
            int latentH = height / 8;
            int steps = Math.max(1, Math.min(request.batch() > 0 ? request.batch() : 4, 8));

            Path base = storage.root().resolve("text-image").resolve("sdxl-turbo");
            Path textEncoder1Path = base.resolve("text_encoder/model.onnx");
            Path textEncoder2Path = base.resolve("text_encoder_2/model.onnx");
            Path unetPath         = base.resolve("unet/model.onnx");
            Path vaeDecoderPath   = base.resolve("vae_decoder/model.onnx");
            Path vocab1 = base.resolve("tokenizer/vocab.json");
            Path merges1 = base.resolve("tokenizer/merges.txt");
            Path vocab2 = base.resolve("tokenizer_2/vocab.json");
            Path merges2 = base.resolve("tokenizer_2/merges.txt");

            for (Path p : List.of(textEncoder1Path, textEncoder2Path, unetPath, vaeDecoderPath,
                    vocab1, merges1, vocab2, merges2)) {
                if (!java.nio.file.Files.exists(p)) {
                    return InferenceResult.fail("SDXL Turbo bundle is incomplete (missing "
                            + p.getFileName() + "). Open Models \u2192 Model Manager and download SDXL Turbo.");
                }
            }

            ClipTokenizer tok1 = getOrCreateTokenizer(vocab1, merges1);
            ClipTokenizer tok2 = getOrCreateTokenizer(vocab2, merges2);
            long[] tokens1 = tok1.encode(request.prompt(), 77);
            long[] tokens2 = tok2.encode(request.prompt(), 77);

            request.reportProgress("Loading SDXL Turbo models (2 text encoders, UNet, VAE)\u2026");
            OrtSession enc1 = getOrCreateSession(environment, textEncoder1Path, sessionOptions);
            OrtSession enc2 = getOrCreateSession(environment, textEncoder2Path, sessionOptions);
            OrtSession unet = getOrCreateSession(environment, unetPath, sessionOptions);
            OrtSession vae  = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);
            {

                request.reportProgress("Encoding text (CLIP-L)\u2026");
                float[][][] embed1 = runTextEncoder(environment, enc1, tokens1); // [1, 77, 768]

                request.reportProgress("Encoding text (OpenCLIP-bigG)\u2026");
                // text_encoder_2 outputs both hidden_states [1, 77, 1280] and pooled [1, 1280]
                float[][][] embed2;
                float[][] pooledOutput;
                {
                    String inName = resolveInputName(enc2, "input_ids", 0);
                    TensorInfo tInfo = (TensorInfo) enc2.getInputInfo().get(inName).getInfo();
                    boolean wantsInt32 = tInfo.type.toString().contains("INT32");
                    OnnxTensor idsTensor;
                    if (wantsInt32) {
                        int[] ids32 = new int[tokens2.length];
                        for (int i = 0; i < tokens2.length; i++) { ids32[i] = (int) tokens2[i]; }
                        idsTensor = OnnxTensor.createTensor(environment, new int[][]{ids32});
                    } else {
                        idsTensor = OnnxTensor.createTensor(environment, new long[][]{tokens2});
                    }
                    Map<String, OnnxTensor> inputs = new HashMap<>();
                    inputs.put(inName, idsTensor);
                    try (OrtSession.Result r = enc2.run(inputs)) {
                        embed2 = null;
                        pooledOutput = null;
                        for (Map.Entry<String, OnnxValue> entry : r) {
                            if (entry.getValue() instanceof OnnxTensor t) {
                                Object v = t.getValue();
                                if (v instanceof float[][][] arr3d && embed2 == null) {
                                    embed2 = arr3d;
                                } else if (v instanceof float[][] arr2d && pooledOutput == null) {
                                    pooledOutput = arr2d;
                                }
                            }
                        }
                        if (embed2 == null) {
                            return InferenceResult.fail("Text encoder 2 produced no hidden states.");
                        }
                        if (pooledOutput == null) {
                            // Fallback: use zeros for pooled output
                            pooledOutput = new float[1][1280];
                        }
                    } finally {
                        idsTensor.close();
                    }
                }

                // Concatenate embeddings: [1, 77, 768] + [1, 77, 1280] → [1, 77, 2048]
                int seqLen = embed1[0].length;
                int dim1 = embed1[0][0].length;
                int dim2 = embed2[0][0].length;
                float[][][] combined = new float[1][seqLen][dim1 + dim2];
                for (int s = 0; s < seqLen; s++) {
                    System.arraycopy(embed1[0][s], 0, combined[0][s], 0, dim1);
                    System.arraycopy(embed2[0][s], 0, combined[0][s], dim1, dim2);
                }

                // time_ids: [original_h, original_w, crop_y, crop_x, target_h, target_w]
                float[][] timeIds = {{height, width, 0, 0, height, width}};

                float[][][][] latents = randomLatents(request.seed(), latentH, latentW);
                int[] timesteps = turboTimesteps(steps);

                request.reportProgress("Denoising: 0/" + steps + " steps (SDXL Turbo) \u2014 EP: " + provider);
                long stepStart = System.currentTimeMillis();
                for (int i = 0; i < timesteps.length; i++) {
                    int t = timesteps[i];

                    OnnxTensor sampleT   = OnnxTensor.createTensor(environment, latents);
                    OnnxTensor tsT       = createTimestepTensor(environment, unet, t);
                    OnnxTensor hiddenT   = OnnxTensor.createTensor(environment, combined);
                    OnnxTensor embedsT   = OnnxTensor.createTensor(environment, pooledOutput);
                    OnnxTensor timeIdsT  = OnnxTensor.createTensor(environment, timeIds);

                    Map<String, OnnxTensor> unetInputs = new HashMap<>();
                    unetInputs.put(resolveInputName(unet, "sample", 0), sampleT);
                    unetInputs.put(resolveInputName(unet, "timestep", 1), tsT);
                    unetInputs.put(resolveInputName(unet, "encoder_hidden_states", 2), hiddenT);
                    // SDXL UNet has additional inputs: text_embeds and time_ids
                    for (String name : unet.getInputNames()) {
                        if (name.contains("text_embeds") || name.contains("added_cond_kwargs.text_embeds")) {
                            unetInputs.put(name, embedsT);
                        } else if (name.contains("time_ids") || name.contains("added_cond_kwargs.time_ids")) {
                            unetInputs.put(name, timeIdsT);
                        }
                    }

                    float[][][][] noise;
                    try (OrtSession.Result unetResult = unet.run(unetInputs)) {
                        noise = extractTensor4d(unetResult);
                    } finally {
                        sampleT.close(); tsT.close(); hiddenT.close(); embedsT.close(); timeIdsT.close();
                    }

                    if (noise == null || noise.length == 0) {
                        return InferenceResult.fail("SDXL Turbo UNet produced invalid output.");
                    }

                    float sigma = turboSigma(t);
                    float sigmaPrev = (i + 1 < timesteps.length) ? turboSigma(timesteps[i + 1]) : 0f;
                    latents = eulerStep(latents, noise[0], sigma, sigmaPrev);

                    long elapsed = System.currentTimeMillis() - stepStart;
                    stepStart = System.currentTimeMillis();
                    request.reportProgress("Denoising: " + (i + 1) + "/" + steps
                            + " steps (" + String.format("%.1f", elapsed / 1000.0) + "s/step)");

                    if (request.isCancelled()) {
                        return InferenceResult.fail("Cancelled by user.");
                    }
                }

                request.reportProgress("Decoding latents with VAE\u2026");
                float[][][][] scaledLatents = scaleLatents(latents, 1f / 0.18215f);
                OnnxTensor latTensor = OnnxTensor.createTensor(environment, scaledLatents);
                Map<String, OnnxTensor> vaeIn = new HashMap<>();
                vaeIn.put(resolveInputName(vae, "latent", 0), latTensor);

                float[][][][] decoded;
                try (OrtSession.Result vaeResult = vae.run(vaeIn)) {
                    decoded = extractTensor4d(vaeResult);
                } finally {
                    latTensor.close();
                }

                if (decoded == null || decoded.length == 0) {
                    return InferenceResult.fail("VAE decoder output is empty.");
                }

                BufferedImage image = tensorToImage(decoded[0]);
                Path outputPath = writeOutputImage(image, "sdxl-turbo");
                return InferenceResult.ok(
                        "Generated image for prompt: \"" + request.prompt() + "\"",
                        "SDXL Turbo pipeline completed (" + steps + " steps) | EP=" + provider,
                        outputPath.toString(), "image");
            }
        } catch (Exception ex) {
            return InferenceResult.fail("SDXL Turbo pipeline failed: " + ex.getMessage());
        }
    }

    /* ================================================================== */
    /*  SDXL Base 1.0 — full SDXL with classifier-free guidance,          */
    /*  dual text encoders (CLIP-L + OpenCLIP-bigG), Euler Discrete       */
    /*  scheduler, 1024×1024 native resolution.                           */
    /* ================================================================== */

    private InferenceResult runSdxlBase(OrtEnvironment environment,
                                        OrtSession.SessionOptions sessionOptions,
                                        InferenceRequest request,
                                        String provider) {
        try {
            int width  = Math.max(512, (request.width()  / 8) * 8);
            int height = Math.max(512, (request.height() / 8) * 8);
            int latentW = width  / 8;
            int latentH = height / 8;
            int steps = Math.max(5, Math.min(request.batch() > 0 ? request.batch() : 30, 50));
            float guidanceScale = request.promptWeight() > 0 ? (float) request.promptWeight() : 7.5f;

            Path base = storage.root().resolve("text-image").resolve("sdxl-base");
            Path textEncoder1Path = base.resolve("text_encoder/model.onnx");
            Path textEncoder2Path = base.resolve("text_encoder_2/model.onnx");
            Path unetPath         = base.resolve("unet/model.onnx");
            Path vaeDecoderPath   = base.resolve("vae_decoder/model.onnx");
            Path schedulerConfig  = base.resolve("scheduler/scheduler_config.json");
            Path vocab1 = base.resolve("tokenizer/vocab.json");
            Path merges1 = base.resolve("tokenizer/merges.txt");
            Path vocab2 = base.resolve("tokenizer_2/vocab.json");
            Path merges2 = base.resolve("tokenizer_2/merges.txt");

            for (Path p : List.of(textEncoder1Path, textEncoder2Path, unetPath, vaeDecoderPath,
                    vocab1, merges1, vocab2, merges2)) {
                if (!java.nio.file.Files.exists(p)) {
                    return InferenceResult.fail("SDXL Base bundle is incomplete (missing "
                            + p.getFileName() + "). Open Models \u2192 Model Manager and download SDXL Base 1.0.");
                }
            }

            ClipTokenizer tok1 = getOrCreateTokenizer(vocab1, merges1);
            ClipTokenizer tok2 = getOrCreateTokenizer(vocab2, merges2);
            long[] tokens1 = tok1.encode(request.prompt(), 77);
            long[] tokens2 = tok2.encode(request.prompt(), 77);
            long[] negTokens1 = tok1.encode(request.negativePrompt() == null ? "" : request.negativePrompt(), 77);
            long[] negTokens2 = tok2.encode(request.negativePrompt() == null ? "" : request.negativePrompt(), 77);

            request.reportProgress("Loading SDXL Base models (2 text encoders, UNet, VAE)\u2026");
            OrtSession enc1 = getOrCreateSession(environment, textEncoder1Path, sessionOptions);
            OrtSession enc2 = getOrCreateSession(environment, textEncoder2Path, sessionOptions);
            OrtSession unet = getOrCreateSession(environment, unetPath, sessionOptions);
            OrtSession vae  = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);
            {
                // ── Encode positive prompt with both encoders ──
                request.reportProgress("Encoding positive prompt (CLIP-L)\u2026");
                float[][][] embed1 = runTextEncoder(environment, enc1, tokens1);

                request.reportProgress("Encoding positive prompt (OpenCLIP-bigG)\u2026");
                float[][][] embed2;
                float[][] pooledPos;
                {
                    String inName = resolveInputName(enc2, "input_ids", 0);
                    TensorInfo tInfo = (TensorInfo) enc2.getInputInfo().get(inName).getInfo();
                    boolean wantsInt32 = tInfo.type.toString().contains("INT32");
                    OnnxTensor idsTensor;
                    if (wantsInt32) {
                        int[] ids32 = new int[tokens2.length];
                        for (int i = 0; i < tokens2.length; i++) { ids32[i] = (int) tokens2[i]; }
                        idsTensor = OnnxTensor.createTensor(environment, new int[][]{ids32});
                    } else {
                        idsTensor = OnnxTensor.createTensor(environment, new long[][]{tokens2});
                    }
                    Map<String, OnnxTensor> inputs = new HashMap<>();
                    inputs.put(inName, idsTensor);
                    try (OrtSession.Result r = enc2.run(inputs)) {
                        embed2 = null; pooledPos = null;
                        for (Map.Entry<String, OnnxValue> entry : r) {
                            if (entry.getValue() instanceof OnnxTensor t) {
                                Object v = t.getValue();
                                if (v instanceof float[][][] a3 && embed2 == null) { embed2 = a3; }
                                else if (v instanceof float[][] a2 && pooledPos == null) { pooledPos = a2; }
                            }
                        }
                        if (embed2 == null) { return InferenceResult.fail("Text encoder 2 produced no hidden states."); }
                        if (pooledPos == null) { pooledPos = new float[1][1280]; }
                    } finally { idsTensor.close(); }
                }

                // ── Encode negative prompt with both encoders ──
                request.reportProgress("Encoding negative prompt\u2026");
                float[][][] negEmbed1 = runTextEncoder(environment, enc1, negTokens1);
                float[][][] negEmbed2;
                float[][] pooledNeg;
                {
                    String inName = resolveInputName(enc2, "input_ids", 0);
                    TensorInfo tInfo = (TensorInfo) enc2.getInputInfo().get(inName).getInfo();
                    boolean wantsInt32 = tInfo.type.toString().contains("INT32");
                    OnnxTensor idsTensor;
                    if (wantsInt32) {
                        int[] ids32 = new int[negTokens2.length];
                        for (int i = 0; i < negTokens2.length; i++) { ids32[i] = (int) negTokens2[i]; }
                        idsTensor = OnnxTensor.createTensor(environment, new int[][]{ids32});
                    } else {
                        idsTensor = OnnxTensor.createTensor(environment, new long[][]{negTokens2});
                    }
                    Map<String, OnnxTensor> inputs = new HashMap<>();
                    inputs.put(inName, idsTensor);
                    try (OrtSession.Result r = enc2.run(inputs)) {
                        negEmbed2 = null; pooledNeg = null;
                        for (Map.Entry<String, OnnxValue> entry : r) {
                            if (entry.getValue() instanceof OnnxTensor t) {
                                Object v = t.getValue();
                                if (v instanceof float[][][] a3 && negEmbed2 == null) { negEmbed2 = a3; }
                                else if (v instanceof float[][] a2 && pooledNeg == null) { pooledNeg = a2; }
                            }
                        }
                        if (negEmbed2 == null) { return InferenceResult.fail("Text encoder 2 negative produced no hidden states."); }
                        if (pooledNeg == null) { pooledNeg = new float[1][1280]; }
                    } finally { idsTensor.close(); }
                }

                // Concatenate embeddings: [1, 77, 768] + [1, 77, 1280] → [1, 77, 2048]
                int seqLen = embed1[0].length;
                int dim1 = embed1[0][0].length;
                int dim2 = embed2[0][0].length;
                float[][][] combinedPos = new float[1][seqLen][dim1 + dim2];
                float[][][] combinedNeg = new float[1][seqLen][dim1 + dim2];
                for (int s = 0; s < seqLen; s++) {
                    System.arraycopy(embed1[0][s], 0, combinedPos[0][s], 0, dim1);
                    System.arraycopy(embed2[0][s], 0, combinedPos[0][s], dim1, dim2);
                    System.arraycopy(negEmbed1[0][s], 0, combinedNeg[0][s], 0, dim1);
                    System.arraycopy(negEmbed2[0][s], 0, combinedNeg[0][s], dim1, dim2);
                }
                // Batch embeddings: [2, 77, 2048] — negative first, positive second
                float[][][] batchedHidden = new float[2][seqLen][dim1 + dim2];
                batchedHidden[0] = combinedNeg[0];
                batchedHidden[1] = combinedPos[0];

                // Batch pooled: [2, 1280]
                int poolDim = pooledPos[0].length;
                float[][] batchedPooled = new float[2][poolDim];
                System.arraycopy(pooledNeg[0], 0, batchedPooled[0], 0, poolDim);
                System.arraycopy(pooledPos[0], 0, batchedPooled[1], 0, poolDim);

                // time_ids: [original_h, original_w, crop_y, crop_x, target_h, target_w] — batched [2, 6]
                float[][] timeIds = {
                    {height, width, 0, 0, height, width},
                    {height, width, 0, 0, height, width}
                };

                // ── Noise schedule from scheduler config ──
                float[] alphaCumprod;
                if (java.nio.file.Files.exists(schedulerConfig)) {
                    alphaCumprod = loadAlphaCumprod(schedulerConfig);
                } else {
                    // Compute default SDXL schedule inline
                    alphaCumprod = computeDefaultAlphaCumprod(1000, 0.00085, 0.012);
                }
                int[] timesteps = createTimesteps(steps, alphaCumprod.length);

                // Convert alphas to sigmas for Euler Discrete
                float[] sigmas = new float[timesteps.length + 1];
                for (int i = 0; i < timesteps.length; i++) {
                    float acp = alphaCumprod[Math.min(timesteps[i], alphaCumprod.length - 1)];
                    sigmas[i] = (float) Math.sqrt((1.0 - acp) / acp);
                }
                sigmas[timesteps.length] = 0f; // final sigma

                float[][][][] latents = randomLatents(request.seed(), latentH, latentW);
                // Scale initial noise by first sigma
                latents = scaleLatents(latents, sigmas[0]);

                request.reportProgress("Denoising: 0/" + steps + " steps (SDXL Base) \u2014 EP: " + provider);
                long stepStart = System.currentTimeMillis();
                long firstStepDuration = 0;
                for (int i = 0; i < timesteps.length; i++) {
                    int t = timesteps[i];
                    float sigma = sigmas[i];
                    float sigmaPrev = sigmas[i + 1];

                    // Scale model input: sample / sqrt(sigma^2 + 1)
                    float scaleFactor = (float) (1.0 / Math.sqrt(sigma * sigma + 1.0));
                    float[][][][] scaledInput = scaleLatents(latents, scaleFactor);

                    // Duplicate for CFG batch [2, 4, H, W]
                    float[][][][] batchInput = duplicateBatch(scaledInput);

                    OnnxTensor sampleT   = OnnxTensor.createTensor(environment, batchInput);
                    OnnxTensor tsT       = createTimestepTensor(environment, unet, t);
                    OnnxTensor hiddenT   = OnnxTensor.createTensor(environment, batchedHidden);
                    OnnxTensor embedsT   = OnnxTensor.createTensor(environment, batchedPooled);
                    OnnxTensor timeIdsT  = OnnxTensor.createTensor(environment, timeIds);

                    Map<String, OnnxTensor> unetInputs = new HashMap<>();
                    unetInputs.put(resolveInputName(unet, "sample", 0), sampleT);
                    unetInputs.put(resolveInputName(unet, "timestep", 1), tsT);
                    unetInputs.put(resolveInputName(unet, "encoder_hidden_states", 2), hiddenT);
                    for (String name : unet.getInputNames()) {
                        if (name.contains("text_embeds") || name.contains("added_cond_kwargs.text_embeds")) {
                            unetInputs.put(name, embedsT);
                        } else if (name.contains("time_ids") || name.contains("added_cond_kwargs.time_ids")) {
                            unetInputs.put(name, timeIdsT);
                        }
                    }

                    float[][][][] noise;
                    try (OrtSession.Result unetResult = unet.run(unetInputs)) {
                        noise = extractTensor4d(unetResult);
                    } finally {
                        sampleT.close(); tsT.close(); hiddenT.close(); embedsT.close(); timeIdsT.close();
                    }

                    if (noise == null || noise.length < 2) {
                        return InferenceResult.fail("SDXL Base UNet produced invalid output.");
                    }

                    // Classifier-free guidance
                    float[][][] guidedNoise = guidance(noise[0], noise[1], guidanceScale);

                    // Euler step
                    latents = eulerStep(latents, guidedNoise, sigma, sigmaPrev);

                    long elapsed = System.currentTimeMillis() - stepStart;
                    stepStart = System.currentTimeMillis();
                    if (i == 0) { firstStepDuration = elapsed; }
                    int remaining = steps - (i + 1);
                    long avgMs = (i == 0) ? firstStepDuration : elapsed;
                    long etaSec = (remaining * avgMs) / 1000;
                    String eta = etaSec > 60
                            ? String.format("%dm %02ds", etaSec / 60, etaSec % 60)
                            : etaSec + "s";
                    request.reportProgress("Denoising: " + (i + 1) + "/" + steps
                            + " steps (" + String.format("%.1f", elapsed / 1000.0) + "s/step, ETA: " + eta + ")");

                    if (request.isCancelled()) {
                        return InferenceResult.fail("Cancelled by user.");
                    }
                }

                // SDXL VAE uses 0.13025 scaling factor
                request.reportProgress("Decoding latents with VAE\u2026");
                float[][][][] scaledLatents = scaleLatents(latents, 1f / 0.13025f);
                OnnxTensor latTensor = OnnxTensor.createTensor(environment, scaledLatents);
                Map<String, OnnxTensor> vaeIn = new HashMap<>();
                vaeIn.put(resolveInputName(vae, "latent", 0), latTensor);

                float[][][][] decoded;
                try (OrtSession.Result vaeResult = vae.run(vaeIn)) {
                    decoded = extractTensor4d(vaeResult);
                } finally {
                    latTensor.close();
                }

                if (decoded == null || decoded.length == 0) {
                    return InferenceResult.fail("VAE decoder output is empty.");
                }

                BufferedImage image = tensorToImage(decoded[0]);
                Path outputPath = writeOutputImage(image, "sdxl-base");
                return InferenceResult.ok(
                        "Generated image for prompt: \"" + request.prompt() + "\"",
                        "SDXL Base 1.0 pipeline completed (" + steps + " steps, CFG=" + guidanceScale + ") | EP=" + provider,
                        outputPath.toString(), "image");
            }
        } catch (Exception ex) {
            return InferenceResult.fail("SDXL Base pipeline failed: " + ex.getMessage());
        }
    }

    /* ================================================================== */
    /*  Image-to-Image / Inpainting (SD v1.5 or SD Turbo ONNX)           */
    /* ================================================================== */

    private InferenceResult runImg2Img(OrtEnvironment environment,
                                       OrtSession.SessionOptions sessionOptions,
                                       InferenceRequest request,
                                       String provider,
                                       boolean turboMode) {
        try {
            String inputPath = request.inputImagePath();
            if (inputPath == null || inputPath.isBlank()) {
                return InferenceResult.fail("Img2Img requires an input image. Choose an image first.");
            }

            BufferedImage inputImage = ImageIO.read(new File(inputPath));
            if (inputImage == null) {
                return InferenceResult.fail("Cannot read image: " + inputPath);
            }

            int width  = Math.max(256, (request.width()  / 8) * 8);
            int height = Math.max(256, (request.height() / 8) * 8);
            int latentW = width  / 8;
            int latentH = height / 8;
            double strength = Math.max(0.01, Math.min(1.0, request.strength()));
            double guidanceScale = request.promptWeight() > 0 ? request.promptWeight() : 7.5;

            String baseDir = turboMode ? "text-image/sd-turbo" : "text-image/stable-diffusion-v15";
            Path base = storage.root().resolve(baseDir);
            Path textEncoderPath = base.resolve("text_encoder/model.onnx");
            Path unetPath        = base.resolve("unet/model.onnx");
            Path vaeDecoderPath  = base.resolve("vae_decoder/model.onnx");

            // VAE encoder: prefer local bundle, fall back to SD v1.5's encoder
            Path vaeEncoderPath = base.resolve("vae_encoder/model.onnx");
            if (!java.nio.file.Files.exists(vaeEncoderPath)) {
                vaeEncoderPath = storage.root()
                        .resolve("text-image/stable-diffusion-v15/vae_encoder/model.onnx");
            }

            Path vocab = resolveTokenizerFile(base, "tokenizer/vocab.json");
            Path merges = resolveTokenizerFile(base, "tokenizer/merges.txt");

            for (Path p : List.of(textEncoderPath, unetPath, vaeDecoderPath, vaeEncoderPath, vocab, merges)) {
                if (!java.nio.file.Files.exists(p)) {
                    return InferenceResult.fail("Img2Img bundle is incomplete (missing "
                            + p.getFileName() + "). Download the model + VAE encoder from Model Manager.");
                }
            }

            // Resize input image to target dimensions
            request.reportProgress("Resizing input image to " + width + "\u00d7" + height + "\u2026");
            BufferedImage resized = resizeImage(inputImage, width, height, "lanczos");

            // Load mask if provided (for inpainting)
            BufferedImage maskImage = null;
            float[][][][] maskLatents = null;
            if (request.maskImagePath() != null && !request.maskImagePath().isBlank()) {
                maskImage = ImageIO.read(new File(request.maskImagePath()));
                if (maskImage != null) {
                    maskImage = resizeImage(maskImage, width, height, "bilinear");
                    request.reportProgress("Mask loaded for inpainting.");
                }
            }

            // Convert input image to tensor [1, 3, H, W] normalized to [-1, 1]
            float[][][][] imageTensor = imageToLatentInput(resized);

            ClipTokenizer tokenizer = getOrCreateTokenizer(vocab, merges);
            long[] tokens = tokenizer.encode(request.prompt(), 77);
            long[] uncondTokens = tokenizer.encode("", 77);

            request.reportProgress("Loading models for Img2Img pipeline\u2026");
            OrtSession textEncoder = getOrCreateSession(environment, textEncoderPath, sessionOptions);
            OrtSession unet = getOrCreateSession(environment, unetPath, sessionOptions);
            OrtSession vaeDecoder = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);
            OrtSession vaeEncoder = getOrCreateSession(environment, vaeEncoderPath, sessionOptions);
            {

                // Text encoding
                request.reportProgress("Encoding prompt\u2026");
                float[][][] textEmbed = runTextEncoder(environment, textEncoder, tokens);

                float[][][] promptEmbedding;
                if (turboMode) {
                    // SD Turbo: no classifier-free guidance
                    promptEmbedding = textEmbed;
                    guidanceScale = 0; // flag for no CFG
                } else {
                    float[][][] uncondEmbed = runTextEncoder(environment, textEncoder, uncondTokens);
                    // Concatenate [uncond, cond] → [2, 77, 768]
                    promptEmbedding = new float[2][textEmbed[0].length][textEmbed[0][0].length];
                    System.arraycopy(uncondEmbed[0], 0, promptEmbedding[0], 0, uncondEmbed[0].length);
                    System.arraycopy(textEmbed[0], 0, promptEmbedding[1], 0, textEmbed[0].length);
                }

                // Encode input image with VAE
                request.reportProgress("Encoding input image with VAE\u2026");
                OnnxTensor imgTensor = OnnxTensor.createTensor(environment, imageTensor);
                Map<String, OnnxTensor> vaeEncIn = new HashMap<>();
                vaeEncIn.put(resolveInputName(vaeEncoder, "sample", 0), imgTensor);
                float[][][][] initLatents;
                try (OrtSession.Result r = vaeEncoder.run(vaeEncIn)) {
                    float[][][][] encoded = extractTensor4d(r);
                    if (encoded == null) {
                        return InferenceResult.fail("VAE encoder produced no output.");
                    }
                    // Scale by VAE scaling factor
                    initLatents = scaleLatents(encoded, 0.18215f);
                } finally {
                    imgTensor.close();
                }

                // Calculate skip steps based on strength
                int totalSteps;
                int skipSteps;
                if (turboMode) {
                    totalSteps = Math.max(1, Math.min(request.batch() > 0 ? request.batch() : 4, 8));
                    skipSteps = Math.max(0, (int) (totalSteps * (1.0 - strength)));
                } else {
                    totalSteps = Math.max(1, Math.min(request.batch() > 0 ? request.batch() : 20, 50));
                    skipSteps = Math.max(0, (int) (totalSteps * (1.0 - strength)));
                }
                int activeSteps = totalSteps - skipSteps;

                // Add noise to initial latents at strength level
                request.reportProgress("Adding noise (strength=" + String.format("%.0f%%", strength * 100) + ")\u2026");
                float[][][][] latents;
                if (turboMode) {
                    int[] timesteps = turboTimesteps(totalSteps);
                    int startTs = timesteps[skipSteps];
                    float sigma = turboSigma(startTs);
                    // noise = random latent × sigma + initLatents
                    float[][][][] noise = randomLatents(request.seed(), latentH, latentW);
                    latents = addNoise(initLatents, noise, sigma);

                    // Create mask in latent space if provided
                    if (maskImage != null) {
                        maskLatents = createLatentMask(maskImage, latentH, latentW);
                    }

                    long stepStart = System.currentTimeMillis();
                    for (int i = skipSteps; i < timesteps.length; i++) {
                        int t = timesteps[i];

                        OnnxTensor sampleT = OnnxTensor.createTensor(environment, latents);
                        OnnxTensor tsT = createTimestepTensor(environment, unet, t);
                        OnnxTensor embedT = OnnxTensor.createTensor(environment, promptEmbedding);

                        Map<String, OnnxTensor> unetInputs = new HashMap<>();
                        unetInputs.put(resolveInputName(unet, "sample", 0), sampleT);
                        unetInputs.put(resolveInputName(unet, "timestep", 1), tsT);
                        unetInputs.put(resolveInputName(unet, "encoder_hidden_states", 2), embedT);

                        float[][][][] noisePred;
                        try (OrtSession.Result unetResult = unet.run(unetInputs)) {
                            noisePred = extractTensor4d(unetResult);
                        } finally {
                            sampleT.close(); tsT.close(); embedT.close();
                        }

                        float sigmaT = turboSigma(t);
                        float sigmaPrev = (i + 1 < timesteps.length) ? turboSigma(timesteps[i + 1]) : 0f;
                        latents = eulerStep(latents, noisePred[0], sigmaT, sigmaPrev);

                        // Apply mask: preserve original latents where mask is black (0)
                        if (maskLatents != null) {
                            latents = applyLatentMask(latents, initLatents, maskLatents);
                        }

                        long elapsed = System.currentTimeMillis() - stepStart;
                        stepStart = System.currentTimeMillis();
                        request.reportProgress("Denoising: " + (i - skipSteps + 1) + "/" + activeSteps
                                + " steps (" + String.format("%.1f", elapsed / 1000.0) + "s/step) [Img2Img]");

                        if (request.isCancelled()) {
                            return InferenceResult.fail("Cancelled by user.");
                        }
                    }
                } else {
                    // SD v1.5 DDIM with img2img
                    Path schedulerConfig = base.resolve("scheduler/scheduler_config.json");
                    float[] alphaCumprod = loadAlphaCumprod(schedulerConfig);
                    int[] fullTimesteps = ddimTimesteps(totalSteps, 1000);
                    int[] timesteps = new int[activeSteps];
                    System.arraycopy(fullTimesteps, skipSteps, timesteps, 0, activeSteps);

                    float startAlpha = alphaCumprod[Math.min(timesteps[0], alphaCumprod.length - 1)];
                    float startSigma = (float) Math.sqrt((1.0 - startAlpha) / startAlpha);
                    float[][][][] noise = randomLatents(request.seed(), latentH, latentW);
                    latents = addNoise(initLatents, noise, startSigma);

                    if (maskImage != null) {
                        maskLatents = createLatentMask(maskImage, latentH, latentW);
                    }

                    long stepStart = System.currentTimeMillis();
                    for (int i = 0; i < timesteps.length; i++) {
                        int t = timesteps[i];
                        float[][][][] doubled = duplicateBatch(latents);
                        OnnxTensor sampleT = OnnxTensor.createTensor(environment, doubled);
                        OnnxTensor tsT = createTimestepTensor(environment, unet, t);
                        OnnxTensor embedT = OnnxTensor.createTensor(environment,
                                new float[][][][]{promptEmbedding});

                        Map<String, OnnxTensor> unetInputs = new HashMap<>();
                        unetInputs.put(resolveInputName(unet, "sample", 0), sampleT);
                        unetInputs.put(resolveInputName(unet, "timestep", 1), tsT);
                        unetInputs.put(resolveInputName(unet, "encoder_hidden_states", 2), embedT);

                        float[][][][] noisePred;
                        try (OrtSession.Result unetResult = unet.run(unetInputs)) {
                            noisePred = extractTensor4d(unetResult);
                        } finally {
                            sampleT.close(); tsT.close(); embedT.close();
                        }

                        // CFG: split batch and guide
                        float[][][] guidedNoise = guidance(noisePred[0], noisePred[1], (float) guidanceScale);

                        // DDIM step
                        int tPrev = (i + 1 < timesteps.length) ? timesteps[i + 1] : 0;
                        float alphaT = alphaCumprod[Math.min(t, alphaCumprod.length - 1)];
                        float alphaPrev = (tPrev > 0) ? alphaCumprod[Math.min(tPrev, alphaCumprod.length - 1)] : 1.0f;
                        latents = ddimStep(latents, guidedNoise, alphaT, alphaPrev);

                        // Apply mask
                        if (maskLatents != null) {
                            // Re-noise initial latents at current noise level for blending
                            float currentSigma = (float) Math.sqrt((1.0 - alphaT) / alphaT);
                            float[][][][] renaised = addNoise(initLatents, noise, currentSigma);
                            latents = applyLatentMask(latents, renaised, maskLatents);
                        }

                        long elapsed = System.currentTimeMillis() - stepStart;
                        stepStart = System.currentTimeMillis();
                        request.reportProgress("Denoising: " + (i + 1) + "/" + activeSteps
                                + " steps (" + String.format("%.1f", elapsed / 1000.0) + "s/step) [Img2Img]");

                        if (request.isCancelled()) {
                            return InferenceResult.fail("Cancelled by user.");
                        }
                    }
                }

                // VAE decode
                request.reportProgress("Decoding latents with VAE\u2026");
                float[][][][] scaledLatents = scaleLatents(latents, 1f / 0.18215f);
                OnnxTensor latTensor = OnnxTensor.createTensor(environment, scaledLatents);
                Map<String, OnnxTensor> vaeDecIn = new HashMap<>();
                vaeDecIn.put(resolveInputName(vaeDecoder, "latent", 0), latTensor);
                float[][][][] decoded;
                try (OrtSession.Result vaeResult = vaeDecoder.run(vaeDecIn)) {
                    decoded = extractTensor4d(vaeResult);
                } finally {
                    latTensor.close();
                }
                if (decoded == null || decoded.length == 0) {
                    return InferenceResult.fail("VAE decoder output is empty.");
                }

                BufferedImage image = tensorToImage(decoded[0]);
                String prefix = turboMode ? "img2img-turbo" : "img2img-sd15";
                Path outputPath = writeOutputImage(image, prefix);
                String mode = (maskImage != null) ? "Inpainting" : "Img2Img";
                return InferenceResult.ok(
                        mode + " result for: \"" + request.prompt() + "\"",
                        mode + " (" + activeSteps + "/" + totalSteps + " steps, strength="
                                + String.format("%.0f%%", strength * 100) + ") | EP=" + provider,
                        outputPath.toString(), "image");
            }
        } catch (Exception ex) {
            return InferenceResult.fail("Img2Img pipeline failed: " + ex.getMessage());
        }
    }

    /** Convert BufferedImage to [1, 3, H, W] tensor normalized to [-1, 1]. */
    private float[][][][] imageToLatentInput(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        float[][][][] tensor = new float[1][3][h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = image.getRGB(x, y);
                tensor[0][0][y][x] = ((rgb >> 16) & 0xFF) / 127.5f - 1.0f;
                tensor[0][1][y][x] = ((rgb >> 8) & 0xFF) / 127.5f - 1.0f;
                tensor[0][2][y][x] = (rgb & 0xFF) / 127.5f - 1.0f;
            }
        }
        return tensor;
    }

    /** Add noise to latents: result = latents + noise × sigma. */
    private float[][][][] addNoise(float[][][][] latents, float[][][][] noise, float sigma) {
        int c = latents[0].length, h = latents[0][0].length, w = latents[0][0][0].length;
        float[][][][] out = new float[1][c][h][w];
        for (int ci = 0; ci < c; ci++)
            for (int yi = 0; yi < h; yi++)
                for (int xi = 0; xi < w; xi++)
                    out[0][ci][yi][xi] = latents[0][ci][yi][xi] + noise[0][ci][yi][xi] * sigma;
        return out;
    }

    /** Downsample mask to latent space [1, 1, latentH, latentW] with values 0..1. */
    private float[][][][] createLatentMask(BufferedImage mask, int latentH, int latentW) {
        // Simple nearest-neighbor downsample; white (255) = repaint area
        int imgW = mask.getWidth(), imgH = mask.getHeight();
        float[][][][] m = new float[1][1][latentH][latentW];
        for (int y = 0; y < latentH; y++) {
            for (int x = 0; x < latentW; x++) {
                int srcX = x * imgW / latentW;
                int srcY = y * imgH / latentH;
                int rgb = mask.getRGB(srcX, srcY);
                // Use luminance; white = 1.0 (repaint), black = 0.0 (keep)
                float lum = (((rgb >> 16) & 0xFF) + ((rgb >> 8) & 0xFF) + (rgb & 0xFF)) / (3f * 255f);
                m[0][0][y][x] = lum;
            }
        }
        return m;
    }

    /** Apply inpainting mask: where mask=1 keep denoised, where mask=0 keep original. */
    private float[][][][] applyLatentMask(float[][][][] denoised, float[][][][] original,
                                          float[][][][] mask) {
        int c = denoised[0].length, h = denoised[0][0].length, w = denoised[0][0][0].length;
        float[][][][] out = new float[1][c][h][w];
        for (int ci = 0; ci < c; ci++)
            for (int yi = 0; yi < h; yi++)
                for (int xi = 0; xi < w; xi++) {
                    float m = mask[0][0][yi][xi]; // 0..1
                    out[0][ci][yi][xi] = denoised[0][ci][yi][xi] * m + original[0][ci][yi][xi] * (1f - m);
                }
        return out;
    }

    /** Generate DDIM timestep schedule (evenly spaced). */
    private int[] ddimTimesteps(int numSteps, int maxTimestep) {
        int[] ts = new int[numSteps];
        for (int i = 0; i < numSteps; i++) {
            ts[i] = (int) ((maxTimestep - 1) * (1.0 - (double) i / numSteps));
        }
        return ts;
    }

    private InferenceResult runRealEsrgan(OrtEnvironment environment, OrtSession session, InferenceRequest request, String provider) {
        String inputPath = request.inputImagePath();
        if (inputPath == null || inputPath.isBlank()) {
            return InferenceResult.fail("Real-ESRGAN requires an input image. Choose an image to upscale.");
        }

        try {
            BufferedImage inputImage = ImageIO.read(new File(inputPath));
            if (inputImage == null) {
                return InferenceResult.fail("Unable to read input image: " + inputPath);
            }

            // Determine the model's expected spatial input size from its metadata.
            String inputName = session.getInputNames().iterator().next();
            TensorInfo inputInfo = (TensorInfo) session.getInputInfo().get(inputName).getInfo();
            long[] inputShape = inputInfo.getShape();  // e.g. [1, 3, 64, 64]
            int tileH = (inputShape.length >= 3 && inputShape[2] > 0) ? (int) inputShape[2] : 0;
            int tileW = (inputShape.length >= 4 && inputShape[3] > 0) ? (int) inputShape[3] : 0;

            // Detect native scale factor.
            int scaleFactor = detectScaleFactor(environment, session, inputName, tileH, tileW);

            // Determine the resize method from the style field (default: Bicubic).
            String resizeMethod = (request.style() != null && !request.style().isBlank())
                    ? request.style() : "Bicubic";

            int imgW = inputImage.getWidth();
            int imgH = inputImage.getHeight();
            int targetW = request.width();
            int targetH = request.height();

            // Calculate how many native ESRGAN passes are needed.
            int passes = 1;
            if ("ESRGAN Multi-Pass".equals(resizeMethod) && targetW > 0 && targetH > 0) {
                int curW = imgW * scaleFactor;
                int curH = imgH * scaleFactor;
                while (curW < targetW || curH < targetH) {
                    curW *= scaleFactor;
                    curH *= scaleFactor;
                    passes++;
                }
            }

            // Run ESRGAN for each pass.
            BufferedImage current = inputImage;
            for (int pass = 0; pass < passes; pass++) {
                if (request.isCancelled()) {
                    return InferenceResult.fail("Cancelled by user.");
                }
                request.reportProgress("Upscaling pass " + (pass + 1) + "/" + passes
                        + " (" + current.getWidth() + "\u00d7" + current.getHeight() + " \u2192 "
                        + (current.getWidth() * scaleFactor) + "\u00d7" + (current.getHeight() * scaleFactor) + ")\u2026");
                current = esrganUpscaleOnce(environment, session, current, inputName, tileH, tileW, scaleFactor);
            }

            // Final resize to exact target dimensions.
            String finalMethod = "ESRGAN Multi-Pass".equals(resizeMethod) ? "Bicubic" : resizeMethod;
            if (targetW > 0 && targetH > 0) {
                current = resizeImage(current, targetW, targetH, finalMethod);
            }

            Path outputPath = writeOutputImage(current, "realesrgan");
            return InferenceResult.ok(
                    "Real-ESRGAN upscale completed (" + scaleFactor + "× scale, " + passes + " pass"
                            + (passes > 1 ? "es" : "") + ", " + resizeMethod
                            + (targetW > 0 ? ", final " + current.getWidth() + "×" + current.getHeight() : "")
                            + ").",
                    "Java-native ONNX Runtime executed Real-ESRGAN | EP=" + provider,
                    outputPath.toString(),
                    "image"
            );
        } catch (Exception ex) {
            return InferenceResult.fail("Real-ESRGAN failed: " + ex.getMessage());
        }
    }

    /**
     * Run a single ESRGAN upscale pass. Uses tiling when the image does not
     * match the model's fixed input size, or runs directly otherwise.
     */
    private BufferedImage esrganUpscaleOnce(OrtEnvironment environment, OrtSession session,
                                            BufferedImage inputImage, String inputName,
                                            int tileH, int tileW, int scaleFactor) throws Exception {
        int imgW = inputImage.getWidth();
        int imgH = inputImage.getHeight();
        boolean needsTiling = tileH > 0 && tileW > 0 && (imgW != tileW || imgH != tileH);

        if (!needsTiling) {
            // Direct inference — image matches model input or model accepts dynamic shapes.
            float[] tensorData = imageToNchw(inputImage);
            OnnxTensor tensor = OnnxTensor.createTensor(environment,
                    FloatBuffer.wrap(tensorData), new long[]{1, 3, imgH, imgW});
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, tensor);
            try (OrtSession.Result result = session.run(inputs)) {
                ImageOutput output = extractFirstImageOutput(result);
                if (output == null) { throw new RuntimeException("Real-ESRGAN produced no output tensor."); }
                return nchwToImage(output.values(), output.width(), output.height());
            } finally {
                tensor.close();
            }
        }

        // Tiled inference.
        int outW = imgW * scaleFactor;
        int outH = imgH * scaleFactor;
        BufferedImage outImage = new BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB);

        int pad = Math.max(4, tileW / 16); // overlap padding to avoid seam artifacts
        for (int ty = 0; ty < imgH; ty += tileH - pad * 2) {
            for (int tx = 0; tx < imgW; tx += tileW - pad * 2) {
                int sx = Math.max(0, tx - pad);
                int sy = Math.max(0, ty - pad);
                int sw = Math.min(tileW, imgW - sx);
                int sh = Math.min(tileH, imgH - sy);

                BufferedImage tile = new BufferedImage(tileW, tileH, BufferedImage.TYPE_INT_RGB);
                java.awt.Graphics2D g = tile.createGraphics();
                g.drawImage(inputImage.getSubimage(sx, sy, sw, sh), 0, 0, null);
                g.dispose();

                float[] tileData = imageToNchw(tile);
                OnnxTensor tensor = OnnxTensor.createTensor(environment,
                        FloatBuffer.wrap(tileData), new long[]{1, 3, tileH, tileW});
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put(inputName, tensor);

                try (OrtSession.Result result = session.run(inputs)) {
                    ImageOutput output = extractFirstImageOutput(result);
                    if (output == null) { continue; }
                    BufferedImage upTile = nchwToImage(output.values(), output.width(), output.height());

                    int dstPadX = (sx == 0 ? 0 : pad) * scaleFactor;
                    int dstPadY = (sy == 0 ? 0 : pad) * scaleFactor;
                    int dstX = sx * scaleFactor + dstPadX;
                    int dstY = sy * scaleFactor + dstPadY;
                    int copyW = Math.min(sw * scaleFactor - dstPadX, outW - dstX);
                    int copyH = Math.min(sh * scaleFactor - dstPadY, outH - dstY);
                    if (copyW <= 0 || copyH <= 0) { continue; }

                    BufferedImage region = upTile.getSubimage(dstPadX, dstPadY, copyW, copyH);
                    java.awt.Graphics2D g2 = outImage.createGraphics();
                    g2.drawImage(region, dstX, dstY, null);
                    g2.dispose();
                } finally {
                    tensor.close();
                }
            }
        }
        return outImage;
    }

    /** Run a 1-pixel tile through the model to discover its scale factor. */
    private int detectScaleFactor(OrtEnvironment environment, OrtSession session,
                                  String inputName, int tileH, int tileW) {
        try {
            float[] probe = new float[3 * tileH * tileW];
            OnnxTensor tensor = OnnxTensor.createTensor(environment,
                    FloatBuffer.wrap(probe), new long[]{1, 3, tileH, tileW});
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, tensor);
            try (OrtSession.Result result = session.run(inputs)) {
                ImageOutput output = extractFirstImageOutput(result);
                if (output != null && output.width() > 0) {
                    return output.width() / tileW;
                }
            } finally {
                tensor.close();
            }
        } catch (Exception ignored) { }
        return 4; // default Real-ESRGAN scale
    }

    /**
     * Resize an image to the given target dimensions using the specified method.
     * Supported methods: ESRGAN Multi-Pass, Bicubic, Bilinear, Nearest Neighbor, Lanczos.
     * "ESRGAN Multi-Pass" falls back to Bicubic for the final adjustment step.
     */
    private BufferedImage resizeImage(BufferedImage image, int targetW, int targetH, String method) {
        if (image.getWidth() == targetW && image.getHeight() == targetH) { return image; }

        Object interpolationHint = switch (method) {
            case "Nearest Neighbor" -> java.awt.RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR;
            case "Bilinear"         -> java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR;
            default                 -> java.awt.RenderingHints.VALUE_INTERPOLATION_BICUBIC;
        };

        if ("Lanczos".equals(method)) {
            return lanczosResize(image, targetW, targetH);
        }

        BufferedImage resized = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = resized.createGraphics();
        g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION, interpolationHint);
        g.setRenderingHint(java.awt.RenderingHints.KEY_RENDERING,
                java.awt.RenderingHints.VALUE_RENDER_QUALITY);
        g.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING,
                java.awt.RenderingHints.VALUE_ANTIALIAS_ON);
        g.drawImage(image, 0, 0, targetW, targetH, null);
        g.dispose();
        return resized;
    }

    /**
     * Lanczos-3 resize (windowed sinc). Produces sharper results than bicubic
     * for large scale factors. Pure Java implementation.
     */
    private BufferedImage lanczosResize(BufferedImage src, int dstW, int dstH) {
        int srcW = src.getWidth();
        int srcH = src.getHeight();
        int a = 3; // Lanczos-3 kernel radius

        // Horizontal pass → intermediate buffer
        BufferedImage hPass = new BufferedImage(dstW, srcH, BufferedImage.TYPE_INT_RGB);
        double xRatio = (double) srcW / dstW;
        for (int y = 0; y < srcH; y++) {
            for (int x = 0; x < dstW; x++) {
                double center = (x + 0.5) * xRatio - 0.5;
                int start = (int) Math.floor(center) - a + 1;
                int end = (int) Math.floor(center) + a;
                double r = 0, g = 0, b = 0, wSum = 0;
                for (int i = start; i <= end; i++) {
                    int clamped = Math.min(Math.max(i, 0), srcW - 1);
                    double w = lanczosWeight(center - i, a);
                    int rgb = src.getRGB(clamped, y);
                    r += ((rgb >> 16) & 0xFF) * w;
                    g += ((rgb >> 8) & 0xFF) * w;
                    b += (rgb & 0xFF) * w;
                    wSum += w;
                }
                if (wSum != 0) { r /= wSum; g /= wSum; b /= wSum; }
                hPass.setRGB(x, y, (clamp8(r) << 16) | (clamp8(g) << 8) | clamp8(b));
            }
        }

        // Vertical pass
        BufferedImage dst = new BufferedImage(dstW, dstH, BufferedImage.TYPE_INT_RGB);
        double yRatio = (double) srcH / dstH;
        for (int x = 0; x < dstW; x++) {
            for (int y = 0; y < dstH; y++) {
                double center = (y + 0.5) * yRatio - 0.5;
                int start = (int) Math.floor(center) - a + 1;
                int end = (int) Math.floor(center) + a;
                double r = 0, g = 0, b = 0, wSum = 0;
                for (int i = start; i <= end; i++) {
                    int clamped = Math.min(Math.max(i, 0), srcH - 1);
                    double w = lanczosWeight(center - i, a);
                    int rgb = hPass.getRGB(x, clamped);
                    r += ((rgb >> 16) & 0xFF) * w;
                    g += ((rgb >> 8) & 0xFF) * w;
                    b += (rgb & 0xFF) * w;
                    wSum += w;
                }
                if (wSum != 0) { r /= wSum; g /= wSum; b /= wSum; }
                dst.setRGB(x, y, (clamp8(r) << 16) | (clamp8(g) << 8) | clamp8(b));
            }
        }
        return dst;
    }

    private static double lanczosWeight(double x, int a) {
        if (x == 0) { return 1.0; }
        if (Math.abs(x) >= a) { return 0.0; }
        double pix = Math.PI * x;
        return (a * Math.sin(pix) * Math.sin(pix / a)) / (pix * pix);
    }

    private static int clamp8(double v) {
        return Math.min(255, Math.max(0, (int) Math.round(v)));
    }

    private float[] imageToNchw(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        float[] tensor = new float[3 * width * height];
        int redOffset = 0;
        int greenOffset = width * height;
        int blueOffset = 2 * width * height;

        int idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255.0f;
                float g = ((rgb >> 8) & 0xFF) / 255.0f;
                float b = (rgb & 0xFF) / 255.0f;
                tensor[redOffset + idx] = r;
                tensor[greenOffset + idx] = g;
                tensor[blueOffset + idx] = b;
                idx++;
            }
        }
        return tensor;
    }

    private ImageOutput extractFirstImageOutput(OrtSession.Result result) throws OrtException {
        for (Map.Entry<String, OnnxValue> entry : result) {
            OnnxValue value = entry.getValue();
            if (value instanceof OnnxTensor tensor) {
                TensorInfo info = (TensorInfo) tensor.getInfo();
                long[] shape = info.getShape();
                if (shape.length == 4 && (shape[1] == 3 || shape[1] == -1)) {
                    Object raw = tensor.getValue();
                    if (raw instanceof float[][][][] arr) {
                        int h = arr[0][0].length;
                        int w = arr[0][0][0].length;
                        float[] out = new float[3 * h * w];
                        int idx = 0;
                        for (int c = 0; c < 3; c++) {
                            for (int y = 0; y < h; y++) {
                                for (int x = 0; x < w; x++) {
                                    out[idx++] = arr[0][c][y][x];
                                }
                            }
                        }
                        return new ImageOutput(out, w, h);
                    }
                }
            }
        }
        return null;
    }

    private record ImageOutput(float[] values, int width, int height) {
    }

    private float[][][] runTextEncoder(OrtEnvironment environment, OrtSession session, long[] tokenIds) throws OrtException {
        String inputName = resolveInputName(session, "input_ids", 0);
        TensorInfo inputInfo = (TensorInfo) session.getInputInfo().get(inputName).getInfo();
        boolean wantsInt32 = inputInfo.type.toString().contains("INT32");

        OnnxTensor idsTensor;
        if (wantsInt32) {
            int[] ids32 = new int[tokenIds.length];
            for (int i = 0; i < tokenIds.length; i++) { ids32[i] = (int) tokenIds[i]; }
            idsTensor = OnnxTensor.createTensor(environment, new int[][]{ids32});
        } else {
            idsTensor = OnnxTensor.createTensor(environment, new long[][]{tokenIds});
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(inputName, idsTensor);
        try (OrtSession.Result result = session.run(inputs)) {
            float[][][] embedding = extractTensor3d(result);
            if (embedding == null) {
                throw new OrtException("Text encoder output is empty.");
            }
            return embedding;
        } finally {
            idsTensor.close();
        }
    }

    private float[][][] extractTensor3d(OrtSession.Result result) throws OrtException {
        for (Map.Entry<String, OnnxValue> entry : result) {
            if (entry.getValue() instanceof OnnxTensor tensor) {
                Object value = tensor.getValue();
                if (value instanceof float[][][] arr) {
                    return arr;
                }
            }
        }
        return null;
    }

    private float[][][][] extractTensor4d(OrtSession.Result result) throws OrtException {
        for (Map.Entry<String, OnnxValue> entry : result) {
            if (entry.getValue() instanceof OnnxTensor tensor) {
                Object value = tensor.getValue();
                if (value instanceof float[][][][] arr) {
                    return arr;
                }
            }
        }
        return null;
    }

    private String resolveInputName(OrtSession session, String preferred, int fallbackIndex) {
        List<String> names = new ArrayList<>(session.getInputNames());
        for (String name : names) {
            if (name.toLowerCase().contains(preferred.toLowerCase())) {
                return name;
            }
        }
        if (fallbackIndex >= 0 && fallbackIndex < names.size()) {
            return names.get(fallbackIndex);
        }
        return names.get(0);
    }

    private OnnxTensor createTimestepTensor(OrtEnvironment environment, OrtSession session, int timestep) throws OrtException {
        String timestepName = resolveInputName(session, "timestep", 1);
        TensorInfo info = (TensorInfo) session.getInputInfo().get(timestepName).getInfo();
        if (info.type.toString().contains("INT64")) {
            return OnnxTensor.createTensor(environment, new long[]{timestep});
        }
        return OnnxTensor.createTensor(environment, new float[]{(float) timestep});
    }

    private float[][][][] randomLatents(long seed, int latentHeight, int latentWidth) {
        Random random = new Random(seed);
        float[][][][] values = new float[1][4][latentHeight][latentWidth];
        for (int c = 0; c < 4; c++) {
            for (int y = 0; y < latentHeight; y++) {
                for (int x = 0; x < latentWidth; x++) {
                    values[0][c][y][x] = (float) random.nextGaussian();
                }
            }
        }
        return values;
    }

    private float[] loadAlphaCumprod(Path schedulerConfigPath) throws Exception {
        Map<String, Object> config = OBJECT_MAPPER.readValue(schedulerConfigPath.toFile(), new TypeReference<>() {
        });
        int trainTimesteps = ((Number) config.getOrDefault("num_train_timesteps", 1000)).intValue();
        double betaStart = ((Number) config.getOrDefault("beta_start", 0.00085)).doubleValue();
        double betaEnd = ((Number) config.getOrDefault("beta_end", 0.012)).doubleValue();
        float[] alphaCumprod = new float[trainTimesteps];
        double cumulative = 1.0;
        for (int i = 0; i < trainTimesteps; i++) {
            double beta = betaStart + (betaEnd - betaStart) * i / Math.max(1, trainTimesteps - 1);
            cumulative *= (1.0 - beta);
            alphaCumprod[i] = (float) cumulative;
        }
        return alphaCumprod;
    }

    /** Compute alpha_cumprod inline when scheduler_config.json is unavailable. */
    private float[] computeDefaultAlphaCumprod(int trainTimesteps, double betaStart, double betaEnd) {
        float[] alphaCumprod = new float[trainTimesteps];
        double cumulative = 1.0;
        for (int i = 0; i < trainTimesteps; i++) {
            double beta = betaStart + (betaEnd - betaStart) * i / Math.max(1, trainTimesteps - 1);
            cumulative *= (1.0 - beta);
            alphaCumprod[i] = (float) cumulative;
        }
        return alphaCumprod;
    }

    private int[] createTimesteps(int steps, int trainTimesteps) {
        int[] timesteps = new int[steps];
        float stride = (float) (trainTimesteps - 1) / Math.max(1, steps - 1);
        for (int i = 0; i < steps; i++) {
            timesteps[i] = Math.max(0, Math.round((steps - 1 - i) * stride));
        }
        return timesteps;
    }

    private float[][][][] duplicateBatch(float[][][][] latents) {
        int channels = latents[0].length;
        int h = latents[0][0].length;
        int w = latents[0][0][0].length;
        float[][][][] out = new float[2][channels][h][w];
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < h; y++) {
                System.arraycopy(latents[0][c][y], 0, out[0][c][y], 0, w);
                System.arraycopy(latents[0][c][y], 0, out[1][c][y], 0, w);
            }
        }
        return out;
    }

    private float[][][] guidance(float[][][] uncond, float[][][] cond, float guidanceScale) {
        int c = uncond.length;
        int h = uncond[0].length;
        int w = uncond[0][0].length;
        float[][][] out = new float[c][h][w];
        for (int ch = 0; ch < c; ch++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    out[ch][y][x] = uncond[ch][y][x] + guidanceScale * (cond[ch][y][x] - uncond[ch][y][x]);
                }
            }
        }
        return out;
    }

    private float[][][][] ddimStep(float[][][][] latents,
                                   float[][][] eps,
                                   float alphaT,
                                   float alphaPrev) {
        int c = latents[0].length;
        int h = latents[0][0].length;
        int w = latents[0][0][0].length;
        float sqrtAlphaT = (float) Math.sqrt(Math.max(1e-6f, alphaT));
        float sqrtOneMinusAlphaT = (float) Math.sqrt(Math.max(1e-6f, 1f - alphaT));
        float sqrtAlphaPrev = (float) Math.sqrt(Math.max(1e-6f, alphaPrev));
        float sqrtOneMinusAlphaPrev = (float) Math.sqrt(Math.max(1e-6f, 1f - alphaPrev));

        float[][][][] out = new float[1][c][h][w];
        for (int ch = 0; ch < c; ch++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float xT = latents[0][ch][y][x];
                    float e = eps[ch][y][x];
                    float x0 = (xT - sqrtOneMinusAlphaT * e) / sqrtAlphaT;
                    out[0][ch][y][x] = sqrtAlphaPrev * x0 + sqrtOneMinusAlphaPrev * e;
                }
            }
        }
        return out;
    }

    private float[][][][] scaleLatents(float[][][][] latents, float scale) {
        float[][][][] out = new float[1][latents[0].length][latents[0][0].length][latents[0][0][0].length];
        for (int c = 0; c < latents[0].length; c++) {
            for (int y = 0; y < latents[0][0].length; y++) {
                for (int x = 0; x < latents[0][0][0].length; x++) {
                    out[0][c][y][x] = latents[0][c][y][x] * scale;
                }
            }
        }
        return out;
    }

    private BufferedImage tensorToImage(float[][][] tensor) {
        int channels = tensor.length;
        int height = tensor[0].length;
        int width = tensor[0][0].length;
        if (channels < 3) {
            throw new IllegalStateException("Decoded image tensor must contain at least 3 channels.");
        }
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = toRgbByte(tensor[0][y][x]);
                int g = toRgbByte(tensor[1][y][x]);
                int b = toRgbByte(tensor[2][y][x]);
                image.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return image;
    }

    private int toRgbByte(float value) {
        float normalized = (value / 2f) + 0.5f;
        float clipped = Math.max(0f, Math.min(1f, normalized));
        return Math.round(clipped * 255f);
    }

    private static final class ClipTokenizer {
        private final Map<String, Integer> vocab;
        private final Map<String, Integer> merges;
        private final Map<Integer, String> byteEncoder;
        private final Map<String, String> cache = new HashMap<>();
        private final int bos;
        private final int eos;

        private ClipTokenizer(Map<String, Integer> vocab, Map<String, Integer> merges) {
            this.vocab = vocab;
            this.merges = merges;
            this.byteEncoder = bytesToUnicode();
            this.bos = vocab.getOrDefault("<|startoftext|>", 49406);
            this.eos = vocab.getOrDefault("<|endoftext|>", 49407);
        }

        static ClipTokenizer load(Path vocabPath, Path mergesPath) throws Exception {
            Map<String, Integer> vocab = OBJECT_MAPPER.readValue(vocabPath.toFile(), new TypeReference<>() {
            });
            List<String> mergeLines = java.nio.file.Files.readAllLines(mergesPath, StandardCharsets.UTF_8);
            Map<String, Integer> ranks = new HashMap<>();
            int rank = 0;
            for (String line : mergeLines) {
                String trimmed = line.trim();
                if (trimmed.isBlank() || trimmed.startsWith("#")) {
                    continue;
                }
                ranks.put(trimmed, rank++);
            }
            return new ClipTokenizer(vocab, ranks);
        }

        long[] encode(String text, int maxLength) {
            List<Integer> ids = new ArrayList<>();
            ids.add(bos);
            String normalized = text == null ? "" : text.toLowerCase();
            Matcher matcher = TOKEN_PATTERN.matcher(normalized);
            while (matcher.find()) {
                String token = matcher.group();
                StringBuilder encoded = new StringBuilder();
                byte[] bytes = token.getBytes(StandardCharsets.UTF_8);
                for (byte value : bytes) {
                    encoded.append(byteEncoder.get(value & 0xFF));
                }
                String bpeToken = bpe(encoded.toString());
                for (String piece : bpeToken.split(" ")) {
                    Integer id = vocab.get(piece);
                    if (id != null) {
                        ids.add(id);
                    }
                }
                if (ids.size() >= maxLength - 1) {
                    break;
                }
            }
            ids.add(eos);
            long[] out = new long[maxLength];
            Arrays.fill(out, eos);
            for (int i = 0; i < Math.min(maxLength, ids.size()); i++) {
                out[i] = ids.get(i);
            }
            return out;
        }

        private String bpe(String token) {
            String cached = cache.get(token);
            if (cached != null) {
                return cached;
            }

            List<String> word = new ArrayList<>();
            for (int i = 0; i < token.length(); i++) {
                word.add(String.valueOf(token.charAt(i)));
            }
            if (word.size() == 1) {
                return token;
            }

            Set<String> pairs = getPairs(word);
            while (true) {
                String bestPair = null;
                int bestRank = Integer.MAX_VALUE;
                for (String pair : pairs) {
                    int rank = merges.getOrDefault(pair, Integer.MAX_VALUE);
                    if (rank < bestRank) {
                        bestRank = rank;
                        bestPair = pair;
                    }
                }
                if (bestPair == null || !merges.containsKey(bestPair)) {
                    break;
                }

                String[] parts = bestPair.split(" ");
                String first = parts[0];
                String second = parts[1];
                List<String> merged = new ArrayList<>();
                int i = 0;
                while (i < word.size()) {
                    if (i < word.size() - 1 && word.get(i).equals(first) && word.get(i + 1).equals(second)) {
                        merged.add(first + second);
                        i += 2;
                    } else {
                        merged.add(word.get(i));
                        i++;
                    }
                }
                word = merged;
                if (word.size() == 1) {
                    break;
                }
                pairs = getPairs(word);
            }

            String out = String.join(" ", word);
            cache.put(token, out);
            return out;
        }

        private Set<String> getPairs(List<String> word) {
            Set<String> pairs = new HashSet<>();
            for (int i = 0; i < word.size() - 1; i++) {
                pairs.add(word.get(i) + " " + word.get(i + 1));
            }
            return pairs;
        }

        private static Map<Integer, String> bytesToUnicode() {
            List<Integer> bs = new ArrayList<>();
            for (int i = '!'; i <= '~'; i++) {
                bs.add(i);
            }
            for (int i = '¡'; i <= '¬'; i++) {
                bs.add(i);
            }
            for (int i = '®'; i <= 'ÿ'; i++) {
                bs.add(i);
            }

            List<Integer> cs = new ArrayList<>(bs);
            int n = 0;
            for (int b = 0; b < 256; b++) {
                if (!bs.contains(b)) {
                    bs.add(b);
                    cs.add(256 + n);
                    n++;
                }
            }

            Map<Integer, String> map = new HashMap<>();
            for (int i = 0; i < bs.size(); i++) {
                map.put(bs.get(i), new String(Character.toChars(cs.get(i))));
            }
            return map;
        }
    }

    private ProviderSelection configureExecutionProvider(OrtSession.SessionOptions options, boolean preferGpu) {
        String os = System.getProperty("os.name", "unknown").toLowerCase();
        String forced = System.getProperty("lumenforge.ep", "").trim().toLowerCase();

        List<String> preference = new ArrayList<>();
        if (!preferGpu) {
            preference.add("cpu");
        } else if (!forced.isBlank()) {
            preference.add(forced);
            preference.add("cpu");
        } else if (os.contains("mac")) {
            preference.add("coreml");
            preference.add("cpu");
        } else if (os.contains("win")) {
            preference.add("directml");
            preference.add("cuda");
            preference.add("cpu");
        } else {
            preference.add("cuda");
            preference.add("rocm");
            preference.add("cpu");
        }

        StringBuilder notes = new StringBuilder();
        for (String candidate : preference) {
            if ("cpu".equals(candidate)) {
                return new ProviderSelection("CPUExecutionProvider", notes.toString());
            }
            if (tryEnableProvider(options, candidate, notes)) {
                return new ProviderSelection(providerDisplayName(candidate), notes.toString());
            }
        }
        return new ProviderSelection("CPUExecutionProvider", notes.toString());
    }

    private boolean tryEnableProvider(OrtSession.SessionOptions options, String candidate, StringBuilder notes) {
        try {
            return switch (candidate) {
                case "cuda" -> invokeNoArg(options, "addCUDA");
                case "directml" -> invokeNoArg(options, "addDirectML") || invokeIntArg(options, "addDirectML", 0);
                case "coreml" -> {
                    // Flags: COREML_FLAG_USE_CPU_AND_GPU (1) enables all ANE/GPU subgraphs
                    boolean ok = invokeIntArg(options, "addCoreML", 1);
                    if (!ok) { ok = invokeNoArg(options, "addCoreML"); }
                    if (!ok) { ok = invokeIntArg(options, "addCoreML", 0); }
                    yield ok;
                }
                case "rocm" -> invokeNoArg(options, "addROCM");
                default -> false;
            };
        } catch (Exception ex) {
            if (!notes.isEmpty()) {
                notes.append("; ");
            }
            notes.append(candidate).append(" unavailable");
            return false;
        }
    }

    private boolean invokeNoArg(OrtSession.SessionOptions options, String methodName) {
        try {
            options.getClass().getMethod(methodName).invoke(options);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    private boolean invokeIntArg(OrtSession.SessionOptions options, String methodName, int arg) {
        try {
            options.getClass().getMethod(methodName, int.class).invoke(options, arg);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    private String providerDisplayName(String candidate) {
        return switch (candidate) {
            case "cuda" -> "CUDAExecutionProvider";
            case "directml" -> "DmlExecutionProvider";
            case "coreml" -> "CoreMLExecutionProvider";
            case "rocm" -> "ROCMExecutionProvider";
            default -> "CPUExecutionProvider";
        };
    }

    private record ProviderSelection(String provider, String notes) {
        String noteSuffix() {
            if (notes == null || notes.isBlank()) {
                return "";
            }
            return " | " + notes;
        }
    }

    private BufferedImage nchwToImage(float[] values, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int redOffset = 0;
        int greenOffset = width * height;
        int blueOffset = 2 * width * height;

        int idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = clampToByte(values[redOffset + idx]);
                int g = clampToByte(values[greenOffset + idx]);
                int b = clampToByte(values[blueOffset + idx]);
                int rgb = (r << 16) | (g << 8) | b;
                image.setRGB(x, y, rgb);
                idx++;
            }
        }
        return image;
    }

    private int clampToByte(float value) {
        float normalized = Math.max(0f, Math.min(1f, value));
        return Math.round(normalized * 255f);
    }

    private Path writeOutputImage(BufferedImage image, String prefix) throws java.io.IOException {
        Path outputDir = storage.root().resolve("outputs").resolve("images");
        java.nio.file.Files.createDirectories(outputDir);
        String fileName = prefix + "-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmssSSS")) + ".png";
        Path out = outputDir.resolve(fileName);
        ImageIO.write(image, "png", out.toFile());
        return out;
    }

    /**
     * An OutputStream that writes to the original stderr AND forwards
     * complete lines to a progress callback (for ONNX Runtime log capture).
     */
    private static class TeeOutputStream extends OutputStream {
        private final PrintStream original;
        private final Consumer<String> callback;
        private final ByteArrayOutputStream lineBuffer = new ByteArrayOutputStream();

        TeeOutputStream(PrintStream original, Consumer<String> callback) {
            this.original = original;
            this.callback = callback;
        }

        @Override
        public void write(int b) {
            original.write(b);
            if (b == '\n') {
                flushLine();
            } else {
                lineBuffer.write(b);
            }
        }

        @Override
        public void write(byte[] buf, int off, int len) {
            original.write(buf, off, len);
            for (int i = off; i < off + len; i++) {
                if (buf[i] == '\n') {
                    flushLine();
                } else {
                    lineBuffer.write(buf[i]);
                }
            }
        }

        @Override
        public void flush() {
            original.flush();
        }

        private void flushLine() {
            String line = lineBuffer.toString(StandardCharsets.UTF_8).trim();
            lineBuffer.reset();
            if (!line.isEmpty() && callback != null) {
                // Filter out noisy/irrelevant lines
                if (line.contains("Context leak detected")) { return; }
                // Clean up ORT timestamp prefix: keep only the message after the bracket
                int bracket = line.indexOf(']');
                String display = (bracket > 0 && bracket < line.length() - 2)
                        ? "ORT: " + line.substring(bracket + 2).trim()
                        : "ORT: " + line;
                callback.accept(display);
            }
        }
    }
}
