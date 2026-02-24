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
import atri.palaash.lumenforge.ui.AppLogger;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.text.Normalizer;
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
    private static final ConcurrentHashMap<String, T5Tokenizer> T5_TOKENIZER_CACHE = new ConcurrentHashMap<>();
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
            AppLogger.model("Session cache hit: " + modelPath.getFileName());
            return existing;
        }
        AppLogger.model("Loading ONNX session: " + modelPath.getFileName());
        OrtSession session = env.createSession(modelPath.toString(), opts);
        SESSION_CACHE.put(key, session);
        AppLogger.model("Session loaded: " + modelPath.getFileName());
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
     * Get or load a T5Tokenizer — tokenizer.json is cached in memory.
     */
    private T5Tokenizer getOrCreateT5Tokenizer(Path tokenizerJsonPath) throws Exception {
        String key = tokenizerJsonPath.toAbsolutePath().toString();
        T5Tokenizer cached = T5_TOKENIZER_CACHE.get(key);
        if (cached != null) {
            return cached;
        }
        T5Tokenizer tokenizer = T5Tokenizer.load(tokenizerJsonPath);
        T5_TOKENIZER_CACHE.put(key, tokenizer);
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
        T5_TOKENIZER_CACHE.clear();
        cachedEpKey = "";
    }

    @Override
    public CompletableFuture<InferenceResult> run(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            if (!storage.isAvailable(request.model())) {
                AppLogger.modelError("Model not found locally: " + request.model().displayName());
                return InferenceResult.fail("Model not found locally. Open Model Manager from the menu bar and download it first.");
            }

            Path modelPath = storage.modelPath(request.model());
            AppLogger.model("Starting inference: " + request.model().displayName()
                    + " (" + request.model().id() + ")");

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
                AppLogger.model("Using EP: " + providerSelection.provider()
                        + (providerSelection.notes().isBlank() ? "" : " (" + providerSelection.notes() + ")"));

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
                // SD 3.x (converted) — detected by transformer/ directory structure
                if (request.model().relativePath() != null
                        && request.model().relativePath().contains("transformer/")) {
                    return runSd3(environment, sessionOptions, request, providerSelection.provider());
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
                AppLogger.modelWarn(details);
                return InferenceResult.fail(details);
            } catch (OrtException ex) {
                String message = ex.getMessage() == null ? "Unknown ONNX Runtime error" : ex.getMessage();
                AppLogger.modelError("ONNX Runtime error: " + message);
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
    /*  Stable Diffusion 3.x — MMDiT transformer, Flow Matching Euler,    */
    /*  dual CLIP encoders (L + G), 16-channel latents, 1024×1024.        */
    /* ================================================================== */

    private InferenceResult runSd3(OrtEnvironment environment,
                                   OrtSession.SessionOptions sessionOptions,
                                   InferenceRequest request,
                                   String provider) {
        try {
            int width  = Math.max(512, (request.width()  / 8) * 8);
            int height = Math.max(512, (request.height() / 8) * 8);
            int latentW = width  / 8;
            int latentH = height / 8;
            int steps = Math.max(5, Math.min(request.batch() > 0 ? request.batch() : 28, 50));
            float guidanceScale = request.promptWeight() > 0 ? (float) request.promptWeight() : 7.0f;
            float shiftFactor = 3.0f; // SD 3.5 medium schedule shift

            // Resolve model directory from the relativePath
            Path modelPath = storage.modelPath(request.model());
            Path base = modelPath.getParent().getParent(); // go up from transformer/model.onnx

            Path transformerPath = base.resolve("transformer/model.onnx");
            Path textEncoder1Path = base.resolve("text_encoder/model.onnx");
            Path textEncoder2Path = base.resolve("text_encoder_2/model.onnx");
            Path vaeDecoderPath   = base.resolve("vae_decoder/model.onnx");
            Path vocab1 = base.resolve("tokenizer/vocab.json");
            Path merges1 = base.resolve("tokenizer/merges.txt");
            Path vocab2 = base.resolve("tokenizer_2/vocab.json");
            Path merges2 = base.resolve("tokenizer_2/merges.txt");

            for (Path p : List.of(transformerPath, textEncoder1Path, textEncoder2Path,
                    vaeDecoderPath, vocab1, merges1, vocab2, merges2)) {
                if (!java.nio.file.Files.exists(p)) {
                    return InferenceResult.fail("SD 3.x bundle is incomplete (missing "
                            + base.relativize(p) + "). Convert the full model first.");
                }
            }

            // Optional T5 text encoder
            Path textEncoder3Path = base.resolve("text_encoder_3/model.onnx");
            Path tokenizer3Json   = base.resolve("tokenizer_3/tokenizer.json");
            boolean hasT5 = java.nio.file.Files.exists(textEncoder3Path)
                    && java.nio.file.Files.exists(tokenizer3Json);

            ClipTokenizer tok1 = getOrCreateTokenizer(vocab1, merges1);
            ClipTokenizer tok2 = getOrCreateTokenizer(vocab2, merges2);
            T5Tokenizer tok3 = hasT5 ? getOrCreateT5Tokenizer(tokenizer3Json) : null;
            long[] tokens1 = tok1.encode(request.prompt(), 77);
            long[] tokens2 = tok2.encode(request.prompt(), 77);
            long[] negTokens1 = tok1.encode(request.negativePrompt() == null ? "" : request.negativePrompt(), 77);
            long[] negTokens2 = tok2.encode(request.negativePrompt() == null ? "" : request.negativePrompt(), 77);
            long[] t5Tokens = hasT5 ? tok3.encode(request.prompt(), 256) : null;
            long[] t5NegTokens = hasT5 ? tok3.encode(
                    request.negativePrompt() == null ? "" : request.negativePrompt(), 256) : null;

            request.reportProgress("Loading SD 3.x models (transformer, "
                    + (hasT5 ? "3" : "2") + " text encoders, VAE)…");
            OrtSession enc1 = getOrCreateSession(environment, textEncoder1Path, sessionOptions);
            OrtSession enc2 = getOrCreateSession(environment, textEncoder2Path, sessionOptions);
            OrtSession transformer = getOrCreateSession(environment, transformerPath, sessionOptions);
            OrtSession vae  = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);
            OrtSession enc3 = hasT5 ? getOrCreateSession(environment, textEncoder3Path, sessionOptions) : null;

            // ── Encode positive prompt ──
            request.reportProgress("Encoding positive prompt (CLIP-L)…");
            float[][][] embed1 = runTextEncoder(environment, enc1, tokens1);

            request.reportProgress("Encoding positive prompt (CLIP-G)…");
            float[][][] embed2;
            float[][] pooledPos;
            {
                String inName = resolveInputName(enc2, "input_ids", 0);
                TensorInfo tInfo = (TensorInfo) enc2.getInputInfo().get(inName).getInfo();
                boolean wantsInt32 = tInfo.type.toString().contains("INT32");
                OnnxTensor idsTensor;
                if (wantsInt32) {
                    int[] ids32 = new int[tokens2.length];
                    for (int i = 0; i < tokens2.length; i++) ids32[i] = (int) tokens2[i];
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
                            if (v instanceof float[][][] a3 && embed2 == null) embed2 = a3;
                            else if (v instanceof float[][] a2 && pooledPos == null) pooledPos = a2;
                        }
                    }
                    if (embed2 == null) return InferenceResult.fail("CLIP-G produced no hidden states.");
                    if (pooledPos == null) pooledPos = new float[1][1280];
                } finally { idsTensor.close(); }
            }

            // ── Encode negative prompt ──
            request.reportProgress("Encoding negative prompt…");
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
                    for (int i = 0; i < negTokens2.length; i++) ids32[i] = (int) negTokens2[i];
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
                            if (v instanceof float[][][] a3 && negEmbed2 == null) negEmbed2 = a3;
                            else if (v instanceof float[][] a2 && pooledNeg == null) pooledNeg = a2;
                        }
                    }
                    if (negEmbed2 == null) return InferenceResult.fail("CLIP-G negative produced no hidden states.");
                    if (pooledNeg == null) pooledNeg = new float[1][1280];
                } finally { idsTensor.close(); }
            }

            // ── Combine CLIP embeddings ──
            // CLIP-L: [1, 77, dim1], CLIP-G: [1, 77, dim2]
            // Concatenate along last dim → [1, 77, dim1+dim2]
            int seqLen = embed1[0].length;
            int dim1 = embed1[0][0].length;  // 768
            int dim2 = embed2[0][0].length;  // 1280
            int clipDim = dim1 + dim2;       // 2048
            int t5Dim = 4096;                // SD3 transformer expects 4096-wide embeddings
            int t5SeqLen = 256;              // T5 max sequence length in SD3

            // Pad CLIP embeddings to t5Dim (4096) → [1, 77, 4096]
            float[][][] clipPosEmbed = new float[1][seqLen][t5Dim];
            float[][][] clipNegEmbed = new float[1][seqLen][t5Dim];
            for (int s = 0; s < seqLen; s++) {
                System.arraycopy(embed1[0][s], 0, clipPosEmbed[0][s], 0, dim1);
                System.arraycopy(embed2[0][s], 0, clipPosEmbed[0][s], dim1, dim2);
                // Rest is zeros (padding to 4096)
                System.arraycopy(negEmbed1[0][s], 0, clipNegEmbed[0][s], 0, dim1);
                System.arraycopy(negEmbed2[0][s], 0, clipNegEmbed[0][s], dim1, dim2);
            }

            // T5 embeddings: real T5-XXL output or zeros fallback
            float[][][] t5PosEmbed;
            float[][][] t5NegEmbed;
            if (hasT5 && enc3 != null && t5Tokens != null) {
                request.reportProgress("Encoding positive prompt (T5-XXL)…");
                t5PosEmbed = runT5Encoder(environment, enc3, t5Tokens, t5SeqLen, t5Dim);
                request.reportProgress("Encoding negative prompt (T5-XXL)…");
                t5NegEmbed = runT5Encoder(environment, enc3, t5NegTokens, t5SeqLen, t5Dim);
            } else {
                t5PosEmbed = new float[1][t5SeqLen][t5Dim]; // zeros fallback
                t5NegEmbed = new float[1][t5SeqLen][t5Dim];
            }

            // Concatenate along sequence: [1, 77+256, 4096] = [1, 333, 4096]
            int totalSeqLen = seqLen + t5SeqLen;
            float[][][] encoderHiddenPos = new float[1][totalSeqLen][t5Dim];
            float[][][] encoderHiddenNeg = new float[1][totalSeqLen][t5Dim];
            for (int s = 0; s < seqLen; s++) {
                System.arraycopy(clipPosEmbed[0][s], 0, encoderHiddenPos[0][s], 0, t5Dim);
                System.arraycopy(clipNegEmbed[0][s], 0, encoderHiddenNeg[0][s], 0, t5Dim);
            }
            for (int s = 0; s < t5SeqLen; s++) {
                System.arraycopy(t5PosEmbed[0][s], 0, encoderHiddenPos[0][seqLen + s], 0, t5Dim);
                System.arraycopy(t5NegEmbed[0][s], 0, encoderHiddenNeg[0][seqLen + s], 0, t5Dim);
            }

            // Batch hidden states [2, 333, 4096] — negative first, positive second (for CFG)
            float[][][] batchedHidden = new float[2][totalSeqLen][t5Dim];
            batchedHidden[0] = encoderHiddenNeg[0];
            batchedHidden[1] = encoderHiddenPos[0];

            // Pooled projections: concat CLIP-L pooled + CLIP-G pooled → [1, 2048]
            // CLIP-L pooled is embed1[0][0] (take first token hidden state as rough pooled — for proper
            // pooling we'd need the model's pooler, but using first text encoder output it's already 768-d
            // hidden state at BOS position). Actually, pooledPos is from CLIP-G which outputs it directly.
            // We need both CLIP-L pooled + CLIP-G pooled = [768 + 1280] = [2048]
            float[][] embed1Pooled = runTextEncoderPooled(environment, enc1, tokens1);
            float[][] negEmbed1Pooled = runTextEncoderPooled(environment, enc1, negTokens1);

            int pooledDim = (embed1Pooled != null ? embed1Pooled[0].length : dim1) + pooledPos[0].length;
            float[][] batchedPooled = new float[2][pooledDim];
            // Negative
            if (negEmbed1Pooled != null) System.arraycopy(negEmbed1Pooled[0], 0, batchedPooled[0], 0, negEmbed1Pooled[0].length);
            System.arraycopy(pooledNeg[0], 0, batchedPooled[0], pooledDim - pooledNeg[0].length, pooledNeg[0].length);
            // Positive
            if (embed1Pooled != null) System.arraycopy(embed1Pooled[0], 0, batchedPooled[1], 0, embed1Pooled[0].length);
            System.arraycopy(pooledPos[0], 0, batchedPooled[1], pooledDim - pooledPos[0].length, pooledPos[0].length);

            // ── Create initial noise (16-channel latents for SD3) ──
            Random random = new Random(request.seed());
            float[][][][] latents = new float[1][16][latentH][latentW];
            for (int c = 0; c < 16; c++)
                for (int y = 0; y < latentH; y++)
                    for (int x = 0; x < latentW; x++)
                        latents[0][c][y][x] = (float) random.nextGaussian();

            // ── Flow Matching Euler schedule ──
            // SD 3.5 uses a shifted sigma schedule: sigma = shift * t / (1 + (shift-1)*t)
            float[] sigmas = new float[steps + 1];
            for (int i = 0; i <= steps; i++) {
                float t = 1.0f - (float) i / steps; // goes from 1.0 to 0.0
                sigmas[i] = shiftFactor * t / (1.0f + (shiftFactor - 1.0f) * t);
            }

            request.reportProgress("Denoising: 0/" + steps + " steps (SD 3.x Flow Matching) — EP: " + provider);
            long stepStart = System.currentTimeMillis();
            long firstStepDuration = 0;

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];
                float sigmaNext = sigmas[i + 1];

                // Scale timestep to 1000-scale for the model
                float timestepVal = sigma * 1000.0f;

                // Duplicate latents for CFG batch [2, 16, H, W]
                float[][][][] batchInput = new float[2][16][latentH][latentW];
                for (int c = 0; c < 16; c++)
                    for (int y = 0; y < latentH; y++)
                        for (int x = 0; x < latentW; x++) {
                            batchInput[0][c][y][x] = latents[0][c][y][x];
                            batchInput[1][c][y][x] = latents[0][c][y][x];
                        }

                OnnxTensor sampleT  = OnnxTensor.createTensor(environment, batchInput);
                OnnxTensor hiddenT  = OnnxTensor.createTensor(environment, batchedHidden);
                OnnxTensor pooledT  = OnnxTensor.createTensor(environment, batchedPooled);

                // Create timestep tensor — SD3 transformer may expect float or int
                OnnxTensor tsT;
                {
                    String tsName = resolveInputName(transformer, "timestep", 1);
                    TensorInfo tsInfo = (TensorInfo) transformer.getInputInfo().get(tsName).getInfo();
                    if (tsInfo.type.toString().contains("INT64")) {
                        tsT = OnnxTensor.createTensor(environment, new long[]{(long) timestepVal, (long) timestepVal});
                    } else if (tsInfo.type.toString().contains("INT32")) {
                        tsT = OnnxTensor.createTensor(environment, new int[]{(int) timestepVal, (int) timestepVal});
                    } else {
                        tsT = OnnxTensor.createTensor(environment, new float[]{timestepVal, timestepVal});
                    }
                }

                Map<String, OnnxTensor> transInputs = new HashMap<>();
                // Map inputs by sniffing the model's input names
                for (String name : transformer.getInputNames()) {
                    String lower = name.toLowerCase();
                    if (lower.contains("hidden_states") || lower.contains("sample")) {
                        transInputs.put(name, sampleT);
                    } else if (lower.contains("timestep")) {
                        transInputs.put(name, tsT);
                    } else if (lower.contains("encoder_hidden") || lower.contains("prompt_embeds")) {
                        transInputs.put(name, hiddenT);
                    } else if (lower.contains("pooled") || lower.contains("text_embeds")) {
                        transInputs.put(name, pooledT);
                    }
                }

                // Fallback: assign by positional index if mapping is empty
                if (transInputs.isEmpty()) {
                    var inputNames = new java.util.ArrayList<>(transformer.getInputNames());
                    if (inputNames.size() >= 4) {
                        transInputs.put(inputNames.get(0), sampleT);
                        transInputs.put(inputNames.get(1), tsT);
                        transInputs.put(inputNames.get(2), hiddenT);
                        transInputs.put(inputNames.get(3), pooledT);
                    }
                }

                float[][][][] noise;
                try (OrtSession.Result transResult = transformer.run(transInputs)) {
                    noise = extractTensor4d(transResult);
                } finally {
                    sampleT.close(); tsT.close(); hiddenT.close(); pooledT.close();
                }

                if (noise == null || noise.length < 2) {
                    return InferenceResult.fail("SD 3.x transformer produced invalid output.");
                }

                // Classifier-free guidance
                float[][][] guidedNoise = guidance(noise[0], noise[1], guidanceScale);

                // Flow matching Euler step: latent = latent + (sigma_next - sigma) * velocity
                float dt = sigmaNext - sigma;
                for (int c = 0; c < 16; c++)
                    for (int y = 0; y < latentH; y++)
                        for (int x = 0; x < latentW; x++)
                            latents[0][c][y][x] += dt * guidedNoise[c][y][x];

                long elapsed = System.currentTimeMillis() - stepStart;
                stepStart = System.currentTimeMillis();
                if (i == 0) firstStepDuration = elapsed;
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

            // ── Decode with VAE ──
            // SD3 VAE: latent = latent / scaling_factor + shift_factor
            // scaling_factor=1.5305, shift_factor=0.0609
            request.reportProgress("Decoding latents with VAE…");
            float[][][][] scaledLatents = new float[1][16][latentH][latentW];
            for (int c = 0; c < 16; c++)
                for (int y = 0; y < latentH; y++)
                    for (int x = 0; x < latentW; x++)
                        scaledLatents[0][c][y][x] = latents[0][c][y][x] / 1.5305f + 0.0609f;

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
            Path outputPath = writeOutputImage(image, "sd3");
            return InferenceResult.ok(
                    "Generated image for prompt: \"" + request.prompt() + "\"",
                    "SD 3.x pipeline completed (" + steps + " steps, CFG=" + guidanceScale
                            + ", shift=" + shiftFactor + ") | EP=" + provider
                            + (hasT5 ? " | T5 encoder active" : " | T5 skipped (CLIP-only)"),
                    outputPath.toString(), "image");
        } catch (Exception ex) {
            return InferenceResult.fail("SD 3.x pipeline failed: " + ex.getMessage());
        }
    }

    /**
     * Run a text encoder and attempt to extract the pooled output (second output).
     * Returns null if the model doesn't produce a 2D pooled output.
     */
    private float[][] runTextEncoderPooled(OrtEnvironment env, OrtSession session, long[] tokenIds) throws OrtException {
        String inName = resolveInputName(session, "input_ids", 0);
        TensorInfo tInfo = (TensorInfo) session.getInputInfo().get(inName).getInfo();
        boolean wantsInt32 = tInfo.type.toString().contains("INT32");
        OnnxTensor idsTensor;
        if (wantsInt32) {
            int[] ids32 = new int[tokenIds.length];
            for (int i = 0; i < tokenIds.length; i++) ids32[i] = (int) tokenIds[i];
            idsTensor = OnnxTensor.createTensor(env, new int[][]{ids32});
        } else {
            idsTensor = OnnxTensor.createTensor(env, new long[][]{tokenIds});
        }
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(inName, idsTensor);
        try (OrtSession.Result r = session.run(inputs)) {
            for (Map.Entry<String, OnnxValue> entry : r) {
                if (entry.getValue() instanceof OnnxTensor t) {
                    Object v = t.getValue();
                    if (v instanceof float[][] a2) return a2;
                }
            }
        } finally { idsTensor.close(); }
        return null;
    }

    /**
     * Run T5-XXL encoder and return hidden states padded/truncated to [1, maxSeqLen, expectedDim].
     */
    private float[][][] runT5Encoder(OrtEnvironment env, OrtSession session,
                                      long[] tokenIds, int maxSeqLen, int expectedDim) throws OrtException {
        String inName = resolveInputName(session, "input_ids", 0);
        TensorInfo tInfo = (TensorInfo) session.getInputInfo().get(inName).getInfo();
        boolean wantsInt32 = tInfo.type.toString().contains("INT32");
        OnnxTensor idsTensor;
        if (wantsInt32) {
            int[] ids32 = new int[tokenIds.length];
            for (int i = 0; i < tokenIds.length; i++) ids32[i] = (int) tokenIds[i];
            idsTensor = OnnxTensor.createTensor(env, new int[][]{ids32});
        } else {
            idsTensor = OnnxTensor.createTensor(env, new long[][]{tokenIds});
        }
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(inName, idsTensor);
        try (OrtSession.Result r = session.run(inputs)) {
            float[][][] raw = null;
            for (Map.Entry<String, OnnxValue> entry : r) {
                if (entry.getValue() instanceof OnnxTensor t) {
                    Object v = t.getValue();
                    if (v instanceof float[][][] a3) { raw = a3; break; }
                }
            }
            if (raw == null) {
                return new float[1][maxSeqLen][expectedDim]; // zeros fallback
            }
            // Pad or truncate to [1, maxSeqLen, expectedDim]
            int srcSeq = raw[0].length;
            int srcDim = raw[0][0].length;
            float[][][] result = new float[1][maxSeqLen][expectedDim];
            int copySeq = Math.min(srcSeq, maxSeqLen);
            int copyDim = Math.min(srcDim, expectedDim);
            for (int s = 0; s < copySeq; s++) {
                System.arraycopy(raw[0][s], 0, result[0][s], 0, copyDim);
            }
            return result;
        } finally {
            idsTensor.close();
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

    /* ================================================================== */
    /*  T5 (SentencePiece / Unigram) Tokenizer for T5-XXL                 */
    /* ================================================================== */

    /**
     * Minimal Unigram (SentencePiece) tokenizer that reads a HuggingFace
     * {@code tokenizer.json} file.  Supports the T5-XXL tokenizer used by
     * Stable Diffusion 3.x.
     *
     * <p>Implements:
     * <ul>
     *   <li>NFKC normalisation</li>
     *   <li>Metaspace pre-tokenisation (spaces → ▁)</li>
     *   <li>Viterbi best-path segmentation over the Unigram log-prob
     *       vocabulary</li>
     *   <li>Special-token handling (pad / eos / unk)</li>
     * </ul>
     */
    @SuppressWarnings("unchecked")
    private static final class T5Tokenizer {

        private final Map<String, Integer> pieceToId;
        private final float[] scores;
        private final int padId;
        private final int eosId;
        private final int unkId;
        private final int maxPieceLen;

        private T5Tokenizer(Map<String, Integer> pieceToId, float[] scores,
                            int padId, int eosId, int unkId, int maxPieceLen) {
            this.pieceToId = pieceToId;
            this.scores    = scores;
            this.padId     = padId;
            this.eosId     = eosId;
            this.unkId     = unkId;
            this.maxPieceLen = maxPieceLen;
        }

        /**
         * Load a T5 tokenizer from a HuggingFace {@code tokenizer.json}.
         *
         * Expected JSON structure (simplified):
         * <pre>{
         *   "model": {
         *     "type": "Unigram",
         *     "unk_id": 2,
         *     "vocab": [ ["▁", -1.23], ["s", -2.34], … ]
         *   },
         *   "added_tokens": [ {"id": 0, "content": "<pad>"}, … ]
         * }</pre>
         */
        static T5Tokenizer load(Path tokenizerJsonPath) throws Exception {
            Map<String, Object> root = OBJECT_MAPPER.readValue(
                    tokenizerJsonPath.toFile(), new TypeReference<>() {});
            Map<String, Object> modelSection = (Map<String, Object>) root.get("model");
            if (modelSection == null) {
                throw new IllegalArgumentException("tokenizer.json has no 'model' section");
            }
            List<List<Object>> vocab = (List<List<Object>>) modelSection.get("vocab");
            if (vocab == null || vocab.isEmpty()) {
                throw new IllegalArgumentException("tokenizer.json model has no 'vocab'");
            }

            int unkIdFromModel = modelSection.get("unk_id") instanceof Number n ? n.intValue() : 2;

            Map<String, Integer> pieceToId = new HashMap<>(vocab.size());
            float[] scoreArr = new float[vocab.size()];
            int maxLen = 1;
            for (int i = 0; i < vocab.size(); i++) {
                List<Object> entry = vocab.get(i);
                String piece = (String) entry.get(0);
                double score = entry.get(1) instanceof Number n ? n.doubleValue() : 0.0;
                pieceToId.put(piece, i);
                scoreArr[i] = (float) score;
                if (piece.length() > maxLen) maxLen = piece.length();
            }

            // Detect special tokens from added_tokens or vocab
            int padId = pieceToId.getOrDefault("<pad>", 0);
            int eosId = pieceToId.getOrDefault("</s>", 1);

            // Merge added_tokens (they may override or supplement vocab)
            if (root.containsKey("added_tokens")) {
                List<Map<String, Object>> addedTokens =
                        (List<Map<String, Object>>) root.get("added_tokens");
                for (Map<String, Object> at : addedTokens) {
                    String content = (String) at.get("content");
                    int id = at.get("id") instanceof Number n ? n.intValue() : -1;
                    if (content != null && id >= 0) {
                        pieceToId.put(content, id);
                        if ("<pad>".equals(content)) padId = id;
                        if ("</s>".equals(content))  eosId = id;
                    }
                }
            }

            return new T5Tokenizer(pieceToId, scoreArr, padId, eosId, unkIdFromModel, maxLen);
        }

        /**
         * Encode {@code text} into token IDs, padded/truncated to {@code maxLength}.
         * Mimics the HuggingFace T5 tokenizer: NFKC normalize → Metaspace
         * pre-tokenize → Unigram Viterbi → append EOS → pad.
         */
        long[] encode(String text, int maxLength) {
            String normalized = text == null ? "" : Normalizer.normalize(text, Normalizer.Form.NFKC);
            // Metaspace pre-tokenization: prefix with ▁, replace spaces with ▁
            String prepared = "\u2581" + normalized.replace(' ', '\u2581');

            List<Integer> ids = viterbi(prepared);
            ids.add(eosId);

            long[] out = new long[maxLength];
            Arrays.fill(out, padId);
            for (int i = 0; i < Math.min(maxLength, ids.size()); i++) {
                out[i] = ids.get(i);
            }
            return out;
        }

        /**
         * Viterbi best-path segmentation.  For each position i in the string,
         * find the best previous position j such that text[j..i] is a known
         * piece and the total log-probability is maximised.
         */
        private List<Integer> viterbi(String text) {
            int n = text.length();
            float[] bestScore = new float[n + 1];
            int[] bestEnd = new int[n + 1];      // best preceding boundary
            int[] bestPieceId = new int[n + 1];   // piece ID for segment [bestEnd[i]..i)
            Arrays.fill(bestScore, Float.NEGATIVE_INFINITY);
            Arrays.fill(bestPieceId, unkId);
            bestScore[0] = 0;

            for (int i = 1; i <= n; i++) {
                // Try all pieces ending at position i
                int lo = Math.max(0, i - maxPieceLen);
                for (int j = lo; j < i; j++) {
                    String sub = text.substring(j, i);
                    Integer id = pieceToId.get(sub);
                    if (id != null) {
                        float candidate = bestScore[j] + scores[id];
                        if (candidate > bestScore[i]) {
                            bestScore[i] = candidate;
                            bestEnd[i] = j;
                            bestPieceId[i] = id;
                        }
                    }
                }
                // If no piece matched, treat single char as unknown
                if (bestScore[i] == Float.NEGATIVE_INFINITY) {
                    bestScore[i] = bestScore[i - 1] + scores[unkId];
                    bestEnd[i] = i - 1;
                    bestPieceId[i] = unkId;
                }
            }

            // Backtrack to recover the segmentation
            List<Integer> ids = new ArrayList<>();
            int pos = n;
            while (pos > 0) {
                ids.add(bestPieceId[pos]);
                pos = bestEnd[pos];
            }
            // The list is in reverse order
            java.util.Collections.reverse(ids);
            return ids;
        }
    }

    /**
     * Build the EP preference list for the current platform, try each in order,
     * and return the first one that the loaded ONNX Runtime native library supports.
     *
     * <p>Priority order per platform (highest → lowest):
     * <ul>
     *   <li><b>macOS</b>: CoreML (GPU+ANE+CPU) → CPU</li>
     *   <li><b>Windows</b>: TensorRT-RTX → TensorRT → CUDA → DirectML → OpenVINO → CPU</li>
     *   <li><b>Linux</b>: TensorRT → CUDA → ROCm → OpenVINO → CPU</li>
     * </ul>
     *
     * <p>Override with {@code -Dlumenforge.ep=cuda} (or any key) to force a specific EP.
     */
    private ProviderSelection configureExecutionProvider(OrtSession.SessionOptions options, boolean preferGpu) {
        String os = System.getProperty("os.name", "unknown").toLowerCase();
        String arch = System.getProperty("os.arch", "unknown").toLowerCase();
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
            preference.add("tensorrt_rtx");   // RTX 30xx+ (Ampere+)
            preference.add("tensorrt");        // any NVIDIA with TensorRT libs
            preference.add("cuda");            // CUDA fallback
            preference.add("directml");        // AMD / Intel / any DX12 GPU
            preference.add("openvino");        // Intel CPUs/GPUs/NPUs
            preference.add("cpu");
        } else {
            // Linux
            preference.add("tensorrt");
            preference.add("cuda");
            preference.add("rocm");            // AMD GPUs
            preference.add("openvino");
            preference.add("cpu");
        }

        StringBuilder notes = new StringBuilder();
        List<String> failReasons = new ArrayList<>();
        AppLogger.app("EP preference order: " + preference
                + "  (os=" + os + ", arch=" + arch + ")");

        if (!preferGpu) {
            AppLogger.app("GPU not requested for this session — using CPUExecutionProvider");
            return new ProviderSelection("CPUExecutionProvider", "GPU not requested");
        }

        for (String candidate : preference) {
            if ("cpu".equals(candidate)) {
                // All GPU EPs exhausted — log summary
                AppLogger.appWarn("Falling back to CPUExecutionProvider");
                if (!failReasons.isEmpty()) {
                    AppLogger.appWarn("Reason: every GPU execution provider was unavailable:");
                    for (String reason : failReasons) {
                        AppLogger.appWarn("  - " + reason);
                    }
                    AppLogger.appWarn("Tip: install the matching GPU runtime "
                            + "(e.g. CUDA/cuDNN for NVIDIA, ROCm for AMD) or use "
                            + "-Dlumenforge.ep=<provider> to force a specific EP.");
                }
                return new ProviderSelection("CPUExecutionProvider", notes.toString());
            }
            String failReason = tryEnableProvider(options, candidate, notes);
            if (failReason == null) {
                String display = providerDisplayName(candidate);
                AppLogger.app("\u2713 Enabled " + display);
                return new ProviderSelection(display, notes.toString());
            }
            failReasons.add(failReason);
        }
        // Preference list didn't include "cpu" explicitly — shouldn't happen, but handle it
        AppLogger.appWarn("No EP available, falling back to CPUExecutionProvider");
        if (!failReasons.isEmpty()) {
            AppLogger.appWarn("Reasons:");
            for (String reason : failReasons) {
                AppLogger.appWarn("  - " + reason);
            }
        }
        return new ProviderSelection("CPUExecutionProvider", notes.toString());
    }

    /**
     * Attempt to enable a single EP via reflection.
     *
     * @return {@code null} if the provider was enabled successfully,
     *         or a human-readable reason string if it could not be enabled.
     */
    private String tryEnableProvider(OrtSession.SessionOptions options, String candidate, StringBuilder notes) {
        String failDetail = null;
        try {
            boolean ok = switch (candidate) {

                /* ── NVIDIA ───────────────────────────────────────────── */
                case "tensorrt_rtx" ->
                    // NvTensorRtRtxExecutionProvider – RTX 30xx+ only
                    // Java method not yet in stock Maven artifacts; try reflection just in case
                    invokeNoArg(options, "addNvTensorRtRtx")
                    || invokeIntArg(options, "addNvTensorRtRtx", 0);

                case "tensorrt" ->
                    // TensorrtExecutionProvider – available in onnxruntime_gpu
                    invokeIntArg(options, "addTensorrt", 0)
                    || invokeNoArg(options, "addTensorrt");

                case "cuda" ->
                    invokeNoArg(options, "addCUDA");

                /* ── Apple ────────────────────────────────────────────── */
                case "coreml" -> {
                    // addCoreML(long flags) — note: parameter is long, not int!
                    // Flag 0x0 = ALL compute units (CPU+GPU+ANE — best for M-series)
                    boolean coreOk = invokeLongArg(options, "addCoreML", 0L);
                    if (!coreOk) { coreOk = invokeNoArg(options, "addCoreML"); }
                    yield coreOk;
                }

                /* ── Microsoft ────────────────────────────────────────── */
                case "directml" ->
                    invokeIntArg(options, "addDirectML", 0)
                    || invokeNoArg(options, "addDirectML");

                /* ── Intel ────────────────────────────────────────────── */
                case "openvino" ->
                    // addOpenVINO(String) — device type "GPU" preferred, "CPU" fallback
                    invokeStringArg(options, "addOpenVINO", "GPU")
                    || invokeStringArg(options, "addOpenVINO", "CPU")
                    || invokeNoArg(options, "addOpenVINO");

                /* ── AMD ──────────────────────────────────────────────── */
                case "rocm" ->
                    invokeNoArg(options, "addROCM");

                default -> false;
            };
            if (!ok) {
                failDetail = candidate + ": native library not found in classpath";
            }
        } catch (Exception ex) {
            String msg = ex.getMessage();
            if (msg == null && ex.getCause() != null) { msg = ex.getCause().getMessage(); }
            failDetail = candidate + ": " + (msg != null ? msg : ex.getClass().getSimpleName());
        }

        if (failDetail != null) {
            if (!notes.isEmpty()) { notes.append("; "); }
            notes.append(candidate).append(" not available");
            AppLogger.appWarn("\u2717 " + failDetail);
        }
        return failDetail;
    }

    /* ── Reflection helpers for EP registration ──────────────────────── */

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

    private boolean invokeLongArg(OrtSession.SessionOptions options, String methodName, long arg) {
        try {
            options.getClass().getMethod(methodName, long.class).invoke(options, arg);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    private boolean invokeStringArg(OrtSession.SessionOptions options, String methodName, String arg) {
        try {
            options.getClass().getMethod(methodName, String.class).invoke(options, arg);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    private String providerDisplayName(String candidate) {
        return switch (candidate) {
            case "tensorrt_rtx" -> "NvTensorRtRtxExecutionProvider";
            case "tensorrt"     -> "TensorrtExecutionProvider";
            case "cuda"         -> "CUDAExecutionProvider";
            case "coreml"       -> "CoreMLExecutionProvider";
            case "directml"     -> "DmlExecutionProvider";
            case "openvino"     -> "OpenVINOExecutionProvider";
            case "rocm"         -> "ROCMExecutionProvider";
            default             -> "CPUExecutionProvider";
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
