package atri.palaash.jforge.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import atri.palaash.jforge.core.SD3Pipeline;
import atri.palaash.jforge.core.SDTurboPipeline;
import atri.palaash.jforge.core.SDXLPipeline;
import atri.palaash.jforge.core.StableDiffusionPipeline;
import atri.palaash.jforge.core.sampler.Sampler;
import atri.palaash.jforge.core.sampler.SamplerContext;
import atri.palaash.jforge.core.sampler.SchedulerSampler;
import atri.palaash.jforge.core.scheduler.EulerAncestralScheduler;
import atri.palaash.jforge.core.scheduler.EulerScheduler;
import atri.palaash.jforge.core.scheduler.FlowMatchingScheduler;
import atri.palaash.jforge.core.scheduler.Scheduler;
import atri.palaash.jforge.core.tensor.FloatTensor;
import atri.palaash.jforge.core.tokenizer.ClipTokenizer;
import atri.palaash.jforge.core.tokenizer.T5Tokenizer;
import atri.palaash.jforge.core.unet.OnnxUNet;
import atri.palaash.jforge.core.unet.UNet;
import atri.palaash.jforge.core.vae.OnnxVAE;
import atri.palaash.jforge.core.vae.VAE;
import atri.palaash.jforge.model.TaskType;
import atri.palaash.jforge.runtime.MemoryGuard;
import atri.palaash.jforge.runtime.SessionManager;
import atri.palaash.jforge.storage.ModelStorage;

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
    /**
     * Maximum number of ONNX sessions to keep cached simultaneously.
     * Each session can consume 200 MB – 6 GB+ of native memory, so we
     * cap the cache to prevent OOM.  An LRU insertion-order policy
     * evicts the oldest session when this limit is reached.
     */
    private static final int MAX_CACHED_SESSIONS = 5;
    private static final SessionManager SESSION_MANAGER = new SessionManager(MAX_CACHED_SESSIONS);
    private static final MemoryGuard MEMORY_GUARD = new MemoryGuard();
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
    private synchronized OrtSession getOrCreateSession(OrtEnvironment env, Path modelPath,
                                           OrtSession.SessionOptions opts) throws OrtException {
        System.out.println("[JForge] Loading ONNX session: " + modelPath.getFileName());
        OrtSession session = SESSION_MANAGER.getSession(env, modelPath, opts);
        System.out.println("[JForge] Session loaded: " + modelPath.getFileName());
        return session;
    }

    /**
     * Close and remove a specific session from the cache to free native memory.
     * Used for sequential model loading in large pipelines (e.g. SD 3.5)
     * where text encoders are only needed temporarily.
     */
    private synchronized void evictSession(Path modelPath) {
        SESSION_MANAGER.evictSession(modelPath);
        System.out.println("[JForge] Evicted session: " + modelPath.getFileName());
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
    public static synchronized void clearCache() {
        SESSION_MANAGER.clearCache();
        TOKENIZER_CACHE.clear();
        T5_TOKENIZER_CACHE.clear();
        cachedEpKey = "";
    }

    @Override
    public CompletableFuture<InferenceResult> run(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            if (!storage.isAvailable(request.model())) {
                System.out.println("[JForge] ERROR: Model not found locally: " + request.model().displayName());
                return InferenceResult.fail("Model not found locally. Open Model Manager from the menu bar and download it first.");
            }

            Path modelPath = storage.modelPath(request.model());
            System.out.println("[JForge] Starting inference: " + request.model().displayName()
                    + " (" + request.model().id() + ")");

            MemoryGuard.Decision memoryDecision = MEMORY_GUARD.assess(request);
            if (!memoryDecision.allowed()) {
                System.out.println("[JForge] ERROR: " + memoryDecision.details());
                return InferenceResult.fail(memoryDecision.details());
            }
            request.reportProgress(memoryDecision.details());

            // Temporarily intercept stderr so ONNX Runtime native warnings
            // appear in the application's Log tab instead of only in the console.
            PrintStream origErr = System.err;
            TeeOutputStream tee = new TeeOutputStream(origErr, request.progressCallback());
            System.setErr(new PrintStream(tee, true));

            try {
                request.reportProgress("Loading ONNX Runtime environment\u2026");
                OrtEnvironment environment = OrtEnvironment.getEnvironment();
                try (OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions()) {
                sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                int cpus = Runtime.getRuntime().availableProcessors();
                sessionOptions.setIntraOpNumThreads(Math.max(1, cpus - 1));
                sessionOptions.setInterOpNumThreads(Math.max(1, Math.min(cpus / 2, 4)));
                ProviderSelection providerSelection = configureExecutionProvider(sessionOptions, request.preferGpu());
                System.out.println("[JForge] Using EP: " + providerSelection.provider()
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
                System.out.println("[JForge] WARN: " + details);
                return InferenceResult.fail(details);
                } // close try(SessionOptions)
            } catch (OrtException ex) {
                String message = ex.getMessage() == null ? "Unknown ONNX Runtime error" : ex.getMessage();
                System.out.println("[JForge] ERROR: ONNX Runtime error: " + message);
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
            int width  = Math.max(256, (request.width()  / 8) * 8);
            int height = Math.max(256, (request.height() / 8) * 8);
            int steps = Math.max(1, request.batch());

            Path base = storage.root().resolve("text-image").resolve("stable-diffusion-v15");
            Path textEncoderPath = base.resolve("text_encoder/model.onnx");
            Path unetPath        = base.resolve("unet/model.onnx");
            Path vaeDecoderPath  = base.resolve("vae_decoder/model.onnx");
            Path vocabPath   = base.resolve("tokenizer/vocab.json");
            Path mergesPath  = base.resolve("tokenizer/merges.txt");

            if (!java.nio.file.Files.exists(textEncoderPath) || !java.nio.file.Files.exists(unetPath) || !java.nio.file.Files.exists(vaeDecoderPath)) {
                return InferenceResult.fail("Stable Diffusion v1.5 bundle is incomplete.");
            }

            ClipTokenizer tokenizer = ClipTokenizer.load(vocabPath, mergesPath);
            long[] promptTokens = tokenizer.encode(request.prompt(), 77);
            long[] negTokens = tokenizer.encode(request.negativePrompt() == null ? "" : request.negativePrompt(), 77);

            request.reportProgress("Loading SD v1.5 models (Modular)\u2026");
            OrtSession textEncoder = getOrCreateSession(environment, textEncoderPath, sessionOptions);
            OrtSession unet = getOrCreateSession(environment, unetPath, sessionOptions);
            OrtSession vaeDecoder = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);

            request.reportProgress("Encoding text prompts\u2026");
            float[][][] posEmbed = runTextEncoder(environment, textEncoder, promptTokens);
            float[][][] negEmbed = runTextEncoder(environment, textEncoder, negTokens);
            
            // Batch for CFG: [neg, pos]
            float[] batchedVals = new float[2 * 77 * 768];
            System.arraycopy(flatten3d(negEmbed), 0, batchedVals, 0, 77 * 768);
            System.arraycopy(flatten3d(posEmbed), 0, batchedVals, 77 * 768, 77 * 768);
            FloatTensor textEmbeddings = FloatTensor.of(new int[]{2, 77, 768}, batchedVals);

            atri.palaash.jforge.core.StableDiffusionPipeline pipeline = new atri.palaash.jforge.core.StableDiffusionPipeline(environment, unet, vaeDecoder);
            
            // SD 1.5 usually uses Euler Ancestral or Euler
            Scheduler scheduler = new EulerAncestralScheduler();
            
            FloatTensor decodedTensor = pipeline.run(request, textEmbeddings, scheduler);

            BufferedImage image = tensorToImage(decodedTensor);
            Path outputPath = writeOutputImage(image, "sd-v15-modular");
            return InferenceResult.ok(
                    "Generated image for prompt: \"" + request.prompt() + "\"",
                    "Modular SD v1.5 pipeline completed | EP=" + provider,
                    outputPath.toString(), "image");
        } catch (Exception ex) {
            return InferenceResult.fail("Stable Diffusion v1.5 pipeline failed: " + ex.getMessage());
        }
    }


    /* ================================================================== */
    /*  SD Turbo — distilled 1–4 step pipeline (no classifier-free        */
    /*  guidance, uses Euler-based single-step scheduler).                 */
    /* ================================================================== */

    private float[] flatten3d(float[][][] arr) {
        int d1 = arr.length, d2 = arr[0].length, d3 = arr[0][0].length;
        float[] flat = new float[d1 * d2 * d3];
        int i = 0;
        for (float[][] m2 : arr) for (float[] m1 : m2) for (float v : m1) flat[i++] = v;
        return flat;
    }

    private float[] flatten4d(float[][][][] arr) {
        int d1 = arr.length, d2 = arr[0].length, d3 = arr[0][0].length, d4 = arr[0][0][0].length;
        float[] flat = new float[d1 * d2 * d3 * d4];
        int i = 0;
        for (float[][][] m3 : arr) for (float[][] m2 : m3) for (float[] m1 : m2) for (float v : m1) flat[i++] = v;
        return flat;
    }

    private BufferedImage tensorToImage(FloatTensor tensor) {
        int channels = tensor.dimension(1);
        int height = tensor.dimension(2);
        int width = tensor.dimension(3);
        float[] vals = tensor.values();
        float[][][] arr = new float[channels][height][width];
        int idx = 0;
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    arr[c][y][x] = vals[idx++];
        return tensorToImage(arr);
    }

    private InferenceResult runStableDiffusionTurbo(OrtEnvironment environment,
                                                    OrtSession.SessionOptions sessionOptions,
                                                    InferenceRequest request,
                                                    String provider) {
        try {
            int width  = Math.max(256, (request.width()  / 8) * 8);
            int height = Math.max(256, (request.height() / 8) * 8);
            int steps = Math.max(1, Math.min(request.batch() > 0 ? request.batch() : 4, 8));

            Path base = storage.root().resolve("text-image").resolve("sd-turbo");
            Path textEncoderPath = base.resolve("text_encoder/model.onnx");
            Path unetPath        = base.resolve("unet/model.onnx");
            Path vaeDecoderPath  = base.resolve("vae_decoder/model.onnx");
            Path vocabPath   = resolveTokenizerFile(base, "tokenizer/vocab.json");
            Path mergesPath  = resolveTokenizerFile(base, "tokenizer/merges.txt");

            for (Path p : List.of(textEncoderPath, unetPath, vaeDecoderPath, vocabPath, mergesPath)) {
                if (!java.nio.file.Files.exists(p)) {
                    return InferenceResult.fail("SD Turbo bundle is incomplete. Download components via Model Manager.");
                }
            }

            ClipTokenizer tokenizer = ClipTokenizer.load(vocabPath, mergesPath);
            long[] promptTokens = tokenizer.encode(request.prompt(), 77);

            request.reportProgress("Loading SD Turbo models (Modular)\u2026");
            OrtSession textEncoder = getOrCreateSession(environment, textEncoderPath, sessionOptions);
            OrtSession unet = getOrCreateSession(environment, unetPath, sessionOptions);
            OrtSession vaeDecoder = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);

            request.reportProgress("Encoding text prompt\u2026");
            float[][][] textEmbeddingsRaw = runTextEncoder(environment, textEncoder, promptTokens);
            FloatTensor textEmbeddings = FloatTensor.of(new int[]{1, 77, 768}, flatten3d(textEmbeddingsRaw));

            atri.palaash.jforge.core.SDTurboPipeline pipeline = new atri.palaash.jforge.core.SDTurboPipeline(environment, unet, vaeDecoder);
            FloatTensor decodedTensor = pipeline.run(request, textEmbeddings);

            BufferedImage image = tensorToImage(decodedTensor);
            Path outputPath = writeOutputImage(image, "sd-turbo-modular");
            return InferenceResult.ok(
                    "Generated image for prompt: \"" + request.prompt() + "\"",
                    "Modular SD Turbo pipeline completed (" + steps + " steps) | EP=" + provider,
                    outputPath.toString(), "image");
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
            int steps = Math.max(1, request.batch());

            Path base = storage.root().resolve("text-image").resolve("sdxl-base");
            Path textEncoder1Path = base.resolve("text_encoder/model.onnx");
            Path textEncoder2Path = base.resolve("text_encoder_2/model.onnx");
            Path unetPath         = base.resolve("unet/model.onnx");
            Path vaeDecoderPath   = base.resolve("vae_decoder/model.onnx");
            Path vocab1 = base.resolve("tokenizer/vocab.json");
            Path merges1 = base.resolve("tokenizer/merges.txt");
            Path vocab2 = base.resolve("tokenizer_2/vocab.json");
            Path merges2 = base.resolve("tokenizer_2/merges.txt");

            if (!java.nio.file.Files.exists(textEncoder1Path) || !java.nio.file.Files.exists(textEncoder2Path) || 
                !java.nio.file.Files.exists(unetPath) || !java.nio.file.Files.exists(vaeDecoderPath)) {
                return InferenceResult.fail("SDXL Base bundle is incomplete.");
            }

            ClipTokenizer tok1 = ClipTokenizer.load(vocab1, merges1);
            ClipTokenizer tok2 = ClipTokenizer.load(vocab2, merges2);
            long[] tokens1 = tok1.encode(request.prompt(), 77);
            long[] tokens2 = tok2.encode(request.prompt(), 77);
            long[] negTokens1 = tok1.encode(request.negativePrompt() == null ? "" : request.negativePrompt(), 77);
            long[] negTokens2 = tok2.encode(request.negativePrompt() == null ? "" : request.negativePrompt(), 77);

            request.reportProgress("Loading SDXL models (Modular)\u2026");
            OrtSession enc1 = getOrCreateSession(environment, textEncoder1Path, sessionOptions);
            OrtSession enc2 = getOrCreateSession(environment, textEncoder2Path, sessionOptions);
            OrtSession unet = getOrCreateSession(environment, unetPath, sessionOptions);
            OrtSession vae  = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);

            request.reportProgress("Encoding prompts (Dual CLIP)\u2026");
            // Pos
            float[][][] embed1 = runTextEncoder(environment, enc1, tokens1);
            var enc2ResPos = runSdxlTextEncoder2(environment, enc2, tokens2);
            // Neg
            float[][][] negEmbed1 = runTextEncoder(environment, enc1, negTokens1);
            var enc2ResNeg = runSdxlTextEncoder2(environment, enc2, negTokens2);

            // Combine embeddings: [1, 77, 768] + [1, 77, 1280] -> [1, 77, 2048]
            float[] combinedPos = combineSdxlEmbeds(embed1, enc2ResPos.hidden);
            float[] combinedNeg = combineSdxlEmbeds(negEmbed1, enc2ResNeg.hidden);
            
            float[] batchedHidden = new float[2 * 77 * 2048];
            System.arraycopy(combinedNeg, 0, batchedHidden, 0, 77 * 2048);
            System.arraycopy(combinedPos, 0, batchedHidden, 77 * 2048, 77 * 2048);
            FloatTensor textEmbeddings = FloatTensor.of(new int[]{2, 77, 2048}, batchedHidden);

            // Batch pooled: [2, 1280]
            float[] batchedPooled = new float[2 * 1280];
            System.arraycopy(enc2ResNeg.pooled[0], 0, batchedPooled, 0, 1280);
            System.arraycopy(enc2ResPos.pooled[0], 0, batchedPooled, 1280, 1280);
            FloatTensor pooledEmbeddings = FloatTensor.of(new int[]{2, 1280}, batchedPooled);

            SDXLPipeline pipeline = new SDXLPipeline(environment, unet, vae);
            Scheduler scheduler = new EulerScheduler();
            
            FloatTensor decodedTensor = pipeline.run(request, textEmbeddings, pooledEmbeddings, scheduler);

            BufferedImage image = tensorToImage(decodedTensor);
            Path outputPath = writeOutputImage(image, "sdxl-base-modular");
            return InferenceResult.ok(
                    "Generated image for prompt: \"" + request.prompt() + "\"",
                    "Modular SDXL Base pipeline completed | EP=" + provider,
                    outputPath.toString(), "image");
        } catch (Exception ex) {
            return InferenceResult.fail("SDXL Base pipeline failed: " + ex.getMessage());
        }
    }

    private record SdxlEnc2Result(float[][][] hidden, float[][] pooled) {}

    private SdxlEnc2Result runSdxlTextEncoder2(OrtEnvironment env, OrtSession session, long[] tokens) throws Exception {
        String inName = resolveInputName(session, "input_ids", 0);
        OnnxTensor idsTensor = OnnxTensor.createTensor(env, new long[][]{tokens});
        try (OrtSession.Result r = session.run(Map.of(inName, idsTensor))) {
            float[][][] hidden = null;
            float[][] pooled = null;
            for (var entry : r) {
                if (entry.getValue() instanceof OnnxTensor t) {
                    Object v = t.getValue();
                    if (v instanceof float[][][] a3) hidden = a3;
                    else if (v instanceof float[][] a2) pooled = a2;
                }
            }
            return new SdxlEnc2Result(hidden, pooled);
        } finally { idsTensor.close(); }
    }

    private float[] combineSdxlEmbeds(float[][][] e1, float[][][] e2) {
        int seqLen = 77;
        float[] combined = new float[seqLen * 2048];
        for (int i = 0; i < seqLen; i++) {
            System.arraycopy(e1[0][i], 0, combined, i * 2048, 768);
            System.arraycopy(e2[0][i], 0, combined, i * 2048 + 768, 1280);
        }
        return combined;
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
            int steps = Math.max(1, request.batch());

            Path modelPath = storage.modelPath(request.model());
            Path base = modelPath.getParent().getParent();

            Path transformerPath = base.resolve("transformer/model.onnx");
            Path textEncoder1Path = base.resolve("text_encoder/model.onnx");
            Path textEncoder2Path = base.resolve("text_encoder_2/model.onnx");
            Path vaeDecoderPath   = base.resolve("vae_decoder/model.onnx");
            Path vocab1 = base.resolve("tokenizer/vocab.json");
            Path merges1 = base.resolve("tokenizer/merges.txt");
            Path vocab2 = base.resolve("tokenizer_2/vocab.json");
            Path merges2 = base.resolve("tokenizer_2/merges.txt");

            // Optional T5
            Path textEncoder3Path = base.resolve("text_encoder_3/model.onnx");
            Path tokenizer3Json   = base.resolve("tokenizer_3/tokenizer.json");
            boolean hasT5 = java.nio.file.Files.exists(textEncoder3Path) && java.nio.file.Files.exists(tokenizer3Json);

            ClipTokenizer tok1 = ClipTokenizer.load(vocab1, merges1);
            ClipTokenizer tok2 = ClipTokenizer.load(vocab2, merges2);
            T5Tokenizer tok3 = hasT5 ? T5Tokenizer.load(tokenizer3Json) : null;

            request.reportProgress("Encoding prompts (Triple Encoders)\u2026");
            
            // Sequential encoding to save memory
            OrtSession enc1 = getOrCreateSession(environment, textEncoder1Path, sessionOptions);
            float[][][] embed1 = runTextEncoder(environment, enc1, tok1.encode(request.prompt(), 77));
            float[][] pool1 = runTextEncoderPooled(environment, enc1, tok1.encode(request.prompt(), 77));
            float[][][] negEmbed1 = runTextEncoder(environment, enc1, tok1.encode(request.negativePrompt(), 77));
            float[][] negPool1 = runTextEncoderPooled(environment, enc1, tok1.encode(request.negativePrompt(), 77));
            
            OrtSession enc2 = getOrCreateSession(environment, textEncoder2Path, sessionOptions);
            var enc2ResPos = runSdxlTextEncoder2(environment, enc2, tok2.encode(request.prompt(), 77));
            var enc2ResNeg = runSdxlTextEncoder2(environment, enc2, tok2.encode(request.negativePrompt(), 77));
            
            float[][][] t5EmbedPos = null;
            float[][][] t5EmbedNeg = null;
            if (hasT5) {
                OrtSession enc3 = getOrCreateSession(environment, textEncoder3Path, sessionOptions);
                t5EmbedPos = runT5Encoder(environment, enc3, tok3.encode(request.prompt(), 256), 256, 4096);
                t5EmbedNeg = runT5Encoder(environment, enc3, tok3.encode(request.negativePrompt(), 256), 256, 4096);
            }

            // Combine into SD3 batched hidden [2, 333, 4096] and pooled [2, 2048]
            float[] batchedHidden = combineSd3Embeds(embed1, negEmbed1, enc2ResPos.hidden, enc2ResNeg.hidden, t5EmbedPos, t5EmbedNeg);
            FloatTensor textEmbeddings = FloatTensor.of(new int[]{2, 333, 4096}, batchedHidden);
            
            float[] batchedPooled = combineSd3Pooled(pool1, negPool1, enc2ResPos.pooled, enc2ResNeg.pooled);
            FloatTensor pooledEmbeddings = FloatTensor.of(new int[]{2, 2048}, batchedPooled);

            OrtSession transformer = getOrCreateSession(environment, transformerPath, sessionOptions);
            OrtSession vae = getOrCreateSession(environment, vaeDecoderPath, sessionOptions);

            SD3Pipeline pipeline = new SD3Pipeline(environment, transformer, vae);
            Scheduler scheduler = new FlowMatchingScheduler();
            
            FloatTensor decodedTensor = pipeline.run(request, textEmbeddings, pooledEmbeddings, scheduler);

            BufferedImage image = tensorToImage(decodedTensor);
            Path outputPath = writeOutputImage(image, "sd3-modular");
            return InferenceResult.ok(
                    "Generated image for prompt: \"" + request.prompt() + "\"",
                    "Modular SD3 pipeline completed | EP=" + provider,
                    outputPath.toString(), "image");
        } catch (Exception ex) {
            return InferenceResult.fail("SD3 pipeline failed: " + ex.getMessage());
        }
    }

    private float[] combineSd3Embeds(float[][][] e1, float[][][] ne1, float[][][] e2, float[][][] ne2, float[][][] e3, float[][][] ne3) {
        float[] combined = new float[2 * 333 * 4096];
        // Combine into negative [0] then positive [1]
        fillSd3Side(combined, 0, ne1, ne2, ne3);
        fillSd3Side(combined, 1, e1, e2, e3);
        return combined;
    }

    private void fillSd3Side(float[] dest, int batchIdx, float[][][] e1, float[][][] e2, float[][][] e3) {
        int offset = batchIdx * 333 * 4096;
        for (int i = 0; i < 77; i++) {
            System.arraycopy(e1[0][i], 0, dest, offset + i * 4096, 768);
            System.arraycopy(e2[0][i], 0, dest, offset + i * 4096 + 768, 1280);
        }
        if (e3 != null) {
            for (int i = 0; i < 256; i++) {
                System.arraycopy(e3[0][i], 0, dest, offset + (77 + i) * 4096, 4096);
            }
        }
    }

    private float[] combineSd3Pooled(float[][] p1, float[][] np1, float[][] p2, float[][] np2) {
        float[] combined = new float[2 * 2048];
        System.arraycopy(np1[0], 0, combined, 0, 768);
        System.arraycopy(np2[0], 0, combined, 768, 1280);
        System.arraycopy(p1[0], 0, combined, 2048, 768);
        System.arraycopy(p2[0], 0, combined, 2048 + 768, 1280);
        return combined;
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
     * <p>Override with {@code -Djforge.ep=cuda} (or any key) to force a specific EP.
     */
    private ProviderSelection configureExecutionProvider(OrtSession.SessionOptions options, boolean preferGpu) {
        String os = System.getProperty("os.name", "unknown").toLowerCase();
        String arch = System.getProperty("os.arch", "unknown").toLowerCase();
        String forced = System.getProperty("jforge.ep", "").trim().toLowerCase();

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
        System.out.println("[JForge] EP preference order: " + preference
                + "  (os=" + os + ", arch=" + arch + ")");

        if (!preferGpu) {
            System.out.println("[JForge] GPU not requested for this session — using CPUExecutionProvider");
            return new ProviderSelection("CPUExecutionProvider", "GPU not requested");
        }

        for (String candidate : preference) {
            if ("cpu".equals(candidate)) {
                // All GPU EPs exhausted — log summary
                System.out.println("[JForge] WARN: Falling back to CPUExecutionProvider");
                if (!failReasons.isEmpty()) {
                    System.out.println("[JForge] WARN: Reason: every GPU execution provider was unavailable:");
                    for (String reason : failReasons) {
                        System.out.println("[JForge] WARN:   - " + reason);
                    }
                    System.out.println("[JForge] WARN: Tip: install the matching GPU runtime "
                            + "(e.g. CUDA/cuDNN for NVIDIA, ROCm for AMD) or use "
                            + "-Djforge.ep=<provider> to force a specific EP.");
                }
                return new ProviderSelection("CPUExecutionProvider", notes.toString());
            }
            String failReason = tryEnableProvider(options, candidate, notes);
            if (failReason == null) {
                String display = providerDisplayName(candidate);
                System.out.println("[JForge] \u2713 Enabled " + display);
                return new ProviderSelection(display, notes.toString());
            }
            failReasons.add(failReason);
        }
        // Preference list didn't include "cpu" explicitly — shouldn't happen, but handle it
        System.out.println("[JForge] WARN: No EP available, falling back to CPUExecutionProvider");
        if (!failReasons.isEmpty()) {
            System.out.println("[JForge] WARN: Reasons:");
            for (String reason : failReasons) {
                System.out.println("[JForge] WARN:   - " + reason);
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
            System.out.println("[JForge] WARN: \u2717 " + failDetail);
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
        private static final int MAX_LINE_BUFFER = 64 * 1024; // 64 KB guard
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
                if (lineBuffer.size() > MAX_LINE_BUFFER) flushLine();
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
                    if (lineBuffer.size() > MAX_LINE_BUFFER) flushLine();
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
