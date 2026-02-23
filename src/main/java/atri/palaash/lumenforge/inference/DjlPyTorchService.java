package atri.palaash.lumenforge.inference;

import atri.palaash.lumenforge.storage.ModelStorage;

import java.awt.image.BufferedImage;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/**
 * Stable Diffusion inference using DJL (Deep Java Library) with PyTorch backend.
 *
 * <p>This backend loads TorchScript-traced models (.pt files) and runs inference
 * using PyTorch's native tensor operations (CPU, CUDA, or MPS). Activate by
 * building with {@code mvn compile -Ddjl=true}.
 *
 * <h3>Required model layout</h3>
 * <pre>
 *   ~/.lumenforge-models/text-image/sd-pytorch/
 *     clip_model.pt           — traced CLIP text encoder
 *     unet_model.pt           — traced UNet2DConditionModel
 *     vae_decoder_model.pt    — traced VAE decoder
 *     tokenizer.json          — HuggingFace tokenizer config
 * </pre>
 *
 * <p>Use the included {@code scripts/export_torchscript.py} helper to export
 * HuggingFace SD checkpoints to TorchScript format.
 */
public class DjlPyTorchService implements InferenceService {

    private final ModelStorage storage;
    private final Executor executor;

    /* ── DJL availability check  ─────────────────────────────────────── */

    private static final boolean DJL_AVAILABLE;
    static {
        boolean found;
        try {
            Class.forName("ai.djl.ndarray.NDManager");
            found = true;
        } catch (ClassNotFoundException e) {
            found = false;
        }
        DJL_AVAILABLE = found;
    }

    /** Returns {@code true} if the DJL PyTorch runtime is on the classpath. */
    public static boolean isAvailable() {
        return DJL_AVAILABLE;
    }

    public DjlPyTorchService(ModelStorage storage, Executor executor) {
        this.storage = storage;
        this.executor = executor;
    }

    @Override
    public CompletableFuture<InferenceResult> run(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            if (!DJL_AVAILABLE) {
                return InferenceResult.fail(
                        "DJL PyTorch is not on the classpath. Rebuild with: mvn compile -Ddjl=true");
            }

            Path base = storage.root().resolve("text-image").resolve("sd-pytorch");
            Path clipPath  = base.resolve("clip_model.pt");
            Path unetPath  = base.resolve("unet_model.pt");
            Path vaePath   = base.resolve("vae_decoder_model.pt");
            Path tokPath   = base.resolve("tokenizer.json");

            for (Path p : new Path[]{clipPath, unetPath, vaePath, tokPath}) {
                if (!Files.exists(p)) {
                    return InferenceResult.fail(
                            "Missing TorchScript component: " + p.getFileName()
                                    + "\nExport models with scripts/export_torchscript.py first.");
                }
            }

            try {
                return runPyTorchPipeline(request, clipPath, unetPath, vaePath, tokPath);
            } catch (Exception ex) {
                return InferenceResult.fail("DJL PyTorch pipeline failed: " + ex.getMessage());
            }
        }, executor);
    }

    /* ── Pipeline (reflection-based to compile without DJL on classpath) ─ */

    @SuppressWarnings("unchecked")
    private InferenceResult runPyTorchPipeline(InferenceRequest request,
                                               Path clipPath, Path unetPath,
                                               Path vaePath, Path tokPath)
            throws Exception {

        // All DJL calls go through reflection so the class compiles even when
        // the djl-pytorch profile is not active.

        Class<?> clsNDManager    = Class.forName("ai.djl.ndarray.NDManager");
        Class<?> clsNDArray      = Class.forName("ai.djl.ndarray.NDArray");
        Class<?> clsNDList       = Class.forName("ai.djl.ndarray.NDList");
        Class<?> clsShape        = Class.forName("ai.djl.ndarray.types.Shape");
        Class<?> clsDataType     = Class.forName("ai.djl.ndarray.types.DataType");
        Class<?> clsDevice       = Class.forName("ai.djl.Device");
        Class<?> clsCriteria     = Class.forName("ai.djl.repository.zoo.Criteria");
        Class<?> clsZooModel     = Class.forName("ai.djl.repository.zoo.ZooModel");
        Class<?> clsPredictor    = Class.forName("ai.djl.inference.Predictor");
        Class<?> clsTokenizer    = Class.forName("ai.djl.huggingface.tokenizers.HuggingFaceTokenizer");
        Class<?> clsEncoding     = Class.forName("ai.djl.huggingface.tokenizers.Encoding");
        Class<?> clsTranslator   = Class.forName("ai.djl.translate.NoopTranslator");

        int width  = Math.max(256, (request.width()  / 8) * 8);
        int height = Math.max(256, (request.height() / 8) * 8);
        int latentW = width  / 8;
        int latentH = height / 8;
        int steps   = Math.max(1, Math.min(request.batch() > 0 ? request.batch() : 20, 50));

        // Choose device — prefer GPU if available
        Object device;
        if (request.preferGpu()) {
            Method gpuMethod = clsDevice.getMethod("gpu");
            device = gpuMethod.invoke(null);
        } else {
            Method cpuMethod = clsDevice.getMethod("cpu");
            device = cpuMethod.invoke(null);
        }

        // Create NDManager
        Method managerOf = clsNDManager.getMethod("newBaseManager", clsDevice);
        Object manager = managerOf.invoke(null, device);

        try {
            /* ── Tokenize ──────────────────────────────────────────────── */
            request.reportProgress("Tokenizing prompt (DJL HuggingFace)\u2026");
            Method tokLoad = clsTokenizer.getMethod("newInstance", Path.class);
            Object tokenizer = tokLoad.invoke(null, tokPath);
            Method tokEncode = clsTokenizer.getMethod("encode", String.class);
            Object encoding = tokEncode.invoke(tokenizer, request.prompt());
            Method getIds = clsEncoding.getMethod("getIds");
            long[] tokenIds = (long[]) getIds.invoke(encoding);

            // Pad/truncate to 77
            long[] padded = new long[77];
            System.arraycopy(tokenIds, 0, padded, 0, Math.min(tokenIds.length, 77));
            if (tokenIds.length < 77) {
                long padId = 49407L; // <|endoftext|>
                for (int i = tokenIds.length; i < 77; i++) padded[i] = padId;
            }

            // Create token tensor [1, 77]
            Object shapeTokens = clsShape.getConstructor(long[].class)
                    .newInstance((Object) new long[]{1, 77});
            Method mgrCreate = clsNDManager.getMethod("create", long[].class,
                    Class.forName("ai.djl.ndarray.types.Shape"));
            Object tokenTensor = mgrCreate.invoke(manager, padded, shapeTokens);

            /* ── CLIP text encoder ──────────────────────────────────── */
            request.reportProgress("Loading CLIP text encoder (PyTorch)\u2026");
            Object clipModel = loadTorchScriptModel(clsCriteria, clsNDList, clsTranslator,
                    clipPath, device);
            Object clipPredictor = clsZooModel.getMethod("newPredictor").invoke(clipModel);

            Object clipInput = clsNDList.getConstructor(clsNDArray).newInstance(tokenTensor);
            request.reportProgress("Running CLIP text encoder\u2026");
            Object clipOutput = clsPredictor.getMethod("predict", Object.class)
                    .invoke(clipPredictor, clipInput);
            Object textEmbedding = clsNDList.getMethod("singletonOrThrow").invoke(clipOutput);

            /* ── Negative prompt (uncond) encoder ───────────────────── */
            long[] emptyTokens = new long[77];
            emptyTokens[0] = 49406L; // <|startoftext|>
            for (int i = 1; i < 77; i++) emptyTokens[i] = 49407L;
            Object uncondTensor = mgrCreate.invoke(manager, emptyTokens, shapeTokens);
            Object uncondInput = clsNDList.getConstructor(clsNDArray).newInstance(uncondTensor);
            request.reportProgress("Running uncond text encoder\u2026");
            Object uncondOutput = clsPredictor.getMethod("predict", Object.class)
                    .invoke(clipPredictor, uncondInput);
            Object uncondEmbedding = clsNDList.getMethod("singletonOrThrow").invoke(uncondOutput);

            // Concatenate [uncond, cond] → [2, 77, 768]
            Method concatMethod = clsNDArray.getMethod("concat", clsNDArray, int.class);
            // Stack along batch dim: uncond embedding [1,77,768] + text embedding [1,77,768]
            Object embeddings = concatMethod.invoke(uncondEmbedding, textEmbedding, 0);

            clsPredictor.getMethod("close").invoke(clipPredictor);
            clsZooModel.getMethod("close").invoke(clipModel);

            /* ── UNet denoising loop ────────────────────────────────── */
            request.reportProgress("Loading UNet (PyTorch)\u2026");
            Object unetModel = loadTorchScriptModel(clsCriteria, clsNDList, clsTranslator,
                    unetPath, device);
            Object unetPredictor = clsZooModel.getMethod("newPredictor").invoke(unetModel);

            // Initialize random latents [1, 4, latentH, latentW]
            Object latentShape = clsShape.getConstructor(long[].class)
                    .newInstance((Object) new long[]{1, 4, latentH, latentW});
            Method mgrRandomNormal = clsNDManager.getMethod("randomNormal",
                    Class.forName("ai.djl.ndarray.types.Shape"));
            Object latents = mgrRandomNormal.invoke(manager, latentShape);

            // Simple linear schedule for beta
            double guidanceScale = request.promptWeight() > 0 ? request.promptWeight() : 7.5;
            float[] timestepSchedule = ddimTimesteps(steps);

            // Initial scale
            Method mulMethod = clsNDArray.getMethod("mul", Number.class);
            latents = mulMethod.invoke(latents, (float) Math.sqrt(1.0 + sigmaForTimestep(timestepSchedule[0])));

            long stepStart = System.currentTimeMillis();
            for (int i = 0; i < timestepSchedule.length; i++) {
                if (request.isCancelled()) {
                    clsPredictor.getMethod("close").invoke(unetPredictor);
                    clsZooModel.getMethod("close").invoke(unetModel);
                    return InferenceResult.fail("Cancelled by user.");
                }

                float t = timestepSchedule[i];

                // Duplicate latents for CFG: [latents, latents] → [2, 4, H, W]
                Object latentDouble = concatMethod.invoke(latents, latents, 0);

                // Create timestep tensor
                Object tsShape = clsShape.getConstructor(long[].class)
                        .newInstance((Object) new long[]{1});
                Object tsTensor = mgrCreate.invoke(manager, new long[]{(long) t}, tsShape);

                // UNet forward: (noisy_latents, timestep, encoder_hidden_states)
                Object unetInput = clsNDList.getConstructor(clsNDArray.arrayType())
                        .newInstance((Object) new Object[]{latentDouble, tsTensor, embeddings});
                Object unetOutput = clsPredictor.getMethod("predict", Object.class)
                        .invoke(unetPredictor, unetInput);
                Object noisePred = clsNDList.getMethod("singletonOrThrow").invoke(unetOutput);

                // Classifier-free guidance: split [2,4,H,W] → uncond, cond
                Method splitMethod = clsNDArray.getMethod("split", long.class, int.class);
                Object splitResult = splitMethod.invoke(noisePred, 2L, 0);
                Object noiseUncond = java.lang.reflect.Array.get(
                        clsNDList.getMethod("toArray").invoke(splitResult), 0);
                Object noiseCond = java.lang.reflect.Array.get(
                        clsNDList.getMethod("toArray").invoke(splitResult), 1);

                // guided = uncond + scale * (cond - uncond)
                Method subMethod = clsNDArray.getMethod("sub", clsNDArray);
                Method addMethod = clsNDArray.getMethod("add", clsNDArray);
                Object diff = subMethod.invoke(noiseCond, noiseUncond);
                Object scaled = mulMethod.invoke(diff, (float) guidanceScale);
                Object guided = addMethod.invoke(noiseUncond, scaled);

                // DDIM step
                float alphaT = alphaForTimestep(t);
                float alphaPrev = (i + 1 < timestepSchedule.length)
                        ? alphaForTimestep(timestepSchedule[i + 1]) : 1.0f;

                // predicted x0 = (latents - sqrt(1-alpha)*noise) / sqrt(alpha)
                Object scaledNoise = mulMethod.invoke(guided, (float) Math.sqrt(1.0 - alphaT));
                Object x0 = subMethod.invoke(latents, scaledNoise);
                x0 = mulMethod.invoke(x0, (float) (1.0 / Math.sqrt(alphaT)));

                // direction pointing to x_t
                Object dirXt = mulMethod.invoke(guided, (float) Math.sqrt(1.0 - alphaPrev));
                // x_{t-1} = sqrt(alpha_prev) * x0 + sqrt(1-alpha_prev) * noise_pred
                Object prevSample = mulMethod.invoke(x0, (float) Math.sqrt(alphaPrev));
                latents = addMethod.invoke(prevSample, dirXt);

                long elapsed = System.currentTimeMillis() - stepStart;
                stepStart = System.currentTimeMillis();
                request.reportProgress("Denoising: " + (i + 1) + "/" + steps
                        + " steps (" + String.format("%.1f", elapsed / 1000.0) + "s/step) [PyTorch]");
            }

            clsPredictor.getMethod("close").invoke(unetPredictor);
            clsZooModel.getMethod("close").invoke(unetModel);

            /* ── VAE decode ─────────────────────────────────────────── */
            request.reportProgress("Loading VAE decoder (PyTorch)\u2026");
            Object vaeModel = loadTorchScriptModel(clsCriteria, clsNDList, clsTranslator,
                    vaePath, device);
            Object vaePredictor = clsZooModel.getMethod("newPredictor").invoke(vaeModel);

            // Scale latents
            latents = mulMethod.invoke(latents, 1.0f / 0.18215f);

            Object vaeInput = clsNDList.getConstructor(clsNDArray).newInstance(latents);
            request.reportProgress("Decoding with VAE\u2026");
            Object vaeOutput = clsPredictor.getMethod("predict", Object.class)
                    .invoke(vaePredictor, vaeInput);
            Object decoded = clsNDList.getMethod("singletonOrThrow").invoke(vaeOutput);

            clsPredictor.getMethod("close").invoke(vaePredictor);
            clsZooModel.getMethod("close").invoke(vaeModel);

            /* ── Convert tensor to BufferedImage ───────────────────── */
            // decoded: [1, 3, H, W] float32, range roughly [-1, 1]
            Method toFloatArray = clsNDArray.getMethod("toFloatArray");
            float[] pixels = (float[]) toFloatArray.invoke(decoded);

            Method getShapeMethod = clsNDArray.getMethod("getShape");
            Object decodedShape = getShapeMethod.invoke(decoded);
            long[] dims = (long[]) clsShape.getMethod("getShape").invoke(decodedShape);
            int imgH = (int) dims[2];
            int imgW = (int) dims[3];

            BufferedImage image = new BufferedImage(imgW, imgH, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < imgH; y++) {
                for (int x = 0; x < imgW; x++) {
                    int rIdx = 0 * imgH * imgW + y * imgW + x;
                    int gIdx = 1 * imgH * imgW + y * imgW + x;
                    int bIdx = 2 * imgH * imgW + y * imgW + x;
                    int r = clamp((int) ((pixels[rIdx] / 2 + 0.5f) * 255));
                    int g = clamp((int) ((pixels[gIdx] / 2 + 0.5f) * 255));
                    int b = clamp((int) ((pixels[bIdx] / 2 + 0.5f) * 255));
                    image.setRGB(x, y, (r << 16) | (g << 8) | b);
                }
            }

            Path outputDir = storage.root().resolve("outputs");
            Files.createDirectories(outputDir);
            String filename = "djl-pytorch_" + System.currentTimeMillis() + ".png";
            Path outputPath = outputDir.resolve(filename);
            javax.imageio.ImageIO.write(image, "PNG", outputPath.toFile());

            return InferenceResult.ok(
                    "Generated image for prompt: \"" + request.prompt() + "\"",
                    "DJL PyTorch pipeline completed (" + steps + " steps)",
                    outputPath.toString(), "image");

        } finally {
            // Close NDManager
            clsNDManager.getMethod("close").invoke(manager);
        }
    }

    /* ── Helpers ──────────────────────────────────────────────────────── */

    /** Load a TorchScript model via DJL Criteria (reflection). */
    private Object loadTorchScriptModel(Class<?> clsCriteria, Class<?> clsNDList,
                                        Class<?> clsTranslator, Path modelPath,
                                        Object device) throws Exception {
        Method builder = clsCriteria.getMethod("builder");
        Object b = builder.invoke(null);

        Method setTypes = b.getClass().getMethod("setTypes", Class.class, Class.class);
        b = setTypes.invoke(b, clsNDList, clsNDList);

        Method optModelPath = b.getClass().getMethod("optModelPath", Path.class);
        b = optModelPath.invoke(b, modelPath);

        Method optEngine = b.getClass().getMethod("optEngine", String.class);
        b = optEngine.invoke(b, "PyTorch");

        Object translator = clsTranslator.getDeclaredConstructor().newInstance();
        Method optTranslator = b.getClass().getMethod("optTranslator",
                Class.forName("ai.djl.translate.Translator"));
        b = optTranslator.invoke(b, translator);

        Method optDevice = b.getClass().getMethod("optDevice",
                Class.forName("ai.djl.Device"));
        b = optDevice.invoke(b, device);

        Method build = b.getClass().getMethod("build");
        Object criteria = build.invoke(b);

        Method loadModel = clsCriteria.getMethod("loadModel");
        return loadModel.invoke(criteria);
    }

    /** DDIM linear timestep schedule. */
    private float[] ddimTimesteps(int numSteps) {
        float[] ts = new float[numSteps];
        for (int i = 0; i < numSteps; i++) {
            ts[i] = 999.0f * (1.0f - (float) i / numSteps);
        }
        return ts;
    }

    /** Linear beta schedule → alpha_bar for a given timestep. */
    private float alphaForTimestep(float t) {
        // Simplified linear schedule: beta from 0.00085 to 0.012 over 1000 steps
        double beta = 0.00085 + (0.012 - 0.00085) * t / 999.0;
        double alpha = 1.0 - beta;
        // Approximate cumulative product
        return (float) Math.pow(alpha, t);
    }

    /** Sigma for a given timestep (derived from alpha). */
    private double sigmaForTimestep(float t) {
        float a = alphaForTimestep(t);
        return Math.sqrt((1.0 - a) / a);
    }

    private static int clamp(int v) {
        return Math.max(0, Math.min(255, v));
    }
}
