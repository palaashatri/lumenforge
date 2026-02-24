package atri.palaash.lumenforge.storage;

import atri.palaash.lumenforge.model.ModelDescriptor;
import atri.palaash.lumenforge.model.TaskType;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ModelDownloader {

    private static final int BUFFER_SIZE = 16 * 1024;
    private static final Pattern HF_MODEL_ID_PATTERN = Pattern.compile("\\\"id\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    private static final Pattern HF_RFILENAME_PATTERN = Pattern.compile("\\\"rfilename\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");

    private final HttpClient httpClient;
    private final ModelStorage storage;
    private final Executor executor;

    public ModelDownloader(HttpClient httpClient, ModelStorage storage, Executor executor) {
        this.httpClient = httpClient;
        this.storage = storage;
        this.executor = executor;
    }

    public boolean canDownload(ModelDescriptor descriptor) {
        String url = descriptor.sourceUrl();
        return url.contains(".onnx") || url.contains("/resolve/") || url.startsWith("hf-pytorch://");
    }

    public CompletableFuture<Path> downloadIfMissing(ModelDescriptor descriptor, Consumer<DownloadProgress> progressConsumer) {
        if (storage.isAvailable(descriptor)) {
            return CompletableFuture.completedFuture(storage.modelPath(descriptor));
        }
        return download(descriptor, progressConsumer);
    }

    public CompletableFuture<Path> download(ModelDescriptor descriptor, Consumer<DownloadProgress> progressConsumer) {
        return CompletableFuture.supplyAsync(() -> {
            Path target = storage.modelPath(descriptor);
            List<Path> writtenFiles = new ArrayList<>();
            System.out.println("[LumenForge] Download started: " + descriptor.displayName());
            try {
                String downloadUrl = descriptor.sourceUrl();
                if (!downloadUrl.toLowerCase(Locale.ROOT).contains(".onnx")
                    && !downloadUrl.toLowerCase(Locale.ROOT).contains("/resolve/")) {
                    throw new IOException("No direct ONNX artifact URL configured for " + descriptor.displayName()
                            + ". Current source is a repository page. Configure a direct .onnx URL first.");
                }

                storage.ensureParentDirectory(descriptor);

                downloadUrlToPath(downloadUrl, target, progressConsumer);
                writtenFiles.add(target);
                downloadCompanionFilesIfNeeded(descriptor, target.getParent(), progressConsumer, writtenFiles);
                downloadKnownBundleFiles(descriptor, progressConsumer, writtenFiles);

                System.out.println("[LumenForge] Download complete: " + descriptor.displayName()
                        + " (" + writtenFiles.size() + " file(s))");
                return target;
            } catch (Exception ex) {
                System.out.println("[LumenForge] ERROR: Download failed: " + descriptor.displayName()
                        + " — " + ex.getMessage());
                for (Path written : writtenFiles) {
                    try {
                        Files.deleteIfExists(written);
                    } catch (IOException ignored) {
                    }
                }
                throw new RuntimeException(ex);
            }
        }, executor);
    }

    private void downloadCompanionFilesIfNeeded(ModelDescriptor descriptor,
                                                Path targetDirectory,
                                                Consumer<DownloadProgress> progressConsumer,
                                                List<Path> writtenFiles) throws IOException, InterruptedException {
        String sourceUrl = descriptor.sourceUrl();
        HfResolvePath hfPath = parseHuggingFaceResolvePath(sourceUrl);
        if (hfPath == null) {
            return;
        }

        String detailsBody = httpGetText("https://huggingface.co/api/models/" + hfPath.repoId());
        if (detailsBody.isBlank()) {
            return;
        }

        String folder = parentFolder(hfPath.filePath());
        Matcher matcher = HF_RFILENAME_PATTERN.matcher(detailsBody);
        while (matcher.find()) {
            String filePath = matcher.group(1);
            if (!filePath.startsWith(folder + "/") || filePath.equals(hfPath.filePath())) {
                continue;
            }
            String fileName = filePath.substring(filePath.lastIndexOf('/') + 1);
            if (!isCompanionArtifact(fileName)) {
                continue;
            }
            String companionUrl = "https://huggingface.co/" + hfPath.repoId()
                    + "/resolve/" + hfPath.revision() + "/" + filePath;
            Path companionTarget = targetDirectory.resolve(fileName);
            downloadUrlToPath(companionUrl, companionTarget, progressConsumer);
            writtenFiles.add(companionTarget);
        }
    }

    private void downloadKnownBundleFiles(ModelDescriptor descriptor,
                                          Consumer<DownloadProgress> progressConsumer,
                                          List<Path> writtenFiles) throws IOException, InterruptedException {
        if ("sd_v15_onnx".equals(descriptor.id())) {
            downloadSdBundle(descriptor, "text-image/stable-diffusion-v15",
                    List.of(
                            "unet/model.onnx", "unet/weights.pb",
                            "text_encoder/model.onnx",
                            "vae_decoder/model.onnx",
                            "scheduler/scheduler_config.json",
                            "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
                            "tokenizer/tokenizer_config.json", "tokenizer/vocab.json"
                    ), progressConsumer, writtenFiles);
        } else if ("sd_turbo_onnx".equals(descriptor.id())) {
            downloadSdBundle(descriptor, "text-image/sd-turbo",
                    List.of(
                            "unet/model.onnx",
                            "text_encoder/model.onnx",
                            "vae_decoder/model.onnx",
                            "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
                            "tokenizer/tokenizer_config.json", "tokenizer/vocab.json"
                    ), progressConsumer, writtenFiles);
        } else if ("sdxl_turbo_onnx".equals(descriptor.id())) {
            downloadSdBundle(descriptor, "text-image/sdxl-turbo",
                    List.of(
                            "unet/model.onnx", "unet/weights.pb",
                            "text_encoder/model.onnx",
                            "text_encoder_2/model.onnx", "text_encoder_2/weights.pb",
                            "vae_decoder/model.onnx", "vae_decoder/weights.pb",
                            "scheduler/scheduler_config.json",
                            "tokenizer/merges.txt", "tokenizer/vocab.json",
                            "tokenizer_2/merges.txt", "tokenizer_2/vocab.json"
                    ), progressConsumer, writtenFiles);
        } else if ("sd_v15_img2img".equals(descriptor.id())) {
            // Img2Img reuses the SD v1.5 bundle + VAE encoder
            downloadSdBundle(descriptor, "text-image/stable-diffusion-v15",
                    List.of(
                            "unet/model.onnx", "unet/weights.pb",
                            "text_encoder/model.onnx",
                            "vae_decoder/model.onnx",
                            "vae_encoder/model.onnx",
                            "scheduler/scheduler_config.json",
                            "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
                            "tokenizer/tokenizer_config.json", "tokenizer/vocab.json"
                    ), progressConsumer, writtenFiles);
        } else if ("sd_turbo_img2img".equals(descriptor.id())) {
            // SD Turbo img2img — reuses SD Turbo bundle + VAE encoder from v1.5
            downloadSdBundle(descriptor, "text-image/sd-turbo",
                    List.of(
                            "unet/model.onnx",
                            "text_encoder/model.onnx",
                            "vae_decoder/model.onnx",
                            "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
                            "tokenizer/tokenizer_config.json", "tokenizer/vocab.json"
                    ), progressConsumer, writtenFiles);
        } else if ("sdxl_base_onnx".equals(descriptor.id())) {
            // SDXL Base 1.0 — dual text encoders, large UNet
            downloadSdBundle(descriptor, "text-image/sdxl-base",
                    List.of(
                            "unet/model.onnx", "unet/model.onnx_data",
                            "text_encoder/model.onnx",
                            "text_encoder_2/model.onnx", "text_encoder_2/model.onnx_data",
                            "vae_decoder/model.onnx",
                            "scheduler/scheduler_config.json",
                            "tokenizer/merges.txt", "tokenizer/vocab.json",
                            "tokenizer_2/merges.txt", "tokenizer_2/vocab.json"
                    ), progressConsumer, writtenFiles);
        }
    }

    private void downloadSdBundle(ModelDescriptor descriptor, String localDir,
                                  List<String> files,
                                  Consumer<DownloadProgress> progressConsumer,
                                  List<Path> writtenFiles) throws IOException, InterruptedException {
        HfResolvePath hfPath = parseHuggingFaceResolvePath(descriptor.sourceUrl());
        if (hfPath == null) { return; }
        String repo = hfPath.repoId();
        String revision = hfPath.revision();
        for (String relative : files) {
            Path target = storage.root().resolve(localDir).resolve(relative);
            Files.createDirectories(target.getParent());
            if (Files.exists(target) && Files.size(target) > 0) { continue; }
            String url = "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + relative;
            downloadUrlToPath(url, target, progressConsumer);
            writtenFiles.add(target);
        }
    }

    private static final int MAX_RETRIES = 5;
    private static final long STALL_TIMEOUT_MS = 90_000; // 90 seconds with no data → stall

    /**
     * Download a URL to a local file with automatic resume and retry.
     * If the file already has partial content, we attempt an HTTP Range request
     * to pick up where we left off. On stall (no data for 90 s) or network
     * error, we retry up to 5 times with exponential back-off.
     */
    private void downloadUrlToPath(String downloadUrl,
                                   Path target,
                                   Consumer<DownloadProgress> progressConsumer) throws IOException, InterruptedException {
        IOException lastException = null;
        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                downloadUrlToPathOnce(downloadUrl, target, progressConsumer);
                return; // success
            } catch (IOException ex) {
                lastException = ex;
                if (attempt < MAX_RETRIES) {
                    long backoff = 2000L * attempt;
                    System.out.println("[LumenForge] WARN: Download stalled/failed for " + target.getFileName()
                            + ", retrying in " + (backoff / 1000) + "s (attempt "
                            + (attempt + 1) + "/" + MAX_RETRIES + ")");
                    if (progressConsumer != null) {
                        progressConsumer.accept(new DownloadProgress(-1, -1,
                                "Download stalled/failed, retrying in " + (backoff / 1000) + "s (attempt "
                                        + (attempt + 1) + "/" + MAX_RETRIES + ")…"));
                    }
                    Thread.sleep(backoff);
                }
            }
        }
        throw lastException;
    }

    private void downloadUrlToPathOnce(String downloadUrl,
                                       Path target,
                                       Consumer<DownloadProgress> progressConsumer) throws IOException, InterruptedException {
        // Check for a partial download we can resume.
        long existingBytes = Files.exists(target) ? Files.size(target) : 0;

        HttpRequest.Builder reqBuilder = HttpRequest.newBuilder()
                .uri(URI.create(downloadUrl))
                .GET();
        if (existingBytes > 0) {
            reqBuilder.header("Range", "bytes=" + existingBytes + "-");
        }

        HttpResponse<InputStream> response = httpClient.send(reqBuilder.build(),
                HttpResponse.BodyHandlers.ofInputStream());
        int statusCode = response.statusCode();

        boolean resuming = (statusCode == 206 && existingBytes > 0);
        if (!resuming) {
            existingBytes = 0; // server does not support Range — start over
        }
        if (statusCode != 200 && statusCode != 206) {
            throw new IOException("Download failed with HTTP status " + statusCode + " for URL " + downloadUrl);
        }

        // Only enforce binary-content checks for ONNX / weight files,
        // not for companion text assets (.json, .txt, etc.).
        String targetName = target.getFileName().toString().toLowerCase(Locale.ROOT);
        boolean expectBinary = targetName.endsWith(".onnx") || targetName.endsWith(".pb")
                || targetName.endsWith(".bin") || targetName.endsWith(".data")
                || targetName.endsWith(".onnx_data");

        if (expectBinary && !resuming) {
            String contentType = response.headers().firstValue("content-type").orElse("").toLowerCase(Locale.ROOT);
            if (contentType.contains("text/html")) {
                throw new IOException("Unexpected response content type for ONNX download: " + contentType);
            }
        }

        long contentLength = response.headers()
                .firstValue("content-length")
                .map(Long::parseLong)
                .orElse(-1L);
        long totalBytes = (contentLength > 0) ? existingBytes + contentLength : -1L;

        StandardOpenOption[] openOptions = resuming
                ? new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND}
                : new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE};

        try (InputStream inputStream = response.body();
             OutputStream outputStream = Files.newOutputStream(target, openOptions)) {

            byte[] buffer = new byte[BUFFER_SIZE];
            long downloaded = existingBytes;
            int read;
            boolean firstChunkChecked = resuming; // skip binary check when resuming
            long lastDataTime = System.currentTimeMillis();

            while (true) {
                // Use available() + sleep to implement stall detection without
                // blocking forever inside read().
                if (inputStream.available() <= 0) {
                    // Attempt a timed read via a virtual thread so we can enforce the stall timeout.
                    final InputStream is = inputStream;
                    var readFuture = java.util.concurrent.CompletableFuture.supplyAsync(() -> {
                        try {
                            return is.read(buffer);
                        } catch (IOException e) {
                            throw new java.util.concurrent.CompletionException(e);
                        }
                    });
                    try {
                        read = readFuture.get(STALL_TIMEOUT_MS, java.util.concurrent.TimeUnit.MILLISECONDS);
                    } catch (java.util.concurrent.TimeoutException te) {
                        throw new IOException("Download stalled — no data received for "
                                + (STALL_TIMEOUT_MS / 1000) + " seconds.");
                    } catch (java.util.concurrent.ExecutionException ee) {
                        Throwable cause = ee.getCause();
                        if (cause instanceof IOException ioe) { throw ioe; }
                        throw new IOException("Read error", cause);
                    }
                } else {
                    read = inputStream.read(buffer);
                }

                if (read < 0) { break; }

                if (!firstChunkChecked && read > 0 && expectBinary) {
                    String prefix = new String(buffer, 0, Math.min(read, 64)).trim().toLowerCase(Locale.ROOT);
                    if (prefix.startsWith("<!doctype html") || prefix.startsWith("<html") || prefix.startsWith("{")) {
                        throw new IOException("Downloaded content does not look like a binary ONNX model.");
                    }
                    firstChunkChecked = true;
                }
                outputStream.write(buffer, 0, read);
                downloaded += read;
                lastDataTime = System.currentTimeMillis();
                if (progressConsumer != null) {
                    progressConsumer.accept(new DownloadProgress(downloaded, totalBytes));
                }
            }
        }
    }

    private boolean isCompanionArtifact(String fileName) {
        String lower = fileName.toLowerCase(Locale.ROOT);
        return lower.endsWith(".pb")
                || lower.endsWith(".bin")
                || lower.endsWith(".data")
                || lower.endsWith(".onnx_data")
                || lower.contains("weight");
    }

    private String parentFolder(String filePath) {
        int slash = filePath.lastIndexOf('/');
        if (slash <= 0) {
            return "";
        }
        return filePath.substring(0, slash);
    }

    private HfResolvePath parseHuggingFaceResolvePath(String sourceUrl) {
        String marker = "https://huggingface.co/";
        if (!sourceUrl.startsWith(marker) || !sourceUrl.contains("/resolve/")) {
            return null;
        }
        String tail = sourceUrl.substring(marker.length());
        int resolveIndex = tail.indexOf("/resolve/");
        if (resolveIndex <= 0) {
            return null;
        }
        String repoId = tail.substring(0, resolveIndex);
        String afterResolve = tail.substring(resolveIndex + "/resolve/".length());
        int slash = afterResolve.indexOf('/');
        if (slash <= 0 || slash >= afterResolve.length() - 1) {
            return null;
        }
        String revision = afterResolve.substring(0, slash);
        String filePath = afterResolve.substring(slash + 1);
        return new HfResolvePath(repoId, revision, filePath);
    }

    private record HfResolvePath(String repoId, String revision, String filePath) {
    }

    public CompletableFuture<List<ModelDescriptor>> discoverTextToImageModels() {
        return CompletableFuture.supplyAsync(() -> {
            try {
                Set<String> modelIds = fetchCandidateModelIds();
                List<ModelDescriptor> discovered = new ArrayList<>();
                for (String modelId : modelIds) {
                    String detailUrl = "https://huggingface.co/api/models/" + modelId;
                    String details = httpGetText(detailUrl);
                    if (details.isBlank()) continue;

                    boolean isGated = details.contains("\"gated\"") &&
                            (details.contains("\"gated\":true") || details.contains("\"gated\":\"auto\"")
                            || details.contains("\"gated\": true") || details.contains("\"gated\": \"auto\""));

                    // ── Determine model type and create descriptor ──
                    boolean hasOnnxUnet = details.contains("unet/model.onnx");
                    boolean hasOnnxTransformer = details.contains("transformer/model.onnx");
                    boolean hasEsrgan = modelId.toLowerCase(Locale.ROOT).contains("esrgan")
                            || modelId.toLowerCase(Locale.ROOT).contains("realesrgan")
                            || modelId.toLowerCase(Locale.ROOT).contains("upscal");
                    boolean hasPyTorch = details.contains("model_index.json")
                            || details.contains("diffusion_pytorch_model")
                            || details.contains(".safetensors")
                            || details.contains("\"pipeline_tag\":\"text-to-image\"");
                    // Don't mark as PyTorch if it already has ONNX artifacts
                    boolean isPyTorchOnly = hasPyTorch && !hasOnnxUnet && !hasOnnxTransformer;

                    if (hasEsrgan) {
                        // ESRGAN upscaler — check for .onnx file
                        boolean hasOnnx = details.contains(".onnx");
                        if (!hasOnnx && !isGated) continue;
                        String onnxFile = findEsrganOnnxFile(details);
                        if (onnxFile == null) onnxFile = "onnx/model.onnx"; // common path
                        String sourceUrl = "https://huggingface.co/" + modelId + "/resolve/main/" + onnxFile;
                        String id = "hf_" + modelId.toLowerCase(Locale.ROOT).replace('/', '_').replace('-', '_');
                        String relativePath = "text-image/huggingface/" + modelId.replace('/', '-') + "/" + onnxFile;
                        discovered.add(new ModelDescriptor(id,
                                "HF: " + modelId + " (ESRGAN ONNX)",
                                TaskType.IMAGE_UPSCALE, relativePath, sourceUrl,
                                "Discovered ESRGAN/upscaler from Hugging Face."));
                    } else if (hasOnnxUnet && !isGated) {
                        // ONNX SD model with UNet
                        String sourceUrl = "https://huggingface.co/" + modelId + "/resolve/main/unet/model.onnx";
                        if (!urlLooksAccessible(sourceUrl)) continue;
                        String id = "hf_" + modelId.toLowerCase(Locale.ROOT).replace('/', '_').replace('-', '_');
                        String displayName = "HF: " + modelId + " (UNet ONNX)";
                        String relativePath = "text-image/huggingface/" + modelId.replace('/', '-') + "/unet/model.onnx";
                        discovered.add(new ModelDescriptor(id, displayName,
                                TaskType.TEXT_TO_IMAGE, relativePath, sourceUrl,
                                "Discovered from Hugging Face ONNX listing."));
                    } else if (hasOnnxTransformer && !isGated) {
                        // ONNX SD3-type model with transformer
                        String sourceUrl = "https://huggingface.co/" + modelId + "/resolve/main/transformer/model.onnx";
                        String id = "hf_" + modelId.toLowerCase(Locale.ROOT).replace('/', '_').replace('-', '_');
                        String displayName = "HF: " + modelId + " (Transformer ONNX)";
                        String relativePath = "text-image/huggingface/" + modelId.replace('/', '-') + "/transformer/model.onnx";
                        discovered.add(new ModelDescriptor(id, displayName,
                                TaskType.TEXT_TO_IMAGE, relativePath, sourceUrl,
                                "Discovered SD 3.x-style ONNX model from Hugging Face."));
                    } else if (isPyTorchOnly) {
                        // PyTorch model — needs conversion to ONNX
                        String sourceUrl = "hf-pytorch://" + modelId;
                        String id = "hf_pt_" + modelId.toLowerCase(Locale.ROOT).replace('/', '_').replace('-', '_');
                        String gatedNote = isGated ? " 🔒" : "";
                        String displayName = "HF: " + modelId + " (PyTorch → convert)" + gatedNote;
                        String relativePath = "text-image/converted-" + modelId.replace('/', '-').toLowerCase() + "/unet/model.onnx";
                        discovered.add(new ModelDescriptor(id, displayName,
                                TaskType.TEXT_TO_IMAGE, relativePath, sourceUrl,
                                "PyTorch model — will be converted to ONNX on download."
                                + (isGated ? " Gated model — requires HF token." : "")));
                    }
                }
                return discovered;
            } catch (Exception ex) {
                throw new RuntimeException("Unable to refresh models from Hugging Face: " + ex.getMessage(), ex);
            }
        }, executor);
    }

    /**
     * Try to find the ONNX file path in an ESRGAN repo's file listing.
     */
    private String findEsrganOnnxFile(String detailsJson) {
        Matcher m = HF_RFILENAME_PATTERN.matcher(detailsJson);
        while (m.find()) {
            String f = m.group(1);
            if (f.endsWith(".onnx")) return f;
        }
        return null;
    }

    private Set<String> fetchCandidateModelIds() throws IOException, InterruptedException {
        Set<String> ids = new LinkedHashSet<>();
        List<String> queries = List.of(
                "text-to-image onnx",
                "stable diffusion onnx",
                "onnxruntime sd",
                "stable-diffusion pytorch",
                "stable-diffusion-xl",
                "stable-diffusion-3",
                "stable-diffusion v1",
                "realesrgan onnx",
                "esrgan onnx",
                "real-esrgan",
                "image super resolution onnx"
        );
        for (String query : queries) {
            String searchUrl = "https://huggingface.co/api/models?search=" +
                    java.net.URLEncoder.encode(query, java.nio.charset.StandardCharsets.UTF_8) +
                    "&limit=60";
            String body = httpGetText(searchUrl);
            Matcher matcher = HF_MODEL_ID_PATTERN.matcher(body);
            while (matcher.find()) {
                String id = matcher.group(1);
                if (id.contains("/") && !id.startsWith("datasets/") && !id.startsWith("spaces/")) {
                    ids.add(id);
                }
            }
        }
        return ids;
    }

    private String httpGetText(String url) throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .GET()
                .build();
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() < 200 || response.statusCode() > 299) {
            return "";
        }
        return response.body() == null ? "" : response.body();
    }

    private boolean urlLooksAccessible(String url) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .method("HEAD", HttpRequest.BodyPublishers.noBody())
                    .build();
            HttpResponse<Void> response = httpClient.send(request, HttpResponse.BodyHandlers.discarding());
            return response.statusCode() >= 200 && response.statusCode() < 400;
        } catch (Exception ex) {
            return false;
        }
    }
}
