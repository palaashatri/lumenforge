package atri.palaash.jforge.storage;

import java.awt.Desktop;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.function.Consumer;

/**
 * Orchestrates PyTorch → ONNX model conversion using a managed Python
 * virtual environment.
 *
 * <p>Workflow:
 * <ol>
 *   <li>Detect a Python 3 installation on the system PATH</li>
 *   <li>Create a virtual environment (if needed) at {@code ~/.jforge-models/.converter-venv/}</li>
 *   <li>Install required packages: torch, onnx, diffusers, transformers, optimum[exporters]</li>
 *   <li>Run the conversion script — either via Optimum (HuggingFace diffusers) or
 *       via {@code torch.onnx.export()} (generic .pt/.pth models)</li>
 * </ol>
 *
 * <p>References:
 * <ul>
 *   <li><a href="https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model">
 *       Microsoft: Convert PyTorch model to ONNX</a></li>
 *   <li><a href="https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model">
 *       Optimum ONNX exporter</a></li>
 * </ul>
 */
public class PyTorchToOnnxConverter {

    /** Root directory for the managed Python venv. */
    private static final Path VENV_DIR = Path.of(
            System.getProperty("user.home"), ".jforge-models", ".converter-venv");

    private static final String PYTHON_DOWNLOAD_URL = "https://www.python.org/downloads/";

    /**
     * Packages installed into the managed venv before conversion.
     *
     * <p>IMPORTANT: use {@code optimum[exporters]}, NOT {@code optimum[onnxruntime]}.
     * The {@code exporters} extra provides {@code optimum.exporters.onnx.main_export};
     * {@code onnxruntime} only adds the runtime inference classes.
     */
    private static final String[] REQUIRED_PACKAGES = {
            "torch",
            "onnx",
            "onnxruntime",
            "diffusers",
            "transformers",
            "accelerate",
            "huggingface_hub",
            "sentencepiece",
            "optimum[exporters]>=1.19.0"
    };

    private final Consumer<String> progressCallback;

    /**
     * @param progressCallback receives human-readable status messages
     *                         (called from the invoking thread)
     */
    public PyTorchToOnnxConverter(Consumer<String> progressCallback) {
        this.progressCallback = progressCallback != null ? progressCallback : s -> {};
    }

    /* ── Static helpers ──────────────────────────────────────────────── */

    /**
     * Probe the system PATH for a Python 3 interpreter.
     *
     * @return the command name ({@code "python3"} or {@code "python"}),
     *         or {@code null} if Python 3 is not installed
     */
    public static String findPython() {
        for (String cmd : new String[]{"python3", "python"}) {
            try {
                Process p = new ProcessBuilder(cmd, "--version")
                        .redirectErrorStream(true)
                        .start();
                String output = new String(p.getInputStream().readAllBytes()).trim();
                int exitCode = p.waitFor();
                if (exitCode == 0 && output.contains("Python 3")) {
                    return cmd;
                }
            } catch (Exception ignored) {
                // Command not found — try next
            }
        }
        return null;
    }

    /**
     * Open the official Python download page in the user's default browser.
     */
    public static void openPythonDownloadPage() {
        try {
            if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                Desktop.getDesktop().browse(URI.create(PYTHON_DOWNLOAD_URL));
            } else {
                // Fallback: try OS-specific openers
                String os = System.getProperty("os.name").toLowerCase();
                if (os.contains("mac")) {
                    Runtime.getRuntime().exec(new String[]{"open", PYTHON_DOWNLOAD_URL});
                } else if (os.contains("linux")) {
                    Runtime.getRuntime().exec(new String[]{"xdg-open", PYTHON_DOWNLOAD_URL});
                }
            }
        } catch (Exception ignored) {
            // Best-effort — caller should show the URL in UI if this fails
        }
    }

    /**
     * Return {@code true} if a model ID or path looks like a PyTorch model
     * (as opposed to an ONNX model that is already ready for inference).
     */
    public static boolean isPyTorchModel(String idOrPath) {
        if (idOrPath == null) return false;
        String lower = idOrPath.toLowerCase();
        return lower.contains("pytorch")
                || lower.endsWith(".pt")
                || lower.endsWith(".pth")
                || lower.endsWith(".safetensors")
                || lower.contains("torchscript");
    }

    /**
     * Return {@code true} if the model ID looks like a gated HuggingFace model
     * that requires authentication to download.
     */
    public static boolean isLikelyGatedModel(String modelId) {
        if (modelId == null) return false;
        String lower = modelId.toLowerCase();
        // Known gated model families
        return lower.contains("stable-diffusion-3")
                || lower.contains("sd3")
                || lower.contains("sd-3")
                || lower.contains("flux")
                || lower.contains("llama")
                || lower.contains("gemma");
    }

    /**
     * Return {@code true} if the model is expected to require large amounts of
     * disk space during conversion (30+ GB for HF cache + ONNX output).
     */
    private static boolean isLargeModel(String modelId) {
        if (modelId == null) return false;
        String lower = modelId.toLowerCase();
        return lower.contains("stable-diffusion-3")
                || lower.contains("sd3")
                || lower.contains("sd-3")
                || lower.contains("sdxl")
                || lower.contains("flux")
                || lower.contains("sd_xl");
    }

    /* ── Main conversion entry point ─────────────────────────────────── */

    /**
     * Run the full PyTorch → ONNX conversion.
     * <p><strong>Call this from a background thread</strong> — it blocks while
     * external processes run (venv creation, pip install, conversion script).
     *
     * @param modelId   HuggingFace model ID (e.g. {@code "runwayml/stable-diffusion-v1-5"})
     *                  or local path to a {@code .pt/.pth} file
     * @param outputDir directory to write converted ONNX file(s)
     * @param mode      {@code "diffusers"} for HuggingFace pipelines,
     *                  {@code "generic"} for standalone .pt/.pth files
     * @return the output directory on success
     * @throws PythonNotFoundException if Python 3 is not installed
     * @throws ConversionException     if any step fails
     */
    public Path convert(String modelId, Path outputDir, String mode) {
        return convert(modelId, outputDir, mode, null, false);
    }

    /** @see #convert(String, Path, String, String, boolean) */
    public Path convert(String modelId, Path outputDir, String mode, String hfToken) {
        return convert(modelId, outputDir, mode, hfToken, false);
    }

    /**
     * Run the full PyTorch → ONNX conversion.
     *
     * @param modelId    HuggingFace model ID or local path to a .pt/.pth file
     * @param outputDir  directory to write converted ONNX file(s)
     * @param mode       {@code "diffusers"} or {@code "generic"}
     * @param hfToken    HuggingFace auth token for gated models (may be null)
     * @param reinstall  if {@code true}, pass {@code --reinstall} to the Python script to
     *                   force-reinstall {@code optimum[exporters]} — fixes broken/partial installs.
     * @return the output directory on success
     */
    public Path convert(String modelId, Path outputDir, String mode,
                        String hfToken, boolean reinstall) {
        // 1. Locate Python
        String pythonCmd = findPython();
        if (pythonCmd == null) {
            throw new PythonNotFoundException();
        }
        report("Found " + getPythonVersion(pythonCmd));

        // 2. Ensure venv exists
        if (!isVenvReady()) {
            report("Creating virtual environment…");
            runProcess(pythonCmd, "-m", "venv", VENV_DIR.toString());
            if (!isVenvReady()) {
                throw new ConversionException("Failed to create Python virtual environment at " + VENV_DIR);
            }
            report("Virtual environment created at " + VENV_DIR);
        } else {
            report("Using existing virtual environment.");
        }

        // 3. Install / upgrade dependencies
        report("Installing Python dependencies (first run may take several minutes)…");
        String venvPython = getVenvPython().toString();
        runProcess(venvPython, "-m", "pip", "install", "--upgrade", "pip");
        String[] installCmd = new String[REQUIRED_PACKAGES.length + 5];
        installCmd[0] = venvPython;
        installCmd[1] = "-m";
        installCmd[2] = "pip";
        installCmd[3] = "install";
        installCmd[4] = "--upgrade";
        System.arraycopy(REQUIRED_PACKAGES, 0, installCmd, 5, REQUIRED_PACKAGES.length);
        runProcess(installCmd);
        report("Dependencies ready.");

        // 4. Write the conversion script to the venv
        Path scriptPath = writeConversionScript();

        // 5. Check disk space before conversion
        //    SD 3.5-class models need ~30 GB (HF cache + ONNX output).
        //    Smaller models (SD 1.5, SDXL) need ~10 GB.
        long minimumBytes = isLargeModel(modelId) ? 30L * 1024 * 1024 * 1024
                                                  : 10L * 1024 * 1024 * 1024;
        try {
            long usable = Files.getFileStore(outputDir.getParent() != null
                    ? outputDir.getParent() : outputDir).getUsableSpace();
            String usableGb = String.format("%.1f", usable / (1024.0 * 1024.0 * 1024.0));
            String requiredGb = String.format("%.0f", minimumBytes / (1024.0 * 1024.0 * 1024.0));
            if (usable < minimumBytes) {
                throw new ConversionException(
                        "Insufficient disk space: " + usableGb + " GB available, ~"
                                + requiredGb + " GB required. Free up space or delete old caches:\n"
                                + "  • HuggingFace cache:  ~/.cache/huggingface/hub/\n"
                                + "  • Pip cache:          ~/Library/Caches/pip/  (macOS)\n"
                                + "  • Converter venv:     " + VENV_DIR);
            }
            report("Disk space OK: " + usableGb + " GB available (need ~" + requiredGb + " GB).");
        } catch (IOException diskErr) {
            report("Warning: could not check disk space — " + diskErr.getMessage());
        }

        // 6. Run the conversion
        report("Starting conversion: " + modelId + " → ONNX");
        try {
            Files.createDirectories(outputDir);
        } catch (IOException ex) {
            throw new ConversionException("Cannot create output directory: " + ex.getMessage());
        }

        // 6b. Run the conversion
        runConversionScript(venvPython, scriptPath, modelId, outputDir, mode,
                            hfToken, reinstall);

        // 7. Post-conversion cleanup — reclaim disk space from intermediate caches
        cleanupAfterConversion(modelId, venvPython);

        report("Conversion complete → " + outputDir);
        return outputDir;
    }

    /* ── Internal helpers ────────────────────────────────────────────── */

    /**
     * Remove intermediate caches that are no longer needed after a successful
     * conversion. This can reclaim 10-20+ GB for large models like SD 3.5.
     *
     * <p>Targets:
     * <ul>
     *   <li>HuggingFace hub cache for the specific model
     *       ({@code ~/.cache/huggingface/hub/models--<org>--<name>/})</li>
     *   <li>pip HTTP cache ({@code ~/Library/Caches/pip/} on macOS,
     *       {@code ~/.cache/pip/} on Linux)</li>
     * </ul>
     */
    private void cleanupAfterConversion(String modelId, String venvPython) {
        // ── 1. Delete the HuggingFace model cache for the converted model ──
        //    HF stores snapshots under ~/.cache/huggingface/hub/models--<org>--<name>/
        if (modelId != null && modelId.contains("/")) {
            String sanitized = "models--" + modelId.replace("/", "--");
            Path hfCacheDir = Path.of(System.getProperty("user.home"),
                    ".cache", "huggingface", "hub", sanitized);
            if (Files.isDirectory(hfCacheDir)) {
                report("Cleaning up HuggingFace cache: " + hfCacheDir);
                try {
                    long freedBytes = deleteDirectoryTree(hfCacheDir);
                    String freedGb = String.format("%.1f", freedBytes / (1024.0 * 1024.0 * 1024.0));
                    report("Freed " + freedGb + " GB from HuggingFace cache.");
                } catch (IOException e) {
                    report("Warning: could not fully clean HF cache — " + e.getMessage());
                }
            }
        }

        // ── 2. Clear pip HTTP cache ──
        try {
            ProcessBuilder pb = new ProcessBuilder(venvPython, "-m", "pip", "cache", "purge");
            pb.redirectErrorStream(true);
            Process p = pb.start();
            // Consume output to prevent blocking
            p.getInputStream().readAllBytes();
            int exit = p.waitFor();
            if (exit == 0) {
                report("Cleared pip cache.");
            }
        } catch (Exception ignored) {
            // pip cache purge is best-effort
        }
    }

    /**
     * Recursively delete a directory tree and return the total bytes freed.
     */
    private static long deleteDirectoryTree(Path root) throws IOException {
        long[] freed = {0};
        java.nio.file.Files.walkFileTree(root, new java.nio.file.SimpleFileVisitor<>() {
            @Override
            public java.nio.file.FileVisitResult visitFile(Path file,
                    java.nio.file.attribute.BasicFileAttributes attrs) throws IOException {
                freed[0] += attrs.size();
                Files.delete(file);
                return java.nio.file.FileVisitResult.CONTINUE;
            }

            @Override
            public java.nio.file.FileVisitResult postVisitDirectory(Path dir, IOException exc)
                    throws IOException {
                Files.delete(dir);
                return java.nio.file.FileVisitResult.CONTINUE;
            }
        });
        return freed[0];
    }

    private boolean isVenvReady() {
        return Files.exists(getVenvPython());
    }

    private Path getVenvPython() {
        if (isWindows()) {
            return VENV_DIR.resolve("Scripts").resolve("python.exe");
        }
        // On macOS/Linux the venv might only create "python" (not "python3")
        Path python3 = VENV_DIR.resolve("bin").resolve("python3");
        if (Files.exists(python3)) return python3;
        return VENV_DIR.resolve("bin").resolve("python");
    }

    private static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase().contains("win");
    }

    private String getPythonVersion(String pythonCmd) {
        try {
            Process p = new ProcessBuilder(pythonCmd, "--version")
                    .redirectErrorStream(true).start();
            return new String(p.getInputStream().readAllBytes()).trim();
        } catch (Exception e) {
            return pythonCmd;
        }
    }

    private void report(String message) {
        progressCallback.accept(message);
    }

    /**
     * Run an external process, streaming its stdout/stderr to the progress
     * callback line by line. Throws on non-zero exit.
     */
    private void runProcess(String... command) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process p = pb.start();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    report("  " + line);
                }
            }
            int exitCode = p.waitFor();
            if (exitCode != 0) {
                throw new ConversionException(
                        "Command exited with code " + exitCode + ": " + String.join(" ", command));
            }
        } catch (ConversionException ex) {
            throw ex;
        } catch (Exception ex) {
            throw new ConversionException("Failed to run: " + String.join(" ", command) + " — " + ex.getMessage());
        }
    }

    /**
     * Run the conversion script and parse PROGRESS / ERROR lines from stdout.
     *
     * @param reinstall if {@code true}, pass {@code --reinstall} to force-reinstall
     *                  {@code optimum[exporters]} before the conversion starts.
     */
    private void runConversionScript(String python, Path script,
                                     String modelId, Path outputDir, String mode,
                                     String hfToken, boolean reinstall) {
        try {
            // Build command list
            java.util.List<String> cmdList = new java.util.ArrayList<>(java.util.List.of(
                    python,
                    script.toString(),
                    "--model_id", modelId,
                    "--output_dir", outputDir.toString(),
                    "--mode", mode
            ));
            // Pass HF token via CLI arg for gated models
            if (hfToken != null && !hfToken.isBlank()) {
                cmdList.add("--hf_token");
                cmdList.add(hfToken);
            }
            // --reinstall forces force-reinstall of optimum[exporters]
            if (reinstall) {
                cmdList.add("--reinstall");
            }
            ProcessBuilder pb = new ProcessBuilder(cmdList);
            // Also expose token as env vars
            if (hfToken != null && !hfToken.isBlank()) {
                pb.environment().put("HF_TOKEN", hfToken);
                pb.environment().put("HUGGING_FACE_HUB_TOKEN", hfToken);
            }
            pb.redirectErrorStream(true);
            Process process = pb.start();

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("PROGRESS: ")) {
                        String payload = line.substring("PROGRESS: ".length());
                        int space = payload.indexOf(' ');
                        if (space > 0) {
                            report(payload.substring(space + 1));
                        }
                    } else if (line.startsWith("ERROR: ")) {
                        throw new ConversionException(line.substring("ERROR: ".length()));
                    } else {
                        report(line);
                    }
                }
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new ConversionException("Conversion script exited with code " + exitCode);
            }
        } catch (ConversionException ex) {
            throw ex;
        } catch (Exception ex) {
            throw new ConversionException("Conversion script failed: " + ex.getMessage());
        }
    }

    /**
     * Write the embedded Python conversion script to the venv directory.
     * This ensures the script is always available regardless of working
     * directory or packaging format.
     */
    private Path writeConversionScript() {
        try {
            // First, try the local project copy
            Path local = Path.of("scripts", "convert_pytorch_to_onnx.py");
            if (Files.exists(local)) {
                return local;
            }

            // Try classpath resource (works when packaged as JAR)
            try (var is = getClass().getResourceAsStream("/scripts/convert_pytorch_to_onnx.py")) {
                if (is != null) {
                    Path target = VENV_DIR.resolve("convert_pytorch_to_onnx.py");
                    Files.createDirectories(target.getParent());
                    Files.copy(is, target, StandardCopyOption.REPLACE_EXISTING);
                    return target;
                }
            }

            // Fallback: write a minimal embedded script
            Path target = VENV_DIR.resolve("convert_pytorch_to_onnx.py");
            Files.createDirectories(target.getParent());
            Files.writeString(target, EMBEDDED_SCRIPT,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            return target;
        } catch (IOException ex) {
            throw new ConversionException("Failed to write conversion script: " + ex.getMessage());
        }
    }

    /* ── Exceptions ──────────────────────────────────────────────────── */

    /** Thrown when Python 3 is not found on the system. */
    public static class PythonNotFoundException extends RuntimeException {
        public PythonNotFoundException() {
            super("Python 3 is required for PyTorch→ONNX conversion but was not found on your system.");
        }
    }

    /** Thrown when any step of the conversion pipeline fails. */
    public static class ConversionException extends RuntimeException {
        public ConversionException(String message) {
            super(message);
        }
    }

    /* ── Embedded fallback conversion script ─────────────────────────── */

    private static final String EMBEDDED_SCRIPT = """
            #!/usr/bin/env python3
            # JForge PyTorch → ONNX converter (embedded fallback)
            # Full version: scripts/convert_pytorch_to_onnx.py
            import argparse, subprocess, sys
            from pathlib import Path

            def p(pct, msg):
                print(f"PROGRESS: {pct} {msg}", flush=True)

            def err(msg):
                print(f"ERROR: {msg}", flush=True)
                sys.exit(1)

            def get_optimum_info(field):
                try:
                    out = subprocess.run([sys.executable,"-m","pip","show","optimum"],
                                        capture_output=True,text=True,timeout=15).stdout
                    for line in out.splitlines():
                        if line.startswith(field+":"):
                            return line.split(":",1)[1].strip()
                except Exception:
                    pass
                return "unknown"

            def reinstall_optimum():
                p(2,"Force-reinstalling optimum[exporters]…")
                subprocess.check_call([sys.executable,"-m","pip","install",
                    "--force-reinstall","--no-cache-dir","optimum[exporters]>=1.19.0"])
                p(4,"optimum[exporters] reinstalled.")

            def import_main_export():
                try:
                    from optimum.exporters.onnx import main_export
                    return main_export
                except ImportError as e1:
                    pass
                try:
                    from optimum.onnx import main_export
                    return main_export
                except ImportError as e2:
                    pass
                ver = get_optimum_info("Version")
                loc = get_optimum_info("Location")
                err(f"Cannot import optimum ONNX exporter.\\n"
                    f"  optimum version  : {ver}\\n"
                    f"  optimum location : {loc}\\n"
                    f"Fix: pip install --force-reinstall 'optimum[exporters]>=1.19.0'\\n"
                    f"Or re-run with --reinstall flag.")

            def convert_diffusers(mid, out):
                p(5,"Importing optimum exporters…")
                main_export = import_main_export()
                p(10,f"Exporting {mid}…")
                tasks = ["auto","text-to-video","stable-diffusion-xl","stable-diffusion"]
                last = None
                for task in tasks:
                    try:
                        p(20,f"Trying task={task}…")
                        main_export(model_name_or_path=mid,output=Path(out),task=task,fp16=False)
                        p(100,f"Done (task={task})")
                        return
                    except Exception as e:
                        last = e
                        p(15,f"Task '{task}' failed ({e}) — trying next…")
                hint = ""
                ml = mid.lower()
                if "vilab" in ml or "text-to-video" in ml or "t2v" in ml:
                    hint = ("\\nHint: text-to-video models use a custom UNet3D not yet "
                            "in optimum's registry; consider running inference via "
                            "the diffusers pipeline directly.")
                err(f"No compatible export task found. Last error: {last}{hint}")

            def convert_generic(mp, out, shape):
                try:
                    import torch,torch.onnx
                except ImportError:
                    err("torch not installed — pip install torch")
                p(10,"Loading model…")
                try:
                    m = torch.load(mp,map_location="cpu",weights_only=False)
                except Exception as e:
                    err(f"Failed to load: {e}")
                if isinstance(m,dict):
                    err("state_dict only — use --mode diffusers for HF models")
                if not hasattr(m,"forward"):
                    err("No forward() method")
                m.eval()
                s = [int(x) for x in shape.split(",")]
                op = Path(out)/"model.onnx"
                op.parent.mkdir(parents=True,exist_ok=True)
                p(50,"Running torch.onnx.export()…")
                try:
                    torch.onnx.export(m,torch.randn(*s),str(op),export_params=True,
                        opset_version=17,do_constant_folding=True,
                        input_names=["modelInput"],output_names=["modelOutput"],
                        dynamic_axes={"modelInput":{0:"batch"},"modelOutput":{0:"batch"}})
                except Exception as e:
                    err(f"torch.onnx.export() failed: {e}")
                p(100,"Done")

            if __name__=="__main__":
                ap = argparse.ArgumentParser()
                ap.add_argument("--model_id",required=True)
                ap.add_argument("--output_dir",required=True)
                ap.add_argument("--mode",default="diffusers")
                ap.add_argument("--input_shape",default="1,3,224,224")
                ap.add_argument("--reinstall",action="store_true")
                a = ap.parse_args()
                Path(a.output_dir).mkdir(parents=True,exist_ok=True)
                if a.reinstall:
                    reinstall_optimum()
                if a.mode=="diffusers":
                    convert_diffusers(a.model_id,a.output_dir)
                else:
                    convert_generic(a.model_id,a.output_dir,a.input_shape)
            """;
}
