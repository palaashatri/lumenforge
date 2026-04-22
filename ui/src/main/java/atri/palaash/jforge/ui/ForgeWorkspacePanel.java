package atri.palaash.jforge.ui;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import atri.palaash.jforge.model.ModelDescriptor;
import atri.palaash.jforge.models.ForgeModelRegistry;
import atri.palaash.jforge.models.ModelCompatibility;
import atri.palaash.jforge.storage.ModelDownloader;
import atri.palaash.jforge.storage.ModelStorage;
import atri.palaash.jforge.inference.InferenceService;
import atri.palaash.jforge.tasks.ForgeTask;
import atri.palaash.jforge.tasks.Img2ImgTask;
import atri.palaash.jforge.tasks.InpaintTask;
import atri.palaash.jforge.tasks.TaskConfig;
import atri.palaash.jforge.tasks.TaskContext;
import atri.palaash.jforge.tasks.TaskResult;
import atri.palaash.jforge.tasks.Txt2ImgTask;
import atri.palaash.jforge.tasks.UpscaleTask;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Three-pane diffusion workspace:
 * left prompt panel, center preview, right settings/actions.
 */
public class ForgeWorkspacePanel extends JPanel {

    public enum Mode {
        TEXT_TO_IMAGE,
        IMAGE_TO_IMAGE,
        INPAINT,
        UPSCALE
    }

    private static final Path HISTORY_PATH = Path.of(
            System.getProperty("user.home"),
            ".jforge-models",
            "history.json");
    private static final ObjectMapper HISTORY_MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);
    private static final Pattern STEP_PATTERN = Pattern.compile("(\\d+)/(\\d+)");
    private static final List<String> DEFAULT_SAMPLERS = List.of("Euler", "Euler A", "DPM++ 2M", "DPM++ SDE");
    private static final List<String> DEFAULT_SCHEDULERS = List.of("Euler", "Euler A", "DPM++ 2M", "DPM++ SDE", "Flow Matching");

    private final Mode mode;
    private final ModelDownloader modelDownloader;
    private final InferenceService inferenceService;
    private ForgeModelRegistry forgeModelRegistry;

    private ModelStorage modelStorage;
    private BooleanSupplier gpuSupplier = () -> true;
    private Runnable openModelManager;

    private final JComboBox<ModelDescriptor> modelCombo;
    private final JComboBox<String> deviceCombo;

    private final JTextArea promptArea;
    private final JTextArea negativePromptArea;
    private final JSpinner seedSpinner;
    private final JSpinner batchSpinner;
    private final JComboBox<String> stylePresetCombo;
    private final JComboBox<String> aspectRatioCombo;

    private final JTextField inputImageField;
    private final JTextField maskImageField;
    private final JSlider strengthSlider;

    private final JSpinner stepsSpinner;
    private final JSpinner cfgSpinner;
    private final JComboBox<String> samplerCombo;
    private final JComboBox<String> schedulerCombo;
    private final JTextField widthField;
    private final JTextField heightField;

    private final ZoomPanImagePanel previewPanel;
    private final JTextArea logArea;

    private final JButton generateButton;
    private final JButton cancelButton;
    private final JButton saveButton;
    private final JButton openFolderButton;
    private final JButton metadataButton;

    private final JProgressBar progressBar;
    private final JLabel statusLabel;

    private AtomicBoolean cancellationFlag;
    private boolean running;
    private String lastArtifactPath = "";
    private Map<String, Object> lastMetadata = Map.of();

    public ForgeWorkspacePanel(Mode mode,
                               List<ModelDescriptor> models,
                               ModelDownloader modelDownloader,
                               InferenceService inferenceService) {
        super(new BorderLayout(12, 12));
        this.mode = mode;
        this.modelDownloader = modelDownloader;
        this.inferenceService = inferenceService;

        setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));

        modelCombo = new JComboBox<>(models.toArray(new ModelDescriptor[0]));
        modelCombo.setRenderer((list, value, index, selected, focus) -> {
            JLabel label = new JLabel(value == null ? "" : value.displayName());
            label.setOpaque(selected);
            if (selected) {
                label.setBackground(list.getSelectionBackground());
                label.setForeground(list.getSelectionForeground());
            }
            label.setBorder(BorderFactory.createEmptyBorder(2, 6, 2, 6));
            return label;
        });
        deviceCombo = new JComboBox<>(new String[]{"Auto", "GPU", "CPU"});

        promptArea = new JTextArea(5, 24);
        promptArea.setLineWrap(true);
        promptArea.setWrapStyleWord(true);

        negativePromptArea = new JTextArea(3, 24);
        negativePromptArea.setLineWrap(true);
        negativePromptArea.setWrapStyleWord(true);

        seedSpinner = new JSpinner(new SpinnerNumberModel(42L, 0L, Long.MAX_VALUE, 1L));
        batchSpinner = new JSpinner(new SpinnerNumberModel(1, 1, 8, 1));
        stylePresetCombo = new JComboBox<>(new String[]{
                "None", "Photorealistic", "Anime", "Illustration", "Cinematic"
        });
        aspectRatioCombo = new JComboBox<>(new String[]{"1:1", "4:3", "16:9", "9:16", "Custom"});

        inputImageField = new JTextField();
        inputImageField.setEditable(false);
        maskImageField = new JTextField();
        maskImageField.setEditable(false);
        strengthSlider = new JSlider(0, 100, 75);
        strengthSlider.setPaintTicks(true);
        strengthSlider.setPaintLabels(true);
        strengthSlider.setMinorTickSpacing(5);
        strengthSlider.setMajorTickSpacing(25);

        stepsSpinner = new JSpinner(new SpinnerNumberModel(20, 1, 100, 1));
        cfgSpinner = new JSpinner(new SpinnerNumberModel(7.5, 1.0, 30.0, 0.5));
        samplerCombo = new JComboBox<>(new String[]{"Euler", "Euler A", "DPM++ 2M", "DPM++ SDE"});
        schedulerCombo = new JComboBox<>(new String[]{"Euler", "Euler A", "DPM++ 2M", "DPM++ SDE", "Flow Matching"});
        widthField = new JTextField("512");
        heightField = new JTextField("512");

        previewPanel = new ZoomPanImagePanel();
        logArea = new JTextArea();
        logArea.setEditable(false);

        generateButton = new JButton(mode == Mode.UPSCALE ? "Upscale" : "Generate");
        cancelButton = new JButton("Cancel");
        cancelButton.setEnabled(false);
        saveButton = new JButton("Save");
        saveButton.setEnabled(false);
        openFolderButton = new JButton("Open Folder");
        openFolderButton.setEnabled(false);
        metadataButton = new JButton("Metadata");
        metadataButton.setEnabled(false);

        progressBar = new JProgressBar();
        progressBar.setIndeterminate(false);
        progressBar.setVisible(false);
        statusLabel = new JLabel("Ready");

        aspectRatioCombo.addActionListener(e -> applyAspectRatioPreset());
        generateButton.addActionListener(e -> runTask());
        cancelButton.addActionListener(e -> cancelTask());
        saveButton.addActionListener(e -> saveOutputImage());
        openFolderButton.addActionListener(e -> openOutputFolder());
        metadataButton.addActionListener(e -> showMetadataDialog());
        modelCombo.addActionListener(e -> applyModelCompatibilityOptions());

        buildLayout();
        applyModeVisibility();
        applyModelCompatibilityOptions();
    }

    public void setModelStorage(ModelStorage storage) {
        this.modelStorage = storage;
    }

    public void setGpuSupplier(BooleanSupplier supplier) {
        this.gpuSupplier = supplier;
    }

    public void setOpenModelManager(Runnable callback) {
        this.openModelManager = callback;
    }

    public void setForgeModelRegistry(ForgeModelRegistry forgeModelRegistry) {
        this.forgeModelRegistry = forgeModelRegistry;
        applyModelCompatibilityOptions();
    }

    public void updateModels(List<ModelDescriptor> models) {
        List<ModelDescriptor> available = models;
        if (modelStorage != null) {
            available = models.stream().filter(modelStorage::isAvailable).toList();
        }

        ModelDescriptor selected = (ModelDescriptor) modelCombo.getSelectedItem();
        modelCombo.removeAllItems();
        for (ModelDescriptor descriptor : available) {
            modelCombo.addItem(descriptor);
        }
        if (selected != null) {
            modelCombo.setSelectedItem(selected);
        }
        if (modelCombo.getItemCount() == 0) {
            generateButton.setEnabled(false);
            statusLabel.setText("No models downloaded. Open Models tab to download one.");
        } else if (!running) {
            generateButton.setEnabled(true);
            statusLabel.setText("Ready");
        }

        applyModelCompatibilityOptions();
    }

    private void buildLayout() {
        JPanel left = buildPromptPanel();
        JPanel center = buildPreviewPanel();
        JPanel right = buildSettingsPanel();

        JPanel main = new JPanel(new GridLayout(1, 3, 12, 12));
        main.add(left);
        main.add(center);
        main.add(right);

        JPanel bottom = new JPanel(new BorderLayout(8, 0));
        bottom.add(progressBar, BorderLayout.CENTER);
        bottom.add(statusLabel, BorderLayout.EAST);

        add(main, BorderLayout.CENTER);
        add(bottom, BorderLayout.SOUTH);
    }

    private JPanel buildPromptPanel() {
        JPanel left = new JPanel();
        left.setLayout(new BoxLayout(left, BoxLayout.Y_AXIS));
        left.setBorder(BorderFactory.createTitledBorder("Prompt"));

        left.add(labelledScroll("Prompt", promptArea, 120));
        left.add(labelledScroll("Negative Prompt", negativePromptArea, 90));
        left.add(labelled("Seed", seedSpinner));
        left.add(labelled("Batch", batchSpinner));
        left.add(labelled("Style", stylePresetCombo));
        left.add(labelled("Aspect", aspectRatioCombo));

        JButton browseInput = new JButton("Choose Input Image");
        browseInput.addActionListener(e -> chooseImage(inputImageField, false));
        left.add(labelled("Input", wrapInputRow(inputImageField, browseInput)));

        JButton browseMask = new JButton("Choose Mask");
        browseMask.addActionListener(e -> chooseImage(maskImageField, true));
        left.add(labelled("Mask", wrapInputRow(maskImageField, browseMask)));

        left.add(labelled("Strength", strengthSlider));

        return left;
    }

    private JPanel buildPreviewPanel() {
        JPanel center = new JPanel(new BorderLayout(8, 8));
        center.setBorder(BorderFactory.createTitledBorder("Output Preview"));

        previewPanel.setPreferredSize(new Dimension(420, 420));
        center.add(previewPanel, BorderLayout.CENTER);

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        JButton resetViewButton = new JButton("Reset View");
        resetViewButton.addActionListener(e -> previewPanel.resetView());
        controls.add(resetViewButton);
        controls.add(saveButton);
        controls.add(openFolderButton);
        controls.add(metadataButton);
        center.add(controls, BorderLayout.SOUTH);

        return center;
    }

    private JPanel buildSettingsPanel() {
        JPanel right = new JPanel();
        right.setLayout(new BoxLayout(right, BoxLayout.Y_AXIS));
        right.setBorder(BorderFactory.createTitledBorder("Settings"));

        right.add(labelled("Model", modelCombo));
        right.add(labelled("Device", deviceCombo));
        right.add(labelled("Steps", stepsSpinner));
        right.add(labelled("CFG", cfgSpinner));
        right.add(labelled("Sampler", samplerCombo));
        right.add(labelled("Scheduler", schedulerCombo));
        right.add(labelled("Width", widthField));
        right.add(labelled("Height", heightField));

        JPanel runRow = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        runRow.add(cancelButton);
        runRow.add(generateButton);
        right.add(runRow);

        JScrollPane logScroll = new JScrollPane(logArea);
        logScroll.setBorder(BorderFactory.createTitledBorder("Live Log"));
        logScroll.setPreferredSize(new Dimension(300, 170));
        right.add(logScroll);

        return right;
    }

    private JPanel labelled(String label, java.awt.Component component) {
        JPanel row = new JPanel(new BorderLayout(0, 4));
        JLabel title = new JLabel(label);
        title.setForeground(new Color(110, 110, 110));
        row.add(title, BorderLayout.NORTH);
        row.add(component, BorderLayout.CENTER);
        row.setMaximumSize(new Dimension(Integer.MAX_VALUE, 100));
        return row;
    }

    private JPanel labelledScroll(String label, JTextArea area, int preferredHeight) {
        JScrollPane scrollPane = new JScrollPane(area);
        scrollPane.setPreferredSize(new Dimension(220, preferredHeight));
        JPanel row = labelled(label, scrollPane);
        row.setMaximumSize(new Dimension(Integer.MAX_VALUE, preferredHeight + 24));
        return row;
    }

    private JPanel wrapInputRow(JTextField field, JButton button) {
        JPanel row = new JPanel(new BorderLayout(6, 0));
        row.add(field, BorderLayout.CENTER);
        row.add(button, BorderLayout.EAST);
        return row;
    }

    private void applyModeVisibility() {
        boolean needsInput = mode == Mode.IMAGE_TO_IMAGE || mode == Mode.INPAINT || mode == Mode.UPSCALE;
        boolean needsMask = mode == Mode.INPAINT;
        boolean needsPrompt = mode != Mode.UPSCALE;

        promptArea.setEnabled(needsPrompt);
        negativePromptArea.setEnabled(needsPrompt);
        stylePresetCombo.setEnabled(needsPrompt);

        inputImageField.setEnabled(needsInput);
        maskImageField.setEnabled(needsMask);
        strengthSlider.setEnabled(mode == Mode.IMAGE_TO_IMAGE || mode == Mode.INPAINT);

        if (mode == Mode.UPSCALE) {
            samplerCombo.setEnabled(false);
            schedulerCombo.setEnabled(false);
            cfgSpinner.setEnabled(false);
            promptArea.setText("");
            negativePromptArea.setText("");
        }
        if (mode == Mode.INPAINT) {
            schedulerCombo.setSelectedItem("Flow Matching");
        }
    }

    private void applyModelCompatibilityOptions() {
        List<String> samplers = new ArrayList<>(DEFAULT_SAMPLERS);
        List<String> schedulers = new ArrayList<>(DEFAULT_SCHEDULERS);

        ModelDescriptor model = (ModelDescriptor) modelCombo.getSelectedItem();
        if (model != null && forgeModelRegistry != null) {
            var maybeModel = forgeModelRegistry.byId(model.id());
            if (maybeModel.isPresent()) {
                var forgeModel = maybeModel.get();
                List<String> modelSamplers = asStringList(forgeModel.metadata().get("supportedSamplers"));
                List<String> modelSchedulers = asStringList(forgeModel.metadata().get("supportedSchedulers"));

                if (!modelSamplers.isEmpty()) {
                    samplers.clear();
                    samplers.addAll(modelSamplers);
                }
                if (!modelSchedulers.isEmpty()) {
                    schedulers.clear();
                    schedulers.addAll(modelSchedulers);
                }
            }
        }

        if (mode == Mode.UPSCALE) {
            samplers = new ArrayList<>(List.of("Euler"));
            schedulers = new ArrayList<>(List.of("Euler"));
        }

        applyComboOptions(samplerCombo, samplers);
        applyComboOptions(schedulerCombo, schedulers);

        if (mode == Mode.INPAINT && containsIgnoreCase(schedulers, "Flow Matching")) {
            schedulerCombo.setSelectedItem("Flow Matching");
        }
    }

    private void applyComboOptions(JComboBox<String> comboBox, List<String> options) {
        List<String> sanitized = options.stream()
                .filter(option -> option != null && !option.isBlank())
                .toList();
        if (sanitized.isEmpty()) {
            return;
        }

        Object selected = comboBox.getSelectedItem();
        comboBox.removeAllItems();
        for (String option : sanitized) {
            comboBox.addItem(option);
        }

        if (selected instanceof String selectedText && containsIgnoreCase(sanitized, selectedText)) {
            comboBox.setSelectedItem(selectedText);
        } else {
            comboBox.setSelectedIndex(0);
        }
    }

    private List<String> asStringList(Object raw) {
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            return List.of();
        }

        List<String> output = new ArrayList<>();
        for (Object item : list) {
            if (item == null) {
                continue;
            }
            String text = item.toString().trim();
            if (!text.isEmpty()) {
                output.add(text);
            }
        }
        return output;
    }

    private boolean containsIgnoreCase(List<String> values, String target) {
        return values.stream().anyMatch(value -> value.equalsIgnoreCase(target));
    }

    private void applyAspectRatioPreset() {
        String selected = String.valueOf(aspectRatioCombo.getSelectedItem());
        switch (selected) {
            case "1:1" -> {
                widthField.setText("512");
                heightField.setText("512");
            }
            case "4:3" -> {
                widthField.setText("640");
                heightField.setText("480");
            }
            case "16:9" -> {
                widthField.setText("768");
                heightField.setText("432");
            }
            case "9:16" -> {
                widthField.setText("432");
                heightField.setText("768");
            }
            default -> {
            }
        }
    }

    private void chooseImage(JTextField targetField, boolean mask) {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileFilter(new FileNameExtensionFilter("Image files", "png", "jpg", "jpeg", "bmp", "webp"));
        chooser.setDialogTitle(mask ? "Choose mask image" : "Choose input image");
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            targetField.setText(chooser.getSelectedFile().getAbsolutePath());
        }
    }

    private void runTask() {
        ModelDescriptor model = (ModelDescriptor) modelCombo.getSelectedItem();
        if (model == null) {
            if (openModelManager != null) {
                statusLabel.setText("No models available. Opening model manager...");
                openModelManager.run();
                return;
            }
            statusLabel.setText("Select a model first.");
            return;
        }

        if ((mode == Mode.IMAGE_TO_IMAGE || mode == Mode.INPAINT || mode == Mode.UPSCALE)
                && inputImageField.getText().isBlank()) {
            statusLabel.setText("Choose an input image for this task.");
            return;
        }

        String prompt = promptArea.getText().trim();
        if (mode != Mode.UPSCALE && prompt.isBlank()) {
            statusLabel.setText("Enter a prompt.");
            return;
        }

        if (!validateCompatibility(model)) {
            return;
        }

        setRunning(true, "Preparing...");
        logArea.setText("");
        previewPanel.clearImage();
        lastArtifactPath = "";
        lastMetadata = Map.of();

        cancellationFlag = new AtomicBoolean(false);

        CompletableFuture<?> downloadFuture;
        if (!modelDownloader.canDownload(model)) {
            Path localPath = Path.of(System.getProperty("user.home"), ".jforge-models", model.relativePath());
            if (!Files.exists(localPath)) {
                setRunning(false, "Model not found locally.");
                appendLog("Model requires manual import/download from Models tab.");
                return;
            }
            downloadFuture = CompletableFuture.completedFuture(null);
        } else {
            downloadFuture = modelDownloader.downloadIfMissing(model, progress -> SwingUtilities.invokeLater(() -> {
                if (progress.isStatusMessage()) {
                    setRunning(true, progress.statusMessage());
                    appendLog(progress.statusMessage());
                } else {
                    String msg = "Download: " + progress.percent() + "%";
                    setRunning(true, msg);
                    appendLog(msg);
                }
            }));
        }

        downloadFuture.thenCompose(ignored -> CompletableFuture.supplyAsync(() -> executeTask(model)))
                .whenComplete((result, error) -> SwingUtilities.invokeLater(() -> {
                    if (error != null) {
                        setRunning(false, "Failed");
                        appendLog("Error: " + error.getMessage());
                        return;
                    }
                    renderResult(result);
                    setRunning(false, result.success() ? "Done" : "Failed");
                }));
    }

    private TaskResult executeTask(ModelDescriptor model) {
        TaskConfig config = buildTaskConfig(model);
        ForgeTask task = switch (mode) {
            case TEXT_TO_IMAGE -> new Txt2ImgTask(config);
            case IMAGE_TO_IMAGE -> new Img2ImgTask(config);
            case INPAINT -> new InpaintTask(config);
            case UPSCALE -> new UpscaleTask(config);
        };

        TaskContext context = new TaskContext(
                model,
                inferenceService,
                message -> SwingUtilities.invokeLater(() -> {
                    setRunning(true, message);
                    appendLog(message);
                }),
                cancellationFlag);

        return task.run(context);
    }

    private TaskConfig buildTaskConfig(ModelDescriptor model) {
        String sampler = String.valueOf(samplerCombo.getSelectedItem());
        String scheduler = String.valueOf(schedulerCombo.getSelectedItem());
        String style = String.valueOf(stylePresetCombo.getSelectedItem())
                + " | sampler=" + sampler
                + " | scheduler=" + scheduler;

        boolean preferGpu = switch (String.valueOf(deviceCombo.getSelectedItem())) {
            case "CPU" -> false;
            case "GPU" -> true;
            default -> gpuSupplier.getAsBoolean();
        };

        return new TaskConfig(
                model.id(),
                promptArea.getText().trim(),
                negativePromptArea.getText().trim(),
                ((Number) seedSpinner.getValue()).longValue(),
                ((Number) batchSpinner.getValue()).intValue(),
                parseInt(widthField.getText(), 512),
                parseInt(heightField.getText(), 512),
                ((Number) stepsSpinner.getValue()).intValue(),
                ((Number) cfgSpinner.getValue()).floatValue(),
                style,
                sampler,
                scheduler,
                preferGpu,
                inputImageField.getText().trim(),
                maskImageField.getText().trim(),
                strengthSlider.getValue() / 100.0
        );
    }

    private void cancelTask() {
        if (cancellationFlag != null) {
            cancellationFlag.set(true);
            setRunning(true, "Cancelling...");
        }
    }

    private void renderResult(TaskResult result) {
        if (!result.success()) {
            appendLog(result.details());
            return;
        }

        appendLog(result.details());
        appendLog(result.output());
        if (result.artifactPath() != null && !result.artifactPath().isBlank()) {
            Path outputPath = Path.of(result.artifactPath());
            try {
                BufferedImage image = ImageIO.read(outputPath.toFile());
                if (image != null) {
                    previewPanel.setImage(image);
                }
                lastArtifactPath = outputPath.toString();
                saveButton.setEnabled(true);
                openFolderButton.setEnabled(true);
                metadataButton.setEnabled(true);

                lastMetadata = buildMetadata(result);
                appendHistory(lastMetadata);
                writeOutputMetadata(outputPath, lastMetadata);
            } catch (Exception ex) {
                appendLog("Preview load failed: " + ex.getMessage());
            }
        }
    }

    private Map<String, Object> buildMetadata(TaskResult result) {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("timestamp", LocalDateTime.now().toString());
        ModelDescriptor model = (ModelDescriptor) modelCombo.getSelectedItem();
        metadata.put("model", model == null ? "" : model.displayName());
        metadata.put("modelId", model == null ? "" : model.id());
        metadata.put("prompt", promptArea.getText().trim());
        metadata.put("negativePrompt", negativePromptArea.getText().trim());
        metadata.put("seed", ((Number) seedSpinner.getValue()).longValue());
        metadata.put("batch", ((Number) batchSpinner.getValue()).intValue());
        metadata.put("size", widthField.getText().trim() + "x" + heightField.getText().trim());
        metadata.put("style", stylePresetCombo.getSelectedItem());
        metadata.put("status", result.success() ? "OK" : "FAIL");
        metadata.put("outputPath", result.artifactPath());
        metadata.put("sampler", samplerCombo.getSelectedItem());
        metadata.put("scheduler", schedulerCombo.getSelectedItem());
        metadata.put("steps", stepsSpinner.getValue());
        metadata.put("cfg", cfgSpinner.getValue());
        metadata.put("task", mode.name());
        metadata.put("details", result.details());

        if (model != null && forgeModelRegistry != null) {
            forgeModelRegistry.byId(model.id()).ifPresent(forgeModel -> {
                metadata.put("modelType", forgeModel.type().name());
                metadata.put("modelPath", forgeModel.path().toString());
                metadata.put("modelVersion", forgeModel.metadata().getOrDefault("version", ""));
                metadata.put("modelTags", forgeModel.metadata().getOrDefault("tags", List.of()));
            });
        }
        return metadata;
    }

    private synchronized void appendHistory(Map<String, Object> entry) {
        try {
            Files.createDirectories(HISTORY_PATH.getParent());
            List<Map<String, Object>> rows;
            if (Files.exists(HISTORY_PATH)) {
                rows = HISTORY_MAPPER.readValue(HISTORY_PATH.toFile(), new TypeReference<>() {});
            } else {
                rows = new ArrayList<>();
            }
            rows.add(0, entry);
            HISTORY_MAPPER.writeValue(HISTORY_PATH.toFile(), rows);
        } catch (Exception ex) {
            appendLog("History write failed: " + ex.getMessage());
        }
    }

    private void writeOutputMetadata(Path imagePath, Map<String, Object> metadata) {
        try {
            String fileName = imagePath.getFileName().toString();
            int dot = fileName.lastIndexOf('.');
            String baseName = dot > 0 ? fileName.substring(0, dot) : fileName;
            Path metadataPath = imagePath.resolveSibling(baseName + ".json");
            HISTORY_MAPPER.writeValue(metadataPath.toFile(), metadata);
            Path genericMetadataPath = imagePath.getParent().resolve("metadata.json");
            HISTORY_MAPPER.writeValue(genericMetadataPath.toFile(), metadata);
        } catch (Exception ex) {
            appendLog("Metadata sidecar write failed: " + ex.getMessage());
        }
    }

    private boolean validateCompatibility(ModelDescriptor model) {
        if (forgeModelRegistry == null) {
            return true;
        }

        String scheduler = String.valueOf(schedulerCombo.getSelectedItem());
        String sampler = String.valueOf(samplerCombo.getSelectedItem());

        return forgeModelRegistry.byId(model.id())
                .map(forgeModel -> {
                    ModelCompatibility compatibility = forgeModelRegistry.checkCompatibility(
                            forgeModel,
                            scheduler,
                            sampler);

                    List<String> issues = new ArrayList<>(compatibility.issues());
                    if (modelDownloader.canDownload(model)) {
                        issues.removeIf(issue -> issue.startsWith("Model file is not available locally:"));
                    }

                    if (!issues.isEmpty()) {
                        String message = String.join("\n", issues);
                        JOptionPane.showMessageDialog(
                                this,
                                message,
                                "Model Compatibility Error",
                                JOptionPane.ERROR_MESSAGE);
                        statusLabel.setText("Compatibility check failed.");
                        return false;
                    }

                    if (!compatibility.warnings().isEmpty()) {
                        String warningMessage = String.join("\n", compatibility.warnings());
                        int choice = JOptionPane.showConfirmDialog(
                                this,
                                warningMessage + "\n\nContinue anyway?",
                                "Model Compatibility Warning",
                                JOptionPane.YES_NO_OPTION,
                                JOptionPane.WARNING_MESSAGE);
                        if (choice != JOptionPane.YES_OPTION) {
                            statusLabel.setText("Cancelled by compatibility warning.");
                            return false;
                        }
                    }
                    return true;
                })
                .orElse(true);
    }

    private void saveOutputImage() {
        if (lastArtifactPath.isBlank()) {
            return;
        }
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Save Output Image");
        chooser.setFileFilter(new FileNameExtensionFilter("PNG Image", "png"));
        chooser.setSelectedFile(Path.of(lastArtifactPath).toFile());
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                Path target = chooser.getSelectedFile().toPath();
                if (!target.toString().toLowerCase().endsWith(".png")) {
                    target = target.resolveSibling(target.getFileName() + ".png");
                }
                Files.copy(Path.of(lastArtifactPath), target, StandardCopyOption.REPLACE_EXISTING);
                statusLabel.setText("Saved to " + target.getFileName());
            } catch (Exception ex) {
                statusLabel.setText("Save failed: " + ex.getMessage());
            }
        }
    }

    private void openOutputFolder() {
        if (lastArtifactPath.isBlank()) {
            return;
        }
        try {
            Path parent = Path.of(lastArtifactPath).getParent();
            if (parent != null) {
                Desktop.getDesktop().open(parent.toFile());
            }
        } catch (Exception ex) {
            statusLabel.setText("Cannot open folder: " + ex.getMessage());
        }
    }

    private void showMetadataDialog() {
        if (lastMetadata.isEmpty()) {
            return;
        }
        StringBuilder content = new StringBuilder();
        content.append("Task: ").append(lastMetadata.getOrDefault("task", "")).append('\n');
        content.append("Model: ").append(lastMetadata.getOrDefault("model", "")).append('\n');
        content.append("Prompt: ").append(lastMetadata.getOrDefault("prompt", "")).append('\n');
        content.append("Negative: ").append(lastMetadata.getOrDefault("negativePrompt", "")).append('\n');
        content.append("Seed: ").append(lastMetadata.getOrDefault("seed", "")).append('\n');
        content.append("Size: ").append(lastMetadata.getOrDefault("size", "")).append('\n');
        content.append("Steps: ").append(lastMetadata.getOrDefault("steps", "")).append('\n');
        content.append("CFG: ").append(lastMetadata.getOrDefault("cfg", "")).append('\n');
        content.append("Sampler: ").append(lastMetadata.getOrDefault("sampler", "")).append('\n');
        content.append("Scheduler: ").append(lastMetadata.getOrDefault("scheduler", "")).append('\n');
        content.append("Output: ").append(lastMetadata.getOrDefault("outputPath", ""));
        JOptionPane.showMessageDialog(this, content.toString(), "Generation Metadata", JOptionPane.INFORMATION_MESSAGE);
    }

    private void setRunning(boolean busy, String status) {
        this.running = busy;
        generateButton.setEnabled(!busy);
        cancelButton.setEnabled(busy);
        progressBar.setVisible(busy);
        statusLabel.setText(status);

        Matcher matcher = STEP_PATTERN.matcher(status);
        if (busy && matcher.find()) {
            int current = parseInt(matcher.group(1), 0);
            int total = parseInt(matcher.group(2), 0);
            if (total > 0) {
                progressBar.setIndeterminate(false);
                progressBar.setMaximum(total);
                progressBar.setValue(current);
            }
        } else if (busy) {
            progressBar.setIndeterminate(true);
        } else {
            progressBar.setValue(0);
            progressBar.setIndeterminate(false);
        }
    }

    private void appendLog(String text) {
        logArea.append(text + "\n");
        logArea.setCaretPosition(logArea.getDocument().getLength());
    }

    private int parseInt(String text, int fallback) {
        try {
            return Integer.parseInt(text.trim());
        } catch (Exception ex) {
            return fallback;
        }
    }
}
