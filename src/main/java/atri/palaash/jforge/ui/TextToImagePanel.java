package atri.palaash.jforge.ui;

import atri.palaash.jforge.inference.InferenceRequest;
import atri.palaash.jforge.inference.InferenceResult;
import atri.palaash.jforge.inference.InferenceService;
import atri.palaash.jforge.inference.PromptEnhancer;
import atri.palaash.jforge.model.ModelDescriptor;
import atri.palaash.jforge.storage.ModelDownloader;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JTextPane;
import javax.swing.JToggleButton;
import javax.swing.KeyStroke;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import atri.palaash.jforge.storage.ModelStorage;

/**
 * Text → Image panel. Noob-friendly by default: just model + prompt + Generate.
 * Advanced options (negative prompt, seed, dimensions, style) hidden behind
 * a disclosure triangle. All download/inference progress goes to the Log tab.
 * GPU preference is injected from the menu bar.
 */
public class TextToImagePanel extends JPanel {

    private final ModelDownloader modelDownloader;
    private final InferenceService inferenceService;
    private ModelStorage modelStorage;
    private Runnable openModelManager;

    /* Controls */
    private final JComboBox<ModelDescriptor> modelCombo;
    private final JTextArea promptField;
    private final JTextField negativePromptField;
    private final JSpinner seedSpinner;
    private final JComboBox<String> aspectRatioBox;
    private final JTextField widthField;
    private final JTextField heightField;
    private final JComboBox<String> stylePresetBox;
    private final JSpinner stepsSpinner;

    /* Output */
    private final JLabel outputPreview;
    private final JLabel statusLabel;
    private final JButton runButton;
    private final JButton cancelButton;
    private final JButton saveButton;
    private final JProgressBar progressBar;

    /* Log / History / Library */
    private final JTextPane logArea;
    private final PromptLibraryPanel promptLibraryPanel;
    private final HistoryPanel historyPanel;

    /* Advanced disclosure */
    private final JToggleButton advancedToggle;
    private final JPanel advancedPanel;

    /* State */
    private String lastArtifactPath = "";
    private boolean running;
    private BooleanSupplier gpuSupplier = () -> true;
    private AtomicBoolean cancellationFlag;
    private static final Pattern STEP_PATTERN = Pattern.compile("Denoising:\\s*(\\d+)/(\\d+)");

    public TextToImagePanel(List<ModelDescriptor> models,
                            ModelDownloader modelDownloader,
                            InferenceService inferenceService) {
        super(new BorderLayout(0, 0));
        this.modelDownloader = modelDownloader;
        this.inferenceService = inferenceService;

        /* ---- init components ---- */
        modelCombo = new JComboBox<>(models.toArray(new ModelDescriptor[0]));
        modelCombo.setRenderer((list, value, idx, sel, focus) -> {
            JLabel lbl = new JLabel(value == null ? "" : value.displayName());
            lbl.setBorder(BorderFactory.createEmptyBorder(4, 8, 4, 8));
            if (sel) {
                lbl.setBackground(list.getSelectionBackground());
                lbl.setForeground(list.getSelectionForeground());
                lbl.setOpaque(true);
            }
            return lbl;
        });

        promptField = new JTextArea(3, 40);
        promptField.setLineWrap(true);
        promptField.setWrapStyleWord(true);
        promptField.putClientProperty("JTextField.placeholderText",
                "Describe the image you want to create\u2026");

        negativePromptField = new JTextField();
        negativePromptField.putClientProperty("JTextField.placeholderText",
                "Things to avoid (optional)");
        seedSpinner = new JSpinner(new SpinnerNumberModel(42, 0, Integer.MAX_VALUE, 1));
        aspectRatioBox = new JComboBox<>(new String[]{"1:1", "4:3", "16:9", "3:2", "Custom"});
        widthField = new JTextField("512");
        heightField = new JTextField("512");
        stylePresetBox = new JComboBox<>(new String[]{
                "None", "Cinematic", "Sketch", "Product", "Illustration"});
        stepsSpinner = new JSpinner(new SpinnerNumberModel(15, 5, 50, 1));
        stepsSpinner.setToolTipText("Fewer steps = faster but lower quality. 10\u201320 recommended.");

        outputPreview = new JLabel("", JLabel.CENTER);
        outputPreview.setVerticalAlignment(JLabel.CENTER);

        statusLabel = new JLabel("Ready");
        statusLabel.setFont(statusLabel.getFont().deriveFont(Font.PLAIN, 11f));
        statusLabel.setForeground(new Color(120, 120, 120));
        statusLabel.setBorder(BorderFactory.createEmptyBorder(6, 20, 8, 20));

        runButton = new JButton("Generate");
        cancelButton = new JButton("Cancel");
        cancelButton.setEnabled(false);
        cancelButton.setToolTipText("Cancel the current generation");
        saveButton = new JButton("Save Result");
        saveButton.setEnabled(false);

        progressBar = new JProgressBar();
        progressBar.setIndeterminate(true);
        progressBar.setVisible(false);
        progressBar.setPreferredSize(new Dimension(0, 4));
        progressBar.setMaximumSize(new Dimension(Integer.MAX_VALUE, 4));

        logArea = new JTextPane();
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        promptLibraryPanel = new PromptLibraryPanel();
        historyPanel = new HistoryPanel();

        advancedToggle = new JToggleButton("Settings \u25B8");
        advancedToggle.setBorderPainted(false);
        advancedToggle.setContentAreaFilled(false);
        advancedToggle.setFocusPainted(false);
        advancedToggle.setFont(advancedToggle.getFont().deriveFont(Font.PLAIN, 12f));
        advancedToggle.setCursor(java.awt.Cursor.getPredefinedCursor(java.awt.Cursor.HAND_CURSOR));

        advancedPanel = new JPanel();
        advancedPanel.setLayout(new BoxLayout(advancedPanel, BoxLayout.Y_AXIS));
        advancedPanel.setVisible(false);

        /* ---- listeners ---- */
        aspectRatioBox.addActionListener(e -> applyAspectRatio());
        runButton.addActionListener(e -> runInference());
        cancelButton.addActionListener(e -> cancelInference());
        saveButton.addActionListener(e -> saveOutput());
        promptLibraryPanel.setOnApply(this::applyPreset);
        advancedToggle.addActionListener(e -> {
            boolean open = advancedToggle.isSelected();
            advancedToggle.setText(open ? "Settings \u25BE" : "Settings \u25B8");
            advancedPanel.setVisible(open);
            revalidate();
        });

        /* ---- keyboard shortcuts ---- */
        int menuMask = Toolkit.getDefaultToolkit().getMenuShortcutKeyMaskEx();
        getInputMap(WHEN_ANCESTOR_OF_FOCUSED_COMPONENT).put(
                KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, menuMask), "generate");
        getActionMap().put("generate", new javax.swing.AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) {
                if (!running) { runInference(); }
            }
        });
        getInputMap(WHEN_ANCESTOR_OF_FOCUSED_COMPONENT).put(
                KeyStroke.getKeyStroke(KeyEvent.VK_S, menuMask), "save");
        getActionMap().put("save", new javax.swing.AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) { saveOutput(); }
        });
        getInputMap(WHEN_ANCESTOR_OF_FOCUSED_COMPONENT).put(
                KeyStroke.getKeyStroke(KeyEvent.VK_PERIOD, menuMask), "cancel");
        getActionMap().put("cancel", new javax.swing.AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) {
                if (running) { cancelInference(); }
            }
        });

        /* ---- tooltips ---- */
        modelCombo.setToolTipText("Choose an ONNX text-to-image model");
        promptField.setToolTipText("Describe the image you want to create");
        negativePromptField.setToolTipText("Concepts to exclude from the generated image");
        seedSpinner.setToolTipText("Random seed for reproducible results");
        aspectRatioBox.setToolTipText("Preset aspect ratios for output dimensions");
        widthField.setToolTipText("Output width in pixels (multiple of 8)");
        heightField.setToolTipText("Output height in pixels (multiple of 8)");
        stylePresetBox.setToolTipText("Optional style modifier for the prompt");
        stepsSpinner.setToolTipText("Fewer steps = faster but lower quality. 10\u201320 recommended.");
        runButton.setToolTipText("Generate image (\u2318Enter)");
        saveButton.setToolTipText("Save output image (\u2318S)");

        if (models.isEmpty()) {
            statusLabel.setText("No models configured \u2014 open Models \u2192 Model Manager to get started.");
            runButton.setEnabled(false);
        }

        buildLayout();
    }

    /* ================================================================== */
    /*  Layout                                                             */
    /* ================================================================== */

    private void buildLayout() {
        JPanel top = new JPanel();
        top.setLayout(new BoxLayout(top, BoxLayout.Y_AXIS));
        top.setBorder(BorderFactory.createEmptyBorder(20, 24, 12, 24));

        /* Model selector */
        JPanel modelRow = new JPanel(new BorderLayout(10, 0));
        modelRow.add(tinyLabel("Model"), BorderLayout.NORTH);
        modelRow.add(modelCombo, BorderLayout.CENTER);
        cap(modelRow, 52);
        top.add(modelRow);
        top.add(Box.createVerticalStrut(12));

        /* Prompt */
        JPanel promptRow = new JPanel(new BorderLayout(0, 4));
        promptRow.add(tinyLabel("Prompt"), BorderLayout.NORTH);
        JScrollPane promptScroll = new JScrollPane(promptField);
        promptScroll.setBorder(BorderFactory.createEmptyBorder());
        promptRow.add(promptScroll, BorderLayout.CENTER);
        promptRow.setPreferredSize(new Dimension(0, 80));
        promptRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 80));
        top.add(promptRow);
        top.add(Box.createVerticalStrut(8));

        /* Disclosure toggle */
        JPanel disclosureRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        disclosureRow.add(advancedToggle);
        cap(disclosureRow, 28);
        top.add(disclosureRow);

        /* Advanced options panel (hidden by default) */
        buildAdvancedPanel();
        top.add(advancedPanel);
        top.add(Box.createVerticalStrut(12));

        /* Actions */
        JPanel actionRow = new JPanel(new BorderLayout(8, 0));
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        saveButton.setPreferredSize(new Dimension(120, 32));
        cancelButton.setPreferredSize(new Dimension(90, 32));
        runButton.setPreferredSize(new Dimension(120, 32));
        buttons.add(saveButton);
        buttons.add(cancelButton);
        buttons.add(runButton);
        actionRow.add(buttons, BorderLayout.EAST);
        cap(actionRow, 40);
        top.add(actionRow);

        /* Progress bar */
        top.add(Box.createVerticalStrut(4));
        top.add(progressBar);

        /* Bottom: output area with tabs */
        JPanel bottom = new JPanel(new BorderLayout());
        bottom.setBorder(BorderFactory.createEmptyBorder(0, 24, 4, 24));

        javax.swing.JTabbedPane tabs = new javax.swing.JTabbedPane();
        JPanel previewCard = new JPanel(new BorderLayout(8, 8));
        previewCard.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        previewCard.add(new JScrollPane(outputPreview), BorderLayout.CENTER);
        tabs.addTab("Output Preview", previewCard);
        tabs.addTab("Generation Log", new JScrollPane(logArea));
        tabs.addTab("Prompt Library", promptLibraryPanel);
        tabs.addTab("History", historyPanel);
        bottom.add(tabs, BorderLayout.CENTER);

        add(top, BorderLayout.NORTH);
        add(bottom, BorderLayout.CENTER);
        add(statusLabel, BorderLayout.SOUTH);
    }

    private void buildAdvancedPanel() {
        advancedPanel.setBorder(BorderFactory.createEmptyBorder(6, 0, 0, 0));

        JPanel negRow = new JPanel(new BorderLayout(0, 4));
        negRow.add(tinyLabel("Negative Prompt"), BorderLayout.NORTH);
        negRow.add(negativePromptField, BorderLayout.CENTER);
        cap(negRow, 44);
        advancedPanel.add(negRow);
        advancedPanel.add(Box.createVerticalStrut(8));

        JPanel grid = new JPanel(new GridLayout(1, 6, 10, 0));
        grid.add(labeled("Steps", stepsSpinner));
        grid.add(labeled("Seed", seedSpinner));
        grid.add(labeled("Aspect", aspectRatioBox));
        grid.add(labeled("Width", widthField));
        grid.add(labeled("Height", heightField));
        grid.add(labeled("Style", stylePresetBox));
        cap(grid, 50);
        advancedPanel.add(grid);
    }

    /* ================================================================== */
    /*  Inference                                                          */
    /* ================================================================== */

    private void runInference() {
        ModelDescriptor selectedModel = (ModelDescriptor) modelCombo.getSelectedItem();
        if (selectedModel == null) {
            if (openModelManager != null) {
                statusLabel.setText("No models available. Opening Model Manager\u2026");
                openModelManager.run();
            } else {
                statusLabel.setText("No models downloaded. Open Model Manager (\u2318M) to download one.");
            }
            return;
        }
        String prompt = promptField.getText().trim();
        if (prompt.isEmpty()) {
            statusLabel.setText("Enter a prompt.");
            return;
        }

        setRunning(true, "Preparing\u2026");
        logArea.setText("");
        lastArtifactPath = "";
        saveButton.setEnabled(false);
        outputPreview.setIcon(null);
        outputPreview.setText("");
        cancellationFlag = new AtomicBoolean(false);

        CompletableFuture<?> downloadFuture;
        if (!modelDownloader.canDownload(selectedModel)) {
            Path localPath = Path.of(System.getProperty("user.home"),
                    ".jforge-models", selectedModel.relativePath());
            if (!Files.exists(localPath)) {
                setRunning(false, "Model not found locally.");
                appendLog("Model requires manual import. Open Models \u2192 Model Manager.");
                return;
            }
            downloadFuture = CompletableFuture.completedFuture(null);
        } else {
            downloadFuture = modelDownloader.downloadIfMissing(selectedModel,
                    progress -> SwingUtilities.invokeLater(() -> {
                        setRunning(true, "Downloading model\u2026");
                        if (progress.isStatusMessage()) {
                            appendLog(progress.statusMessage() + "\n");
                        } else {
                            appendLog("Download: " + progress.percent() + "% ("
                                    + formatBytes(progress.bytesRead()) + " / "
                                    + formatBytes(progress.totalBytes()) + ")\n");
                        }
                    }));
        }

        downloadFuture.thenCompose(ignored -> {
            SwingUtilities.invokeLater(() -> setRunning(true, "Generating image\u2026"));
            return inferenceService.run(new InferenceRequest(
                    selectedModel,
                    PromptEnhancer.enhanceOriginal(prompt),
                    PromptEnhancer.enhanceNegative(negativePromptField.getText().trim()),
                    1.0,
                    ((Number) seedSpinner.getValue()).longValue(),
                    ((Number) stepsSpinner.getValue()).intValue(),
                    parseInt(widthField.getText(), 512),
                    parseInt(heightField.getText(), 512),
                    String.valueOf(stylePresetBox.getSelectedItem()),
                    false,
                    "",
                    gpuSupplier.getAsBoolean(),
                    msg -> SwingUtilities.invokeLater(() -> {
                        setRunning(true, msg);
                        appendLog(msg);
                    }),
                    cancellationFlag
            ));
        }).whenComplete((result, error) -> SwingUtilities.invokeLater(() -> {
            if (error != null) {
                setRunning(false, "Failed");
                appendLog("Error: " + error.getMessage());
                return;
            }
            renderResult(result);
            addHistoryEntry(
                    ((ModelDescriptor) modelCombo.getSelectedItem()).displayName(),
                    prompt, result);
            setRunning(false, result.success() ? "Done" : "Failed");
        }));
    }

    private void setRunning(boolean busy, String message) {
        running = busy;
        runButton.setEnabled(!busy);
        cancelButton.setEnabled(busy);
        runButton.setText(busy ? message : "Generate");
        statusLabel.setText(message);
        progressBar.setVisible(busy);
        if (busy) {
            Matcher m = STEP_PATTERN.matcher(message);
            if (m.find()) {
                int current = Integer.parseInt(m.group(1));
                int total = Integer.parseInt(m.group(2));
                progressBar.setIndeterminate(false);
                progressBar.setMaximum(total);
                progressBar.setValue(current);
            }
        } else {
            progressBar.setIndeterminate(true);
        }
    }

    private void cancelInference() {
        if (cancellationFlag != null) {
            cancellationFlag.set(true);
        }
        cancelButton.setEnabled(false);
        statusLabel.setText("Cancelling\u2026");
    }

    private void renderResult(InferenceResult result) {
        if (result.success()) {
            appendLog(result.details() + "\n" + result.output());
            if (result.artifactPath() != null && !result.artifactPath().isBlank()) {
                lastArtifactPath = result.artifactPath();
                saveButton.setEnabled(true);
                renderPreview(lastArtifactPath);
            }
        } else {
            appendLog(result.details());
        }
    }

    private void renderPreview(String path) {
        try {
            Image image = ImageIO.read(Path.of(path).toFile());
            if (image == null) { return; }
            int maxW = Math.max(256, getWidth() - 80);
            Image scaled = scaleToFit(image, maxW, 480);
            outputPreview.setIcon(new ImageIcon(scaled));
            outputPreview.setText("");
        } catch (Exception ex) {
            appendLog("Preview error: " + ex.getMessage());
        }
    }

    /* ================================================================== */
    /*  Save output                                                        */
    /* ================================================================== */

    private void saveOutput() {
        if (lastArtifactPath == null || lastArtifactPath.isBlank()) { return; }
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Save Generated Image");
        chooser.setFileFilter(new FileNameExtensionFilter("PNG Image", "png"));
        chooser.setSelectedFile(new File(Path.of(lastArtifactPath).getFileName().toString()));
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

    /* ================================================================== */
    /*  Presets & history                                                   */
    /* ================================================================== */

    private void applyPreset(PromptPreset preset) {
        promptField.setText(preset.prompt());
        negativePromptField.setText(preset.negativePrompt());
        stylePresetBox.setSelectedItem(preset.style());
        statusLabel.setText("Preset loaded: " + preset.name());
    }

    public void saveCurrentPreset() {
        String name = "Preset " + (System.currentTimeMillis() % 10000);
        PromptPreset preset = new PromptPreset(
                name, promptField.getText().trim(),
                negativePromptField.getText().trim(),
                "custom", String.valueOf(stylePresetBox.getSelectedItem()));
        promptLibraryPanel.addPreset(preset);
        statusLabel.setText("Preset saved: " + name);
    }

    private void addHistoryEntry(String modelName, String prompt, InferenceResult result) {
        String size = widthField.getText().trim() + "\u00D7" + heightField.getText().trim();
        long seed = ((Number) seedSpinner.getValue()).longValue();
        HistoryEntry entry = new HistoryEntry(
                LocalDateTime.now(), modelName, prompt,
                negativePromptField.getText().trim(),
                seed, 1, size, String.valueOf(stylePresetBox.getSelectedItem()),
                result.success() ? "OK" : "FAIL", result.artifactPath());
        historyPanel.addEntry(entry);
        writeLog(entry, result);
    }

    private void writeLog(HistoryEntry entry, InferenceResult result) {
        try {
            Path logDir = Path.of(System.getProperty("user.home"),
                    ".jforge-models", "outputs", "logs");
            Files.createDirectories(logDir);
            Path logFile = logDir.resolve("jforge.log");
            String line = entry.timestamp() + " | " + entry.model() + " | "
                    + entry.prompt() + " | " + entry.status() + " | "
                    + result.details().replace("\n", " ") + "\n";
            Files.writeString(logFile, line, StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (Exception ignored) { }
    }

    public void openLogsFolder() {
        try {
            Path logDir = Path.of(System.getProperty("user.home"),
                    ".jforge-models", "outputs", "logs");
            Files.createDirectories(logDir);
            Desktop.getDesktop().open(logDir.toFile());
        } catch (Exception ex) {
            statusLabel.setText("Unable to open logs: " + ex.getMessage());
        }
    }

    /* ================================================================== */
    /*  GPU (injected from MainFrame menu)                                 */
    /* ================================================================== */

    public void setGpuSupplier(BooleanSupplier supplier) {
        this.gpuSupplier = supplier;
    }

    public void setModelStorage(ModelStorage storage) {
        this.modelStorage = storage;
    }

    public void setOpenModelManager(Runnable callback) {
        this.openModelManager = callback;
    }

    /* ================================================================== */
    /*  Model updates                                                      */
    /* ================================================================== */

    public void updateModels(List<ModelDescriptor> models) {
        List<ModelDescriptor> available = models;
        if (modelStorage != null) {
            available = models.stream()
                    .filter(m -> modelStorage.isAvailable(m))
                    .collect(java.util.stream.Collectors.toList());
        }
        modelCombo.removeAllItems();
        for (ModelDescriptor m : available) { modelCombo.addItem(m); }
        if (modelCombo.getItemCount() > 0) {
            modelCombo.setSelectedIndex(0);
            if (!running) { runButton.setEnabled(true); }
            statusLabel.setText("Ready");
        } else {
            runButton.setEnabled(false);
            statusLabel.setText("No models downloaded. Open Model Manager to download one.");
        }
    }

    /* ================================================================== */
    /*  Helpers                                                            */
    /* ================================================================== */

    private void appendLog(String text) {
        try {
            var doc = logArea.getStyledDocument();
            doc.insertString(doc.getLength(), text + "\n", null);
        } catch (Exception ignored) { }
    }

    private void applyAspectRatio() {
        switch (String.valueOf(aspectRatioBox.getSelectedItem())) {
            case "1:1" -> { widthField.setText("512"); heightField.setText("512"); }
            case "4:3" -> { widthField.setText("640"); heightField.setText("480"); }
            case "16:9" -> { widthField.setText("768"); heightField.setText("432"); }
            case "3:2" -> { widthField.setText("768"); heightField.setText("512"); }
            default -> { }
        }
    }

    private static JLabel tinyLabel(String text) {
        JLabel lbl = new JLabel(text);
        lbl.setFont(lbl.getFont().deriveFont(Font.PLAIN, 11f));
        lbl.setForeground(new Color(130, 130, 130));
        lbl.setBorder(BorderFactory.createEmptyBorder(0, 2, 2, 0));
        return lbl;
    }

    private static JPanel labeled(String text, java.awt.Component comp) {
        JPanel p = new JPanel(new BorderLayout(0, 4));
        p.add(tinyLabel(text), BorderLayout.NORTH);
        p.add(comp, BorderLayout.CENTER);
        return p;
    }

    private static void cap(JPanel panel, int height) {
        panel.setMaximumSize(new Dimension(Integer.MAX_VALUE, height));
    }

    private static Image scaleToFit(Image img, int maxW, int maxH) {
        int w = img.getWidth(null), h = img.getHeight(null);
        double s = Math.min((double) maxW / w, (double) maxH / h);
        if (s >= 1.0) { return img; }
        return img.getScaledInstance(Math.max(1, (int)(w * s)),
                Math.max(1, (int)(h * s)), Image.SCALE_SMOOTH);
    }

    private static int parseInt(String t, int fb) {
        try { return Integer.parseInt(t.trim()); } catch (Exception e) { return fb; }
    }

    private static String formatBytes(long bytes) {
        if (bytes < 0) { return "?"; }
        if (bytes < 1024) { return bytes + " B"; }
        if (bytes < 1024 * 1024) { return String.format("%.1f KB", bytes / 1024.0); }
        return String.format("%.1f MB", bytes / (1024.0 * 1024));
    }
}
