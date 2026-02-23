package atri.palaash.lumenforge.ui;

import atri.palaash.lumenforge.inference.InferenceRequest;
import atri.palaash.lumenforge.inference.InferenceResult;
import atri.palaash.lumenforge.inference.InferenceService;
import atri.palaash.lumenforge.model.ModelDescriptor;
import atri.palaash.lumenforge.storage.ModelDownloader;

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
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.JTextPane;
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
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.function.BooleanSupplier;

/**
 * Image Upscale panel. Clean two-step UI: pick a model, browse an input image,
 * and run. Before / after preview. Download & processing details go to the Log
 * tab. GPU preference is injected from the menu bar.
 */
public class ImageUpscalePanel extends JPanel {

    private final ModelDownloader modelDownloader;
    private final InferenceService inferenceService;

    /* Controls */
    private final JComboBox<ModelDescriptor> modelCombo;
    private final JLabel inputFileLabel;
    private final JButton browseButton;
    private final JComboBox<String> outputResolutionBox;
    private final JComboBox<String> resizeMethodBox;
    private final JTextField customWidthField;
    private final JTextField customHeightField;
    private final JPanel customDimsPanel;
    private String inputImagePath = "";
    private int inputImgW, inputImgH;

    /* Output */
    private final JLabel beforePreview;
    private final JLabel afterPreview;
    private final JLabel statusLabel;
    private final JButton runButton;
    private final JButton saveButton;

    /* Log */
    private final JTextPane logArea;

    /* State */
    private String lastArtifactPath = "";
    private boolean running;
    private BooleanSupplier gpuSupplier = () -> true;

    public ImageUpscalePanel(List<ModelDescriptor> models,
                             ModelDownloader modelDownloader,
                             InferenceService inferenceService) {
        super(new BorderLayout(0, 0));
        this.modelDownloader = modelDownloader;
        this.inferenceService = inferenceService;

        /* ---- init ---- */
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

        inputFileLabel = new JLabel("No file selected");
        inputFileLabel.setFont(inputFileLabel.getFont().deriveFont(Font.PLAIN, 12f));
        inputFileLabel.setForeground(new Color(130, 130, 130));
        browseButton = new JButton("Choose Image\u2026");

        outputResolutionBox = new JComboBox<>(new String[]{
                "Auto (4\u00d7)", "512\u00d7512", "1024\u00d71024",
                "2048\u00d72048", "4096\u00d74096", "Custom"});
        resizeMethodBox = new JComboBox<>(new String[]{
                "ESRGAN Multi-Pass", "Bicubic", "Bilinear", "Nearest Neighbor", "Lanczos"});
        customWidthField = new JTextField("1024", 5);
        customHeightField = new JTextField("1024", 5);
        customDimsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));
        customDimsPanel.add(customWidthField);
        customDimsPanel.add(new JLabel("\u00d7"));
        customDimsPanel.add(customHeightField);
        customDimsPanel.setVisible(false);
        outputResolutionBox.addActionListener(e -> {
            customDimsPanel.setVisible("Custom".equals(outputResolutionBox.getSelectedItem()));
            revalidate();
        });

        beforePreview = new JLabel("", JLabel.CENTER);
        beforePreview.setVerticalAlignment(JLabel.CENTER);
        afterPreview = new JLabel("", JLabel.CENTER);
        afterPreview.setVerticalAlignment(JLabel.CENTER);

        statusLabel = new JLabel("Ready");
        statusLabel.setFont(statusLabel.getFont().deriveFont(Font.PLAIN, 11f));
        statusLabel.setForeground(new Color(120, 120, 120));
        statusLabel.setBorder(BorderFactory.createEmptyBorder(6, 20, 8, 20));

        runButton = new JButton("Upscale");
        saveButton = new JButton("Save Output");
        saveButton.setEnabled(false);

        logArea = new JTextPane();
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));

        /* ---- listeners ---- */
        browseButton.addActionListener(e -> browseInput());
        runButton.addActionListener(e -> runInference());
        saveButton.addActionListener(e -> saveOutput());

        if (models.isEmpty()) {
            statusLabel.setText("No upscale models \u2014 open Models \u2192 Model Manager to get started.");
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

        /* Model */
        JPanel modelRow = new JPanel(new BorderLayout(10, 0));
        modelRow.add(tinyLabel("Model"), BorderLayout.NORTH);
        modelRow.add(modelCombo, BorderLayout.CENTER);
        cap(modelRow, 52);
        top.add(modelRow);
        top.add(Box.createVerticalStrut(12));

        /* Input image */
        JPanel inputRow = new JPanel(new BorderLayout(10, 0));
        inputRow.add(tinyLabel("Input Image"), BorderLayout.NORTH);
        JPanel browseRow = new JPanel(new BorderLayout(8, 0));
        browseRow.add(inputFileLabel, BorderLayout.CENTER);
        browseRow.add(browseButton, BorderLayout.EAST);
        inputRow.add(browseRow, BorderLayout.CENTER);
        cap(inputRow, 52);
        top.add(inputRow);
        top.add(Box.createVerticalStrut(12));

        /* Output resolution */
        JPanel resRow = new JPanel(new BorderLayout(10, 0));
        resRow.add(tinyLabel("Output Resolution"), BorderLayout.NORTH);
        JPanel resControls = new JPanel(new BorderLayout(8, 0));
        resControls.add(outputResolutionBox, BorderLayout.CENTER);
        resControls.add(customDimsPanel, BorderLayout.EAST);
        resRow.add(resControls, BorderLayout.CENTER);
        cap(resRow, 52);
        top.add(resRow);
        top.add(Box.createVerticalStrut(12));

        /* Resize method */
        JPanel methodRow = new JPanel(new BorderLayout(10, 0));
        methodRow.add(tinyLabel("Resize Method"), BorderLayout.NORTH);
        methodRow.add(resizeMethodBox, BorderLayout.CENTER);
        cap(methodRow, 52);
        top.add(methodRow);
        top.add(Box.createVerticalStrut(16));

        /* Buttons */
        JPanel actionRow = new JPanel(new BorderLayout(8, 0));
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        saveButton.setPreferredSize(new Dimension(120, 32));
        runButton.setPreferredSize(new Dimension(120, 32));
        buttons.add(saveButton);
        buttons.add(runButton);
        actionRow.add(buttons, BorderLayout.EAST);
        cap(actionRow, 40);
        top.add(actionRow);

        /* Bottom: preview + log */
        JPanel bottom = new JPanel(new BorderLayout());
        bottom.setBorder(BorderFactory.createEmptyBorder(0, 24, 4, 24));

        javax.swing.JTabbedPane tabs = new javax.swing.JTabbedPane();

        /* Before / After */
        JPanel previewCard = new JPanel(new GridLayout(1, 2, 12, 0));
        previewCard.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        JPanel beforePanel = borderedPreview("Before", beforePreview);
        JPanel afterPanel  = borderedPreview("After", afterPreview);
        previewCard.add(beforePanel);
        previewCard.add(afterPanel);
        tabs.addTab("Preview", new JScrollPane(previewCard));
        tabs.addTab("Log", new JScrollPane(logArea));
        bottom.add(tabs, BorderLayout.CENTER);

        add(top, BorderLayout.NORTH);
        add(bottom, BorderLayout.CENTER);
        add(statusLabel, BorderLayout.SOUTH);
    }

    private static JPanel borderedPreview(String title, JLabel label) {
        JPanel p = new JPanel(new BorderLayout(0, 4));
        JLabel t = new JLabel(title, JLabel.CENTER);
        t.setFont(t.getFont().deriveFont(Font.PLAIN, 11f));
        t.setForeground(new Color(130, 130, 130));
        p.add(t, BorderLayout.NORTH);
        p.add(new JScrollPane(label), BorderLayout.CENTER);
        return p;
    }

    /* ================================================================== */
    /*  Browse input                                                       */
    /* ================================================================== */

    private void browseInput() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select Image to Upscale");
        chooser.setFileFilter(new FileNameExtensionFilter("Images", "png", "jpg", "jpeg", "bmp"));
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File f = chooser.getSelectedFile();
            inputImagePath = f.getAbsolutePath();
            showBeforePreview(f);
        }
    }

    private void showBeforePreview(File file) {
        try {
            BufferedImage img = ImageIO.read(file);
            if (img != null) {
                inputImgW = img.getWidth();
                inputImgH = img.getHeight();
                inputFileLabel.setText(file.getName() + "  (" + inputImgW + "\u00d7" + inputImgH + ")");
                Image scaled = scaleToFit(img, 360, 360);
                beforePreview.setIcon(new ImageIcon(scaled));
                beforePreview.setText("");
            }
        } catch (Exception ex) {
            appendLog("Cannot preview: " + ex.getMessage());
        }
    }

    private int[] resolveOutputDimensions() {
        String selected = String.valueOf(outputResolutionBox.getSelectedItem());
        return switch (selected) {
            case "512\u00d7512"   -> new int[]{512, 512};
            case "1024\u00d71024" -> new int[]{1024, 1024};
            case "2048\u00d72048" -> new int[]{2048, 2048};
            case "4096\u00d74096" -> new int[]{4096, 4096};
            case "Custom"        -> new int[]{
                    parseInt(customWidthField.getText(), 0),
                    parseInt(customHeightField.getText(), 0)};
            default              -> new int[]{0, 0}; // Auto
        };
    }

    private static int parseInt(String t, int fb) {
        try { return Integer.parseInt(t.trim()); } catch (Exception e) { return fb; }
    }

    /* ================================================================== */
    /*  Inference                                                          */
    /* ================================================================== */

    private void runInference() {
        ModelDescriptor model = (ModelDescriptor) modelCombo.getSelectedItem();
        if (model == null) { statusLabel.setText("Select a model first."); return; }
        if (inputImagePath.isBlank()) { statusLabel.setText("Choose an input image first."); return; }

        setRunning(true, "Preparing\u2026");
        logArea.setText("");
        lastArtifactPath = "";
        saveButton.setEnabled(false);
        afterPreview.setIcon(null);
        afterPreview.setText("");

        CompletableFuture<?> downloadFuture;
        if (!modelDownloader.canDownload(model)) {
            Path localPath = Path.of(System.getProperty("user.home"),
                    ".lumenforge-models", model.relativePath());
            if (!Files.exists(localPath)) {
                setRunning(false, "Model not found locally.");
                appendLog("Model requires manual import. Open Models \u2192 Model Manager.");
                return;
            }
            downloadFuture = CompletableFuture.completedFuture(null);
        } else {
            downloadFuture = modelDownloader.downloadIfMissing(model,
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

        int[] outDims = resolveOutputDimensions();
        String resizeMethod = String.valueOf(resizeMethodBox.getSelectedItem());
        downloadFuture.thenCompose(ignored -> {
            SwingUtilities.invokeLater(() -> setRunning(true, "Upscaling\u2026"));
            return inferenceService.run(new InferenceRequest(
                    model, "", "", 1.0, 42, 1,
                    outDims[0], outDims[1],
                    resizeMethod, true,
                    inputImagePath, gpuSupplier.getAsBoolean(),
                    msg -> SwingUtilities.invokeLater(() -> {
                        setRunning(true, msg);
                        appendLog(msg);
                    })
            ));
        }).whenComplete((result, error) -> SwingUtilities.invokeLater(() -> {
            if (error != null) {
                setRunning(false, "Failed");
                appendLog("Error: " + error.getMessage());
                return;
            }
            renderResult(result);
            setRunning(false, result.success() ? "Done" : "Failed");
        }));
    }

    private void setRunning(boolean busy, String message) {
        running = busy;
        runButton.setEnabled(!busy);
        runButton.setText(busy ? message : "Upscale");
        statusLabel.setText(message);
    }

    private void renderResult(InferenceResult result) {
        if (result.success()) {
            appendLog(result.details() + "\n" + result.output());
            if (result.artifactPath() != null && !result.artifactPath().isBlank()) {
                lastArtifactPath = result.artifactPath();
                saveButton.setEnabled(true);
                showAfterPreview(lastArtifactPath);
            }
        } else {
            appendLog(result.details());
        }
    }

    private void showAfterPreview(String path) {
        try {
            Image img = ImageIO.read(Path.of(path).toFile());
            if (img != null) {
                Image scaled = scaleToFit(img, 360, 360);
                afterPreview.setIcon(new ImageIcon(scaled));
                afterPreview.setText("");
            }
        } catch (Exception ex) {
            appendLog("Preview error: " + ex.getMessage());
        }
    }

    /* ================================================================== */
    /*  Save                                                               */
    /* ================================================================== */

    private void saveOutput() {
        if (lastArtifactPath == null || lastArtifactPath.isBlank()) { return; }
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Save Upscaled Image");
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
    /*  GPU (injected from MainFrame menu)                                 */
    /* ================================================================== */

    public void setGpuSupplier(BooleanSupplier supplier) {
        this.gpuSupplier = supplier;
    }

    /* ================================================================== */
    /*  Model updates                                                      */
    /* ================================================================== */

    public void updateModels(List<ModelDescriptor> models) {
        modelCombo.removeAllItems();
        for (ModelDescriptor m : models) { modelCombo.addItem(m); }
        if (modelCombo.getItemCount() > 0) {
            modelCombo.setSelectedIndex(0);
            if (!running) { runButton.setEnabled(true); }
        }
    }

    /* ================================================================== */
    /*  Logs                                                               */
    /* ================================================================== */

    public void openLogsFolder() {
        try {
            Path logDir = Path.of(System.getProperty("user.home"),
                    ".lumenforge-models", "outputs", "logs");
            Files.createDirectories(logDir);
            Desktop.getDesktop().open(logDir.toFile());
        } catch (Exception ex) {
            statusLabel.setText("Unable to open logs: " + ex.getMessage());
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

    private static JLabel tinyLabel(String text) {
        JLabel lbl = new JLabel(text);
        lbl.setFont(lbl.getFont().deriveFont(Font.PLAIN, 11f));
        lbl.setForeground(new Color(130, 130, 130));
        lbl.setBorder(BorderFactory.createEmptyBorder(0, 2, 2, 0));
        return lbl;
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

    private static String formatBytes(long bytes) {
        if (bytes < 0) { return "?"; }
        if (bytes < 1024) { return bytes + " B"; }
        if (bytes < 1024 * 1024) { return String.format("%.1f KB", bytes / 1024.0); }
        return String.format("%.1f MB", bytes / (1024.0 * 1024));
    }
}
