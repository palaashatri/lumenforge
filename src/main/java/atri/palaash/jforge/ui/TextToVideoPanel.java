package atri.palaash.jforge.ui;

import atri.palaash.jforge.inference.InferenceRequest;
import atri.palaash.jforge.inference.InferenceResult;
import atri.palaash.jforge.inference.InferenceService;
import atri.palaash.jforge.inference.PromptEnhancer;
import atri.palaash.jforge.model.ModelDescriptor;
import atri.palaash.jforge.model.TaskType;
import atri.palaash.jforge.storage.ModelDownloader;
import atri.palaash.jforge.storage.ModelStorage;
import atri.palaash.jforge.storage.PyTorchToOnnxConverter;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TextToVideoPanel extends JPanel {

    private final InferenceService inferenceService;
    private final ModelStorage modelStorage;
    private final ModelDownloader modelDownloader;
    private final JTextArea promptField;
    private final JTextField negativePromptField;
    private final JComboBox<String> resolutionBox;
    private final JSlider durationSlider;
    private final JSlider fpsSlider;
    private final JComboBox<String> motionBox;
    private final JComboBox<String> styleBox;
    private final JCheckBox upscaleCheck;
    private final JCheckBox interpolateCheck;
    
    private final JLabel outputPreview;
    private final JLabel statusLabel;
    private final JButton runButton;
    private final JButton cancelButton;
    private final JButton saveButton;
    private final JProgressBar progressBar;
    private final JTextPane logArea;

    private String lastArtifactPath = "";
    private boolean running;
    private AtomicBoolean cancellationFlag;
    private static final Pattern STEP_PATTERN = Pattern.compile("Denoising:\\s*(\\d+)/(\\d+)");

    public TextToVideoPanel(InferenceService inferenceService, ModelStorage modelStorage, ModelDownloader modelDownloader) {
        super(new BorderLayout());
        this.inferenceService = inferenceService;
        this.modelStorage = modelStorage;
        this.modelDownloader = modelDownloader;

        promptField = new JTextArea(3, 40);
        promptField.setLineWrap(true);
        promptField.setWrapStyleWord(true);
        promptField.putClientProperty("JTextField.placeholderText", "Describe the video you want to create...");

        negativePromptField = new JTextField();
        negativePromptField.putClientProperty("JTextField.placeholderText", "Things to avoid (optional)");

        resolutionBox = new JComboBox<>(new String[]{"256p", "320p", "448p", "512p"});
        
        durationSlider = new JSlider(JSlider.HORIZONTAL, 1, 6, 3);
        durationSlider.setMajorTickSpacing(1);
        durationSlider.setPaintTicks(true);
        durationSlider.setPaintLabels(true);

        fpsSlider = new JSlider(JSlider.HORIZONTAL, 8, 24, 8);
        fpsSlider.setMajorTickSpacing(8);
        fpsSlider.setMinorTickSpacing(2);
        fpsSlider.setPaintTicks(true);
        fpsSlider.setPaintLabels(true);

        motionBox = new JComboBox<>(new String[]{"None", "pan left", "zoom in", "orbit", "dolly shot"});
        styleBox = new JComboBox<>(new String[]{"None", "cinematic", "anime", "documentary", "Pixar-style"});
        
        upscaleCheck = new JCheckBox("Upscale (Real-ESRGAN)");
        upscaleCheck.setFont(upscaleCheck.getFont().deriveFont(11f));
        interpolateCheck = new JCheckBox("Interpolate FPS (RIFE/FILM)");
        interpolateCheck.setFont(interpolateCheck.getFont().deriveFont(11f));

        outputPreview = new JLabel("Video output will appear here", JLabel.CENTER);
        outputPreview.setVerticalAlignment(JLabel.CENTER);

        statusLabel = new JLabel("Ready");
        statusLabel.setFont(statusLabel.getFont().deriveFont(Font.PLAIN, 11f));
        statusLabel.setForeground(new Color(120, 120, 120));
        statusLabel.setBorder(BorderFactory.createEmptyBorder(6, 20, 8, 20));

        runButton = new JButton("Generate Video");
        cancelButton = new JButton("Cancel");
        cancelButton.setEnabled(false);
        saveButton = new JButton("Save Video");
        saveButton.setEnabled(false);

        progressBar = new JProgressBar();
        progressBar.setIndeterminate(true);
        progressBar.setVisible(false);
        progressBar.setPreferredSize(new Dimension(0, 4));

        logArea = new JTextPane();
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));

        runButton.addActionListener(e -> runInference());
        cancelButton.addActionListener(e -> cancelInference());
        saveButton.addActionListener(e -> saveOutput());

        buildLayout();
        
        int menuMask = Toolkit.getDefaultToolkit().getMenuShortcutKeyMaskEx();
        getInputMap(WHEN_ANCESTOR_OF_FOCUSED_COMPONENT).put(
                KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, menuMask), "generate");
        getActionMap().put("generate", new javax.swing.AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) {
                if (!running) { runInference(); }
            }
        });
    }

    private void buildLayout() {
        JPanel top = new JPanel();
        top.setLayout(new BoxLayout(top, BoxLayout.Y_AXIS));
        top.setBorder(BorderFactory.createEmptyBorder(20, 24, 12, 24));

        JPanel headerRow = new JPanel(new BorderLayout());
        JLabel title = new JLabel("Local Text-to-Video Engine");
        title.setFont(title.getFont().deriveFont(Font.BOLD, 14f));
        headerRow.add(title, BorderLayout.WEST);
        cap(headerRow, 30);
        top.add(headerRow);
        top.add(Box.createVerticalStrut(12));

        JPanel promptRow = new JPanel(new BorderLayout(0, 4));
        promptRow.add(tinyLabel("Prompt"), BorderLayout.NORTH);
        JScrollPane promptScroll = new JScrollPane(promptField);
        promptScroll.setBorder(BorderFactory.createEmptyBorder());
        promptRow.add(promptScroll, BorderLayout.CENTER);
        promptRow.setPreferredSize(new Dimension(0, 80));
        promptRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 80));
        top.add(promptRow);
        top.add(Box.createVerticalStrut(8));

        JPanel negRow = new JPanel(new BorderLayout(0, 4));
        negRow.add(tinyLabel("Negative Prompt"), BorderLayout.NORTH);
        negRow.add(negativePromptField, BorderLayout.CENTER);
        cap(negRow, 44);
        top.add(negRow);
        top.add(Box.createVerticalStrut(8));

        JPanel topGrid = new JPanel(new GridLayout(1, 3, 10, 0));
        topGrid.add(labeled("Resolution", resolutionBox));
        topGrid.add(labeled("Duration (s)", durationSlider));
        topGrid.add(labeled("FPS", fpsSlider));
        cap(topGrid, 50);
        top.add(topGrid);
        top.add(Box.createVerticalStrut(10));

        JPanel midGrid = new JPanel(new GridLayout(1, 2, 10, 0));
        midGrid.add(labeled("Camera Motion", motionBox));
        midGrid.add(labeled("Style Preset", styleBox));
        cap(midGrid, 40);
        top.add(midGrid);
        top.add(Box.createVerticalStrut(8));
        
        JPanel bottomGrid = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 0));
        bottomGrid.add(upscaleCheck);
        bottomGrid.add(interpolateCheck);
        cap(bottomGrid, 30);
        top.add(bottomGrid);
        top.add(Box.createVerticalStrut(8));

        JPanel actionRow = new JPanel(new BorderLayout(8, 0));
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        buttons.add(saveButton);
        buttons.add(cancelButton);
        buttons.add(runButton);
        actionRow.add(buttons, BorderLayout.EAST);
        cap(actionRow, 40);
        top.add(actionRow);

        top.add(Box.createVerticalStrut(4));
        top.add(progressBar);

        JPanel bottom = new JPanel(new BorderLayout());
        bottom.setBorder(BorderFactory.createEmptyBorder(0, 24, 4, 24));

        JTabbedPane tabs = new JTabbedPane();
        JPanel previewCard = new JPanel(new BorderLayout(8, 8));
        previewCard.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        previewCard.add(new JScrollPane(outputPreview), BorderLayout.CENTER);
        tabs.addTab("Output Preview", previewCard);
        tabs.addTab("Generation Log", new JScrollPane(logArea));
        bottom.add(tabs, BorderLayout.CENTER);

        add(top, BorderLayout.NORTH);
        add(bottom, BorderLayout.CENTER);
        add(statusLabel, BorderLayout.SOUTH);
    }

    private void runInference() {
        String prompt = promptField.getText().trim();
        if (prompt.isEmpty()) {
            statusLabel.setText("Enter a prompt.");
            return;
        }

        setRunning(true, "Preparing...");
        logArea.setText("");
        lastArtifactPath = "";
        saveButton.setEnabled(false);
        outputPreview.setText("");
        cancellationFlag = new AtomicBoolean(false);

        String rawId = "damo-vilab/text-to-video-ms-1.7b";
        ModelDescriptor videoModel = new ModelDescriptor(
                "hf_pt_damo_text_to_video", 
                "ModelScope T2V", 
                TaskType.TEXT_TO_VIDEO, 
                "video/converted-damo-vilab-text-to-video/unet/model.onnx", 
                "hf-pytorch://" + rawId, 
                "Offline text-to-video generator."
        );
        
        String motion = (String) motionBox.getSelectedItem();
        String style = (String) styleBox.getSelectedItem();

        String enhancedPrompt = PromptEnhancer.enhanceOriginal(prompt, motion, style);
        String enhancedNegative = PromptEnhancer.enhanceNegative(negativePromptField.getText().trim());

        if (!modelStorage.isAvailable(videoModel)) {
            setRunning(true, "Downloading & Converting Model...");
            appendLog("Model missing. Starting automatic PyTorch to ONNX conversion for " + rawId + "...");
            
            String sanitized = rawId.replaceAll("[^a-zA-Z0-9._-]", "-").toLowerCase();
            Path outputDir = modelStorage.root().resolve("video").resolve("converted-" + sanitized);
            
            CompletableFuture.supplyAsync(() -> {
                PyTorchToOnnxConverter converter = new PyTorchToOnnxConverter(
                        msg -> SwingUtilities.invokeLater(() -> appendLog("Converter: " + msg))
                );
                return converter.convert(rawId, outputDir, "generic"); 
            }).whenComplete((path, error) -> SwingUtilities.invokeLater(() -> {
                if (error != null) {
                    setRunning(false, "Conversion Failed");
                    appendLog("Error: " + error.getMessage());
                } else {
                    appendLog("Successfully converted " + rawId + " to ONNX.");
                    executeInference(videoModel, enhancedPrompt, enhancedNegative);
                }
            }));
            return;
        }

        executeInference(videoModel, enhancedPrompt, enhancedNegative);
    }
    
    private void executeInference(ModelDescriptor videoModel, String enhancedPrompt, String enhancedNegative) {
        setRunning(true, "Preparing...");
        inferenceService.run(new InferenceRequest(
                videoModel,
                enhancedPrompt,
                enhancedNegative,
                1.0, 42, 1, 1024, 576, "", false, "", false,
                msg -> SwingUtilities.invokeLater(() -> {
                    setRunning(true, msg);
                    appendLog(msg);
                }),
                cancellationFlag
        )).whenComplete((result, error) -> SwingUtilities.invokeLater(() -> {
            if (error != null) {
                setRunning(false, "Failed");
                appendLog("Error: " + error.getMessage());
                return;
            }
            if (result.success()) {
                appendLog(result.details() + "\n" + result.output());
                lastArtifactPath = result.artifactPath();
                saveButton.setEnabled(true);
                outputPreview.setText("<html><center>Video Generated Successfully!<br/>Saved to: " + lastArtifactPath + "</center></html>");
            } else {
                appendLog(result.details());
            }
            setRunning(false, result.success() ? "Done" : "Failed");
        }));
    }

    private void setRunning(boolean busy, String message) {
        running = busy;
        runButton.setEnabled(!busy);
        cancelButton.setEnabled(busy);
        runButton.setText(busy ? message : "Generate Video");
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
    }

    private void saveOutput() {
        if (lastArtifactPath == null || lastArtifactPath.isBlank()) return;
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Save Generated Video");
        chooser.setSelectedFile(new File("sora_output.mp4"));
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                Path target = chooser.getSelectedFile().toPath();
                if (!target.toString().toLowerCase().endsWith(".mp4")) {
                    target = target.resolveSibling(target.getFileName() + ".mp4");
                }
                Files.copy(Path.of(lastArtifactPath), target, StandardCopyOption.REPLACE_EXISTING);
                statusLabel.setText("Saved to " + target.getFileName());
            } catch (Exception ex) {
                statusLabel.setText("Save failed: " + ex.getMessage());
            }
        }
    }

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

    private static JPanel labeled(String text, JComponent comp) {
        JPanel p = new JPanel(new BorderLayout(0, 4));
        p.add(tinyLabel(text), BorderLayout.NORTH);
        p.add(comp, BorderLayout.CENTER);
        return p;
    }

    private static void cap(JPanel panel, int height) {
        panel.setMaximumSize(new Dimension(Integer.MAX_VALUE, height));
    }
}
