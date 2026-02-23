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
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JTextPane;
import javax.swing.JToggleButton;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.datatransfer.DataFlavor;
import java.awt.dnd.DnDConstants;
import java.awt.dnd.DropTarget;
import java.awt.dnd.DropTargetAdapter;
import java.awt.dnd.DropTargetDropEvent;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Image-to-Image / Inpainting panel.
 *
 * <p>Features:
 * <ul>
 *   <li>Load an input image (browse or drag-and-drop)</li>
 *   <li>Optional mask drawing (brush-based painting) for inpainting</li>
 *   <li>Strength slider (0–100%) controls how much of the image to regenerate</li>
 *   <li>Prompt-guided transformation of the input image</li>
 *   <li>Cancel support with progress bar</li>
 * </ul>
 */
public class Img2ImgPanel extends JPanel {

    /* ── UI components ────────────────────────────────────────────── */
    private final JComboBox<ModelDescriptor> modelCombo;
    private final JTextArea promptField;
    private final JSlider strengthSlider;
    private final JLabel strengthLabel;
    private final JTextField widthField;
    private final JTextField heightField;
    private final JTextField seedField;
    private final JTextField stepsField;
    private final JButton browseButton;
    private final JButton generateButton;
    private final JButton cancelButton;
    private final JButton saveButton;
    private final JButton clearMaskButton;
    private final JToggleButton maskToggle;
    private final JLabel previewLabel;
    private final JTextPane statusArea;
    private final JLabel statusLabel;
    private final JProgressBar progressBar;
    private final MaskCanvas maskCanvas;

    /* ── State ────────────────────────────────────────────────────── */
    private final ModelDownloader modelDownloader;
    private final InferenceService inferenceService;
    private File inputImageFile;
    private BufferedImage inputImage;
    private BufferedImage resultImage;
    private File lastSaved;
    private boolean running;
    private BooleanSupplier gpuSupplier = () -> true;
    private AtomicBoolean cancellationFlag;
    private static final Pattern STEP_PATTERN = Pattern.compile("Denoising:\\s*(\\d+)/(\\d+)");

    /* ── Mask drawing state ──────────────────────────────────────── */
    private BufferedImage maskImage; // white = repaint, black = keep
    private int brushSize = 30;
    private boolean maskMode = false;

    public Img2ImgPanel(List<ModelDescriptor> models,
                        ModelDownloader modelDownloader,
                        InferenceService inferenceService) {
        super(new BorderLayout(0, 0));
        this.modelDownloader = modelDownloader;
        this.inferenceService = inferenceService;

        /* ── Init components ─────────────────────────────────────── */
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
        modelCombo.setToolTipText("Select the SD model to use for Img2Img");

        promptField = new JTextArea(3, 30);
        promptField.setLineWrap(true);
        promptField.setWrapStyleWord(true);
        promptField.setToolTipText("Describe the transformation you want to apply to the image");

        strengthSlider = new JSlider(0, 100, 75);
        strengthSlider.setToolTipText("Denoise strength: higher = more transformation (0% = no change, 100% = full generation)");
        strengthLabel = new JLabel("75%");
        strengthSlider.addChangeListener(e -> strengthLabel.setText(strengthSlider.getValue() + "%"));

        widthField  = new JTextField("512", 5);
        heightField = new JTextField("512", 5);
        seedField   = new JTextField("-1", 8);
        stepsField  = new JTextField("20", 4);
        widthField.setToolTipText("Output width (must be divisible by 8)");
        heightField.setToolTipText("Output height (must be divisible by 8)");
        seedField.setToolTipText("Random seed (-1 = random)");
        stepsField.setToolTipText("Number of denoising steps");

        browseButton = new JButton("Browse\u2026");
        browseButton.setToolTipText("Choose an input image");

        generateButton = new JButton("Generate");
        generateButton.setToolTipText("Run Img2Img transformation (\u2318Enter)");

        cancelButton = new JButton("Cancel");
        cancelButton.setEnabled(false);
        cancelButton.setToolTipText("Cancel the current generation (\u2318.)");

        saveButton = new JButton("Save");
        saveButton.setEnabled(false);
        saveButton.setToolTipText("Save the result image (\u2318S)");

        previewLabel = new JLabel("", JLabel.CENTER);
        previewLabel.setPreferredSize(new Dimension(400, 400));
        previewLabel.setText("Drop an image here or click Browse");
        previewLabel.setFont(previewLabel.getFont().deriveFont(Font.ITALIC, 12f));

        maskCanvas = new MaskCanvas();

        maskToggle = new JToggleButton("Paint Mask");
        maskToggle.setToolTipText("Toggle mask drawing mode for inpainting (white areas = regenerate)");
        maskToggle.addActionListener(e -> {
            maskMode = maskToggle.isSelected();
            maskCanvas.setMaskMode(maskMode);
            if (maskMode && maskImage == null && inputImage != null) {
                maskImage = new BufferedImage(inputImage.getWidth(), inputImage.getHeight(),
                        BufferedImage.TYPE_INT_RGB);
                Graphics2D g = maskImage.createGraphics();
                g.setColor(Color.BLACK);
                g.fillRect(0, 0, maskImage.getWidth(), maskImage.getHeight());
                g.dispose();
            }
            maskCanvas.repaint();
        });

        clearMaskButton = new JButton("Clear Mask");
        clearMaskButton.setToolTipText("Erase the inpainting mask");
        clearMaskButton.addActionListener(e -> {
            if (maskImage != null) {
                Graphics2D g = maskImage.createGraphics();
                g.setColor(Color.BLACK);
                g.fillRect(0, 0, maskImage.getWidth(), maskImage.getHeight());
                g.dispose();
                maskCanvas.repaint();
            }
        });

        statusLabel = new JLabel("Ready");
        statusLabel.setFont(statusLabel.getFont().deriveFont(Font.PLAIN, 12f));

        statusArea = new JTextPane();
        statusArea.setEditable(false);
        statusArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));

        progressBar = new JProgressBar();
        progressBar.setPreferredSize(new Dimension(0, 4));
        progressBar.setMaximumSize(new Dimension(Integer.MAX_VALUE, 4));
        progressBar.setVisible(false);

        /* ── Layout ──────────────────────────────────────────────── */
        buildLayout();
        wireActions();
        wireKeyboardShortcuts();
        wireDragAndDrop();
    }

    /* ---- Accessors ---- */
    public void setGpuSupplier(BooleanSupplier supplier) { this.gpuSupplier = supplier; }

    public void updateModels(List<ModelDescriptor> models) {
        ModelDescriptor selected = (ModelDescriptor) modelCombo.getSelectedItem();
        modelCombo.removeAllItems();
        models.forEach(modelCombo::addItem);
        if (selected != null) modelCombo.setSelectedItem(selected);
    }

    /* ================================================================== */
    /*  Layout                                                             */
    /* ================================================================== */

    private void buildLayout() {
        JPanel top = new JPanel();
        top.setLayout(new BoxLayout(top, BoxLayout.Y_AXIS));
        top.setBorder(BorderFactory.createEmptyBorder(20, 24, 12, 24));

        /* Model row */
        JPanel modelRow = new JPanel(new BorderLayout(8, 0));
        modelRow.add(new JLabel("Model:"), BorderLayout.WEST);
        modelRow.add(modelCombo, BorderLayout.CENTER);
        modelRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 36));
        top.add(modelRow);
        top.add(Box.createVerticalStrut(8));

        /* Input image row */
        JPanel imageRow = new JPanel(new BorderLayout(8, 0));
        imageRow.add(new JLabel("Input:"), BorderLayout.WEST);
        JPanel browsePanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));
        browsePanel.add(browseButton);
        browsePanel.add(maskToggle);
        browsePanel.add(clearMaskButton);
        imageRow.add(browsePanel, BorderLayout.CENTER);
        imageRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 36));
        top.add(imageRow);
        top.add(Box.createVerticalStrut(8));

        /* Prompt */
        JPanel promptRow = new JPanel(new BorderLayout(8, 0));
        promptRow.add(new JLabel("Prompt:"), BorderLayout.WEST);
        JScrollPane promptScroll = new JScrollPane(promptField);
        promptScroll.setBorder(BorderFactory.createEmptyBorder());
        promptRow.add(promptScroll, BorderLayout.CENTER);
        promptRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 80));
        top.add(promptRow);
        top.add(Box.createVerticalStrut(8));

        /* Strength slider */
        JPanel strengthRow = new JPanel(new BorderLayout(8, 0));
        strengthRow.add(new JLabel("Strength:"), BorderLayout.WEST);
        JPanel sliderPanel = new JPanel(new BorderLayout(4, 0));
        sliderPanel.add(strengthSlider, BorderLayout.CENTER);
        sliderPanel.add(strengthLabel, BorderLayout.EAST);
        strengthRow.add(sliderPanel, BorderLayout.CENTER);
        strengthRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 36));
        top.add(strengthRow);
        top.add(Box.createVerticalStrut(8));

        /* Settings row */
        JPanel settingsRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        settingsRow.add(new JLabel("W:"));  settingsRow.add(widthField);
        settingsRow.add(new JLabel("H:"));  settingsRow.add(heightField);
        settingsRow.add(new JLabel("Seed:")); settingsRow.add(seedField);
        settingsRow.add(new JLabel("Steps:")); settingsRow.add(stepsField);
        settingsRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 36));
        top.add(settingsRow);
        top.add(Box.createVerticalStrut(12));

        /* Buttons row */
        JPanel btnRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        btnRow.add(generateButton);
        btnRow.add(cancelButton);
        btnRow.add(saveButton);
        btnRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 36));
        top.add(btnRow);

        /* Status */
        statusLabel.setBorder(BorderFactory.createEmptyBorder(6, 20, 8, 20));
        top.add(Box.createVerticalStrut(4));
        top.add(progressBar);
        top.add(statusLabel);

        /* Bottom — preview / mask canvas + status/log */
        JPanel bottom = new JPanel(new GridLayout(1, 2, 8, 0));
        bottom.setBorder(BorderFactory.createEmptyBorder(0, 24, 4, 24));

        JPanel previewCard = new JPanel(new BorderLayout());
        previewCard.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        previewCard.add(maskCanvas, BorderLayout.CENTER);
        bottom.add(previewCard);

        JPanel resultCard = new JPanel(new BorderLayout());
        resultCard.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        resultCard.add(previewLabel, BorderLayout.CENTER);
        bottom.add(resultCard);

        JPanel logPanel = new JPanel(new BorderLayout());
        JScrollPane logScroll = new JScrollPane(statusArea);
        logScroll.setPreferredSize(new Dimension(0, 80));
        logPanel.add(logScroll, BorderLayout.CENTER);
        logPanel.setBorder(BorderFactory.createEmptyBorder(4, 24, 12, 24));

        add(top, BorderLayout.NORTH);
        add(bottom, BorderLayout.CENTER);
        add(logPanel, BorderLayout.SOUTH);
    }

    /* ================================================================== */
    /*  Actions                                                            */
    /* ================================================================== */

    private void wireActions() {
        browseButton.addActionListener(e -> browseImage());
        generateButton.addActionListener(e -> generate());
        cancelButton.addActionListener(e -> cancelInference());
        saveButton.addActionListener(e -> saveResult());
    }

    private void browseImage() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileFilter(new FileNameExtensionFilter("Images", "png", "jpg", "jpeg", "bmp", "webp"));
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            loadInputImage(chooser.getSelectedFile());
        }
    }

    private void loadInputImage(File file) {
        try {
            inputImage = ImageIO.read(file);
            inputImageFile = file;
            if (inputImage != null) {
                maskImage = new BufferedImage(inputImage.getWidth(), inputImage.getHeight(),
                        BufferedImage.TYPE_INT_RGB);
                Graphics2D g = maskImage.createGraphics();
                g.setColor(Color.BLACK);
                g.fillRect(0, 0, maskImage.getWidth(), maskImage.getHeight());
                g.dispose();
                maskCanvas.repaint();

                // Auto-fill width/height from image
                widthField.setText(String.valueOf((inputImage.getWidth() / 8) * 8));
                heightField.setText(String.valueOf((inputImage.getHeight() / 8) * 8));
                statusLabel.setText("Loaded: " + file.getName() + " (" + inputImage.getWidth()
                        + "\u00d7" + inputImage.getHeight() + ")");
            }
        } catch (Exception ex) {
            statusLabel.setText("Error loading image: " + ex.getMessage());
        }
    }

    private void generate() {
        if (running || inputImage == null) {
            if (inputImage == null) statusLabel.setText("Load an input image first.");
            return;
        }

        ModelDescriptor model = (ModelDescriptor) modelCombo.getSelectedItem();
        if (model == null) return;

        setRunning(true);

        // Save mask to temp file if mask has been drawn
        String maskPath = null;
        if (maskImage != null && hasMaskContent()) {
            try {
                Path tmpMask = Files.createTempFile("lumenforge-mask-", ".png");
                ImageIO.write(maskImage, "PNG", tmpMask.toFile());
                maskPath = tmpMask.toString();
            } catch (Exception ex) {
                statusLabel.setText("Warning: Could not save mask. Proceeding without mask.");
            }
        }

        long seed;
        try { seed = Long.parseLong(seedField.getText().trim()); }
        catch (NumberFormatException e) { seed = -1; }
        if (seed < 0) seed = System.nanoTime();

        int w = parseIntOr(widthField.getText(), 512);
        int h = parseIntOr(heightField.getText(), 512);
        int steps = parseIntOr(stepsField.getText(), 20);
        double strength = strengthSlider.getValue() / 100.0;

        cancellationFlag = new AtomicBoolean(false);

        CompletableFuture<InferenceResult> future = inferenceService.run(new InferenceRequest(
                model, promptField.getText().trim(), "",
                7.5, seed, steps, w, h, "img2img", false,
                inputImageFile.getAbsolutePath(), gpuSupplier.getAsBoolean(),
                msg -> SwingUtilities.invokeLater(() -> {
                    statusLabel.setText(msg);
                    appendLog(msg);
                    Matcher m2 = STEP_PATTERN.matcher(msg);
                    if (m2.find()) {
                        int cur = Integer.parseInt(m2.group(1));
                        int tot = Integer.parseInt(m2.group(2));
                        progressBar.setIndeterminate(false);
                        progressBar.setMaximum(tot);
                        progressBar.setValue(cur);
                    }
                }),
                cancellationFlag,
                strength,
                maskPath
        ));

        future.thenAccept(result -> SwingUtilities.invokeLater(() -> {
            setRunning(false);
            if (result.success()) {
                statusLabel.setText(result.output());
                appendLog(result.details());
                if (result.artifactPath() != null && !result.artifactPath().isEmpty()) {
                    showResult(result.artifactPath());
                }
            } else {
                statusLabel.setText("Error: " + result.details());
                appendLog("ERROR: " + result.details());
            }
        }));
    }

    private boolean hasMaskContent() {
        if (maskImage == null) return false;
        for (int y = 0; y < maskImage.getHeight(); y++)
            for (int x = 0; x < maskImage.getWidth(); x++)
                if ((maskImage.getRGB(x, y) & 0xFFFFFF) != 0) return true;
        return false;
    }

    private void cancelInference() {
        if (cancellationFlag != null) {
            cancellationFlag.set(true);
            statusLabel.setText("Cancelling\u2026");
        }
    }

    private void saveResult() {
        if (resultImage == null) return;
        JFileChooser chooser = new JFileChooser();
        chooser.setSelectedFile(new File("img2img_result.png"));
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File dest = chooser.getSelectedFile();
                ImageIO.write(resultImage, "PNG", dest);
                lastSaved = dest;
                statusLabel.setText("Saved to " + dest.getName());
            } catch (Exception ex) {
                statusLabel.setText("Save failed: " + ex.getMessage());
            }
        }
    }

    private void showResult(String path) {
        try {
            resultImage = ImageIO.read(new File(path));
            if (resultImage != null) {
                Image scaled = resultImage.getScaledInstance(400, 400, Image.SCALE_SMOOTH);
                previewLabel.setIcon(new ImageIcon(scaled));
                previewLabel.setText(null);
                saveButton.setEnabled(true);
            }
        } catch (Exception ex) {
            statusLabel.setText("Cannot show result: " + ex.getMessage());
        }
    }

    private void setRunning(boolean r) {
        running = r;
        generateButton.setEnabled(!r);
        cancelButton.setEnabled(r);
        browseButton.setEnabled(!r);
        modelCombo.setEnabled(!r);
        progressBar.setVisible(r);
        if (r) {
            progressBar.setIndeterminate(true);
        } else {
            progressBar.setIndeterminate(false);
            progressBar.setValue(0);
        }
    }

    /* ================================================================== */
    /*  Keyboard shortcuts                                                 */
    /* ================================================================== */

    private void wireKeyboardShortcuts() {
        int menuMask = Toolkit.getDefaultToolkit().getMenuShortcutKeyMaskEx();

        getInputMap(WHEN_IN_FOCUSED_WINDOW).put(
                KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, menuMask), "generate");
        getActionMap().put("generate", new javax.swing.AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) { generate(); }
        });

        getInputMap(WHEN_IN_FOCUSED_WINDOW).put(
                KeyStroke.getKeyStroke(KeyEvent.VK_S, menuMask), "save");
        getActionMap().put("save", new javax.swing.AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) { saveResult(); }
        });

        getInputMap(WHEN_IN_FOCUSED_WINDOW).put(
                KeyStroke.getKeyStroke(KeyEvent.VK_PERIOD, menuMask), "cancel");
        getActionMap().put("cancel", new javax.swing.AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) { cancelInference(); }
        });
    }

    /* ================================================================== */
    /*  Drag and drop                                                      */
    /* ================================================================== */

    private void wireDragAndDrop() {
        new DropTarget(maskCanvas, DnDConstants.ACTION_COPY, new DropTargetAdapter() {
            @Override public void drop(DropTargetDropEvent dtde) {
                try {
                    dtde.acceptDrop(DnDConstants.ACTION_COPY);
                    @SuppressWarnings("unchecked")
                    List<File> files = (List<File>) dtde.getTransferable()
                            .getTransferData(DataFlavor.javaFileListFlavor);
                    if (!files.isEmpty()) loadInputImage(files.get(0));
                    dtde.dropComplete(true);
                } catch (Exception ex) {
                    dtde.dropComplete(false);
                }
            }
        });
    }

    /* ================================================================== */
    /*  Helpers                                                            */
    /* ================================================================== */

    private void appendLog(String msg) {
        try {
            var doc = statusArea.getDocument();
            doc.insertString(doc.getLength(), msg + "\n", null);
            statusArea.setCaretPosition(doc.getLength());
        } catch (Exception ignored) {}
    }

    private int parseIntOr(String s, int fallback) {
        try { return Integer.parseInt(s.trim()); }
        catch (NumberFormatException e) { return fallback; }
    }

    /* ================================================================== */
    /*  Mask drawing canvas                                                */
    /* ================================================================== */

    private class MaskCanvas extends JPanel {
        private boolean drawMode = false;
        private int lastX, lastY;

        MaskCanvas() {
            setPreferredSize(new Dimension(400, 400));
            setBackground(Color.DARK_GRAY);
            setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));

            addMouseListener(new MouseAdapter() {
                @Override public void mousePressed(MouseEvent e) {
                    if (maskMode && maskImage != null) {
                        drawMode = true;
                        lastX = e.getX();
                        lastY = e.getY();
                        paintOnMask(e.getX(), e.getY(), e.getX(), e.getY(), e.isShiftDown());
                    }
                }
                @Override public void mouseReleased(MouseEvent e) {
                    drawMode = false;
                }
            });

            addMouseMotionListener(new MouseMotionAdapter() {
                @Override public void mouseDragged(MouseEvent e) {
                    if (drawMode && maskMode && maskImage != null) {
                        paintOnMask(lastX, lastY, e.getX(), e.getY(), e.isShiftDown());
                        lastX = e.getX();
                        lastY = e.getY();
                    }
                }
            });

            // Mouse wheel to adjust brush size
            addMouseWheelListener(e -> {
                brushSize = Math.max(5, Math.min(200, brushSize - e.getWheelRotation() * 5));
                repaint();
            });
        }

        void setMaskMode(boolean mode) {
            drawMode = false;
        }

        private void paintOnMask(int x1, int y1, int x2, int y2, boolean eraseMode) {
            if (maskImage == null || inputImage == null) return;
            // Map canvas coordinates to mask coordinates
            double scaleX = (double) maskImage.getWidth() / getWidth();
            double scaleY = (double) maskImage.getHeight() / getHeight();
            int mx1 = (int) (x1 * scaleX), my1 = (int) (y1 * scaleY);
            int mx2 = (int) (x2 * scaleX), my2 = (int) (y2 * scaleY);
            int mBrush = (int) (brushSize * scaleX);

            Graphics2D g = maskImage.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(eraseMode ? Color.BLACK : Color.WHITE);
            g.setStroke(new BasicStroke(mBrush, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
            g.drawLine(mx1, my1, mx2, my2);
            g.dispose();
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            int w = getWidth(), h = getHeight();

            if (inputImage != null) {
                // Draw input image scaled to fill
                g2.drawImage(inputImage, 0, 0, w, h, null);

                // Overlay mask semi-transparently
                if (maskImage != null && maskMode) {
                    g2.setComposite(java.awt.AlphaComposite.getInstance(
                            java.awt.AlphaComposite.SRC_OVER, 0.4f));
                    g2.drawImage(maskImage, 0, 0, w, h, null);
                    g2.setComposite(java.awt.AlphaComposite.SrcOver);
                }

                // Draw brush cursor if in mask mode
                if (maskMode) {
                    java.awt.Point mouse = getMousePosition();
                    if (mouse != null) {
                        g2.setColor(new Color(255, 255, 255, 100));
                        g2.drawOval(mouse.x - brushSize / 2, mouse.y - brushSize / 2,
                                brushSize, brushSize);
                    }
                }
            } else {
                g2.setColor(new Color(100, 100, 100));
                g2.setFont(g2.getFont().deriveFont(Font.ITALIC, 12f));
                String hint = "Drop image here or Browse";
                int sw = g2.getFontMetrics().stringWidth(hint);
                g2.drawString(hint, (w - sw) / 2, h / 2);
            }
        }
    }
}
