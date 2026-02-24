package atri.palaash.lumenforge.ui;

import atri.palaash.lumenforge.model.ModelDescriptor;
import atri.palaash.lumenforge.model.ModelRegistry;
import atri.palaash.lumenforge.model.TaskType;
import atri.palaash.lumenforge.storage.DownloadProgress;
import atri.palaash.lumenforge.storage.ModelDownloader;
import atri.palaash.lumenforge.storage.ModelStorage;
import atri.palaash.lumenforge.storage.PyTorchToOnnxConverter;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.AbstractTableModel;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

public class ModelManagerPanel extends JPanel {

    private final ModelRegistry modelRegistry;
    private final ModelStorage modelStorage;
    private final ModelDownloader modelDownloader;
    private final ModelTableModel tableModel;
    private final JLabel statusLabel;
    private final JProgressBar progressBar;
    private final JTable table;
    private final Set<String> promptedForConversion = new HashSet<>();
    private Runnable onModelsUpdated;

    public ModelManagerPanel(ModelRegistry modelRegistry, ModelStorage modelStorage, ModelDownloader modelDownloader) {
        super(new BorderLayout(12, 12));
        this.modelRegistry = modelRegistry;
        this.modelStorage = modelStorage;
        this.modelDownloader = modelDownloader;
        this.tableModel = new ModelTableModel(modelRegistry.allModels(), modelStorage);
        this.statusLabel = new JLabel("Select a model and download if needed");
        this.statusLabel.setFont(this.statusLabel.getFont().deriveFont(Font.PLAIN, 12f));
        this.progressBar = new JProgressBar(0, 100);
        this.progressBar.setStringPainted(true);
        this.progressBar.setVisible(false);

        this.table = new JTable(tableModel);
        table.setFillsViewportHeight(true);
        table.setRowHeight(26);
        JScrollPane tableScroll = new JScrollPane(table);

        JButton refreshButton = new JButton("Refresh from Hugging Face");
        refreshButton.setPreferredSize(new Dimension(200, 32));
        refreshButton.addActionListener(e -> refreshModelsFromHuggingFace());

        JButton importButton = new JButton("Import ONNX File");
        importButton.setPreferredSize(new Dimension(160, 32));
        importButton.addActionListener(e -> {
            int row = table.getSelectedRow();
            if (row < 0) {
                statusLabel.setText("Select a row first.");
                return;
            }
            importModelFromFile(tableModel.modelAt(row), row);
        });

        JButton downloadButton = new JButton("Download / Redownload");
        downloadButton.setPreferredSize(new Dimension(190, 32));
        downloadButton.addActionListener(e -> {
            int row = table.getSelectedRow();
            if (row < 0) {
                statusLabel.setText("Select a row first.");
                return;
            }
            ModelDescriptor descriptor = tableModel.modelAt(row);
            startDownload(descriptor, row);
        });

        JButton convertButton = new JButton("Convert to ONNX");
        convertButton.setPreferredSize(new Dimension(160, 32));
        convertButton.setToolTipText("Convert a PyTorch / HuggingFace model to ONNX format");
        convertButton.addActionListener(e -> handleConvertToOnnx());

        /* Prompt the user when they select a PyTorch model row */
        table.getSelectionModel().addListSelectionListener(e -> {
            if (e.getValueIsAdjusting()) return;
            int row = table.getSelectedRow();
            if (row < 0) return;
            ModelDescriptor desc = tableModel.modelAt(row);
            if (PyTorchToOnnxConverter.isPyTorchModel(desc.id())
                    || PyTorchToOnnxConverter.isPyTorchModel(desc.relativePath())) {
                statusLabel.setText("\u2139 This is a PyTorch model — click 'Convert to ONNX' to create an ONNX version.");
                if (promptedForConversion.add(desc.id())) {
                    SwingUtilities.invokeLater(() -> offerConversion(desc));
                }
            }
        });

        JPanel top = new JPanel(new BorderLayout(8, 8));
        top.add(refreshButton, BorderLayout.WEST);
        JPanel rightActions = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        rightActions.add(convertButton);
        rightActions.add(importButton);
        rightActions.add(downloadButton);
        top.add(rightActions, BorderLayout.EAST);

        add(top, BorderLayout.NORTH);
        add(tableScroll, BorderLayout.CENTER);
        JPanel bottom = new JPanel(new BorderLayout(8, 8));
        bottom.add(statusLabel, BorderLayout.CENTER);
        bottom.add(progressBar, BorderLayout.EAST);

        add(bottom, BorderLayout.SOUTH);
        setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
    }

    private void refreshTable() {
        tableModel.setRows(modelRegistry.allModels());
        tableModel.refreshAvailability();
        statusLabel.setText("Availability refreshed. Root: " + modelStorage.root());
    }

    public void refreshModelsFromHuggingFace() {
        statusLabel.setText("Refreshing Hugging Face ONNX text→image models...");
        progressBar.setVisible(true);
        progressBar.setIndeterminate(true);
        modelDownloader.discoverTextToImageModels()
                .whenComplete((models, error) -> SwingUtilities.invokeLater(() -> {
                    progressBar.setIndeterminate(false);
                    progressBar.setVisible(false);
                    if (error != null) {
                        statusLabel.setText("Refresh failed: " + error.getMessage());
                        return;
                    }
                    int added = modelRegistry.mergeDownloadableAssets(models);
                    refreshTable();
                    statusLabel.setText("Model refresh complete. Added " + added + " new model(s).");
                    if (onModelsUpdated != null) {
                        onModelsUpdated.run();
                    }
                }));
    }

    public void setOnModelsUpdated(Runnable onModelsUpdated) {
        this.onModelsUpdated = onModelsUpdated;
    }

    private void startDownload(ModelDescriptor descriptor, int row) {
        statusLabel.setText("Downloading " + descriptor.displayName() + "...");
        progressBar.setVisible(true);
        progressBar.setIndeterminate(true);
        progressBar.setValue(0);
        tableModel.updateProgress(descriptor.id(), 0);
        modelDownloader.download(descriptor, progress -> onProgress(descriptor, progress))
                .whenComplete((path, error) -> SwingUtilities.invokeLater(() -> {
                    progressBar.setIndeterminate(false);
                    progressBar.setVisible(false);
                    if (error != null) {
                        statusLabel.setText("Download failed: " + error.getMessage());
                        return;
                    }
                    tableModel.setAvailable(row, true);
                    tableModel.updateProgress(descriptor.id(), 100);
                    statusLabel.setText("Downloaded: " + path);
                }));
    }

    private void onProgress(ModelDescriptor descriptor, DownloadProgress progress) {
        SwingUtilities.invokeLater(() -> {
            if (progress.isStatusMessage()) {
                statusLabel.setText(progress.statusMessage());
                return;
            }
            progressBar.setIndeterminate(false);
            progressBar.setValue(progress.percent());
            tableModel.updateProgress(descriptor.id(), progress.percent());
        });
    }

    private void importModelFromFile(ModelDescriptor descriptor, int row) {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select model file for " + descriptor.displayName());
        chooser.setFileFilter(new FileNameExtensionFilter(
                "Model files (*.onnx, *.pt, *.pth)", "onnx", "pt", "pth"));
        int selection = chooser.showOpenDialog(this);
        if (selection != JFileChooser.APPROVE_OPTION) {
            statusLabel.setText("Import canceled.");
            return;
        }

        Path source = chooser.getSelectedFile().toPath();
        String fileName = source.getFileName().toString().toLowerCase();

        // If user imports a PyTorch file, offer to convert it to ONNX
        if (fileName.endsWith(".pt") || fileName.endsWith(".pth")) {
            int choice = JOptionPane.showConfirmDialog(this,
                    "'" + source.getFileName() + "' is a PyTorch model file.\n\n"
                    + "Would you like to convert it to ONNX format?\n"
                    + "(Requires Python 3 with torch and onnx packages)",
                    "Convert to ONNX?",
                    JOptionPane.YES_NO_CANCEL_OPTION,
                    JOptionPane.QUESTION_MESSAGE);
            if (choice == JOptionPane.YES_OPTION) {
                startConversion(source.toString(), source.getFileName().toString(), "generic");
                return;
            } else if (choice == JOptionPane.CANCEL_OPTION) {
                statusLabel.setText("Import canceled.");
                return;
            }
            // NO → import as-is (fall through)
        }

        Path target = modelStorage.modelPath(descriptor);
        try {
            modelStorage.ensureParentDirectory(descriptor);
            Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING);
            tableModel.setAvailable(row, true);
            tableModel.updateProgress(descriptor.id(), 100);
            statusLabel.setText("Imported: " + target);
        } catch (IOException ex) {
            statusLabel.setText("Import failed: " + ex.getMessage());
        }
    }

    /* ── PyTorch → ONNX conversion ───────────────────────────────────── */

    /**
     * Handle the "Convert to ONNX" button click:
     * if the selected model is PyTorch → convert that model,
     * otherwise → ask the user for a HuggingFace model ID.
     */
    private void handleConvertToOnnx() {
        int row = table.getSelectedRow();
        if (row >= 0) {
            ModelDescriptor selected = tableModel.modelAt(row);
            if (PyTorchToOnnxConverter.isPyTorchModel(selected.id())
                    || PyTorchToOnnxConverter.isPyTorchModel(selected.relativePath())) {
                String modelId = selected.sourceUrl().isBlank()
                        ? selected.id() : selected.sourceUrl();
                startConversion(modelId, selected.displayName(), "diffusers");
                return;
            }
        }

        // No PyTorch model selected — ask for a HuggingFace model ID
        String modelId = (String) JOptionPane.showInputDialog(
                this,
                "Enter a HuggingFace model ID to convert to ONNX:\n\n"
                + "Examples:\n"
                + "  \u2022 runwayml/stable-diffusion-v1-5\n"
                + "  \u2022 stabilityai/stable-diffusion-xl-base-1.0\n"
                + "  \u2022 CompVis/stable-diffusion-v1-4\n",
                "Convert HuggingFace Model to ONNX",
                JOptionPane.PLAIN_MESSAGE,
                null, null, "");
        if (modelId != null && !modelId.isBlank()) {
            startConversion(modelId.trim(), modelId.trim(), "diffusers");
        }
    }

    /**
     * Offer conversion when the user selects a PyTorch model row (once per model).
     */
    private void offerConversion(ModelDescriptor desc) {
        int choice = JOptionPane.showConfirmDialog(this,
                "\"" + desc.displayName() + "\" is a PyTorch model.\n\n"
                + "Would you like to convert it to ONNX format\n"
                + "for use with LumenForge's ONNX Runtime engine?\n\n"
                + "(Requires Python 3 — dependencies are installed automatically)",
                "Convert to ONNX?",
                JOptionPane.YES_NO_OPTION,
                JOptionPane.QUESTION_MESSAGE);
        if (choice == JOptionPane.YES_OPTION) {
            String modelId = desc.sourceUrl().isBlank() ? desc.id() : desc.sourceUrl();
            startConversion(modelId, desc.displayName(), "diffusers");
        }
    }

    /**
     * Launch the full conversion pipeline on a background thread.
     */
    private void startConversion(String modelId, String displayName, String mode) {
        // 1. Check for Python
        String python = PyTorchToOnnxConverter.findPython();
        if (python == null) {
            int choice = JOptionPane.showConfirmDialog(this,
                    "Python 3 is required for PyTorch \u2192 ONNX conversion\n"
                    + "but was not found on your system.\n\n"
                    + "Would you like to open the Python download page?",
                    "Python Not Found",
                    JOptionPane.YES_NO_OPTION,
                    JOptionPane.WARNING_MESSAGE);
            if (choice == JOptionPane.YES_OPTION) {
                PyTorchToOnnxConverter.openPythonDownloadPage();
            }
            return;
        }

        // 2. Determine output directory
        String sanitized = modelId.replaceAll("[^a-zA-Z0-9._-]", "-").toLowerCase();
        Path outputDir = modelStorage.root().resolve("text-image").resolve("converted-" + sanitized);

        // 3. Show progress
        statusLabel.setText("Converting " + displayName + " to ONNX\u2026");
        progressBar.setVisible(true);
        progressBar.setIndeterminate(true);

        // 4. Run on background thread
        final String conversionMode = mode;
        CompletableFuture.supplyAsync(() -> {
            PyTorchToOnnxConverter converter = new PyTorchToOnnxConverter(
                    msg -> SwingUtilities.invokeLater(() -> statusLabel.setText(msg))
            );
            return converter.convert(modelId, outputDir, conversionMode);
        }).whenComplete((path, error) -> SwingUtilities.invokeLater(() -> {
            progressBar.setIndeterminate(false);
            progressBar.setVisible(false);
            if (error != null) {
                Throwable cause = error.getCause() != null ? error.getCause() : error;
                if (cause instanceof PyTorchToOnnxConverter.PythonNotFoundException) {
                    int choice = JOptionPane.showConfirmDialog(this,
                            cause.getMessage() + "\n\nOpen the Python download page?",
                            "Python Not Found",
                            JOptionPane.YES_NO_OPTION,
                            JOptionPane.WARNING_MESSAGE);
                    if (choice == JOptionPane.YES_OPTION) {
                        PyTorchToOnnxConverter.openPythonDownloadPage();
                    }
                } else {
                    statusLabel.setText("Conversion failed: " + cause.getMessage());
                    JOptionPane.showMessageDialog(this,
                            "Conversion failed:\n" + cause.getMessage(),
                            "Conversion Error",
                            JOptionPane.ERROR_MESSAGE);
                }
                return;
            }
            registerConvertedModel(modelId, sanitized, path);
        }));
    }

    /**
     * After a successful conversion, register the new ONNX model in the registry
     * and refresh the table so it appears immediately.
     */
    private void registerConvertedModel(String modelId, String sanitized, Path outputDir) {
        // Determine the ONNX artifact path
        Path unetOnnx = outputDir.resolve("unet").resolve("model.onnx");
        Path genericOnnx = outputDir.resolve("model.onnx");

        String relativePath;
        if (Files.exists(unetOnnx)) {
            relativePath = "text-image/converted-" + sanitized + "/unet/model.onnx";
        } else if (Files.exists(genericOnnx)) {
            relativePath = "text-image/converted-" + sanitized + "/model.onnx";
        } else {
            statusLabel.setText("Conversion produced output but no ONNX files were found.");
            return;
        }

        String id = "converted_" + sanitized.replace("-", "_");
        ModelDescriptor desc = new ModelDescriptor(
                id,
                "Converted: " + modelId,
                TaskType.TEXT_TO_IMAGE,
                relativePath,
                "",
                "Auto-converted from PyTorch via LumenForge."
        );

        modelRegistry.mergeDownloadableAssets(List.of(desc));
        refreshTable();
        statusLabel.setText("\u2713 Converted and registered: " + modelId);
        if (onModelsUpdated != null) {
            onModelsUpdated.run();
        }
    }

    private static final class ModelTableModel extends AbstractTableModel {

        private final String[] columns = {"Task", "Model", "Available", "Progress", "Source URL"};
        private final java.util.ArrayList<ModelDescriptor> rows;
        private final ModelStorage storage;
        private final Map<String, Integer> progressById = new HashMap<>();
        private final Map<String, Boolean> availableById = new HashMap<>();

        private ModelTableModel(List<ModelDescriptor> rows, ModelStorage storage) {
            this.rows = new java.util.ArrayList<>(rows);
            this.storage = storage;
            refreshAvailability();
        }

        private void setRows(List<ModelDescriptor> newRows) {
            rows.clear();
            rows.addAll(newRows);
            refreshAvailability();
        }

        private ModelDescriptor modelAt(int row) {
            return rows.get(row);
        }

        private void refreshAvailability() {
            for (ModelDescriptor descriptor : rows) {
                availableById.put(descriptor.id(), storage.isAvailable(descriptor));
                progressById.putIfAbsent(descriptor.id(), 0);
            }
            fireTableDataChanged();
        }

        private void setAvailable(int row, boolean available) {
            ModelDescriptor descriptor = rows.get(row);
            availableById.put(descriptor.id(), available);
            fireTableRowsUpdated(row, row);
        }

        private void updateProgress(String modelId, int percent) {
            progressById.put(modelId, percent);
            fireTableDataChanged();
        }

        @Override
        public int getRowCount() {
            return rows.size();
        }

        @Override
        public int getColumnCount() {
            return columns.length;
        }

        @Override
        public String getColumnName(int column) {
            return columns[column];
        }

        @Override
        public Object getValueAt(int rowIndex, int columnIndex) {
            ModelDescriptor descriptor = rows.get(rowIndex);
            return switch (columnIndex) {
                case 0 -> descriptor.taskType().displayName();
                case 1 -> descriptor.displayName();
                case 2 -> availableById.getOrDefault(descriptor.id(), false) ? "Yes" : "No";
                case 3 -> progressById.getOrDefault(descriptor.id(), 0) + "%";
                case 4 -> descriptor.sourceUrl();
                default -> "";
            };
        }
    }
}
