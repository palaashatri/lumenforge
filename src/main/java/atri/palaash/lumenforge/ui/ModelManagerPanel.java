package atri.palaash.lumenforge.ui;

import atri.palaash.lumenforge.model.ModelDescriptor;
import atri.palaash.lumenforge.model.ModelRegistry;
import atri.palaash.lumenforge.storage.DownloadProgress;
import atri.palaash.lumenforge.storage.ModelDownloader;
import atri.palaash.lumenforge.storage.ModelStorage;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.AbstractTableModel;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelManagerPanel extends JPanel {

    private final ModelRegistry modelRegistry;
    private final ModelStorage modelStorage;
    private final ModelDownloader modelDownloader;
    private final ModelTableModel tableModel;
    private final JLabel statusLabel;
    private final JProgressBar progressBar;
    private final JTable table;
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


        JPanel top = new JPanel(new BorderLayout(8, 8));
        top.add(refreshButton, BorderLayout.WEST);
        JPanel rightActions = new JPanel(new BorderLayout(8, 8));
        rightActions.add(importButton, BorderLayout.WEST);
        rightActions.add(downloadButton, BorderLayout.EAST);
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
        chooser.setDialogTitle("Select ONNX file for " + descriptor.displayName());
        chooser.setFileFilter(new FileNameExtensionFilter("ONNX model (*.onnx)", "onnx"));
        int selection = chooser.showOpenDialog(this);
        if (selection != JFileChooser.APPROVE_OPTION) {
            statusLabel.setText("Import canceled.");
            return;
        }

        Path source = chooser.getSelectedFile().toPath();
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
