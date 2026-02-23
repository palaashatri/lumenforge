package atri.palaash.lumenforge.ui;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JPanel;
import javax.swing.table.AbstractTableModel;
import java.awt.BorderLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class HistoryPanel extends JPanel {

    private static final Path PERSIST_PATH = Path.of(
            System.getProperty("user.home"), ".lumenforge-models", "history.json");
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    private final HistoryTableModel tableModel;

    public HistoryPanel() {
        super(new BorderLayout());
        this.tableModel = new HistoryTableModel();
        JTable table = new JTable(tableModel);
        table.setFillsViewportHeight(true);
        table.setRowHeight(24);
        add(new JScrollPane(table), BorderLayout.CENTER);
        loadFromDisk();
    }

    public void addEntry(HistoryEntry entry) {
        tableModel.addEntry(entry);
        saveToDisk();
    }

    /* ---- persistence ---- */

    private void saveToDisk() {
        try {
            Files.createDirectories(PERSIST_PATH.getParent());
            List<Map<String, Object>> list = new ArrayList<>();
            for (HistoryEntry e : tableModel.rows) {
                list.add(Map.of(
                        "timestamp", e.timestamp().toString(),
                        "model", e.model(),
                        "prompt", e.prompt(),
                        "negativePrompt", e.negativePrompt(),
                        "seed", e.seed(),
                        "batch", e.batch(),
                        "size", e.size(),
                        "style", e.style(),
                        "status", e.status(),
                        "outputPath", e.outputPath() == null ? "" : e.outputPath()
                ));
            }
            MAPPER.writeValue(PERSIST_PATH.toFile(), list);
        } catch (Exception ignored) { }
    }

    private void loadFromDisk() {
        try {
            if (!Files.exists(PERSIST_PATH)) { return; }
            List<Map<String, Object>> list = MAPPER.readValue(
                    PERSIST_PATH.toFile(), new TypeReference<>() {});
            for (Map<String, Object> m : list) {
                HistoryEntry entry = new HistoryEntry(
                        LocalDateTime.parse((String) m.get("timestamp")),
                        (String) m.get("model"),
                        (String) m.get("prompt"),
                        (String) m.getOrDefault("negativePrompt", ""),
                        ((Number) m.get("seed")).longValue(),
                        ((Number) m.get("batch")).intValue(),
                        (String) m.get("size"),
                        (String) m.getOrDefault("style", ""),
                        (String) m.get("status"),
                        (String) m.getOrDefault("outputPath", "")
                );
                tableModel.rows.add(entry);
            }
            tableModel.fireTableDataChanged();
        } catch (Exception ignored) { }
    }

    private static final class HistoryTableModel extends AbstractTableModel {

        private final String[] columns = {
                "Time", "Model", "Prompt", "Negative", "Seed", "Batch", "Size", "Style", "Status", "Output"
        };
        final List<HistoryEntry> rows = new ArrayList<>();
        private final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");

        void addEntry(HistoryEntry entry) {
            rows.add(0, entry);
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
            HistoryEntry entry = rows.get(rowIndex);
            return switch (columnIndex) {
                case 0 -> formatter.format(entry.timestamp());
                case 1 -> entry.model();
                case 2 -> entry.prompt();
                case 3 -> entry.negativePrompt();
                case 4 -> entry.seed();
                case 5 -> entry.batch();
                case 6 -> entry.size();
                case 7 -> entry.style();
                case 8 -> entry.status();
                case 9 -> entry.outputPath();
                default -> "";
            };
        }
    }
}
