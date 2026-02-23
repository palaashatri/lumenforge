package atri.palaash.lumenforge.ui;

import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JPanel;
import javax.swing.table.AbstractTableModel;
import java.awt.BorderLayout;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public class HistoryPanel extends JPanel {

    private final HistoryTableModel tableModel;

    public HistoryPanel() {
        super(new BorderLayout());
        this.tableModel = new HistoryTableModel();
        JTable table = new JTable(tableModel);
        table.setFillsViewportHeight(true);
        table.setRowHeight(24);
        add(new JScrollPane(table), BorderLayout.CENTER);
    }

    public void addEntry(HistoryEntry entry) {
        tableModel.addEntry(entry);
    }

    private static final class HistoryTableModel extends AbstractTableModel {

        private final String[] columns = {
                "Time", "Model", "Prompt", "Negative", "Seed", "Batch", "Size", "Style", "Status", "Output"
        };
        private final List<HistoryEntry> rows = new ArrayList<>();
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
