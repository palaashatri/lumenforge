package atri.palaash.lumenforge.ui;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import javax.swing.BorderFactory;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PromptLibraryPanel extends JPanel {

    private static final Path PERSIST_PATH = Path.of(
            System.getProperty("user.home"), ".lumenforge-models", "library.json");
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    private final DefaultListModel<PromptPreset> listModel = new DefaultListModel<>();
    private final JList<PromptPreset> list = new JList<>(listModel);
    private final JTextField searchField = new JTextField();
    private final List<PromptPreset> allPresets = new ArrayList<>();

    public PromptLibraryPanel() {
        super(new BorderLayout(8, 8));
        list.setCellRenderer((lst, value, index, isSelected, cellHasFocus) ->
                new JLabel(value == null ? "" : value.name() + "  |  " + value.tags()));
        list.setFixedCellHeight(24);

        JPanel top = new JPanel(new BorderLayout(6, 6));
        top.add(new JLabel("Search"), BorderLayout.WEST);
        top.add(searchField, BorderLayout.CENTER);

        JPanel buttons = new JPanel(new BorderLayout(6, 6));
        JButton applyButton = new JButton("Apply");
        JButton deleteButton = new JButton("Delete");
        buttons.add(applyButton, BorderLayout.WEST);
        buttons.add(deleteButton, BorderLayout.EAST);

        add(top, BorderLayout.NORTH);
        add(new JScrollPane(list), BorderLayout.CENTER);
        add(buttons, BorderLayout.SOUTH);
        setBorder(BorderFactory.createEmptyBorder(6, 6, 6, 6));
        setPreferredSize(new Dimension(320, 260));

        searchField.getDocument().addDocumentListener((SimpleDocumentListener) e -> applyFilter());
        deleteButton.addActionListener(e -> deleteSelected());
        applyButton.addActionListener(e -> {
            PromptPreset preset = list.getSelectedValue();
            if (preset != null && onApply != null) {
                onApply.onApply(preset);
            }
        });

        loadFromDisk();
    }

    public interface ApplyListener {
        void onApply(PromptPreset preset);
    }

    private ApplyListener onApply;

    public void setOnApply(ApplyListener listener) {
        this.onApply = listener;
    }

    public void addPreset(PromptPreset preset) {
        allPresets.add(0, preset);
        applyFilter();
        saveToDisk();
    }

    private void applyFilter() {
        String query = searchField.getText().trim().toLowerCase();
        listModel.clear();
        for (PromptPreset preset : allPresets) {
            String haystack = (preset.name() + " " + preset.prompt() + " " + preset.tags()).toLowerCase();
            if (query.isEmpty() || haystack.contains(query)) {
                listModel.addElement(preset);
            }
        }
    }

    private void deleteSelected() {
        PromptPreset selected = list.getSelectedValue();
        if (selected == null) {
            return;
        }
        allPresets.remove(selected);
        applyFilter();
        saveToDisk();
    }

    /* ---- persistence ---- */

    private void saveToDisk() {
        try {
            Files.createDirectories(PERSIST_PATH.getParent());
            List<Map<String, String>> list = new ArrayList<>();
            for (PromptPreset p : allPresets) {
                list.add(Map.of(
                        "name", p.name(),
                        "prompt", p.prompt(),
                        "negativePrompt", p.negativePrompt(),
                        "tags", p.tags(),
                        "style", p.style()
                ));
            }
            MAPPER.writeValue(PERSIST_PATH.toFile(), list);
        } catch (Exception ignored) { }
    }

    private void loadFromDisk() {
        try {
            if (!Files.exists(PERSIST_PATH)) { return; }
            List<Map<String, String>> list = MAPPER.readValue(
                    PERSIST_PATH.toFile(), new TypeReference<>() {});
            for (Map<String, String> m : list) {
                allPresets.add(new PromptPreset(
                        m.getOrDefault("name", ""),
                        m.getOrDefault("prompt", ""),
                        m.getOrDefault("negativePrompt", ""),
                        m.getOrDefault("tags", ""),
                        m.getOrDefault("style", "")
                ));
            }
            applyFilter();
        } catch (Exception ignored) { }
    }
}
