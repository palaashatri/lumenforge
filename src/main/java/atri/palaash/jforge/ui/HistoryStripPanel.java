package atri.palaash.jforge.ui;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingConstants;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class HistoryStripPanel extends JPanel {

    private static final Path HISTORY_PATH = Path.of(
            System.getProperty("user.home"),
            ".jforge-models",
            "history.json");
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final DateTimeFormatter TS_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final JPanel strip;

    public HistoryStripPanel() {
        super(new BorderLayout());
        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(1, 0, 0, 0, new Color(220, 220, 220)),
                BorderFactory.createEmptyBorder(6, 8, 6, 8)));

        JLabel title = new JLabel("History");
        title.setBorder(BorderFactory.createEmptyBorder(0, 0, 4, 0));
        add(title, BorderLayout.NORTH);

        strip = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 2));
        strip.setOpaque(false);

        JScrollPane scrollPane = new JScrollPane(
                strip,
                JScrollPane.VERTICAL_SCROLLBAR_NEVER,
                JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        scrollPane.setBorder(BorderFactory.createEmptyBorder());
        scrollPane.getHorizontalScrollBar().setUnitIncrement(24);
        add(scrollPane, BorderLayout.CENTER);

        setPreferredSize(new Dimension(0, 138));
        reload();
    }

    public final void reload() {
        strip.removeAll();
        List<Map<String, Object>> entries = loadEntries();
        if (entries.isEmpty()) {
            JLabel empty = new JLabel("No generations yet.", SwingConstants.LEFT);
            empty.setForeground(new Color(120, 120, 120));
            strip.add(empty);
            revalidate();
            repaint();
            return;
        }

        int count = 0;
        for (Map<String, Object> entry : entries) {
            if (count >= 24) {
                break;
            }
            Object outputPathValue = entry.getOrDefault("outputPath", "");
            if (!(outputPathValue instanceof String outputPath) || outputPath.isBlank()) {
                continue;
            }
            File outputFile = Path.of(outputPath).toFile();
            if (!outputFile.exists()) {
                continue;
            }

            JButton thumbnailButton = createThumbnailButton(outputFile, entry);
            strip.add(thumbnailButton);
            count++;
        }

        if (count == 0) {
            JLabel empty = new JLabel("History file found, but no output images are available.");
            empty.setForeground(new Color(120, 120, 120));
            strip.add(empty);
        }

        revalidate();
        repaint();
    }

    private JButton createThumbnailButton(File outputFile, Map<String, Object> entry) {
        JButton button = new JButton();
        button.setPreferredSize(new Dimension(96, 96));
        button.setBorder(BorderFactory.createLineBorder(new Color(190, 190, 190)));
        button.setToolTipText(String.valueOf(entry.getOrDefault("prompt", "")));
        button.setContentAreaFilled(false);

        try {
            Image image = ImageIO.read(outputFile);
            if (image != null) {
                Image scaled = image.getScaledInstance(92, 92, Image.SCALE_SMOOTH);
                button.setIcon(new ImageIcon(scaled));
            } else {
                button.setText("Image");
            }
        } catch (Exception ex) {
            button.setText("Image");
        }

        button.addActionListener(e -> showMetadata(entry));
        button.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                maybeOpenFolder(e);
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                maybeOpenFolder(e);
            }

            private void maybeOpenFolder(MouseEvent e) {
                if (!e.isPopupTrigger()) {
                    return;
                }
                try {
                    File parent = outputFile.getParentFile();
                    if (parent != null && parent.exists()) {
                        Desktop.getDesktop().open(parent);
                    }
                } catch (Exception ignored) {
                }
            }
        });
        return button;
    }

    private void showMetadata(Map<String, Object> entry) {
        StringBuilder content = new StringBuilder();
        content.append("Time: ").append(entry.getOrDefault("timestamp", "")).append('\n');
        content.append("Model: ").append(entry.getOrDefault("model", "")).append('\n');
        content.append("Prompt: ").append(entry.getOrDefault("prompt", "")).append('\n');
        content.append("Negative: ").append(entry.getOrDefault("negativePrompt", "")).append('\n');
        content.append("Seed: ").append(entry.getOrDefault("seed", "")).append('\n');
        content.append("Batch: ").append(entry.getOrDefault("batch", "")).append('\n');
        content.append("Size: ").append(entry.getOrDefault("size", "")).append('\n');
        content.append("Style: ").append(entry.getOrDefault("style", "")).append('\n');
        content.append("Status: ").append(entry.getOrDefault("status", "")).append('\n');
        content.append("Output: ").append(entry.getOrDefault("outputPath", ""));

        JOptionPane.showMessageDialog(this, content.toString(), "Generation Metadata", JOptionPane.INFORMATION_MESSAGE);
    }

    private List<Map<String, Object>> loadEntries() {
        if (!Files.exists(HISTORY_PATH)) {
            return List.of();
        }
        try {
            List<Map<String, Object>> entries = OBJECT_MAPPER.readValue(
                    HISTORY_PATH.toFile(),
                    new TypeReference<>() {
                    });
            List<Map<String, Object>> copy = new ArrayList<>(entries);
            copy.sort(Comparator.comparing(this::timestampOrMin).reversed());
            return copy;
        } catch (Exception ex) {
            return List.of();
        }
    }

    private LocalDateTime timestampOrMin(Map<String, Object> entry) {
        Object timestamp = entry.get("timestamp");
        if (timestamp instanceof String text) {
            try {
                return LocalDateTime.parse(text);
            } catch (Exception ignored) {
                try {
                    return LocalDateTime.parse(text, TS_FORMAT);
                } catch (Exception ignoredAgain) {
                    return LocalDateTime.MIN;
                }
            }
        }
        return LocalDateTime.MIN;
    }
}
