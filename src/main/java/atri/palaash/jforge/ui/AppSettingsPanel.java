package atri.palaash.jforge.ui;

import atri.palaash.jforge.inference.GenericOnnxService;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.nio.file.Path;

/**
 * Application-wide settings tab.
 */
public class AppSettingsPanel extends JPanel {

    public AppSettingsPanel(Runnable onHistoryRefresh) {
        super(new BorderLayout(12, 12));
        setBorder(BorderFactory.createEmptyBorder(16, 16, 16, 16));

        JPanel top = new JPanel(new BorderLayout(8, 8));
        JLabel title = new JLabel("Application Settings");
        title.setFont(title.getFont().deriveFont(16f));
        top.add(title, BorderLayout.NORTH);

        JTextArea details = new JTextArea();
        details.setEditable(false);
        details.setLineWrap(true);
        details.setWrapStyleWord(true);
        details.setText("Use this tab to manage global UI and runtime behavior. "
                + "Changes are applied immediately.");
        details.setOpaque(false);
        top.add(details, BorderLayout.CENTER);

        add(top, BorderLayout.NORTH);

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 8));

        JCheckBox darkMode = new JCheckBox("Dark Mode", NativeLookAndFeel.isDarkMode());
        darkMode.addActionListener(e -> {
            NativeLookAndFeel.setDarkMode(darkMode.isSelected());
            for (java.awt.Window window : java.awt.Window.getWindows()) {
                SwingUtilities.updateComponentTreeUI(window);
            }
        });
        controls.add(darkMode);

        JButton clearRuntimeCache = new JButton("Clear Runtime Cache");
        clearRuntimeCache.addActionListener(e -> {
            GenericOnnxService.clearCache();
            JOptionPane.showMessageDialog(
                    this,
                    "ONNX Runtime session and tokenizer caches cleared.",
                    "Runtime Cache",
                    JOptionPane.INFORMATION_MESSAGE);
        });
        controls.add(clearRuntimeCache);

        JButton refreshHistory = new JButton("Refresh History Strip");
        refreshHistory.addActionListener(e -> {
            if (onHistoryRefresh != null) {
                onHistoryRefresh.run();
            }
        });
        controls.add(refreshHistory);

        JButton openOutputFolder = new JButton("Open Outputs Folder");
        openOutputFolder.addActionListener(e -> {
            try {
                Path outputPath = Path.of(System.getProperty("user.home"), ".jforge-models", "outputs", "images");
                java.nio.file.Files.createDirectories(outputPath);
                java.awt.Desktop.getDesktop().open(outputPath.toFile());
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(
                        this,
                        "Could not open outputs folder: " + ex.getMessage(),
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
        });
        controls.add(openOutputFolder);

        JPanel body = new JPanel(new BorderLayout());
        body.add(controls, BorderLayout.NORTH);
        body.setPreferredSize(new Dimension(900, 420));

        add(body, BorderLayout.CENTER);
    }
}
