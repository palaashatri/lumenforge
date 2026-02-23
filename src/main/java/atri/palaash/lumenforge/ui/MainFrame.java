package atri.palaash.lumenforge.ui;

import atri.palaash.lumenforge.inference.InferenceService;
import atri.palaash.lumenforge.model.ModelDescriptor;
import atri.palaash.lumenforge.model.ModelRegistry;
import atri.palaash.lumenforge.model.TaskType;
import atri.palaash.lumenforge.storage.ModelDownloader;
import atri.palaash.lumenforge.storage.ModelStorage;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.DefaultListCellRenderer;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.ListSelectionModel;
import javax.swing.UIManager;
import javax.swing.border.Border;
import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.KeyboardFocusManager;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Main application window.
 * <ul>
 *   <li>Left sidebar for navigation (Generate / Upscale)</li>
 *   <li>Card layout content area</li>
 *   <li>Menu bar: View (Dark Mode), Models (Model Manager), Inference (Use GPU), Tools</li>
 * </ul>
 */
public class MainFrame extends JFrame {

    private static final String CARD_GENERATE = "Generate";
    private static final String CARD_UPSCALE  = "Upscale";

    private final CardLayout cardLayout = new CardLayout();
    private final JPanel contentPanel = new JPanel(cardLayout);

    private final TextToImagePanel textToImagePanel;
    private final ImageUpscalePanel imageUpscalePanel;
    private final ModelManagerPanel modelManagerPanel;

    /* GPU state (shared across panels via supplier) */
    private boolean gpuEnabled = true;

    public MainFrame(ModelRegistry registry,
                     ModelStorage storage,
                     ModelDownloader downloader,
                     Map<TaskType, InferenceService> services) {

        setTitle("LumenForge");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(980, 720);
        setMinimumSize(new Dimension(760, 520));
        setLocationRelativeTo(null);

        /* ---- panels ---- */
        List<ModelDescriptor> t2iModels = registry.allModels().stream()
                .filter(m -> m.taskType() == TaskType.TEXT_TO_IMAGE)
                .collect(Collectors.toList());
        List<ModelDescriptor> upscaleModels = registry.allModels().stream()
                .filter(m -> m.taskType() == TaskType.IMAGE_UPSCALE)
                .collect(Collectors.toList());

        textToImagePanel = new TextToImagePanel(
                t2iModels, downloader, services.get(TaskType.TEXT_TO_IMAGE));
        imageUpscalePanel = new ImageUpscalePanel(
                upscaleModels, downloader, services.get(TaskType.IMAGE_UPSCALE));

        textToImagePanel.setGpuSupplier(() -> gpuEnabled);
        imageUpscalePanel.setGpuSupplier(() -> gpuEnabled);

        modelManagerPanel = new ModelManagerPanel(registry, storage, downloader);
        modelManagerPanel.setOnModelsUpdated(() -> {
            textToImagePanel.updateModels(
                    registry.allModels().stream()
                            .filter(m -> m.taskType() == TaskType.TEXT_TO_IMAGE)
                            .collect(Collectors.toList()));
            imageUpscalePanel.updateModels(
                    registry.allModels().stream()
                            .filter(m -> m.taskType() == TaskType.IMAGE_UPSCALE)
                            .collect(Collectors.toList()));
        });

        /* ---- sidebar ---- */
        String[] navItems = {CARD_GENERATE, CARD_UPSCALE};
        JList<String> sidebar = new JList<>(navItems);
        sidebar.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        sidebar.setSelectedIndex(0);
        sidebar.setFixedCellHeight(40);
        sidebar.setCellRenderer(new SidebarRenderer());
        sidebar.setBorder(BorderFactory.createEmptyBorder(12, 0, 0, 0));
        sidebar.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                cardLayout.show(contentPanel, sidebar.getSelectedValue());
            }
        });

        JPanel sidebarPanel = new JPanel(new BorderLayout());
        sidebarPanel.setPreferredSize(new Dimension(150, 0));
        sidebarPanel.setBorder(new SeparatorBorder());
        sidebarPanel.add(new JScrollPane(sidebar, JScrollPane.VERTICAL_SCROLLBAR_NEVER,
                JScrollPane.HORIZONTAL_SCROLLBAR_NEVER), BorderLayout.CENTER);

        JLabel brand = new JLabel("LumenForge");
        brand.setFont(brand.getFont().deriveFont(Font.BOLD, 14f));
        brand.setBorder(BorderFactory.createEmptyBorder(14, 16, 8, 16));
        sidebarPanel.add(brand, BorderLayout.NORTH);

        /* ---- content cards ---- */
        contentPanel.add(textToImagePanel, CARD_GENERATE);
        contentPanel.add(imageUpscalePanel, CARD_UPSCALE);
        cardLayout.show(contentPanel, CARD_GENERATE);

        /* ---- assemble ---- */
        JPanel root = new JPanel(new BorderLayout());
        root.add(sidebarPanel, BorderLayout.WEST);
        root.add(contentPanel, BorderLayout.CENTER);
        setContentPane(root);

        /* ---- menu bar ---- */
        setJMenuBar(buildMenuBar());
    }

    /* ================================================================== */
    /*  Menu bar                                                           */
    /* ================================================================== */

    private JMenuBar buildMenuBar() {
        JMenuBar bar = new JMenuBar();

        /* View */
        JMenu viewMenu = new JMenu("View");
        JCheckBoxMenuItem darkModeItem = new JCheckBoxMenuItem("Dark Mode");
        darkModeItem.setSelected(NativeLookAndFeel.isDarkMode());
        darkModeItem.addActionListener(e -> {
            NativeLookAndFeel.setDarkMode(darkModeItem.isSelected());
            repaintAll();
        });
        viewMenu.add(darkModeItem);
        bar.add(viewMenu);

        /* Models */
        JMenu modelsMenu = new JMenu("Models");
        JMenuItem openManager = new JMenuItem("Open Model Manager\u2026");
        openManager.addActionListener(e -> showModelManager());
        modelsMenu.add(openManager);
        bar.add(modelsMenu);

        /* Inference */
        JMenu inferenceMenu = new JMenu("Inference");
        JCheckBoxMenuItem gpuItem = new JCheckBoxMenuItem("Use GPU");
        gpuItem.setSelected(gpuEnabled);
        gpuItem.addActionListener(e -> gpuEnabled = gpuItem.isSelected());
        inferenceMenu.add(gpuItem);
        bar.add(inferenceMenu);

        /* Tools */
        JMenu toolsMenu = new JMenu("Tools");
        JMenuItem savePreset = new JMenuItem("Save Current Preset");
        savePreset.addActionListener(e -> textToImagePanel.saveCurrentPreset());
        JMenuItem openLogs = new JMenuItem("Open Logs Folder");
        openLogs.addActionListener(e -> textToImagePanel.openLogsFolder());
        toolsMenu.add(savePreset);
        toolsMenu.add(openLogs);
        bar.add(toolsMenu);

        return bar;
    }

    /* ================================================================== */
    /*  Model Manager dialog                                               */
    /* ================================================================== */

    private void showModelManager() {
        JDialog dialog = new JDialog(this, "Model Manager", true);
        dialog.setSize(720, 480);
        dialog.setLocationRelativeTo(this);
        dialog.setContentPane(modelManagerPanel);
        dialog.setVisible(true);
    }

    /* ================================================================== */
    /*  L&F helpers                                                        */
    /* ================================================================== */

    private void repaintAll() {
        for (java.awt.Window w : java.awt.Window.getWindows()) {
            javax.swing.SwingUtilities.updateComponentTreeUI(w);
        }
    }

    /* ================================================================== */
    /*  Sidebar renderer                                                   */
    /* ================================================================== */

    private static class SidebarRenderer extends DefaultListCellRenderer {
        @Override
        public Component getListCellRendererComponent(JList<?> list, Object value,
                                                       int index, boolean isSelected,
                                                       boolean cellHasFocus) {
            JLabel lbl = (JLabel) super.getListCellRendererComponent(
                    list, value, index, isSelected, cellHasFocus);
            lbl.setBorder(BorderFactory.createEmptyBorder(0, 18, 0, 18));
            lbl.setFont(lbl.getFont().deriveFont(Font.PLAIN, 13f));
            if (isSelected) {
                lbl.setBackground(UIManager.getColor("List.selectionBackground"));
                lbl.setForeground(UIManager.getColor("List.selectionForeground"));
            }
            return lbl;
        }
    }

    /* ================================================================== */
    /*  Separator border (right edge of sidebar)                           */
    /* ================================================================== */

    private static class SeparatorBorder implements Border {
        @Override public java.awt.Insets getBorderInsets(Component c) {
            return new java.awt.Insets(0, 0, 0, 1);
        }
        @Override public boolean isBorderOpaque() { return true; }
        @Override public void paintBorder(Component c, java.awt.Graphics g,
                                           int x, int y, int width, int height) {
            g.setColor(UIManager.getColor("Separator.foreground"));
            g.drawLine(x + width - 1, y, x + width - 1, y + height);
        }
    }
}
