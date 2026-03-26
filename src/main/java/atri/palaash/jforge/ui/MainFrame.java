package atri.palaash.jforge.ui;

import atri.palaash.jforge.inference.InferenceService;
import atri.palaash.jforge.model.ModelDescriptor;
import atri.palaash.jforge.model.ModelRegistry;
import atri.palaash.jforge.model.TaskType;
import atri.palaash.jforge.storage.ModelDownloader;
import atri.palaash.jforge.storage.ModelStorage;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.DefaultListCellRenderer;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.KeyStroke;
import javax.swing.ListSelectionModel;
import javax.swing.UIManager;
import javax.swing.border.Border;
import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Insets;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.event.KeyEvent;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Main application window — Apple Human Interface Guidelines–inspired layout.
 * <ul>
 *   <li>Left sidebar with grouped navigation (Generate / Img2Img / Upscale) and Models at the bottom</li>
 *   <li>Card-layout content area</li>
 *   <li>Menu bar: View, Inference, Tools</li>
 * </ul>
 */
public class MainFrame extends JFrame {

    /* Card names */
    private static final String CARD_IMAGE_STUDIO = "Image Studio";
    private static final String CARD_MODELS   = "Models";

    private final CardLayout cardLayout = new CardLayout();
    private final JPanel contentPanel = new JPanel(cardLayout);

    private final TextToImagePanel textToImagePanel;
    private final TextToVideoPanel textToVideoPanel;
    private final ImageUpscalePanel imageUpscalePanel;
    private final Img2ImgPanel img2ImgPanel;
    private final ImageStudioPanel imageStudioPanel;
    private final ModelManagerPanel modelManagerPanel;
    private final JLabel statusBarLabel;

    /* Sidebar lists */
    private JList<String> imageList;
    private JList<String> videoList;
    private JList<String> managementList;

    /* GPU state (shared across panels via supplier) */
    private boolean gpuEnabled = true;

    public MainFrame(ModelRegistry registry,
                     ModelStorage storage,
                     ModelDownloader downloader,
                     Map<TaskType, InferenceService> services) {

        setTitle("JForge");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1060, 740);
        setMinimumSize(new Dimension(800, 540));
        setLocationRelativeTo(null);

        /* ── Panels ──────────────────────────────────────────────── */
        List<ModelDescriptor> t2iModels = registry.allModels().stream()
                .filter(m -> m.taskType() == TaskType.TEXT_TO_IMAGE)
                .collect(Collectors.toList());
        List<ModelDescriptor> upscaleModels = registry.allModels().stream()
                .filter(m -> m.taskType() == TaskType.IMAGE_UPSCALE)
                .collect(Collectors.toList());
        List<ModelDescriptor> img2imgModels = registry.allModels().stream()
                .filter(m -> m.taskType() == TaskType.IMAGE_TO_IMAGE)
                .collect(Collectors.toList());

        List<ModelDescriptor> t2vModels = registry.allModels().stream()
                .filter(m -> m.taskType() == TaskType.TEXT_TO_VIDEO)
                .collect(Collectors.toList());

        textToImagePanel = new TextToImagePanel(
                t2iModels, downloader, services.get(TaskType.TEXT_TO_IMAGE));
        textToVideoPanel = new TextToVideoPanel(t2vModels, services.get(TaskType.TEXT_TO_VIDEO), storage, downloader);
        imageUpscalePanel = new ImageUpscalePanel(
                upscaleModels, downloader, services.get(TaskType.IMAGE_UPSCALE));
        img2ImgPanel = new Img2ImgPanel(
                img2imgModels, downloader, services.get(TaskType.IMAGE_TO_IMAGE));

        /* Inject storage so panels filter to downloaded-only models */
        Runnable goToModels = () -> {
            if (managementList != null) switchToCard(CARD_MODELS, managementList, 0);
        };
        textToImagePanel.setModelStorage(storage);
        textToImagePanel.setOpenModelManager(goToModels);
        imageUpscalePanel.setModelStorage(storage);
        imageUpscalePanel.setOpenModelManager(goToModels);
        img2ImgPanel.setModelStorage(storage);
        img2ImgPanel.setOpenModelManager(goToModels);

        textToImagePanel.setGpuSupplier(() -> gpuEnabled);
        imageUpscalePanel.setGpuSupplier(() -> gpuEnabled);
        img2ImgPanel.setGpuSupplier(() -> gpuEnabled);

        /* Apply initial downloaded-only filter */
        textToImagePanel.updateModels(t2iModels);
        imageUpscalePanel.updateModels(upscaleModels);
        img2ImgPanel.updateModels(img2imgModels);

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
            img2ImgPanel.updateModels(
                    registry.allModels().stream()
                            .filter(m -> m.taskType() == TaskType.IMAGE_TO_IMAGE)
                            .collect(Collectors.toList()));
            textToVideoPanel.updateModels(
                    registry.allModels().stream()
                            .filter(m -> m.taskType() == TaskType.TEXT_TO_VIDEO)
                            .collect(Collectors.toList()));
        });

        /* ── Content cards ─────────────────────────────────────── */
        imageStudioPanel = new ImageStudioPanel(textToImagePanel, img2ImgPanel, imageUpscalePanel);
        contentPanel.add(imageStudioPanel, CARD_IMAGE_STUDIO);
        contentPanel.add(textToVideoPanel, "Video Studio");
        contentPanel.add(modelManagerPanel, CARD_MODELS);

        /* ── Sidebar ─────────────────────────────────────────────── */
        // Create all lists first, then wire listeners
        String[] imageItems = {CARD_IMAGE_STUDIO};
        imageList = new JList<>(imageItems);
        imageList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        imageList.setFixedCellHeight(32);
        imageList.setCellRenderer(new SidebarRenderer());
        imageList.setOpaque(false);

        String[] videoItems = {"Video Studio"};
        videoList = new JList<>(videoItems);
        videoList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        videoList.setFixedCellHeight(32);
        videoList.setCellRenderer(new SidebarRenderer());
        videoList.setOpaque(false);

        String[] managementItems = {CARD_MODELS};
        managementList = new JList<>(managementItems);
        managementList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        managementList.setFixedCellHeight(32);
        managementList.setCellRenderer(new SidebarRenderer());
        managementList.setOpaque(false);

        // Wire listeners after all lists exist
        imageList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting() && imageList.getSelectedIndex() >= 0) {
                String val = imageList.getSelectedValue();
                switchToCard(val, imageList, imageList.getSelectedIndex());
                if (!val.equals("Video Studio")) videoList.clearSelection();
                if (!val.equals(CARD_MODELS)) managementList.clearSelection();
            }
        });
        videoList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting() && videoList.getSelectedIndex() >= 0) {
                String val = videoList.getSelectedValue();
                switchToCard(val, videoList, videoList.getSelectedIndex());
                if (!val.equals(CARD_IMAGE_STUDIO)) imageList.clearSelection();
                if (!val.equals(CARD_MODELS)) managementList.clearSelection();
            }
        });
        managementList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting() && managementList.getSelectedIndex() >= 0) {
                videoList.clearSelection();
                cardLayout.show(contentPanel, managementList.getSelectedValue());
            }
        });

        // Set initial selection after listeners are wired
        imageList.setSelectedIndex(0);

        JPanel sidebarPanel = new JPanel(new BorderLayout(0, 0));
        sidebarPanel.setPreferredSize(new Dimension(170, 0));
        sidebarPanel.setBorder(new SeparatorBorder());

        // Brand label
        JLabel brand = new JLabel("JForge");
        brand.setFont(brand.getFont().deriveFont(Font.BOLD, 13f));
        brand.setBorder(BorderFactory.createEmptyBorder(14, 20, 12, 20));
        sidebarPanel.add(brand, BorderLayout.NORTH);

        // Image section
        JPanel imageSection = new JPanel();
        imageSection.setLayout(new BoxLayout(imageSection, BoxLayout.Y_AXIS));
        imageSection.setOpaque(false);
        imageSection.setBorder(BorderFactory.createEmptyBorder(4, 0, 0, 0));

        JLabel imageHeader = new JLabel("IMAGE");
        imageHeader.setFont(imageHeader.getFont().deriveFont(Font.BOLD, 10f));
        imageHeader.setForeground(UIManager.getColor("Label.disabledForeground"));
        imageHeader.setBorder(BorderFactory.createEmptyBorder(4, 20, 4, 20));
        imageHeader.setAlignmentX(Component.LEFT_ALIGNMENT);
        imageSection.add(imageHeader);
        imageList.setAlignmentX(Component.LEFT_ALIGNMENT);
        imageSection.add(imageList);

        // Video section
        JPanel videoSection = new JPanel();
        videoSection.setLayout(new BoxLayout(videoSection, BoxLayout.Y_AXIS));
        videoSection.setOpaque(false);
        videoSection.setBorder(BorderFactory.createEmptyBorder(4, 0, 0, 0));

        JLabel videoHeader = new JLabel("VIDEO");
        videoHeader.setFont(videoHeader.getFont().deriveFont(Font.BOLD, 10f));
        videoHeader.setForeground(UIManager.getColor("Label.disabledForeground"));
        videoHeader.setBorder(BorderFactory.createEmptyBorder(8, 20, 4, 20));
        videoHeader.setAlignmentX(Component.LEFT_ALIGNMENT);
        videoSection.add(videoHeader);
        videoList.setAlignmentX(Component.LEFT_ALIGNMENT);
        videoSection.add(videoList);

        // Management section
        JPanel managementSection = new JPanel();
        managementSection.setLayout(new BoxLayout(managementSection, BoxLayout.Y_AXIS));
        managementSection.setOpaque(false);

        JLabel managementHeader = new JLabel("MANAGEMENT");
        managementHeader.setFont(managementHeader.getFont().deriveFont(Font.BOLD, 10f));
        managementHeader.setForeground(UIManager.getColor("Label.disabledForeground"));
        managementHeader.setBorder(BorderFactory.createEmptyBorder(8, 20, 4, 20));
        managementHeader.setAlignmentX(Component.LEFT_ALIGNMENT);
        managementSection.add(managementHeader);
        managementList.setAlignmentX(Component.LEFT_ALIGNMENT);
        managementSection.add(managementList);

        // Bottom panel: management
        JPanel bottomSection = new JPanel();
        bottomSection.setLayout(new BoxLayout(bottomSection, BoxLayout.Y_AXIS));
        bottomSection.setOpaque(false);
        bottomSection.setBorder(BorderFactory.createEmptyBorder(0, 0, 12, 0));
        bottomSection.add(managementSection);

        // Combine: workflow on top, management at bottom
        JPanel sidebarContent = new JPanel(new BorderLayout());
        sidebarContent.setOpaque(false);
        JPanel topSections = new JPanel();
        topSections.setLayout(new BoxLayout(topSections, BoxLayout.Y_AXIS));
        topSections.setOpaque(false);
        topSections.add(imageSection);
        topSections.add(videoSection);
        sidebarContent.add(topSections, BorderLayout.NORTH);
        sidebarContent.add(bottomSection, BorderLayout.SOUTH);

        sidebarPanel.add(sidebarContent, BorderLayout.CENTER);

        /* ── Status bar ──────────────────────────────────────────── */
        statusBarLabel = new JLabel(detectEpInfo());
        statusBarLabel.setFont(statusBarLabel.getFont().deriveFont(Font.PLAIN, 11f));
        statusBarLabel.setForeground(UIManager.getColor("Label.disabledForeground"));
        statusBarLabel.setBorder(BorderFactory.createCompoundBorder(
                new SeparatorBorder(true),
                BorderFactory.createEmptyBorder(4, 16, 4, 16)));

        /* ── Root assembly ───────────────────────────────────────── */
        JPanel root = new JPanel(new BorderLayout());
        root.add(sidebarPanel, BorderLayout.WEST);
        root.add(contentPanel, BorderLayout.CENTER);
        root.add(statusBarLabel, BorderLayout.SOUTH);
        setContentPane(root);

        /* ── Menu bar ────────────────────────────────────────────── */
        setJMenuBar(buildMenuBar());

        /* Select first card */
        cardLayout.show(contentPanel, CARD_IMAGE_STUDIO);
    }

    /* ================================================================== */
    /*  Menu bar                                                           */
    /* ================================================================== */

    private JMenuBar buildMenuBar() {
        JMenuBar bar = new JMenuBar();
        int menuMask = Toolkit.getDefaultToolkit().getMenuShortcutKeyMaskEx();

        /* View */
        JMenu viewMenu = new JMenu("View");

        JCheckBoxMenuItem darkModeItem = new JCheckBoxMenuItem("Dark Mode");
        darkModeItem.setSelected(NativeLookAndFeel.isDarkMode());
        darkModeItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_D, menuMask | KeyEvent.SHIFT_DOWN_MASK));
        darkModeItem.addActionListener(e -> {
            NativeLookAndFeel.setDarkMode(darkModeItem.isSelected());
            repaintAll();
        });
        viewMenu.add(darkModeItem);

        viewMenu.addSeparator();

        JMenuItem showGenerate = new JMenuItem("Generate");
        showGenerate.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_1, menuMask));
        showGenerate.addActionListener(e -> {
            imageStudioPanel.selectTab(0);
            switchToCard(CARD_IMAGE_STUDIO, imageList, 0);
        });
        viewMenu.add(showGenerate);

        JMenuItem showImg2Img = new JMenuItem("Img2Img");
        showImg2Img.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_2, menuMask));
        showImg2Img.addActionListener(e -> {
            imageStudioPanel.selectTab(1);
            switchToCard(CARD_IMAGE_STUDIO, imageList, 0);
        });
        viewMenu.add(showImg2Img);

        JMenuItem showUpscale = new JMenuItem("Upscale");
        showUpscale.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_3, menuMask));
        showUpscale.addActionListener(e -> {
            imageStudioPanel.selectTab(2);
            switchToCard(CARD_IMAGE_STUDIO, imageList, 0);
        });
        viewMenu.add(showUpscale);
        
        JMenuItem showVideo = new JMenuItem("Video");
        showVideo.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_4, menuMask));
        showVideo.addActionListener(e -> switchToCard("Video", videoList, 0));
        viewMenu.add(showVideo);

        viewMenu.addSeparator();

        JMenuItem showModels = new JMenuItem("Models");
        showModels.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_M, menuMask));
        showModels.addActionListener(e -> switchToCard(CARD_MODELS, managementList, 0));
        viewMenu.add(showModels);

        bar.add(viewMenu);

        /* Inference */
        JMenu inferenceMenu = new JMenu("Inference");
        JCheckBoxMenuItem gpuItem = new JCheckBoxMenuItem("Use GPU");
        gpuItem.setSelected(gpuEnabled);
        gpuItem.addActionListener(e -> {
            gpuEnabled = gpuItem.isSelected();
            statusBarLabel.setText(detectEpInfo());
        });
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
    /*  Navigation helpers                                                 */
    /* ================================================================== */

    private void switchToCard(String card, JList<String> targetList, int index) {
        cardLayout.show(contentPanel, card);
        if (targetList != imageList) imageList.clearSelection();
        if (targetList != videoList) videoList.clearSelection();
        if (targetList != managementList) managementList.clearSelection();
        targetList.setSelectedIndex(index);
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
    /*  Sidebar renderer — HIG-inspired with rounded selection             */
    /* ================================================================== */

    private static class SidebarRenderer extends DefaultListCellRenderer {
        @Override
        public Component getListCellRendererComponent(JList<?> list, Object value,
                                                       int index, boolean isSelected,
                                                       boolean cellHasFocus) {
            JLabel lbl = (JLabel) super.getListCellRendererComponent(
                    list, value, index, isSelected, cellHasFocus);
            lbl.setFont(lbl.getFont().deriveFont(Font.PLAIN, 13f));
            lbl.setOpaque(false);
            lbl.setBorder(BorderFactory.createEmptyBorder(0, 20, 0, 20));
            if (isSelected) {
                lbl.setForeground(UIManager.getColor("List.selectionForeground"));
            }
            return new RoundedSelectionPanel(lbl, isSelected);
        }
    }

    /**
     *  Wrapper panel that draws a rounded-rect selection highlight behind
     *  the label — mimics the native macOS sidebar selection style.
     */
    private static class RoundedSelectionPanel extends JPanel {
        private final boolean selected;
        RoundedSelectionPanel(JLabel content, boolean selected) {
            super(new BorderLayout());
            this.selected = selected;
            setOpaque(false);
            setBorder(BorderFactory.createEmptyBorder(1, 8, 1, 8));
            add(content, BorderLayout.CENTER);
        }
        @Override
        protected void paintComponent(Graphics g) {
            if (selected) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                Color bg = UIManager.getColor("List.selectionBackground");
                if (bg == null) bg = new Color(56, 117, 215);
                g2.setColor(bg);
                g2.fillRoundRect(0, 0, getWidth(), getHeight(), 8, 8);
                g2.dispose();
            }
            super.paintComponent(g);
        }
    }

    /* ================================================================== */
    /*  Status bar EP detection                                            */
    /* ================================================================== */

    private String detectEpInfo() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String ep;
        if (gpuEnabled) {
            if (os.contains("mac")) { ep = "CoreML"; }
            else if (os.contains("win")) { ep = "DirectML / CUDA"; }
            else { ep = "CUDA / ROCm"; }
        } else {
            ep = "CPU";
        }
        int cores = Runtime.getRuntime().availableProcessors();
        long mem = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        return "EP: " + ep + "  \u2502  Cores: " + cores + "  \u2502  Heap: " + mem
                + " MB  \u2502  Java " + Runtime.version();
    }

    /* ================================================================== */
    /*  Separator border (top or right edge)                               */
    /* ================================================================== */

    private static class SeparatorBorder implements Border {
        private final boolean top;
        SeparatorBorder() { this(false); }
        SeparatorBorder(boolean top) { this.top = top; }
        @Override public Insets getBorderInsets(Component c) {
            return top ? new Insets(1, 0, 0, 0) : new Insets(0, 0, 0, 1);
        }
        @Override public boolean isBorderOpaque() { return true; }
        @Override public void paintBorder(Component c, Graphics g,
                                           int x, int y, int width, int height) {
            g.setColor(UIManager.getColor("Separator.foreground"));
            if (top) {
                g.drawLine(x, y, x + width, y);
            } else {
                g.drawLine(x + width - 1, y, x + width - 1, y + height);
            }
        }
    }
}
