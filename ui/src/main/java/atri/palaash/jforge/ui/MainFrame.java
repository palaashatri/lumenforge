package atri.palaash.jforge.ui;

import atri.palaash.jforge.inference.InferenceService;
import atri.palaash.jforge.model.ModelDescriptor;
import atri.palaash.jforge.model.ModelRegistry;
import atri.palaash.jforge.model.TaskType;
import atri.palaash.jforge.models.ForgeModelRegistry;
import atri.palaash.jforge.storage.ModelDownloader;
import atri.palaash.jforge.storage.ModelStorage;

import javax.swing.BorderFactory;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Toolkit;
import java.awt.event.KeyEvent;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Main application window with top task tabs and a bottom history strip.
 */
public class MainFrame extends JFrame {

    private final ModelRegistry modelRegistry;

    private final ForgeWorkspacePanel textToImagePanel;
    private final ForgeWorkspacePanel img2ImgPanel;
    private final ForgeWorkspacePanel inpaintPanel;
    private final ForgeWorkspacePanel imageUpscalePanel;
    private final AppSettingsPanel settingsPanel;
    private final ModelManagerPanel modelManagerPanel;

    private final JTabbedPane taskTabs;
    private final HistoryStripPanel historyStripPanel;
    private final JLabel statusBarLabel;

    private boolean gpuEnabled = true;
    private final int modelsTabIndex;

    public MainFrame(ModelRegistry registry,
                     ModelStorage storage,
                     ModelDownloader downloader,
                     Map<TaskType, InferenceService> services) {

        this.modelRegistry = registry;

        setTitle("JForge");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1180, 820);
        setMinimumSize(new Dimension(980, 640));
        setLocationRelativeTo(null);

        List<ModelDescriptor> textToImageModels = filterModels(TaskType.TEXT_TO_IMAGE);
        List<ModelDescriptor> upscaleModels = filterModels(TaskType.IMAGE_UPSCALE);
        List<ModelDescriptor> img2imgModels = filterModels(TaskType.IMAGE_TO_IMAGE);
        if (img2imgModels.isEmpty()) {
            img2imgModels = textToImageModels;
        }

        textToImagePanel = new ForgeWorkspacePanel(
            ForgeWorkspacePanel.Mode.TEXT_TO_IMAGE,
            textToImageModels,
            downloader,
            services.get(TaskType.TEXT_TO_IMAGE));
        img2ImgPanel = new ForgeWorkspacePanel(
            ForgeWorkspacePanel.Mode.IMAGE_TO_IMAGE,
            img2imgModels,
            downloader,
            services.get(TaskType.IMAGE_TO_IMAGE));
        inpaintPanel = new ForgeWorkspacePanel(
            ForgeWorkspacePanel.Mode.INPAINT,
            img2imgModels,
            downloader,
            services.get(TaskType.IMAGE_TO_IMAGE));
        imageUpscalePanel = new ForgeWorkspacePanel(
            ForgeWorkspacePanel.Mode.UPSCALE,
            upscaleModels,
            downloader,
            services.get(TaskType.IMAGE_UPSCALE));

        ForgeModelRegistry forgeModelRegistry = new ForgeModelRegistry(registry, storage);
        textToImagePanel.setForgeModelRegistry(forgeModelRegistry);
        img2ImgPanel.setForgeModelRegistry(forgeModelRegistry);
        inpaintPanel.setForgeModelRegistry(forgeModelRegistry);
        imageUpscalePanel.setForgeModelRegistry(forgeModelRegistry);

        textToImagePanel.setModelStorage(storage);
        img2ImgPanel.setModelStorage(storage);
        inpaintPanel.setModelStorage(storage);
        imageUpscalePanel.setModelStorage(storage);

        textToImagePanel.setGpuSupplier(() -> gpuEnabled);
        img2ImgPanel.setGpuSupplier(() -> gpuEnabled);
        inpaintPanel.setGpuSupplier(() -> gpuEnabled);
        imageUpscalePanel.setGpuSupplier(() -> gpuEnabled);

        modelManagerPanel = new ModelManagerPanel(registry, storage, downloader);
        historyStripPanel = new HistoryStripPanel();
        settingsPanel = new AppSettingsPanel(historyStripPanel::reload);

        taskTabs = new JTabbedPane();
        taskTabs.setFont(taskTabs.getFont().deriveFont(Font.PLAIN, 13f));
        taskTabs.addTab("Text -> Image", textToImagePanel);
        taskTabs.addTab("Image -> Image", img2ImgPanel);
        taskTabs.addTab("Inpaint", inpaintPanel);
        taskTabs.addTab("Upscale", imageUpscalePanel);
        taskTabs.addTab("Settings", settingsPanel);
        taskTabs.addTab("Models", modelManagerPanel);
        modelsTabIndex = taskTabs.getTabCount() - 1;

        Runnable openModels = () -> switchToTab(modelsTabIndex);
        textToImagePanel.setOpenModelManager(openModels);
        img2ImgPanel.setOpenModelManager(openModels);
        inpaintPanel.setOpenModelManager(openModels);
        imageUpscalePanel.setOpenModelManager(openModels);

        modelManagerPanel.setOnModelsUpdated(() -> {
            refreshModelLists();
            historyStripPanel.reload();
        });

        statusBarLabel = new JLabel();
        statusBarLabel.setFont(statusBarLabel.getFont().deriveFont(Font.PLAIN, 11f));
        statusBarLabel.setBorder(BorderFactory.createEmptyBorder(4, 10, 4, 10));

        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.add(statusBarLabel, BorderLayout.NORTH);
        bottomPanel.add(historyStripPanel, BorderLayout.CENTER);

        JPanel root = new JPanel(new BorderLayout());
        root.add(taskTabs, BorderLayout.CENTER);
        root.add(bottomPanel, BorderLayout.SOUTH);
        setContentPane(root);

        setJMenuBar(buildMenuBar());

        taskTabs.addChangeListener(e -> updateStatusBar());
        updateStatusBar();

        Timer historyRefresh = new Timer(3500, e -> historyStripPanel.reload());
        historyRefresh.start();
    }

    private JMenuBar buildMenuBar() {
        JMenuBar bar = new JMenuBar();
        int menuMask = Toolkit.getDefaultToolkit().getMenuShortcutKeyMaskEx();

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

        JMenuItem txt2ImgTab = new JMenuItem("Text -> Image");
        txt2ImgTab.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_1, menuMask));
        txt2ImgTab.addActionListener(e -> switchToTab(0));
        viewMenu.add(txt2ImgTab);

        JMenuItem img2ImgTab = new JMenuItem("Image -> Image");
        img2ImgTab.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_2, menuMask));
        img2ImgTab.addActionListener(e -> switchToTab(1));
        viewMenu.add(img2ImgTab);

        JMenuItem inpaintTab = new JMenuItem("Inpaint");
        inpaintTab.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_3, menuMask));
        inpaintTab.addActionListener(e -> switchToTab(2));
        viewMenu.add(inpaintTab);

        JMenuItem upscaleTab = new JMenuItem("Upscale");
        upscaleTab.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_4, menuMask));
        upscaleTab.addActionListener(e -> switchToTab(3));
        viewMenu.add(upscaleTab);

        JMenuItem settingsTab = new JMenuItem("Settings");
        settingsTab.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_5, menuMask));
        settingsTab.addActionListener(e -> switchToTab(4));
        viewMenu.add(settingsTab);

        viewMenu.addSeparator();

        JMenuItem modelsTab = new JMenuItem("Models");
        modelsTab.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_M, menuMask));
        modelsTab.addActionListener(e -> switchToTab(modelsTabIndex));
        viewMenu.add(modelsTab);

        bar.add(viewMenu);

        JMenu inferenceMenu = new JMenu("Inference");
        JCheckBoxMenuItem gpuItem = new JCheckBoxMenuItem("Use GPU");
        gpuItem.setSelected(gpuEnabled);
        gpuItem.addActionListener(e -> {
            gpuEnabled = gpuItem.isSelected();
            updateStatusBar();
        });
        inferenceMenu.add(gpuItem);
        bar.add(inferenceMenu);

        JMenu toolsMenu = new JMenu("Tools");

        JMenuItem refreshHistory = new JMenuItem("Refresh History Strip");
        refreshHistory.addActionListener(e -> historyStripPanel.reload());
        toolsMenu.add(refreshHistory);

        bar.add(toolsMenu);
        return bar;
    }

    private void refreshModelLists() {
        List<ModelDescriptor> textToImageModels = filterModels(TaskType.TEXT_TO_IMAGE);
        List<ModelDescriptor> upscaleModels = filterModels(TaskType.IMAGE_UPSCALE);
        List<ModelDescriptor> img2imgModels = filterModels(TaskType.IMAGE_TO_IMAGE);
        if (img2imgModels.isEmpty()) {
            img2imgModels = textToImageModels;
        }

        textToImagePanel.updateModels(textToImageModels);
        img2ImgPanel.updateModels(img2imgModels);
        inpaintPanel.updateModels(img2imgModels);
        imageUpscalePanel.updateModels(upscaleModels);
    }

    private List<ModelDescriptor> filterModels(TaskType type) {
        return modelRegistry.allModels().stream()
                .filter(model -> model.taskType() == type)
                .collect(Collectors.toList());
    }

    private void switchToTab(int tabIndex) {
        if (tabIndex < 0 || tabIndex >= taskTabs.getTabCount()) {
            return;
        }
        taskTabs.setSelectedIndex(tabIndex);
        updateStatusBar();
    }

    private void repaintAll() {
        for (java.awt.Window window : java.awt.Window.getWindows()) {
            SwingUtilities.updateComponentTreeUI(window);
        }
    }

    private void updateStatusBar() {
        String tabName = taskTabs.getSelectedIndex() >= 0
                ? taskTabs.getTitleAt(taskTabs.getSelectedIndex())
                : "Ready";
        statusBarLabel.setText("Task: " + tabName + " | " + detectEpInfo());
    }

    private String detectEpInfo() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String ep;
        if (gpuEnabled) {
            if (os.contains("mac")) {
                ep = "CoreML";
            } else if (os.contains("win")) {
                ep = "DirectML/CUDA";
            } else {
                ep = "CUDA/ROCm";
            }
        } else {
            ep = "CPU";
        }

        int cores = Runtime.getRuntime().availableProcessors();
        long mem = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        return "EP: " + ep + " | Cores: " + cores + " | Heap: " + mem + " MB | Java " + Runtime.version();
    }
}
