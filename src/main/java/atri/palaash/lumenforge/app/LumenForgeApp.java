package atri.palaash.lumenforge.app;

import atri.palaash.lumenforge.inference.InferenceService;
import atri.palaash.lumenforge.inference.ServiceFactory;
import atri.palaash.lumenforge.model.ModelRegistry;
import atri.palaash.lumenforge.model.TaskType;
import atri.palaash.lumenforge.storage.ModelDownloader;
import atri.palaash.lumenforge.storage.ModelStorage;
import atri.palaash.lumenforge.ui.MainFrame;
import atri.palaash.lumenforge.ui.NativeLookAndFeel;

import javax.swing.SwingUtilities;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class LumenForgeApp {

    public static void main(String[] args) {
        configureDesktopIntegration();

        ExecutorService workerPool = Executors.newVirtualThreadPerTaskExecutor();
        ModelStorage storage = new ModelStorage();
        ModelRegistry registry = new ModelRegistry();
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(20))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
        ModelDownloader downloader = new ModelDownloader(httpClient, storage, workerPool);
        Map<TaskType, InferenceService> services = new EnumMap<>(TaskType.class);
        ServiceFactory serviceFactory = new ServiceFactory(registry, storage, workerPool);
        for (TaskType type : TaskType.values()) {
            services.put(type, serviceFactory.create(type));
        }

        Runtime.getRuntime().addShutdownHook(new Thread(workerPool::close));

        SwingUtilities.invokeLater(() -> {
            NativeLookAndFeel.apply();
            MainFrame frame = new MainFrame(registry, storage, downloader, services);
            frame.setVisible(true);
        });
    }

    private static void configureDesktopIntegration() {
        String osName = System.getProperty("os.name", "").toLowerCase();
        if (osName.contains("mac")) {
            System.setProperty("apple.laf.useScreenMenuBar", "true");
            System.setProperty("apple.awt.application.name", "LumenForge");
            System.setProperty("apple.awt.application.appearance", "system");
            // Enable FlatLaf embedded title bar on macOS for a unified toolbar look
            System.setProperty("flatlaf.menuBarEmbedded", "true");
            System.setProperty("flatlaf.useWindowDecorations", "false");
        }
    }
}
