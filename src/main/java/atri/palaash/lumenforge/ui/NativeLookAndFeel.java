package atri.palaash.lumenforge.ui;

import com.formdev.flatlaf.FlatDarkLaf;
import com.formdev.flatlaf.FlatLightLaf;

import javax.swing.SwingUtilities;
import javax.swing.UIManager;
import java.awt.Insets;
import java.awt.Window;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Locale;

/**
 * Centralized look-and-feel management with dark mode auto-detection
 * and runtime toggling. Uses FlatLaf on every platform, with macOS-specific
 * themes when available (FlatMacLightLaf / FlatMacDarkLaf).
 */
public final class NativeLookAndFeel {

    private static boolean darkMode;

    private NativeLookAndFeel() {
    }

    /** Apply the look-and-feel once at startup (auto-detects dark mode). */
    public static void apply() {
        darkMode = isSystemDarkMode();
        applyTheme(darkMode);
    }

    public static boolean isDarkMode() {
        return darkMode;
    }

    /** Switch theme at runtime and refresh every open window. */
    public static void setDarkMode(boolean dark) {
        if (darkMode == dark) {
            return;
        }
        darkMode = dark;
        applyTheme(dark);
        for (Window window : Window.getWindows()) {
            SwingUtilities.updateComponentTreeUI(window);
            window.repaint();
        }
    }

    public static void toggleDarkMode() {
        setDarkMode(!darkMode);
    }

    /* ------------------------------------------------------------------ */

    private static void applyTheme(boolean dark) {
        try {
            String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
            if (os.contains("mac")) {
                try {
                    String className = dark
                            ? "com.formdev.flatlaf.themes.FlatMacDarkLaf"
                            : "com.formdev.flatlaf.themes.FlatMacLightLaf";
                    Class.forName(className).getMethod("setup").invoke(null);
                    configureDefaults();
                    return;
                } catch (Exception ignored) {
                    // Mac themes unavailable in this FlatLaf build — fall through.
                }
            }
            if (dark) {
                FlatDarkLaf.setup();
            } else {
                FlatLightLaf.setup();
            }
            configureDefaults();
        } catch (Exception ex) {
            try {
                UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
            } catch (Exception fallback) {
                throw new IllegalStateException("Unable to initialize look and feel", fallback);
            }
        }
    }

    /** Tweak FlatLaf UI defaults for a polished, modern feel. */
    private static void configureDefaults() {
        UIManager.put("Component.arc", 8);
        UIManager.put("Button.arc", 8);
        UIManager.put("TextComponent.arc", 6);
        UIManager.put("CheckBox.arc", 4);
        UIManager.put("ScrollBar.thumbArc", 999);
        UIManager.put("ScrollBar.thumbInsets", new Insets(2, 2, 2, 2));
        UIManager.put("ScrollPane.smoothScrolling", true);
        UIManager.put("TitlePane.unifiedBackground", true);
        UIManager.put("Table.showHorizontalLines", true);
        UIManager.put("Table.showVerticalLines", false);
    }

    /* ------------------------------------------------------------------ */

    /** Best-effort detection of the OS-level dark mode setting. */
    public static boolean isSystemDarkMode() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);

        if (os.contains("mac")) {
            try {
                Process process = new ProcessBuilder(
                        "defaults", "read", "-g", "AppleInterfaceStyle").start();
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String line = reader.readLine();
                    process.waitFor();
                    return line != null && line.toLowerCase(Locale.ROOT).contains("dark");
                }
            } catch (Exception ignored) {
            }
        }

        if (os.contains("win")) {
            try {
                Process process = new ProcessBuilder(
                        "reg", "query",
                        "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize",
                        "/v", "AppsUseLightTheme").start();
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String output = reader.lines().reduce("", (a, b) -> a + " " + b);
                    process.waitFor();
                    return output.contains("0x0");
                }
            } catch (Exception ignored) {
            }
        }

        // Linux / GTK
        String gtkTheme = System.getenv("GTK_THEME");
        if (gtkTheme != null && gtkTheme.toLowerCase(Locale.ROOT).contains("dark")) {
            return true;
        }

        String override = System.getProperty("lumenforge.theme", "").toLowerCase(Locale.ROOT);
        return "dark".equals(override);
    }
}
