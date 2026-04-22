package atri.palaash.jforge.ui;

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
        /* Shape — rounded corners following macOS conventions */
        UIManager.put("Component.arc", 8);
        UIManager.put("Button.arc", 8);
        UIManager.put("TextComponent.arc", 6);
        UIManager.put("CheckBox.arc", 4);

        /* Scrollbar — thin thumb, rounded */
        UIManager.put("ScrollBar.thumbArc", 999);
        UIManager.put("ScrollBar.thumbInsets", new Insets(2, 2, 2, 2));
        UIManager.put("ScrollBar.width", 10);
        UIManager.put("ScrollPane.smoothScrolling", true);

        /* Title bar — unified background (macOS style) */
        UIManager.put("TitlePane.unifiedBackground", true);

        /* Table */
        UIManager.put("Table.showHorizontalLines", true);
        UIManager.put("Table.showVerticalLines", false);
        UIManager.put("Table.intercellSpacing", new java.awt.Dimension(0, 1));

        /* Focus indicators — subtle blue ring */
        UIManager.put("Component.focusWidth", 2);
        UIManager.put("Component.innerFocusWidth", 0);

        /* Button padding — Apple-style generous padding */
        UIManager.put("Button.margin", new Insets(4, 14, 4, 14));

        /* ComboBox */
        UIManager.put("ComboBox.padding", new Insets(3, 6, 3, 6));

        /* List — sidebar-friendly defaults */
        UIManager.put("List.selectionArc", 8);

        /* Separator */
        UIManager.put("Separator.stripeWidth", 1);
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

        String override = System.getProperty("jforge.theme", "").toLowerCase(Locale.ROOT);
        return "dark".equals(override);
    }
}
