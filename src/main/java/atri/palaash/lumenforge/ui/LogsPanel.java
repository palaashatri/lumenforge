package atri.palaash.lumenforge.ui;

import atri.palaash.lumenforge.ui.AppLogger.Channel;
import atri.palaash.lumenforge.ui.AppLogger.LogEntry;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JToggleButton;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Logs panel with two toggle-filtered views: <b>Application</b> and <b>Model</b>.
 * <p>
 * Entries are appended in real-time via {@link AppLogger} and auto-scroll
 * to the bottom. A <i>Clear</i> button wipes the visible log.
 */
public class LogsPanel extends JPanel {

    private final JTextArea logArea;
    private final List<LogEntry> allEntries = new ArrayList<>();
    private boolean showApp = true;
    private boolean showModel = true;

    /* Coalesce rapid-fire updates (e.g. per-step progress) to ~60 fps. */
    private boolean dirty = false;
    private final Timer coalesceTimer;

    public LogsPanel() {
        super(new BorderLayout(0, 0));

        /* ── Log text area (init first — referenced by toolbar buttons) ── */
        logArea = new JTextArea();
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        logArea.setLineWrap(true);
        logArea.setWrapStyleWord(true);
        JScrollPane scroll = new JScrollPane(logArea);
        scroll.setBorder(BorderFactory.createEmptyBorder());

        /* ── Toolbar ─────────────────────────────────────────────── */
        JPanel toolbar = new JPanel();
        toolbar.setLayout(new BoxLayout(toolbar, BoxLayout.X_AXIS));
        toolbar.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));

        JToggleButton appBtn = new JToggleButton("Application", showApp);
        JToggleButton modelBtn = new JToggleButton("Model", showModel);
        appBtn.setFocusable(false);
        modelBtn.setFocusable(false);
        appBtn.addActionListener(e -> { showApp = appBtn.isSelected(); rebuild(); });
        modelBtn.addActionListener(e -> { showModel = modelBtn.isSelected(); rebuild(); });

        JButton clearBtn = new JButton("Clear");
        clearBtn.setFocusable(false);
        clearBtn.addActionListener(e -> {
            synchronized (allEntries) { allEntries.clear(); }
            logArea.setText("");
        });

        toolbar.add(appBtn);
        toolbar.add(Box.createRigidArea(new Dimension(4, 0)));
        toolbar.add(modelBtn);
        toolbar.add(Box.createHorizontalGlue());
        toolbar.add(clearBtn);

        add(toolbar, BorderLayout.NORTH);
        add(scroll, BorderLayout.CENTER);

        /* ── Wire up listener ────────────────────────────────────── */
        Consumer<LogEntry> listener = entry -> {
            synchronized (allEntries) { allEntries.add(entry); }
            scheduleDirty();
        };
        AppLogger.addListener(listener);

        /* Coalesce timer fires every 16 ms (~60 fps) if dirty */
        coalesceTimer = new Timer(16, e -> {
            if (dirty) {
                dirty = false;
                rebuild();
            }
        });
        coalesceTimer.setRepeats(true);
        coalesceTimer.start();
    }

    private void scheduleDirty() {
        dirty = true;
    }

    /** Rebuild the text area contents from the filtered entry list. */
    private void rebuild() {
        SwingUtilities.invokeLater(() -> {
            StringBuilder sb = new StringBuilder();
            List<LogEntry> snapshot;
            synchronized (allEntries) { snapshot = new ArrayList<>(allEntries); }
            for (LogEntry entry : snapshot) {
                if (entry.channel() == Channel.APPLICATION && !showApp) continue;
                if (entry.channel() == Channel.MODEL && !showModel) continue;
                String tag = entry.channel() == Channel.APPLICATION ? "[APP] " : "[MDL] ";
                sb.append(tag).append(AppLogger.format(entry)).append('\n');
            }
            logArea.setText(sb.toString());
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }
}
