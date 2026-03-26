package atri.palaash.jforge.ui;

import javax.swing.*;
import java.awt.*;

/**
 * A unified wrapper panel that houses all Image-related tasks:
 * Text-to-Image, Image-to-Image, and Image Upscaling.
 */
public class ImageStudioPanel extends JPanel {
    private final JTabbedPane tabs;

    public ImageStudioPanel(TextToImagePanel t2i, Img2ImgPanel i2i, ImageUpscalePanel upscale) {
        super(new BorderLayout());

        tabs = new JTabbedPane(JTabbedPane.TOP);
        
        // Optional FlatLaf styling: segmented or underlined
        tabs.putClientProperty("JTabbedPane.style", "underlined"); 
        tabs.putClientProperty("JTabbedPane.tabHeight", 40);
        
        tabs.addTab("Text to Image", t2i);
        tabs.addTab("Image to Image", i2i);
        tabs.addTab("Upscale", upscale);

        // Add 12px padding identical to the content padding of Sidebar in MainFrame
        setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        add(tabs, BorderLayout.CENTER);
    }
    
    public void selectTab(int index) {
        if (index >= 0 && index < tabs.getTabCount()) {
            tabs.setSelectedIndex(index);
        }
    }
}
