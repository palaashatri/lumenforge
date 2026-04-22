package atri.palaash.jforge.ui;

import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.event.MouseWheelEvent;
import java.awt.image.BufferedImage;

/**
 * Preview panel supporting wheel zoom and drag pan.
 */
public class ZoomPanImagePanel extends JPanel {

    private BufferedImage image;
    private double zoom = 1.0;
    private double offsetX = 0.0;
    private double offsetY = 0.0;
    private int dragStartX;
    private int dragStartY;

    public ZoomPanImagePanel() {
        setBackground(new Color(25, 25, 25));

        addMouseWheelListener(this::onZoom);
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                dragStartX = e.getX();
                dragStartY = e.getY();
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                dragStartX = 0;
                dragStartY = 0;
            }
        });
        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                int dx = e.getX() - dragStartX;
                int dy = e.getY() - dragStartY;
                offsetX += dx;
                offsetY += dy;
                dragStartX = e.getX();
                dragStartY = e.getY();
                repaint();
            }
        });
    }

    public void setImage(BufferedImage image) {
        this.image = image;
        this.zoom = 1.0;
        this.offsetX = 0.0;
        this.offsetY = 0.0;
        repaint();
    }

    public BufferedImage getImage() {
        return image;
    }

    public void clearImage() {
        this.image = null;
        this.zoom = 1.0;
        this.offsetX = 0.0;
        this.offsetY = 0.0;
        repaint();
    }

    public void resetView() {
        this.zoom = 1.0;
        this.offsetX = 0.0;
        this.offsetY = 0.0;
        repaint();
    }

    private void onZoom(MouseWheelEvent event) {
        if (image == null) {
            return;
        }
        double oldZoom = zoom;
        if (event.getWheelRotation() < 0) {
            zoom = Math.min(10.0, zoom * 1.12);
        } else {
            zoom = Math.max(0.1, zoom / 1.12);
        }

        double factor = zoom / oldZoom;
        double mouseX = event.getX() - getWidth() / 2.0 - offsetX;
        double mouseY = event.getY() - getHeight() / 2.0 - offsetY;
        offsetX -= mouseX * (factor - 1.0);
        offsetY -= mouseY * (factor - 1.0);

        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        if (image == null) {
            g2.setColor(new Color(130, 130, 130));
            g2.drawString("No preview yet", getWidth() / 2 - 40, getHeight() / 2);
            g2.dispose();
            return;
        }

        double fitScale = Math.min((double) getWidth() / image.getWidth(), (double) getHeight() / image.getHeight());
        double drawScale = fitScale * zoom;
        int drawWidth = (int) Math.round(image.getWidth() * drawScale);
        int drawHeight = (int) Math.round(image.getHeight() * drawScale);

        int x = (getWidth() - drawWidth) / 2 + (int) Math.round(offsetX);
        int y = (getHeight() - drawHeight) / 2 + (int) Math.round(offsetY);
        g2.drawImage(image, x, y, drawWidth, drawHeight, null);
        g2.dispose();
    }
}
