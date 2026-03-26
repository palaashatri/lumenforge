package atri.palaash.jforge.inference;

public class PromptEnhancer {
    
    /**
     * Enhances the original prompt to be much more concise, targeting high quality and efficiency.
     */
    public static String enhanceOriginal(String prompt) {
        return enhanceOriginal(prompt, "None", "None");
    }

    /**
     * Enhances the original prompt with video-specific modifiers.
     */
    public static String enhanceOriginal(String prompt, String motion, String style) {
        if (prompt == null || prompt.isBlank()) {
            return "";
        }
        
        // Remove conversational fluff
        String refined = prompt.replaceAll("(?i)\\b(please|can you|draw me|create|a picture of|make)\\b", "")
                               .replaceAll("\\s+", " ")
                               .trim();
                               
        if (motion != null && !motion.equals("None")) {
            refined += ", camera motion: " + motion;
        }
        if (style != null && !style.equals("None")) {
            refined += ", style: " + style;
        }
                               
        // Ensure prompt is concise yet detailed
        return refined + ", highest quality, highly detailed, clean, precise, masterpiece, 8k resolution, concise";
    }
    
    /**
     * Enhances the negative prompt to remove verbosity and noise.
     */
    public static String enhanceNegative(String negative) {
        if (negative == null) {
            negative = "";
        }
        String refined = negative.trim();
        if (!refined.isEmpty()) {
            refined += ", ";
        }
        // Force concise outputs by removing clutter
        return refined + "verbose, cluttered, text, watermark, bad anatomy, ugly, messy";
    }
}
