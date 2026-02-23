package atri.palaash.lumenforge.ui;

import java.time.LocalDateTime;

public record HistoryEntry(
        LocalDateTime timestamp,
        String model,
        String prompt,
        String negativePrompt,
        long seed,
        int batch,
        String size,
        String style,
        String status,
        String outputPath
) {
}
