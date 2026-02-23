package atri.palaash.lumenforge.storage;

public record DownloadProgress(long bytesRead, long totalBytes, String statusMessage) {

    /** Standard progress constructor (no status message). */
    public DownloadProgress(long bytesRead, long totalBytes) {
        this(bytesRead, totalBytes, null);
    }

    public int percent() {
        if (totalBytes <= 0) {
            return 0;
        }
        long value = (bytesRead * 100) / totalBytes;
        return (int) Math.max(0, Math.min(100, value));
    }

    /** Returns true if this is a status/retry notification rather than normal progress. */
    public boolean isStatusMessage() {
        return statusMessage != null && !statusMessage.isEmpty();
    }
}
