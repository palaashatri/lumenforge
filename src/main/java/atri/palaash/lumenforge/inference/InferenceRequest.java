package atri.palaash.lumenforge.inference;

import atri.palaash.lumenforge.model.ModelDescriptor;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

public record InferenceRequest(
	ModelDescriptor model,
	String prompt,
	String negativePrompt,
	double promptWeight,
	long seed,
	int batch,
	int width,
	int height,
	String style,
	boolean upscale,
	String inputImagePath,
	boolean preferGpu,
	Consumer<String> progressCallback,
	AtomicBoolean cancellationFlag,
	double strength,         // img2img denoise strength (0.0–1.0, default 0.75)
	String maskImagePath     // optional mask for inpainting (white = repaint)
) {
	/** Convenience constructor without progress callback (backward compatible). */
	public InferenceRequest(
			ModelDescriptor model, String prompt, String negativePrompt,
			double promptWeight, long seed, int batch,
			int width, int height, String style, boolean upscale,
			String inputImagePath, boolean preferGpu) {
		this(model, prompt, negativePrompt, promptWeight, seed, batch,
				width, height, style, upscale, inputImagePath, preferGpu, null, null, 0.75, null);
	}

	/** Constructor with progress callback but no cancellation flag. */
	public InferenceRequest(
			ModelDescriptor model, String prompt, String negativePrompt,
			double promptWeight, long seed, int batch,
			int width, int height, String style, boolean upscale,
			String inputImagePath, boolean preferGpu,
			Consumer<String> progressCallback) {
		this(model, prompt, negativePrompt, promptWeight, seed, batch,
				width, height, style, upscale, inputImagePath, preferGpu,
				progressCallback, null, 0.75, null);
	}

	/** Full constructor with cancellation flag (txt2img panels). */
	public InferenceRequest(
			ModelDescriptor model, String prompt, String negativePrompt,
			double promptWeight, long seed, int batch,
			int width, int height, String style, boolean upscale,
			String inputImagePath, boolean preferGpu,
			Consumer<String> progressCallback,
			AtomicBoolean cancellationFlag) {
		this(model, prompt, negativePrompt, promptWeight, seed, batch,
				width, height, style, upscale, inputImagePath, preferGpu,
				progressCallback, cancellationFlag, 0.75, null);
	}

	/** Fire a progress message if a callback is set. */
	public void reportProgress(String message) {
		if (progressCallback != null) {
			progressCallback.accept(message);
		}
	}

	/** Check if the user has requested cancellation. */
	public boolean isCancelled() {
		return cancellationFlag != null && cancellationFlag.get();
	}
}
