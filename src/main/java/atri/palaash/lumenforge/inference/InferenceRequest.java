package atri.palaash.lumenforge.inference;

import atri.palaash.lumenforge.model.ModelDescriptor;

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
	Consumer<String> progressCallback
) {
	/** Convenience constructor without progress callback (backward compatible). */
	public InferenceRequest(
			ModelDescriptor model, String prompt, String negativePrompt,
			double promptWeight, long seed, int batch,
			int width, int height, String style, boolean upscale,
			String inputImagePath, boolean preferGpu) {
		this(model, prompt, negativePrompt, promptWeight, seed, batch,
				width, height, style, upscale, inputImagePath, preferGpu, null);
	}

	/** Fire a progress message if a callback is set. */
	public void reportProgress(String message) {
		if (progressCallback != null) {
			progressCallback.accept(message);
		}
	}
}
