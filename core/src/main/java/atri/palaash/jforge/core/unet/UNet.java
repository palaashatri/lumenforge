package atri.palaash.jforge.core.unet;

import atri.palaash.jforge.core.tensor.FloatTensor;
import java.util.Map;

public interface UNet {
    FloatTensor predict(FloatTensor sample, long timestep, FloatTensor encoderHiddenStates, Map<String, Object> additionalInputs);
}
