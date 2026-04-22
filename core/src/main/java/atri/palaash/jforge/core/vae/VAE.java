package atri.palaash.jforge.core.vae;

import atri.palaash.jforge.core.tensor.FloatTensor;

public interface VAE {
    FloatTensor encode(FloatTensor image);
    FloatTensor decode(FloatTensor latents);
}
