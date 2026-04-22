package atri.palaash.jforge.core.sampler;

import atri.palaash.jforge.core.tensor.FloatTensor;

public interface Sampler {
    FloatTensor sample(SamplerContext ctx);
}
