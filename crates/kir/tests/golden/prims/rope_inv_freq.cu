__device__ __forceinline__ float rope_inv_freq(int i, int layer) {
    if (LOCAL_ROPE[layer]) return 1.0f / powf(LOCAL_THETA, (float)(2 * i) / (float)ROT);
    float f = 1.0f / powf(THETA, (float)(2 * i) / (float)(ROT));
    if (ROPE_KIND == 1) {
        f = f / ROPE_FACTOR;
    } else if (ROPE_KIND == 2) {
        float wavelen = 6.283185307179586f / f;
        float low = ROPE_ORIG_POS / ROPE_LOW_FREQ;
        float high = ROPE_ORIG_POS / ROPE_HIGH_FREQ;
        if (wavelen > low) {
            f = f / ROPE_FACTOR;
        } else if (wavelen >= high) {
            float sm = (ROPE_ORIG_POS / wavelen - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ);
            f = (1.0f - sm) * f / ROPE_FACTOR + sm * f;
        }
    } else if (ROPE_KIND == 3) {
        float ramp = fminf(fmaxf(((float)i - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ), 0.0f), 1.0f);
        f = ramp * f / ROPE_FACTOR + (1.0f - ramp) * f;
    } else if (ROPE_KIND == 4) {
        f = f / ROPE_FREQ_FACTORS[i];
    }
    return f;
}
