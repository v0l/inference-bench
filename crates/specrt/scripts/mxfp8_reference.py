import sys

import ml_dtypes
import numpy as np
from safetensors.numpy import save_file


def main(out):
    rng = np.random.default_rng(1190)
    rows, cols, group = 64, 256, 32
    codes = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] &= 0xF0
    scales = rng.integers(118, 138, size=(rows, cols // group), dtype=np.uint8)
    values = codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    factor = scales.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    expected = (values * np.repeat(factor, group, axis=1)).astype(ml_dtypes.bfloat16)
    save_file(
        {
            "weight": codes,
            "weight_scale": scales,
            "expected": expected.view(np.uint16),
        },
        out,
    )


if __name__ == "__main__":
    main(sys.argv[1])
