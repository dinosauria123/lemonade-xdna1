#!/usr/bin/env python3
"""Isolate getExpBf16 behaviour via a @iron.jit wrapper that calls the LUT
exp on a 16-wide vector.  Reads back what the LUT actually returns for known
negative bf16 inputs.  Decides scaling-vs-replace for the softmax fix.

Run: python examples/test_lut_exp.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
import numpy as np
import npu_gemm
import aie.iron as iron
from aie.iron import CompileTime
from aie.utils import config
from pathlib import Path
from aie.iron.kernel import ExternalFunction
from aie.iron.device import Tile
from ml_dtypes import bfloat16 as bf

runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
include = [str(config.cxx_header_path()),
           str(Path(config.cxx_header_path()) / "aie_kernels" / "aie2"),
           str(runtime_dir)]

SRC = '''
#include "lut_based_ops.h"
extern "C" void lut_exp_test(const bfloat16* in, bfloat16* out) {
  aie::vector<bfloat16, 16> x = aie::load_v<16>(in);
  aie::vector<bfloat16, 16> e = to_v16bfloat16(getExpBf16(x));
  aie::store_v(out, e);
}
'''

ef = ExternalFunction("lut_exp_test", source_string=SRC,
                      arg_types=[np.ndarray[(16,), np.dtype[bf]],
                                 np.ndarray[(16,), np.dtype[bf]]],
                      include_dirs=include, compile_flags=["-O2"])

In_ty = np.ndarray[(16,), np.dtype[bf]]
Out_ty = np.ndarray[(16,), np.dtype[bf]]

@iron.jit
def run_lut(in_b: In_ty, out_b: Out_ty):
    ef(in_b, out_b)


def main():
    if not npu_gemm.available():
        print("NPU not available"); return
    vals = np.array([-1.0, -5.0, -10.0, -20.0, -30.0, -50.0, -70.0, -80.0,
                     -88.0, -100.0, 0.0, 1.0, 2.0, -1813.0, -2630.0, 5.0],
                    dtype=np.float32)
    In = vals.astype(bf)
    In_t = iron.tensor(In.reshape(-1), dtype=bf, device="npu")
    Out_t = iron.zeros(16, dtype=bf, device="npu")
    run_lut(In_t, Out_t)
    got = Out_t.numpy().copy().astype(np.float32)
    print(f"{'input':>10} {'exp(in)':>14} {'LUT_out':>14} {'ratio':>10}")
    for v, g in zip(vals, got):
        ref = float(np.exp(v))
        print(f"{v:10.2f} {ref:14.6g} {g:14.6g} {g/ref:10.4f}")


if __name__ == "__main__":
    main()
