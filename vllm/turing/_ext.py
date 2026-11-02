# SPDX-License-Identifier: Apache-2.0
"""Load the prebuilt CUDA extensions of vllm.turing.

The extensions are built ahead of time (``python -m vllm.turing._ext build <csrc/turing> <out_dir>``,
done by docker/Dockerfile.turing) with sm_75 SASS plus compute_75 PTX, so nothing is compiled at runtime.
They are looked up in ``VLLM_TURING_EXT_DIR`` (default: ``vllm/turing/_ext``). In a source checkout
without prebuilt files they are JIT-built from ``csrc/turing`` into ``~/.cache/vllm/turing_ext`` (needs nvcc).
"""

import glob
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOADED: dict = {}

# extension module name -> source file in csrc/turing
SOURCES = {
    "cuda_attn_ext": "decode_attn_sm75.cu",
    "cuda_w4_ext": "w4a16_sm75.cu",
}
ARCH = "7.5+PTX"
CUDA_FLAGS = ["-O3", "-lineinfo", "--expt-relaxed-constexpr"]


def ext_dir() -> str:
    return os.environ.get("VLLM_TURING_EXT_DIR", os.path.join(_HERE, "_ext"))


def _src_dir() -> str:
    return os.path.normpath(os.path.join(_HERE, "..", "..", "csrc", "turing"))


def _build(name: str, src: str, out_dir: str, verbose: bool = False):
    from torch.utils.cpp_extension import load

    os.makedirs(out_dir, exist_ok=True)
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH
    try:
        return load(
            name=name,
            sources=[src],
            build_directory=out_dir,
            verbose=verbose,
            extra_cflags=["-O3"],
            extra_cuda_cflags=CUDA_FLAGS,
        )
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old


def load_ext(name: str):
    mod = _LOADED.get(name)
    if mod is not None:
        return mod
    so = sorted(glob.glob(os.path.join(ext_dir(), name + "*.so")))
    if so:
        spec = importlib.util.spec_from_file_location(name, so[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        src = os.path.join(_src_dir(), SOURCES[name])
        if not os.path.exists(src):
            raise RuntimeError(
                f"vllm.turing: prebuilt extension {name} not found in {ext_dir()} "
                f"(set VLLM_TURING_EXT_DIR, or build it with `python -m vllm.turing._ext build`)"
            )
        cache = os.path.join(os.path.expanduser("~"), ".cache", "vllm", "turing_ext", name)
        mod = _build(name, src, cache)
    _LOADED[name] = mod
    return mod


def _main(argv):
    if len(argv) != 3 or argv[0] != "build":
        print("usage: python -m vllm.turing._ext build <csrc/turing> <out_dir>", file=sys.stderr)
        return 2
    src_dir, out_dir = argv[1], argv[2]
    for name, src in SOURCES.items():
        _build(name, os.path.join(src_dir, src), out_dir, verbose=True)
    # keep only the shared objects
    for f in os.listdir(out_dir):
        if not f.endswith(".so"):
            p = os.path.join(out_dir, f)
            if os.path.isfile(p):
                os.remove(p)
    print("built:", sorted(os.listdir(out_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
