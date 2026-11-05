"""
Environment checks for the Gu Lab Minian pipeline.

Run inside the minian-gu-lab environment:

    python env_test.py            # run every check
    python env_test.py minian     # only check that minian is installed
    python env_test.py cuda       # only check GPU / CUDA support

The exit code says which checks failed, so scripts can rely on it:
    0  every requested check passed
    1  minian check failed
    2  CUDA check failed
    3  both failed

CUDA is optional: without it the pipeline still runs, with motion correction
on the CPU (``use_gpu: "auto"`` in the notebook picks this automatically).
"""

import sys

MINIAN_FAILED = 1
CUDA_FAILED = 2

# CUDA runtime version that environment.yml installs (cudatoolkit=11.8).
# The NVIDIA driver must support at least this version.
REQUIRED_CUDA = 11080


def _version(v):
    """Format a CUDA version integer such as 11080 as '11.8'."""
    return f"{v // 1000}.{(v % 1000) // 10}"


def check_minian():
    """Return (ok, message) for whether minian is importable."""
    try:
        import minian
    except Exception as err:
        return False, f"cannot import minian ({type(err).__name__}: {err}). Run 'pip install -e .' in the repo root."
    version = getattr(minian, "__version__", None)
    if version is None:
        return False, "minian imported but has no __version__; the install looks incomplete."
    return True, f"minian {version} ({minian.__file__})"


def check_cuda():
    """Return (ok, message) for whether GPU motion correction can run."""
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cp_ndi
    except Exception as err:
        return False, f"cannot import cupy ({type(err).__name__}: {err})."

    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
    except Exception as err:
        return False, (
            f"no usable NVIDIA GPU or driver ({type(err).__name__}: {err}). "
            "Check that the machine has an NVIDIA GPU and that 'nvidia-smi' works."
        )
    if n_devices == 0:
        return False, "cupy is installed but no CUDA device is visible."

    driver = cp.cuda.runtime.driverGetVersion()
    runtime = cp.cuda.runtime.runtimeGetVersion()
    if driver < REQUIRED_CUDA:
        return False, (
            f"NVIDIA driver supports CUDA {_version(driver)}, but the environment "
            f"needs CUDA {_version(REQUIRED_CUDA)}. Update the NVIDIA driver."
        )

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    name = name.decode() if isinstance(name, bytes) else name

    # Exercise the same CUDA libraries motion correction uses: plain kernels,
    # cuFFT, cuBLAS (einsum) and runtime-compiled kernels (ndimage.shift).
    import numpy as np

    steps = [
        ("array maths", lambda: int(cp.asnumpy(cp.arange(4).sum())) == 6),
        ("cuFFT", lambda: np.allclose(
            cp.asnumpy(cp.fft.ifft(cp.fft.fft(cp.arange(8.0))).real), np.arange(8.0))),
        ("einsum", lambda: float(cp.einsum("ij,ij->", cp.ones((4, 4)), cp.ones((4, 4)))) == 16.0),
        ("ndimage.shift", lambda: float(cp_ndi.shift(cp.ones((8, 8)), (1, 1), order=1).sum()) > 0),
    ]
    for label, step in steps:
        try:
            if not step():
                return False, f"GPU {name}: {label} returned a wrong result."
        except Exception as err:
            return False, f"GPU {name}: {label} failed ({type(err).__name__}: {err})."

    return True, (
        f"{name} ({n_devices} device(s)), driver CUDA {_version(driver)}, "
        f"runtime CUDA {_version(runtime)}, cupy {cp.__version__}"
    )


CHECKS = {
    "minian": (check_minian, MINIAN_FAILED, "required"),
    "cuda": (check_cuda, CUDA_FAILED, "optional: CPU fallback is used without it"),
}


def main(argv):
    requested = [a.lower() for a in argv] or list(CHECKS)
    unknown = [a for a in requested if a not in CHECKS]
    if unknown:
        print(f"Unknown check(s): {', '.join(unknown)}. Choose from: {', '.join(CHECKS)}")
        return 64

    code = 0
    for name in requested:
        func, flag, note = CHECKS[name]
        ok, message = func()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {message}")
        if not ok:
            print(f"       ({note})")
            code |= flag
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
