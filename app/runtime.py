"""Configure TensorFlow before any models or tensors are created."""
import ctypes
from pathlib import Path
import sys
import warnings

import tensorflow as tf


def _configure_device():
    # TensorFlow 2.21's Linux wheel omits nvidia/cusolver/lib from its
    # loader search path. Load the pip-installed library by its full path;
    # its own RUNPATH resolves the other NVIDIA package dependencies.
    if sys.platform == "linux":
        solver = Path(tf.__file__).parent.parent / "nvidia/cusolver/lib/libcusolver.so.11"
        if solver.exists():
            try:
                ctypes.CDLL(str(solver))
            except OSError as exc:
                warnings.warn(f"Could not load CUDA solver library: {exc}", RuntimeWarning)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # Allocate memory as needed instead of reserving the entire GPU.
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        name = tf.config.experimental.get_device_details(gpus[0]).get(
            "device_name", gpus[0].name
        )
        return "/GPU:0", name
    return "/CPU:0", "CPU (TensorFlow could not detect a GPU)"


TRAINING_DEVICE, DEVICE_NAME = _configure_device()
