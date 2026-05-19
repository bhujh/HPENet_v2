"""
TensorRT 8.6 utility layer — engine loading, GPU memory management, inference.

Uses the NEW TensorRT 8.6 API exclusively:
  - get_tensor_name / get_tensor_mode / get_tensor_dtype / get_tensor_shape
  - set_input_shape / set_tensor_address / execute_async_v3
  - num_io_tensors / MemoryPoolType

NO deprecated API (get_binding_index, enqueue, max_workspace_size, etc.)
"""

import os
import sys
import ctypes
import numpy as np
import torch
import tensorrt as trt

# ---------------------------------------------------------------------------
#  TRT dtype ↔ torch dtype ↔ numpy dtype
# ---------------------------------------------------------------------------

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF:  torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL:  torch.bool,
}

TRT_TO_NUMPY = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF:  np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL:  np.bool_,
}

# ---------------------------------------------------------------------------
#  Environment setup
# ---------------------------------------------------------------------------

_TRT_LIB_PATH = "/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib"
_CUDA_LIB_PATH = "/usr/local/cuda-11.8/lib64"


def _is_cuda_available():
    """Return True if CUDA / nvidia-smi is available."""
    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


def setup_trt_env(lib_path=_TRT_LIB_PATH, cuda_path=_CUDA_LIB_PATH):
    """Ensure TensorRT and CUDA shared libraries are on LD_LIBRARY_PATH.

    Must be called BEFORE any `import tensorrt` if the libraries aren't
    already on the system path.
    """
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    for path in [lib_path, cuda_path]:
        if path not in existing.split(":"):
            existing = path + ":" + existing if existing else path
    os.environ["LD_LIBRARY_PATH"] = existing


# ---------------------------------------------------------------------------
#  TRTSession — load engine, manage GPU buffers, run inference
# ---------------------------------------------------------------------------

class TRTSession:
    """Load a serialized TensorRT engine and provide a simple `run()` API.

    Handles:
      - Dynamic input shapes (variable point counts)
      - GPU memory allocation via torch tensors (no cupy/pycuda needed)
      - Asynchronous execution on the current CUDA stream
    """

    def __init__(self, engine_path):
        """Load TRT engine from a .engine file.

        Args:
            engine_path: Path to a serialized TensorRT engine file.
        """
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)

        with open(engine_path, "rb") as f:
            serialized = f.read()

        self.engine = self._runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Discover I/O tensor metadata (new TRT 8.6 API)
        self._input_specs = {}   # name -> {"shape": (...), "dtype": trt.DataType}
        self._output_name = None
        self._output_dtype = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)

            if mode == trt.TensorIOMode.INPUT:
                self._input_specs[name] = {"shape": shape, "dtype": dtype}
            else:  # OUTPUT
                self._output_name = name
                self._output_dtype = dtype
                self._output_shape = shape

        # Cached GPU buffers (reused across calls when shapes match)
        self._input_buffers = {}    # name -> torch.Tensor on CUDA
        self._output_buffer = None  # torch.Tensor on CUDA

        # Engine info
        self._engine_path = engine_path

    def run(self, pos, x):
        """Run inference with dynamic point count.

        Args:
            pos: numpy array of shape (1, N, 3), dtype float32
            x:   numpy array of shape (1, 4, N), dtype float32

        Returns:
            output: numpy array of shape (1, 2, N), dtype float32
        """
        assert pos.ndim == 3 and pos.shape[0] == 1, f"pos must be (1, N, 3), got {pos.shape}"
        assert x.ndim == 3 and x.shape[0] == 1, f"x must be (1, 4, N), got {x.shape}"
        N = pos.shape[1]
        assert x.shape[2] == N, f"pos N ({N}) != x N ({x.shape[2]})"

        # --- set dynamic input shapes ---
        self.context.set_input_shape("pos", (1, N, 3))
        self.context.set_input_shape("x", (1, 4, N))

        # --- transfer inputs to GPU ---
        if "pos" not in self._input_buffers or self._input_buffers["pos"].shape != pos.shape:
            self._input_buffers["pos"] = torch.empty(pos.shape, dtype=torch.float32, device="cuda")
        if "x" not in self._input_buffers or self._input_buffers["x"].shape != x.shape:
            self._input_buffers["x"] = torch.empty(x.shape, dtype=torch.float32, device="cuda")

        pos_gpu = self._input_buffers["pos"]
        x_gpu = self._input_buffers["x"]
        pos_gpu.copy_(torch.from_numpy(pos), non_blocking=True)
        x_gpu.copy_(torch.from_numpy(x), non_blocking=True)

        # --- bind input pointers ---
        self.context.set_tensor_address("pos", pos_gpu.data_ptr())
        self.context.set_tensor_address("x", x_gpu.data_ptr())

        # --- allocate / reuse output buffer ---
        out_shape = (1, 2, N)
        if self._output_buffer is None or self._output_buffer.shape != out_shape:
            torch_dtype = TRT_TO_TORCH.get(self._output_dtype, torch.float32)
            self._output_buffer = torch.empty(out_shape, dtype=torch_dtype, device="cuda")
        self.context.set_tensor_address(self._output_name, self._output_buffer.data_ptr())

        # --- execute on current CUDA stream ---
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        # --- return as CPU numpy ---
        return self._output_buffer.cpu().numpy()

    def __repr__(self):
        info = [f"TRTSession(engine={self._engine_path!r})"]
        for name, spec in self._input_specs.items():
            info.append(f"  input  {name}: {spec['shape']} ({spec['dtype']})")
        info.append(f"  output {self._output_name}: {self._output_shape} ({self._output_dtype})")
        return "\n".join(info)
