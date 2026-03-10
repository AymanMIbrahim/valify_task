import onnxruntime as ort

from inference.helpers.config import ONNX_MODEL_PATH


_session = None


def get_onnx_session():
    global _session

    if _session is None:
        providers = ["CPUExecutionProvider"]

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        _session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            providers=providers,
        )

    return _session