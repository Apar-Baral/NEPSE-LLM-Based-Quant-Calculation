"""Multi-scale temporal CNN + LLM semantic fusion + graph propagation."""

from backend.models.multimodal.architecture import torch_available

__all__ = ["train_multimodal", "predict_multimodal", "torch_available"]


def __getattr__(name: str):
    if name == "train_multimodal":
        from backend.models.multimodal.train import train_multimodal
        return train_multimodal
    if name == "predict_multimodal":
        from backend.models.multimodal.predict import predict_multimodal
        return predict_multimodal
    raise AttributeError(name)
