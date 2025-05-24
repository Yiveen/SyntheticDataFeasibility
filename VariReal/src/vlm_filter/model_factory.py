"""
Code mainly from https://github.com/rabiulcste/vismin/blob/master/ctrl_edit/filters/vqa_models.py
"""

import logging
from functools import partial

from .vlm.idesfics2 import Idefics2Model
from .vlm.internvl import InternVLModel
from .vlm.llava import LlaVaModel


class ModelFactory:
    """
    Factory class to manage and instantiate supported models.
    """

    MLLM_MODELS = {
        "HuggingFaceM4/idefics2-8b": partial(Idefics2Model, model_name_or_path="HuggingFaceM4/idefics2-8b"),
        "llava-hf/llava-v1.6-mistral-7b-hf": partial(
            LlaVaModel, model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf"
        ),
        "OpenGVLab/InternVL2-8B": partial(InternVLModel, model_name="InternVL2-8B"),
    }

    SUPPORTED_MODELS = {**MLLM_MODELS}

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = self.get_model(model_name, **kwargs)

    def is_model_supported(self, model_name: str) -> bool:
        """
        Check if the given model name is supported.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model is supported, False otherwise.
        """
        is_supported = any(model in model_name for model in self.SUPPORTED_MODELS)
        return is_supported

    def get_model(self, model_name: str, **kwargs):
        """
        Get an instance of the specified model.

        Args:
            model_name (str): The name of the model to instantiate.

        Returns:
            An instance of the specified model.

        Raises:
            ValueError: If the model name is not supported.
        """
        pretrained = kwargs.pop("pretrained", None)
        if not self.is_model_supported(model_name):
            raise ValueError(f"Unsupported model name: {model_name}")
        if pretrained:
            model = self.SUPPORTED_MODELS[model_name](pretrained=pretrained, **kwargs)
        else:
            model = self.SUPPORTED_MODELS[model_name](**kwargs)
        return model

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_batch(self, *args, **kwargs):
        return self.model.predict_batch(*args, **kwargs)
