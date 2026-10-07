"""统一入口，保持 from model import ModelConfig, Transformer 兼容。"""
from .config import ModelConfig
from .transformer import Transformer

__all__ = ["ModelConfig", "Transformer"]
