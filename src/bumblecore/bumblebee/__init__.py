"""
BumbleCore - 大语言模型训练框架
"""

__version__ = "0.1.0"

from .model import BumblebeeForCausalLM,BumblebeeConfig
from transformers import AutoModelForCausalLM,AutoConfig
AutoModelForCausalLM.register(BumblebeeConfig, BumblebeeForCausalLM)
AutoConfig.register("bumblebee", BumblebeeConfig)

__all__ = ["BumblebeeForCausalLM","BumblebeeConfig"]
