from enum import Enum


class ModelName(str, Enum):
    DEEPSEEK_V4_PRO = "deepseek-ai/DeepSeek-V4-Pro"


class ModelHost(str, Enum):
    HUGGING_FACE = "hugging_face"
