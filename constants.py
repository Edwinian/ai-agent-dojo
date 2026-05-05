from enum import Enum


class ModelName(str, Enum):
    DEEPSEEK_V4_PRO = "deepseek-ai/DeepSeek-V4-Pro"
    QWEN_3_6_27B = "Qwen/Qwen3.6-27B"
    GROK_4_FAST_NON_REASONING = "grok-4-fast-non-reasoning"
    QWEN2_VL_OCR2_2B_INSTRUCT = "prithivMLmods/Qwen2-VL-OCR2-2B-Instruct"
    QWEN2_5_VL_7B_INSTRUCT = "Qwen/Qwen2.5-VL-7B-Instruct"


class RouteKey(str, Enum):
    end = "__end__"

