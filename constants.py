from enum import Enum


class NodeName(str, Enum):
    READ_EMAIL = "read_email"
    CLASSIFY_EMAIL = "classify_email"
    HANDLE_SPAM = "handle_spam"
    DRAFT_RESPONSE = "draft_response"
    NOTIFY_MR_HUGG = "notify_mr_hugg"


class ModelName(str, Enum):
    DEEPSEEK_V4_PRO = "deepseek-ai/DeepSeek-V4-Pro"


class ModelHost(str, Enum):
    HUGGING_FACE = "hugging_face"
