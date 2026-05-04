from llm_service import LLMService


def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b


def extract_text(img_path: str) -> str:
    """
    Extract text from an image file.
    """
    return LLMService().extract_text(img_path)
