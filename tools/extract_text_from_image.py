import base64
import mimetypes
from pathlib import Path
from typing import Annotated, Any

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

from xai_service import XaiService

_IMAGE_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def _local_image_path_to_data_url(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        ext = Path(file_path).suffix.lower()
        mime_type = _IMAGE_EXT_TO_MIME.get(ext, "image/png")
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


@tool
def extract_text_from_image(
    img_path: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any] | str:
    """Extract text from a local image file path (PNG, JPEG, WebP, etc.) using vision; not for PDFs."""
    mime, _ = mimetypes.guess_type(img_path)
    if img_path.lower().endswith(".pdf") or mime == "application/pdf":
        return (
            "[extract_text_from_image] Only image files are supported (e.g. PNG, JPEG), not PDF."
        )

    image_url = _local_image_path_to_data_url(img_path)
    text = XaiService().extract_text_from_image(image_url)
    return Command(
        update={
            "input_file": img_path,
            "files": {img_path: text},
            "messages": [ToolMessage(text, tool_call_id=tool_call_id)],
        }
    )


extract_text_from_image_tool = extract_text_from_image

__all__ = ["extract_text_from_image", "extract_text_from_image_tool"]
