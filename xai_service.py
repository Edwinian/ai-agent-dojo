"""Text and vision calls via the xAI SDK."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import image, user

load_dotenv()


class XaiService:
    def __init__(self) -> None:
        self._api_key = os.getenv("XAI_API_KEY")
        self.x_client: Client | None = (
            Client(api_key=self._api_key) if self._api_key else None
        )
        self.TEXT_MODEL = "grok-3-mini"
        self.VISION_MODEL = "grok-4-fast-non-reasoning"

    def _get_response_content(self, chat: object) -> str:
        response = chat.sample()
        return str(response.content)

    def generate_text(self, prompt: str) -> str:
        try:
            if not self.x_client:
                return ""
            chat = self.x_client.chat.create(model=self.TEXT_MODEL)
            chat.append(user(prompt))
            return self._get_response_content(chat)
        except Exception as e:
            print(f"XaiService LLM API error: {e!s}")
            return ""

    def extract_texts(self, image_url: str) -> str:
        try:
            if not self.x_client:
                return "[extract_text failed] XAI_API_KEY is not set."
            chat = self.x_client.chat.create(model=self.VISION_MODEL)
            prompt_splits: list[str] = ["Extract the texts from the image"]
            chat.append(
                user(prompt_splits[0], image(image_url=image_url, detail="low"))
            )
            return self._get_response_content(chat).strip()
        except Exception as e:
            print(f"XaiService LLM API error: {e!s}")
            return ""
