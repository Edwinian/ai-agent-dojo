"""Text and vision calls via the xAI SDK."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import image, user, Sequence

from llm_service import LlmService

load_dotenv()


class XaiService(LlmService):
    def __init__(
        self,
        model_name: str = "grok-3-mini",
        tools: list[Any] | None = None,
    ) -> None:
        super().__init__(model_name=model_name, tools=tools)
        self._api_key = os.getenv("XAI_API_KEY")
        self.x_client: Client | None = (
            Client(api_key=self._api_key) if self._api_key else None
        )
        self.VISION_MODEL = "grok-4-fast-non-reasoning"

    def init_model(self) -> Client | None:
        """Initialize and return the xAI SDK client."""
        return self.x_client

    def _get_response_content(self, chat: object) -> str:
        response = chat.sample()
        return str(response.content)

    def _create_chat(self, model: str):
        if self.tools is not None:
            return self.x_client.chat.create(
                model=model,
                tools=self.tools,
                tool_choice="auto",
            )
        return self.x_client.chat.create(model=model)

    def invoke(self, prompt: str) -> str:
        try:
            if not self.x_client:
                return ""
            chat = self._create_chat(model=self.model_name)
            chat.append(user(prompt))
            return self._get_response_content(chat)
        except Exception as e:
            print(f"XaiService LLM API error: {e!s}")
            return ""

    def extract_text_from_image(self, image_url: str) -> str:
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
