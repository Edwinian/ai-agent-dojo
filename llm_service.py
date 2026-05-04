import base64
import json
import mimetypes
import os
from typing import Any, Callable, Generic, Protocol

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from constants import ModelName
from type import GRAPH_STATE

load_dotenv()


class Invokable(Protocol):
    def invoke(self, prompt: str) -> Any:
        ...


class LLMService(Generic[GRAPH_STATE]):
    def __init__(
        self,
        model_name: ModelName = ModelName.DEEPSEEK_V4_PRO,
        vision_model_name: str | None = ModelName.GROK_4_FAST_NON_REASONING.value,
        tools: list[Callable[..., Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.vision_model_name = vision_model_name
        self.max_output_length = 512
        self.request_timeout_seconds = int(os.getenv("HF_REQUEST_TIMEOUT_SECONDS", "240"))
        self.chat_llm = self.init_model()

        if tools is not None:
            self.chat_llm = self.chat_llm.bind_tools(tools)

        self.vision_llm: ChatHuggingFace = self.init_vision_model() 
        self.output_parser = StrOutputParser()
        self.prompt_template = PromptTemplate.from_template(
            "## Current graph state\n{state}\n\n## Task\n{input}"
        )

    def init_model(self) -> ChatHuggingFace:
        """Initialize and return the ChatHuggingFace LLM."""
        endpoint = HuggingFaceEndpoint(
                repo_id=self.model_name.value,
                huggingfacehub_api_token=os.getenv("HF_TOKEN"),
                max_new_tokens=self.max_output_length,
                temperature=0.6,
                top_p=0.95,
                return_full_text=False,
                timeout=self.request_timeout_seconds,
            )
        return ChatHuggingFace(llm=endpoint)

    def init_vision_model(self) -> ChatHuggingFace:
        """Initialize Hugging Face chat model for image / multimodal input."""
        endpoint = HuggingFaceEndpoint(
            repo_id=self.vision_model_name,
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            max_new_tokens=self.max_output_length,
            temperature=0.6,
            top_p=0.95,
            return_full_text=False,
            timeout=self.request_timeout_seconds,
        )
        return ChatHuggingFace(llm=endpoint)

    def extract_text(self, img_path: str) -> str:
        """
        Extract text from an image file using a multimodal model.
        """
        all_text = ""

        try:
            with open(img_path, "rb") as image_file:
                image_bytes = image_file.read()

            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(img_path)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/png"

            message = [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Extract all the text from this image. "
                                "Return only the extracted text, no explanations."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            },
                        },
                    ]
                )
            ]

            response = self.vision_llm.invoke(message)
            raw_content = response.content
            if isinstance(raw_content, str):
                all_text = raw_content
            elif isinstance(raw_content, list):
                parts: list[str] = []
                for block in raw_content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                all_text = "\n".join(parts)
            else:
                all_text = str(raw_content)

            return all_text.strip()
        except Exception as e:
            error_msg = f"Error extracting text: {e!s}"
            print(error_msg)
            return ""

    def invoke(self, state: GRAPH_STATE, prompt: str) -> str:
        state_text = json.dumps(dict(state), indent=2, default=str)
        chain = self.prompt_template | self.chat_llm | self.output_parser
        return chain.invoke({"input": prompt, "state": state_text})
