import json
import os
from typing import Any, Callable, Generic

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from constants import ModelName
from type import GRAPH_STATE

load_dotenv()


class HuggingFaceService(Generic[GRAPH_STATE]):
    def __init__(
        self,
        model_name: ModelName = ModelName.DEEPSEEK_V4_PRO,
        tools: list[Callable[..., Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_output_length = 512
        self.request_timeout_seconds = int(os.getenv("HF_REQUEST_TIMEOUT_SECONDS", "240"))
        self.chat_llm = self.init_model()

        if tools is not None:
            self.chat_llm = self.chat_llm.bind_tools(tools)

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

    def invoke(self, state: GRAPH_STATE, prompt: str) -> str:
        state_text = json.dumps(dict(state), indent=2, default=str)
        chain = self.prompt_template | self.chat_llm | self.output_parser
        return chain.invoke({"input": prompt, "state": state_text})
