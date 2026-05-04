import json
import os
from typing import Any, Callable, Generic, Protocol

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from constants import ModelHost, ModelName
from type import GRAPH_STATE


class Invokable(Protocol):
    def invoke(self, prompt: str) -> Any:
        ...


class LLMService(Generic[GRAPH_STATE]):
    def __init__(self, model_name: ModelName = ModelName.DEEPSEEK_V4_PRO) -> None:
        load_dotenv()
        self.model_name = model_name
        self.max_output_length = 512
        self.model_host_map: dict[ModelName, ModelHost] = {
            ModelName.DEEPSEEK_V4_PRO: ModelHost.HUGGING_FACE
        }
        self.host_init_map: dict[ModelHost, Callable[[], Invokable]] = {
            ModelHost.HUGGING_FACE: self.init_hf_mode
        }
        selected_host = self.model_host_map[self.model_name]
        self.chat_llm: Invokable = self.host_init_map[selected_host]()
        self.output_parser = StrOutputParser()
        self.prompt_template = PromptTemplate.from_template(
            "## Current graph state\n{state}\n\n## Task\n{input}"
        )

    def init_hf_mode(self):
        """Initialize and return the ChatHuggingFace LLM."""
        endpoint = HuggingFaceEndpoint(
                repo_id=self.model_name.value,
                huggingfacehub_api_token=os.getenv("HF_TOKEN"),
                max_new_tokens=self.max_output_length,
                temperature=0.6,
                top_p=0.95,
                return_full_text=False,
            )
        return ChatHuggingFace(llm=endpoint)

    def invoke(self, state: GRAPH_STATE, prompt: str) -> str:
        state_text = json.dumps(dict(state), indent=2, default=str)
        chain = self.prompt_template | self.chat_llm | self.output_parser
        return chain.invoke({"input": prompt, "state": state_text})
