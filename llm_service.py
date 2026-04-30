import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from constants import ModelName


class LLMService:
    def __init__(self) -> None:
        load_dotenv()
        self.model_name = ModelName.DEEPSEEK_V4_PRO.value
        self.max_output_length = 512
        self.chat_llm = self.init_model()
        self.output_parser = StrOutputParser()
        self.prompt_template = PromptTemplate.from_template("{input}")

    def init_model(self):
        """Initialize and return the ChatHuggingFace LLM."""
        try:
            endpoint = HuggingFaceEndpoint(
                repo_id=self.model_name,
                huggingfacehub_api_token=os.getenv("HF_TOKEN"),
                max_new_tokens=self.max_output_length,
                temperature=0.6,
                top_p=0.95,
                return_full_text=False,
            )
            return ChatHuggingFace(llm=endpoint)
        except Exception as e:
            raise Exception(f"Failed to initialize LLM {self.model_name}: {str(e)}")

    def invoke(self, prompt: str) -> str:
        chain = self.prompt_template | self.chat_llm | self.output_parser
        return chain.invoke({"input": prompt})
