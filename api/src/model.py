from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel
from config import MODEL_PATH, DEVICE, SYSTEM_PROMPT


class InputPrompt(BaseModel):
    message: str


class GeneratedText(BaseModel):
    generated_text: str
    generated_token_ids: str


class Model():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        self.pipeline = pipeline(task="text2text-generation", model=self.model, device=DEVICE,
                                 tokenizer=self.tokenizer, max_new_tokens=128)

    def generate(self, prompt: InputPrompt) -> List[GeneratedText]:
        return self.pipeline(SYSTEM_PROMPT + prompt.message)[0]["generated_text"]
