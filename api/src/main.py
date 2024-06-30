from typing import Union
from fastapi import FastAPI
from model import Model, InputPrompt

model = Model()
app = FastAPI()


@app.post("/generate")
def generate(prompt: InputPrompt):
    return {"response": model.generate(prompt)}
