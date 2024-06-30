from typing import Union
from fastapi import FastAPI
from model import Model, InputPrompt

model = Model()
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/generate")
def generate(prompt: InputPrompt):
    return {"response": model.generate(prompt)}
