import os

MODEL_PATH = os.getenv("MODEL_PATH", f"models/saved_model/")
DEVICE = os.getenv("DEVICE", "cpu")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "translate to SQL: ")
