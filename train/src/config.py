import os

MODEL_ID = os.getenv("MODEL_ID", "google/flan-t5-small")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/app/shared/data/train_data")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "/app/shared/data/test_data")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH", "/app/sharedmodels/saved_model")
DEVICE = os.getenv("DEVICE", "cpu")
ITERATIONS = int(os.getenv("ITERATIONS", 10))