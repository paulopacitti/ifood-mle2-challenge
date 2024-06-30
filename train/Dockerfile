FROM python:3.11.9-bullseye
WORKDIR /app

VOLUME shared/
ENV MODEL_ID="google/flan-t5-small"
ENV TRAIN_DATA_PATH="/app/shared/data/train_data"
ENV TEST_DATA_PATH="/app/shared/data/test_data"
ENV BATCH_SIZE=1
ENV SAVE_MODEL_PATH="/app/shared/models/saved_model"
ENV DEVICE="cpu"
ENV ITERATIONS=10

COPY requirements.txt requirements.txt
COPY src src/
RUN pip install -r requirements.txt

CMD ["python", "src/main.py"]

EXPOSE 8000