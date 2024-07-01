FROM python:3.11.9-bullseye
WORKDIR /app

VOLUME models/
ENV MODEL_PATH="models/saved_model"
ENV SYSTEM_PROMPT="translate to SQL: "
ENV DEVICE="cpu"

COPY requirements.txt requirements.txt
COPY src src/
RUN pip install -r requirements.txt

CMD ["fastapi", "run", "src/main.py"]

EXPOSE 8000