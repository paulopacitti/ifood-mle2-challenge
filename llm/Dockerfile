FROM huggingface/transformers-pytorch-gpu
WORKDIR /app

COPY requirements.txt .
COPY *.py .
COPY entrypoint.sh .

RUN pip install -r requirements.txt

ENTRYPOINT ["/app/entrypoint.sh"]