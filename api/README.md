# API

Docs on how serving the model works

Run with docker:
```sh
 docker run -v ./models:/app/models --env MODEL_PATH="/app/models/saved_model" -p 8000:8000 paulopacitti/t5-api
```