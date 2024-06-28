python train.py
python llama.cpp/convert.py models/$MODEL_NAME \
  --outfile "${MODEL_NAME}".gguf \
  --outtype f16
