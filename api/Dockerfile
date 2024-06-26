FROM ollama/ollama:latest
WORKDIR /app

ENV model "example.gguf"
ENV system_prompt "translate to SQL: "
ENV template "
{{ if .System }}{{ .System }}{{ end }}
{{ if .Prompt }}{{ .Prompt }}{{ end }}
"
COPY model/$model /app/model.gguf

# create Modelfile
RUN RUN <<EOF
echo "
FROM $model

TEMPLATE """$template""""

SYSTEM """$system_prompt"""
"""" > Modelfile
EOF
RUN ollama create main -f Modelfile

# Serve the model
CMD ollama serve

EXPOSE 11434