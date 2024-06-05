from transformers import pipeline

# load model and tokenizer from huggingface hub with pipeline
summarizer = pipeline(
    "summarization", model="philschmid/flan-t5-base-samsum", device="mps")

# select a random test sample
sample = dataset['test'][randrange(len(dataset["test"]))]
print(f"dialogue: \n{sample['dialogue']}\n---------------")

# summarize dialogue
res = summarizer(sample["dialogue"])

print(f"flan-t5-base summary:\n{res[0]['summary_text']}")
