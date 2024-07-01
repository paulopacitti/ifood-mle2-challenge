from datasets import load_dataset


def format_dataset(example):
    return {"input": "translate to SQL: " + example["question"], "target": example["sql"]["human_readable"]}


print("> Loading dataset...")
train_data = load_dataset(
    "wikisql", split="train+validation", trust_remote_code=True)
test_data = load_dataset("wikisql", split="test", trust_remote_code=True)

print("> Formating dataset...")
train_data = train_data.map(
    format_dataset, remove_columns=train_data.column_names)
test_data = test_data.map(
    format_dataset, remove_columns=test_data.column_names)

print("> Saving dataset...")
train_data.save_to_disk("shared/data/train_data")
test_data.save_to_disk("shared/data/test_data")
