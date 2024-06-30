import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
from evaluate import load
import numpy as np
import nltk
nltk.download("punkt")


class Trainer():
    def __init__(self, model_id, train_dataset, test_dataset, batch_size, iterations, device="cpu"):
        # load model architecture and tokenizer
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        # tokenize train dataset
        self.train_data = load_from_disk(train_dataset)
        self.train_data = self.train_data.map(
            self._convert_to_features, batched=True, remove_columns=self.train_data.column_names)
        self.train_data.set_format(type="torch", columns=[
                                   "input_ids", "attention_mask", "labels", "decoder_attention_mask"])

        # tokenize test dataset
        self.test_data = load_from_disk(test_dataset)
        self.test_data = self.test_data.map(
            self._convert_to_features, batched=True, remove_columns=self.test_data.column_names)
        self.test_data.set_format(type="torch", columns=[
            "input_ids", "attention_mask", "labels", "decoder_attention_mask"])

        self.batch_size = batch_size
        self.iterations = iterations
        self.device = torch.device(device)
        self.model.to(self.device)

        self.trainer = None
        self.rouge = load("rouge")

    def _convert_to_features(self, example_batch):
        input_encodings = self.tokenizer.batch_encode_plus(
            example_batch["input"], padding="longest")
        target_encodings = self.tokenizer.batch_encode_plus(
            example_batch["target"], padding="longest")

        encodings = {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
            "decoder_attention_mask": target_encodings["attention_mask"]
        }

        return encodings

    def _compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        return result

    def train(self):
        training_args = Seq2SeqTrainingArguments(
            "output",
            eval_strategy="no",
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            load_best_model_at_end=True,
            logging_steps=2,
            max_steps=self.iterations,
            num_train_epochs=1,
            per_device_train_batch_size=self.batch_size,
            predict_with_generate=True,
            push_to_hub=False,
            save_strategy="no",
            weight_decay=0.01,
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            compute_metrics=self._compute_metrics,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
        )
        self.trainer.train()

    def evaluate(self):
        self.trainer.evaluate()

    def save(self, path):
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
