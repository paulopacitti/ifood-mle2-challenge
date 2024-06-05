import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets
from evaluate import load
import nltk
import numpy as np


class Trainer():
    def __init__(self, model_id, dataset_id, input_label, target_label, prefix="", batch_size=16, eval_metric="", device="cpu"):

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        self.dataset_id = dataset_id
        self.raw_dataset = load_dataset(dataset_id)
        self.input_label = input_label
        self.target_label = target_label
        self.max_input_length, self.max_target_length = self._get_dataset_max_length(
            self.raw_dataset)

        self.prefix = prefix
        self.batch_size = batch_size
        self.eval_matric = eval_metric
        self.device = torch.device(device)

        self.trainer = None

    def _get_dataset_max_length(self, dataset):
        tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: self.tokenizer(
            x[self.input_label], truncation=True), batched=True, remove_columns=[self.target_label])
        max_input_length = max([len(x)
                                for x in tokenized_inputs["input_ids"]])

        tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: self.tokenizer(
            x[self.target_label], truncation=True), batched=True, remove_columns=[self.input_label])
        max_target_length = max([len(x)
                                for x in tokenized_targets["input_ids"]])
        return max_input_length, max_target_length

    def _preprocess_function(self, sample):
        inputs = [self.prefix + doc for doc in sample[self.input_label]]
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        labels = self.tokenizer(
            text_target=sample[self.target_label], max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                         for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                          for label in decoded_labels]

        # Note that other metrics may not have a `use_aggregator` parameter
        # and thus will return a list, computing a metric for each sentence.
        result = self.metric.compute(predictions=decoded_preds,
                                     references=decoded_labels, use_stemmer=True, use_aggregator=True)
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(
            pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model)
        args = Seq2SeqTrainingArguments(
            f"{self.model_id}-finetuned-xsum",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            predict_with_generate=True,
        )

        self.trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=self.tokenized_dataset["train"].select(range(100)),
            eval_dataset=self.tokenized_dataset["validation"].select(
                range(100)),
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )
        self.trainer.train()

    def evaluate(self):
        self.trainer.evaluate()

    def save(self, path):
        self.trainer.save_model(path)
