from train import Trainer
import os


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    model_id = os.environ["MODEL_ID"]
    dataset_id = os.environ["DATASET_ID"]
    input_label = os.environ["INPUT_LABEL"]
    target_label = os.environ["TARGET_LABEL"]
    prefix = os.environ.get("PREFIX", "")
    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    eval_metric = os.environ.get("EVAL_METRIC")
    model_path = os.environ.get("MODEL_PATH")
    device = os.environ.get("DEVICE")

    trainer = Trainer(
        model_id=model_id,
        dataset_id=dataset_id,
        input_label=input_label,
        target_label=target_label,
        prefix=prefix,
        batch_size=batch_size,
        eval_metric=eval_metric,
        device=device
    )
    trainer.train()
    trainer.save_model(model_path)


if __name__ == "__main__":
    main()
