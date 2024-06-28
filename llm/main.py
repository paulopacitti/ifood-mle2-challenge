from train import Trainer
import os


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    model_id = "google/flan-t5-small"
    train_data_path = "data/train_data"
    test_data_path = "data/test_data"
    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    save_model_path = os.environ.get("MODEL_PATH")
    device = os.environ.get("DEVICE")

    trainer = Trainer(
        model_id=model_id,
        train_dataset=train_data_path,
        test_dataset=test_data_path,
        batch_size=batch_size,
        device=device
    )
    trainer.train()
    trainer.save_model(model_path)


if __name__ == "__main__":
    main()
