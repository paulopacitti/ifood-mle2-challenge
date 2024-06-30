from train import Trainer
import os
import config


def main():
    print("> Training model...")
    trainer = Trainer(
        model_id=config.MODEL_ID,
        train_dataset=config.TRAIN_DATA_PATH,
        test_dataset=config.TEST_DATA_PATH,
        batch_size=config.BATCH_SIZE,
        iterations=config.ITERATIONS,
        device=config.DEVICE
    )
    trainer.train()
    trainer.save(config.SAVE_MODEL_PATH)
    print("> Model fine-tuned and saved.")


if __name__ == "__main__":
    main()
