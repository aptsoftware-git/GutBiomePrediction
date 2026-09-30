"""Build once, then reuse: python app/train_default.py [--force]."""
import argparse
import json
from main import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR, TrainingConfig, train_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Retrain even when the dataset and settings match")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--validation", choices=["holdout", "kfold", "loo", "none"], default="holdout")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    info = train_controller(args.data, model_dir=args.output, force=args.force,
                            config=TrainingConfig(validation=args.validation, epochs=args.epochs, batch_size=args.batch_size),
                            progress=lambda message: print(message, flush=True))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
