"""Build both default models once, then reuse: python app/train_default.py [--force]."""
import argparse
import json
from main import (DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR, MICROBIOME_DATA_PATH,
                  DEFAULT_MICROBIOME_MODEL_DIR, TrainingConfig, MicrobiomeConfig,
                  train_controller, train_microbiome_controller)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Retrain even when data and settings match")
    parser.add_argument("--task", choices=["both", "metabolite", "microbiome"], default="both")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Metabolite training file")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_DIR), help="Metabolite model directory")
    parser.add_argument("--microbiome-data", default=str(MICROBIOME_DATA_PATH))
    parser.add_argument("--microbiome-output", default=str(DEFAULT_MICROBIOME_MODEL_DIR))
    parser.add_argument("--validation", choices=["holdout", "kfold", "loo", "none"], default="holdout",
                        help="Metabolite evaluation")
    parser.add_argument("--microbiome-validation", choices=["holdout", "kfold", "loo", "none"], default="kfold")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    progress = lambda message: print(message, flush=True)
    results = {}
    if args.task in {"both", "metabolite"}:
        results["metabolite"] = train_controller(args.data, model_dir=args.output, force=args.force,
            config=TrainingConfig(validation=args.validation, epochs=args.epochs, batch_size=args.batch_size),
            progress=progress)
    if args.task in {"both", "microbiome"}:
        results["microbiome"] = train_microbiome_controller(args.microbiome_data, model_dir=args.microbiome_output,
            force=args.force, config=MicrobiomeConfig(validation=args.microbiome_validation), progress=progress)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
