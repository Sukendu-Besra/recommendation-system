import argparse
import logging
import os

from src.recsys import ModelHyperParams, build_engine_and_artifacts


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recommendation models and export artifacts.")
    parser.add_argument("--data-dir", default="data", help="Where to store/download MovieLens data.")
    parser.add_argument("--model-path", default="models/artifacts.joblib", help="Output path for saved artifacts.")
    parser.add_argument("--retrain", action="store_true", help="Train even if artifacts already exist.")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-user-ratings", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)

    if os.path.exists(args.model_path) and not args.retrain:
        logging.getLogger(__name__).info("Artifacts already exist at %s. Use --retrain to rebuild.", args.model_path)
        return

    hyperparams = ModelHyperParams()
    engine, report = build_engine_and_artifacts(
        data_dir=args.data_dir,
        model_path=args.model_path,
        hyperparams=hyperparams,
        test_fraction=args.test_fraction,
        seed=args.seed,
        min_user_ratings=args.min_user_ratings,
    )

    logging.getLogger(__name__).info("Training done. Report: %s", report)


if __name__ == "__main__":
    main()

