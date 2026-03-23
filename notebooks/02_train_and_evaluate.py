import json
import logging
import os

from src.evaluate import main as eval_main
from src.train import main as train_main


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    # Train (build artifacts)
    train_main()

    # Evaluate (writes reports/eval.json by default)
    eval_main()

    print("Training + evaluation done.")


if __name__ == "__main__":
    main()

