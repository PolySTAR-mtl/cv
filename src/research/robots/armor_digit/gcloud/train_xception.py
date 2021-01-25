import logging
import warnings
from argparse import ArgumentParser

from tensorflow.python.keras.applications.xception import Xception

from research.robots.armor_digit.gcloud.train import train_evaluate_digit_pipeline
from research.robots.armor_digit.pipeline import ArmorDigitKerasPipeline

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.getLogger("tensorflow").setLevel("ERROR")
    warnings.filterwarnings("ignore")

    _parser = ArgumentParser()
    _parser.add_argument("--job-dir", type=str, required=True)
    _parser.add_argument("--lr", type=float, required=True)
    _parser.add_argument("--dropout", type=float, required=True)
    _parser.add_argument("--dense-size", type=int, required=True)
    _args = _parser.parse_args()

    _pipeline = ArmorDigitKerasPipeline.from_transfer_learning(
        model_factory=Xception,
        logs_dir=_args.job_dir,
        input_size=72,
        lr=_args.lr,
        dense_size=_args.dense_size,
        dropout=_args.dropout,
    )

    train_evaluate_digit_pipeline(_pipeline, _args.job_dir)
