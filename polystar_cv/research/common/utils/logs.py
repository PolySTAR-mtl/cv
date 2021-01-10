import logging
import warnings


def setup_dev_logs():
    logging.getLogger().setLevel("INFO")
    logging.getLogger("tensorflow").setLevel("ERROR")
    warnings.filterwarnings("ignore")
    logging.info("Dev logs setup")
