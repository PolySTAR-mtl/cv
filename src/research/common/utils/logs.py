import logging


def setup_dev_logs():
    logging.getLogger().setLevel("INFO")
    logging.info("logs started")
