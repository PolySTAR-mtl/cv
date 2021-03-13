import logging

from research.roco_detection.training.tf1.pretrained_models import PretrainedModels
from research.roco_detection.training.tf1.records import Records

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    PretrainedModels.SSD_MOBILENET_V2.setup(Records.TWITCH).train_and_export(nb_steps=6)
