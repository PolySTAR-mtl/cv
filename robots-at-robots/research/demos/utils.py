import tensorflow as tf

from polystar.common.constants import MODELS_DIR


def load_tf_model():
    model = tf.saved_model.load(export_dir=str(MODELS_DIR / "robots/ssd_mobilenet_v2_coco_2018_03_29" / "saved_model"))
    return model.signatures["serving_default"]
