import tensorflow as tf

from polystar.common.constants import MODELS_DIR
from polystar.robots_at_robots.globals import settings


def load_tf_model():
    model = tf.saved_model.load(export_dir=str(MODELS_DIR / settings.MODEL_NAME / "saved_model"))
    return model.signatures["serving_default"]
