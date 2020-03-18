import tensorflow as tf

from polystar.common.constants import MODELS_DIR


def load_tf_model():
    model = tf.saved_model.load(
        export_dir=str(
            MODELS_DIR
            / "robots"
            / "ssd_mobilenet_v2_roco_2018_03_29_20200314_015411_TWITCH_TEMP_733_IMGS_29595steps"
            / "saved_model"
        )
    )
    return model.signatures["serving_default"]
