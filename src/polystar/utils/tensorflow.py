import tensorflow as tf

# from object_detection.utils import ops as utils_ops


def patch_tf_v2():
    # # patch tf1 into `utils.ops`
    # utils_ops.tf = tf.compat.v1

    # Patch the location of gfile
    tf.gfile = tf.io.gfile
