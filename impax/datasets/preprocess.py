"""Code for preprocessing training examples.

This code can be aware of the existence of individual datasets, but it can't be
aware of their internals.
"""

import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from impax.datasets import shapenet
from impax.utils import random_util

# pylint: enable=g-bad-import-order


# Main entry point for preprocessing. This code uses the model config to
# generate an appropriate training example. It uses duck typing, in that
# it should dispatch generation to a dataset to handle internals, and verify
# that the training example contains the properties associated with the config.
# But it doesn't care about anything else.
def preprocess(model_config, dataset, split):
    """Generates a training example object from the model config."""
    # TODO(kgenova) Check if dataset is shapenet. If so, return a ShapeNet
    # training example.
    training_example = shapenet.ShapeNetExample(model_config, dataset, split)

    # TODO(kgenova) Look at the model config and verify that nothing is missing.
    if model_config.data_augmentation == "f":
        return training_example
    elif model_config.data_augmentation == "p":  # Pan:
        tx = random_util.random_pan_rotations(model_config.batch_size)
    elif model_config.data_augmentation == "r":  # SO(3):
        tx = random_util.random_rotations(model_config.batch_size)
    elif model_config.data_augmentation == "t":
        origin = training_example.sample_sdf_near_surface(1)[0]
        origin = tf.reshape(origin, [model_config.batch_size, 3])
        tx = random_util.random_transformation(origin)
    elif model_config.data_augmentation == "z":
        origin = training_example.sample_sdf_near_surface(1)[0]
        origin = tf.reshape(origin, [model_config.batch_size, 3])
        tx = random_util.random_zoom_transformation(origin)
    training_example.apply_transformation(tx)
    if model_config.crop_input:
        training_example.crop_input(model_config.num_input_crops)
    if model_config.num_supervision_crops and (model_config.train or model_config.eval):
        training_example.crop_supervision(model_config.num_supervision_crops)
    return training_example
