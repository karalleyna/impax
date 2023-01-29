"""
Code for preprocessing training examples.

This code can be aware of the existence of individual datasets, but it can't be
aware of their internals.
"""

import jax.numpy as jnp
import tensorflow as tf

from impax.datasets import shapenet
from impax.utils import random_util


# Main entry point for preprocessing. This code uses the model config to
# generate an appropriate training example. It uses duck typing, in that
# it should dispatch generation to a dataset to handle internals, and verify
# that the training example contains the properties associated with the config.
# But it doesn't care about anything else.
def preprocess(model_config, dataset, split):
    """Generates a training example object from the model config."""
    # TODO(kgenova) Check if dataset is shapenet. If so, return a ShapeNet
    # training example.

    # Get the input data from the input_fn.
    if split != "train":
        model_config.batch_size = 1

    if model_config.rescaling != 1.0:
        def new_dataset():
            return 0
        factor = model_config.rescaling
        new_dataset.factor = factor
        new_dataset.xyz_render = dataset.xyz_render * factor
        new_dataset.near_surface_samples = dataset.near_surface_samples * factor
        new_dataset.bounding_box_samples = dataset.bounding_box_samples * factor
        xyz = dataset.surface_point_samples[:, :, :3] * factor
        nrm = dataset.surface_point_samples[:, :, 3:]
        new_dataset.surface_point_samples = jnp.concatenate([xyz, nrm], axis=-1)
        new_dataset.grid = dataset.grid * factor
        to_old_world = jnp.array(
            [
                [
                    [1.0 / factor, 0.0, 0.0, 0.0],
                    [0.0, 1.0 / factor, 0.0, 0.0],
                    [0.0, 0.0, 1.0 / factor, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=jnp.float32,
        )
        to_old_world = jnp.tile(to_old_world, [model_config.batch_size, 1, 1])
        new_dataset.world2grid = jnp.matmul(dataset.world2grid, to_old_world)
        new_dataset.mesh_name = dataset.mesh_name
        new_dataset.depth_render = dataset.depth_render
        dataset = new_dataset

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
