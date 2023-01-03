"""
Checks whether or not the original function implementation and ours return the same values.

References:
https://github.com/google/ldif/blob/master/ldif/util/np_util.py 
"""

import jax.numpy as jnp
# global
import numpy as np
import pytest
import tensorflow as tf
from jax import random

# local
from impax.utils import jnp_util


def batch_np(arr, batch_size):
    s = arr.shape
    arr = np.expand_dims(arr, 0)
    tile = [batch_size] + [1] * len(s)
    return np.tile(arr, tile)


def make_coordinate_grid(height, width, is_screen_space, is_homogeneous):
    with tf.name_scope("make_coordinate_grid"):
        x_coords = np.linspace(0.5, width - 0.5, width)
        y_coords = np.linspace(0.5, height - 0.5, height)
        if not is_screen_space:
            x_coords /= width
            y_coords /= height
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, sparse=False, indexing="xy")
        if is_homogeneous:
            homogeneous_coords = np.ones_like(grid_x)
            return np.stack([grid_x, grid_y, homogeneous_coords], axis=2)
        return np.stack([grid_x, grid_y], axis=2)


def make_coordinate_grid_3d(length, height, width, is_screen_space, is_homogeneous):
    x_coords = np.linspace(0.5, width - 0.5, width)
    y_coords = np.linspace(0.5, height - 0.5, height)
    z_coords = np.linspace(0.5, length - 0.5, length)
    if not is_screen_space:
        x_coords /= width
        y_coords /= height
        z_coords /= length
    grid_x, grid_y, grid_z = np.meshgrid(x_coords, y_coords, z_coords, sparse=False, indexing="ij")
    if is_homogeneous:
        homogeneous_coords = np.ones_like(grid_x)
        grid = np.stack([grid_x, grid_y, grid_z, homogeneous_coords], axis=3)
    else:
        grid = np.stack([grid_x, grid_y, grid_z], axis=3)
    return np.swapaxes(grid, 0, 2)


def filter_valid(mask, vals):
    if mask.shape[-1] == 1:
        assert len(mask.shape) == len(vals.shape)
        mask = np.reshape(mask, mask.shape[:-1])
    else:
        assert len(mask.shape) == len(vals.shape) - 1
    return vals[mask, :]


def zero_by_mask(mask, vals, replace_with=0.0):
    vals = vals.copy()
    mask = np.reshape(mask, vals.shape[:-1])
    vals[np.logical_not(mask), :] = replace_with
    return vals


def make_mask(im, thresh=0.0):
    mv = np.min(im)
    assert mv >= 0.0
    return im > thresh


def make_pixel_mask(im):
    channels_valid = im.astype(bool)
    mask = np.any(channels_valid, axis=2)
    assert len(mask.shape) == 2
    return mask


def thresh_and_radius_to_distance(radius, thresh):
    return np.sqrt(-2.0 * radius * np.log(thresh))


def sample_surface(quadrics, centers, radii, length, height, width, renormalize):
    quadric_count = quadrics.shape[0]
    homogeneous_coords = make_coordinate_grid_3d(length, height, width, is_screen_space=False, is_homogeneous=True)
    homogeneous_coords = np.reshape(homogeneous_coords, [length, height, width, 4])
    homogeneous_coords[:, :, :, :3] -= 0.5
    flat_coords = np.reshape(homogeneous_coords, [length * height * width, 4])

    surface_volume = np.zeros([length, height, width, 1], dtype=np.float32)

    max_bf_weights = np.zeros([length, height, width, 1], dtype=np.float32)
    total_bf_weights = np.zeros([length, height, width, 1], dtype=np.float32)
    for qi in range(quadric_count):
        quadric = quadrics[qi, :, :]
        center = centers[qi, :]
        radius = radii[qi, :3]
        offset_coords = flat_coords.copy()
        offset_coords[:, :3] -= np.reshape(center, [1, 3])
        half_distance = np.matmul(quadric, offset_coords.T).T
        algebraic_distance = np.sum(offset_coords * half_distance, axis=1)

        squared_diff = offset_coords[:, :3] * offset_coords[:, :3]
        scale = np.reciprocal(np.minimum(-2 * radius, 1e-6))
        bf_weights = np.exp(np.sum(scale * squared_diff, axis=1))
        volume_addition = np.reshape(algebraic_distance * bf_weights, [length, height, width, 1])
        max_bf_weights = np.maximum(np.reshape(bf_weights, [length, height, width, 1]), max_bf_weights)
        total_bf_weights += np.reshape(bf_weights, [length, height, width, 1])
        surface_volume += volume_addition
    if renormalize:
        surface_volume /= total_bf_weights
    surface_volume[max_bf_weights < 0.0001] = 1.0
    return surface_volume


@pytest.mark.parametrize("batch_size", [2, 4, 8, 16])
def test_batch_np(batch_size, key=random.PRNGKey(0)):
    x = random.uniform(key, shape=(batch_size,))
    ret = jnp_util.batch_jnp(x, batch_size)
    ground_truth = batch_np(np.array(x), batch_size)
    assert jnp.allclose(ret, ground_truth)


@pytest.mark.parametrize("height", [8, 16])
@pytest.mark.parametrize("width", [8])
@pytest.mark.parametrize("is_screen_space", [True, False])
@pytest.mark.parametrize("is_homogeneous", [True, False])
def test_make_coordinate_grid(height, width, is_screen_space, is_homogeneous):
    ret = jnp_util.make_coordinate_grid(height, width, is_screen_space, is_homogeneous)
    ground_truth = make_coordinate_grid(height, width, is_screen_space, is_homogeneous)
    assert jnp.allclose(ret, ground_truth)


@pytest.mark.parametrize("length", [8, 16])
@pytest.mark.parametrize("height", [8])
@pytest.mark.parametrize("width", [8, 16])
@pytest.mark.parametrize("is_screen_space", [True, False])
@pytest.mark.parametrize("is_homogeneous", [True, False])
def test_make_coordinate_grid_3d(length, height, width, is_screen_space, is_homogeneous):
    ret = jnp_util.make_coordinate_grid_3d(length, height, width, is_screen_space, is_homogeneous)
    ground_truth = make_coordinate_grid_3d(length, height, width, is_screen_space, is_homogeneous)
    assert jnp.allclose(ret, ground_truth)


@pytest.mark.parametrize("mask_shape", [(8, 1), (8,)])
@pytest.mark.parametrize("vals_shape", [(8, 1)])
def test_filter_valid(mask_shape, vals_shape, key=random.PRNGKey(0)):
    key0, key1 = random.split(key)
    mask = random.randint(key0, shape=mask_shape, minval=0, maxval=2).astype(bool)
    vals = random.normal(key1, shape=vals_shape)
    ret = jnp_util.filter_valid(mask, vals)
    ground_truth = filter_valid(np.array(mask), np.array(vals))
    assert jnp.allclose(ret, ground_truth)


@pytest.mark.parametrize("mask_shape", [(8,), (8, 1)])
@pytest.mark.parametrize("vals_shape", [(8, 3)])
def test_zero_by_mask(mask_shape, vals_shape, replace_with=0.0, key=random.PRNGKey(0)):
    key0, key1 = random.split(key)
    mask = random.randint(key0, shape=mask_shape, minval=0, maxval=2).astype(bool)
    vals = random.normal(key1, shape=vals_shape)
    ret = jnp_util.zero_by_mask(mask, vals, replace_with)
    ground_truth = zero_by_mask(np.array(mask), np.array(vals), replace_with)
    assert jnp.allclose(ret, ground_truth)


@pytest.mark.parametrize("im_shape", [(8, 3)])
def test_make_mask(im_shape, thresh=0.0, key=random.PRNGKey(0)):
    im = random.uniform(key, shape=im_shape)
    ret = jnp_util.make_mask(im, thresh)
    ground_truth = make_mask(np.array(im), thresh)
    assert jnp.allclose(ret, ground_truth)


@pytest.mark.parametrize("im_shape", [(5, 8, 3)])
def test_make_pixel_mask(im_shape, key=random.PRNGKey(0)):
    im = random.uniform(key, shape=im_shape)
    ret = jnp_util.make_pixel_mask(im)
    ground_truth = make_pixel_mask(np.array(im))
    assert jnp.allclose(ret, ground_truth)


@pytest.mark.parametrize("radius", [8, 3])
def test_thresh_and_radius_to_distance(radius, thresh=0.0):
    ret = jnp_util.thresh_and_radius_to_distance(radius, thresh)
    ground_truth = thresh_and_radius_to_distance(radius, thresh)
    assert jnp.allclose(ret, ground_truth)
