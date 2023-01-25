import pytest
from jax import random
import tensorflow as tf

import ldif.ldif.util.random_util as original
from impax.utils import random_util


@pytest.mark.parametrize("dim", [0, 1])
def test_random_shuffle_along_dim(dim, key=random.PRNGKey(0)):
    x = random.uniform(key, shape=(12, 21))
    ground_truth = original.random_shuffle_along_dim(tf.convert_to_tensor(x), dim)
    ret = random_util.random_shuffle_along_dim(x, dim)
    assert ret.shape == ground_truth.shape


@pytest.mark.parametrize("batch_size", [24, 1])
def test_random_pan_rotations(batch_size, key=random.PRNGKey(0)):
    ground_truth = original.random_pan_rotations(batch_size)
    ret = random_util.random_pan_rotations(batch_size)
    assert ret.shape == ground_truth.shape


def test_random_pan_rotation():
    ground_truth = original.random_pan_rotation_np()
    ret = random_util.random_pan_rotation_jnp()
    assert ret.shape == ground_truth.shape


@pytest.mark.parametrize("batch_size", [24, 1])
def test_random_rotations(batch_size, key=random.PRNGKey(0)):
    ground_truth = original.random_rotations(batch_size)
    ret = random_util.random_rotations(batch_size)
    assert ret.shape == ground_truth.shape


def test_random_rotation_jnp():
    ground_truth = original.random_rotation_np()
    ret = random_util.random_rotation_jnp()
    assert ret.shape == ground_truth.shape


@pytest.mark.parametrize("batch_size", [24, 1])
def test_random_scales(batch_size, minval=0.0, maxval=5.0, key=random.PRNGKey(0)):
    ground_truth = original.random_scales(batch_size, minval, maxval)
    ret = random_util.random_scales(batch_size, minval, maxval)
    assert ret.shape == ground_truth.shape


@pytest.mark.parametrize("batch_size", [24, 1])
def test_random_transformation(batch_size, key=random.PRNGKey(0)):
    origin = random.uniform(key, shape=(batch_size, 3))
    ground_truth = original.random_transformation(tf.convert_to_tensor(origin))
    ret = random_util.random_transformation(origin)
    assert ret.shape == ground_truth.shape


@pytest.mark.parametrize("batch_size", [24, 1])
def test_random_zoom_transformation(batch_size, key=random.PRNGKey(0)):
    origin = random.uniform(key, shape=(batch_size, 3))
    ground_truth = original.random_zoom_transformation(tf.convert_to_tensor(origin))
    ret = random_util.random_zoom_transformation(origin)
    assert ret.shape == ground_truth.shape


@pytest.mark.parametrize("batch_size", [24, 1])
def test_translation_to_tx(batch_size, key=random.PRNGKey(0)):
    x = random.uniform(key, shape=(batch_size, 3))
    ground_truth = original.translation_to_tx(tf.convert_to_tensor(x))
    ret = random_util.translation_to_tx(x)
    assert ret.shape == ground_truth.shape


@pytest.mark.parametrize("batch_size", [24, 1])
def test_rotation_to_tx(batch_size, key=random.PRNGKey(0)):
    x = random.uniform(key, shape=(batch_size, 3, 3))
    ground_truth = original.rotation_to_tx(tf.convert_to_tensor(x))
    ret = random_util.rotation_to_tx(x)
    assert ret.shape == ground_truth.shape
