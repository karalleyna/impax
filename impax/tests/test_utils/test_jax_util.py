"""Tests for ldif.util.jax_util."""

import jax
import jax.numpy as jnp
import pytest
import tensorflow as tf
from absl.testing import absltest, parameterized

from impax.utils import jax_util
from ldif.ldif.util import tf_util as orig


@pytest.mark.parametrize("seed", [2, 4, 6, 8])
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_tile_new_axis(seed, axis):
    key = jax.random.PRNGKey(seed)
    key, key2 = jax.random.split(key)

    t = jax.random.normal(key, shape=(10, 2, 3, 4))
    lenght = int(jax.random.randint(key2, (1,), 5, 10))

    gnd = orig.tile_new_axis(tf.convert_to_tensor(t), axis, lenght)
    ret = jax_util.tile_new_axis(t, axis, lenght)

    assert jnp.allclose(gnd.numpy(), ret)


@pytest.mark.parametrize("seed", [2, 4, 6, 8])
def test_zero_by_mask(seed):
    key = jax.random.PRNGKey(seed)
    key, key2 = jax.random.split(key)

    mask = jax.random.uniform(key, shape=(5, 4, 3, 2, 1)) > 0.5
    vals = jax.random.normal(key2, shape=(5, 4, 3, 2, 100))

    gnd = orig.zero_by_mask(tf.convert_to_tensor(mask), tf.convert_to_tensor(vals))
    ret = jax_util.zero_by_mask(mask, vals)

    assert jnp.allclose(gnd.numpy(), ret)
