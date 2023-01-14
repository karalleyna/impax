"""
Checks whether or not the original function implementation and ours return the same values.

References:
https://github.com/google/ldif/blob/master/ldif/util/geom_util_jnp.py
"""

import math

import jax.numpy as jnp
# global
import numpy as np
import pytest
#import tensorflow as tf
from jax import random

from impax.utils import geom_util_jnp
# ldif is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
#from ldif.ldif.util import geom_util_np

# pylint: enable=g-bad-import-order


@pytest.mark.parametrize("feature_count", [16, 8])
@pytest.mark.parametrize("are_points", [True, False])
def test_apply_4x4(feature_count, are_points, key=random.PRNGKey(0)):
    key0, key1 = random.split(key)
    # apply_4x4(arr, m, are_points=True, feature_count=0):
    """Applies a 4x4 matrix to 3D points/vectors.
    Args:
      arr: Numpy array with shape [..., 3 + feature_count].
      m: Matrix with shape [4, 4].
      are_points: Boolean. Whether to treat arr as points or vectors.
      feature_count: Int. The number of extra features after the points."""

    arr = random.normal(key0, shape=(16, feature_count + 3))
    m = random.normal(key1, shape=(4, 4))

    ret = geom_util_jnp.apply_4x4(arr, m, are_points, feature_count)
    ground_truth = geom_util_jnp.apply_4x4(arr, m, are_points, feature_count)

    assert jnp.allclose(ret, ground_truth, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("batch_size", [8, 16])
@pytest.mark.parametrize("are_points", [True, False])
def test_batch_apply_4x4(batch_size, are_points, key=random.PRNGKey(0)):
    key0, key1 = random.split(key)

    arrs = random.normal(key0, shape=(batch_size, 3))
    ms = random.normal(key1, shape=(batch_size, 4, 4))

    ret = geom_util_jnp.batch_apply_4x4(arrs, ms, are_points)
    ground_truth = geom_util_jnp.batch_apply_4x4(arrs, ms, are_points)

    assert jnp.allclose(ret, ground_truth, atol=1e-4, rtol=1e-4)
