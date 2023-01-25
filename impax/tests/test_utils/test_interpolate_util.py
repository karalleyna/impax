import jax
import jax.numpy as jnp
import pytest
from jax import random

from impax.utils import interpolate_util
from ldif.ldif.util import interpolate_util as orig


@pytest.mark.parametrize("seed", [2, 4, 6, 8])
def test_interpolate_np(seed):
    key = random.PRNGKey(seed)
    key, key2, key3 = random.split(key, 3)
    depth = 3
    sample_c = 100
    height = 32
    width = 32

    grid = jax.random.normal(key, (depth, height, width))
    samples = jax.random.normal(key2, (sample_c, 3))
    world2grid = jax.random.normal(key3, (4, 4))

    gnd = orig.interpolate_np(grid, samples, world2grid)

    ret = interpolate_util.interpolate_np(grid, samples, world2grid)

    assert jnp.allclose(gnd, ret)
