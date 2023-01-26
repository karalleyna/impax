import pytest
from jax import random
import jax.numpy as jnp
import tensorflow as tf
import ldif.ldif.inference.extract_mesh as original
from impax.inference import extract_mesh


def test_marching_cubes(
    resolution=4, mcubes_extent=1.0, key=random.PRNGKey(0)
):
    volume = random.uniform(key, shape=(resolution, resolution, resolution))
    ground_truth0, ground_truth1 = original.marching_cubes(tf.convert_to_tensor(volume), mcubes_extent)

    ret0, ret1 = extract_mesh.marching_cubes(volume, mcubes_extent)
    
    assert ret0 == ground_truth0
    assert ret1.area == ground_truth1.area
    assert jnp.allclose(ret1.center, ground_truth1.center)