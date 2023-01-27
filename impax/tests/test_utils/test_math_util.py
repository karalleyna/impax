import pytest
from impax.utils.math_util import increase_frequency
import jax.numpy as jnp


@pytest.mark.parametrize("flatten", [True, False])
def test_increase_frequency(flatten):
    x = jnp.array([1.0, 0.1123, 0.7463], dtype=jnp.float32)
    output_dim = 3
    expected = jnp.array(
        [
            [0, -1, 0, 1, 0, 1],
            [0.345528, 0.938409, 0.648492, 0.761221, 0.987292, 0.158916],
            [0.715278, -0.69884, -0.99973, -0.0232457, 0.0464788, -0.998919],
        ],
        dtype=jnp.float32,
    )
    expected = jnp.array(expected, dtype=jnp.float32)

    if flatten:
        expected = expected.flatten()

    output = increase_frequency(x, output_dim, flatten=flatten, interleave=True)

    assert output.shape == expected.shape
    assert jnp.allclose(output, expected, atol=1e-2, rtol=1e-3)
