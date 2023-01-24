"""Tests for ldif.util.jax_util."""

import jax.numpy as jnp
from absl.testing import parameterized
from absl.testing import absltest


from impax.utils import jax_util


DISTANCE_EPS = 1e-6
EXPECTED = {
    (1, 0): jnp.array([[1.0, 2.0]]),
    (0, 0): jnp.array([[3.0, 4.0]]),
    (0, 1): jnp.array([[2.0], [4.0]]),
    (1, 1): jnp.array([[1.0], [3.0]]),
}


class JAXUtilTest(parameterized.TestCase):
    @parameterized.product(elt=[0, 1], axis=[0, 1])
    def testRemoveElement(self, elt, axis):
        expected = EXPECTED[(elt, axis)]
        initial = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        removed = jax_util.remove_element(
            initial, jnp.array([elt], dtype=jnp.int32), axis
        )

        distance = float(jnp.sum(jnp.abs(expected - removed)))
        self.assertLess(
            distance,
            DISTANCE_EPS,
            "Expected \n%s\n but got \n%s"
            % (jnp.array_str(expected), jnp.array_str(removed)),
        )


if __name__ == "__main__":
    absltest.main()
