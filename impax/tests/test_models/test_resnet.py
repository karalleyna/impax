"""Tests for ResNet"""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp

from impax.models.resnet_v1 import ResNet18 as ResNet18V1
from impax.models.resnet_v2 import ResNet18 as ResNet18V2

from impax.models.resnet_v1 import ResNet50 as ResNet50V1
from impax.models.resnet_v2 import ResNet50 as ResNet50V2


class ResNetTest(parameterized.TestCase):
    """Test cases for ResNet model definitions."""

    @parameterized.product(model=(ResNet50V1, ResNet50V2))
    def test_resnet_v1_model(self, model):
        """Tests ResNet model definitions and outputs (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = model(num_classes=10, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((8, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # Resnet50 model will create parameters for the following layers:
        #   conv + batch_norm = 2
        #   BottleneckResNetBlock in stages: [3, 4, 6, 3] = 16
        #   Followed by a Dense layer = 1
        self.assertLen(variables["params"], 19)

    @parameterized.product(model=(ResNet18V1, ResNet18V2))
    def test_resnet_18_v1_model(self, model):
        """Tests ResNet18 model definitions and outputs (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = model(num_classes=2, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((1, 64, 64, 3), jnp.float32))

        self.assertLen(variables, 2)
        self.assertLen(variables["params"], 11)


if __name__ == "__main__":
    absltest.main()
