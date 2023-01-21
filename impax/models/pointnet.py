from typing import Any, Optional, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any


class AffineTransformation(nn.Module):
    @nn.compact
    def __call__(self, x):
        """Maps a point set to an affine transformation and a translation."""
        batch_size, point_count, _ = x.shape

        x = jnp.expand_dims(x, axis=2)

        x = nn.Conv(64, kernel_size=[1, 1], padding="VALID", strides=[1, 1])(x)

        x = nn.Conv(128, kernel_size=[1, 1], padding="VALID", strides=[1, 1])(x)
        x = nn.Conv(1024, kernel_size=[1, 1], padding="VALID", strides=[1, 1])(x)

        x = nn.max_pool(x, window_shape=[point_count, 1], padding="VALID")
        # flatten
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        return x


class OrthogonalTransformation(nn.Module):
    """A block to learn an orthogonal feature transformation matrix."""

    @nn.compact
    def __call__(x) -> Any:
        batch_size, point_count, _, input_feature_count = x.shape
        assert input_feature_count == 64

        x = nn.Conv(
            64,
            kernel_size=[1, 1],
            padding="VALID",
            stride=[1, 1],
        )(x)
        x = nn.Conv(
            128,
            kernel_size=[1, 1],
            padding="VALID",
            stride=[1, 1],
        )(x)
        x = nn.Conv(
            1024,
            kernel_size=[1, 1],
            padding="VALID",
            stride=[1, 1],
        )(x)
        x = nn.max_pool(x, kernel_size=[point_count, 1], padding="VALID")
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)

        return x


class MLP(nn.Module):
    layer_sizes: Sequence[int]
    activation: Optional[Any] = nn.relu
    activation_final: Optional[Any] = None

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for layer_size in self.layer_sizes[:-1]:
            x = nn.Dense(features=layer_size)(x)
            if self.activation is not None:
                x = self.activation(x)
        x = nn.Dense(features=self.layer_sizes[-1])(x)
        if self.activation_final is None:
            return x
        return self.activation_final(x)


class CNN(nn.Module):
    """A multi-layer perceptron."""

    features: Sequence[Tuple[int, int]]

    @nn.compact
    def __call__(self, x):

        for feature in self.features:
            x = nn.Conv(feature, kernel_size=[1, 1], padding="VALID", strides=1)(x)

        return x


class PointNet(nn.Module):
    """Bottleneck ResNet block."""

    output_dim: int
    maxpool_feature_count: int = 1024
    affine_transformation: bool = True
    orthogonal_transformation: bool = False

    @nn.compact
    def __call__(self, x):
        batch_size, point_count, feature_count = x.shape
        point_positions = x[..., 0:3].copy()
        point_features = x[..., 3:].copy()
        feature_count -= 3
        if self.affine_transformation:
            x = AffineTransformation()(x)
            # transformation
            transformation_bias = jnp.array(
                [1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=jnp.float32
            )
            transformation = jnp.zeros((*x.shape[:-1], 9))
            transformation = transformation + transformation_bias
            transformation = transformation.reshape((batch_size, 3, 3))
            translation = jnp.zeros(([x.shape[0], 1, 3]))
            transformed_points = point_positions + translation
            transformed_points = transformed_points @ transformation
            if feature_count > 0:
                x = jnp.concat([transformed_points, point_features], axis=2)

        # Go from NWC to NCW so that the final reduce can be faster.
        assert len(x.shape) == 3

        x = CNN([64, 64])(x)

        if self.orthogonal_transformation:
            x = OrthogonalTransformation()(x)
            transformation = jnp.add(
                jnp.zeros((*x.shape[:-1], self.output_dim**2), dtype=jnp.float32),
                jnp.eye(64).flatten().astype(jnp.float32),
            )
            transformation = jnp.reshape(
                transformation, [batch_size, self.output_dim, 64]
            )

            x = jnp.matmul(
                jnp.reshape(x, [batch_size, point_count, 64]),
                transformation,
            )
            x = jnp.expand_dims(x, axis=2)

        x = CNN([64, 128, self.maxpool_feature_count])(x)

        assert len(x.shape) == 3

        x = jnp.max(x, axis=1)
        x = MLP([512, 256, self.output_dim])(x)
        return x


if __name__ == "__main__":
    module = PointNet(2, 2)
    import jax

    params = module.init(jax.random.PRNGKey(0), jnp.zeros((2, 3, 4)))

    print(module.apply(params, jnp.zeros((2, 3, 4))).shape)
