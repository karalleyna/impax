"""Utilities for building neural networks."""
from typing import Any, Callable, Sequence, Union

import jax
import jax.numpy as jnp
from flax import linen as nn

# local
from impax.utils.logging_util import log

ModuleDef = Any


class TwinInference(nn.Module):
    inference_fn: Callable
    element_count: int
    element_embedding_length: int
    flat_element_length: int

    @nn.compact
    def __call__(self, observation):
        remaining_length = self.flat_element_length - self.element_embedding_length
        if remaining_length <= 0:
            log.warning("Using less-tested option: single-tower in twin-tower.")
            explicit_embedding_length = self.flat_element_length
        else:
            explicit_embedding_length = remaining_length

        assert self.element_embedding_length > 10
        # "Unsafe code: May not be possible to determine. Presence/absence of implicit parameters."

        prediction, embedding = self.inference_fn(
            observation, explicit_embedding_length
        )

        if remaining_length > 0:
            implicit_parameters, implicit_embedding = self.inference_fn(
                observation,
                self.element_embedding_length,
            )
            prediction = jnp.concat([prediction, implicit_parameters], axis=2)
            embedding = jnp.concat([embedding, implicit_embedding], axis=1)
        return prediction, embedding


class EncoderLayer(nn.Module):
    """A single encoder layer."""

    output_dim: int
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu
    name: str = ""

    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.Conv(self.output_dim, kernel_size=[3, 3], strides=2, padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = self.activation(x)
        return x


class DecoderLayer(nn.Module):
    output_dim: int
    output_shape: Union[int, Sequence[int]]
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu
    name: str = ""

    @nn.compact
    def __call__(self, x) -> Any:

        assert len(x.shape) in [4, 5]
        # "Unexpected input dimensionality / 3D Upsampling has not been implemented.: %i" % len(x.shape),

        x = nn.Conv(self.output_dim, kernel_size=[5, 5], padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = self.activation(x)
        x = jax.image.resize(x, shape=self.output_shape, method="cubic", antialias=True)

        return x


class Encoder(nn.Module):
    convolution_layers: Sequence[int]
    dense_layers: Sequence[int]
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x) -> Any:
        batch_size = x.shape[0]

        spatial_shapes = [x.shape]
        for i, output_dim in enumerate(self.convolution_layers):
            x = EncoderLayer(
                output_dim,
                use_running_average=self.use_running_average,
                activation=self.activation,
                name=f"encoder_{i}",
            )(x)
            # remove batch
            spatial_shapes.append(x.shape)

        x = jnp.reshape(x, (batch_size, -1))

        for features in self.dense_layers:
            x = nn.Dense(features)(x)
            x = nn.BatchNorm(self.use_running_average)(x)
            x = self.activation(x)

        return x, spatial_shapes


class Decoder(nn.Module):
    """Decode a latent vector into an image."""

    output_dim: int
    convolution_layers: Sequence[int]
    dense_layers: Sequence[int]
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x, spatial_shapes) -> Any:
        # add another dense layer to map output and remove first dense_layer
        for features in reversed(self.dense_layers[:-1]):
            x = nn.Dense(features)(x)
            x = nn.BatchNorm(self.use_running_average)(x)
            x = self.activation(x)

        x = nn.Dense(
            spatial_shapes[-1][1] * spatial_shapes[-1][2] * spatial_shapes[-1][3]
        )(x)
        x = nn.BatchNorm(self.use_running_average)(x)
        x = self.activation(x)

        x = jnp.reshape(x, spatial_shapes[-1])

        for i, (output_dim, output_shape) in enumerate(
            zip(reversed(self.convolution_layers[:-1]), reversed(spatial_shapes[1:-1]))
        ):
            # remove batch_size, number_of_channels from output_shape
            x = DecoderLayer(
                output_dim,
                output_shape,
                use_running_average=self.use_running_average,
                activation=self.activation,
                name=f"decoder_{i}",
            )(x)

        # remove batch_size, number_of_channels from output_shape
        x = DecoderLayer(
            spatial_shapes[0][-1],
            spatial_shapes[0],
            use_running_average=self.use_running_average,
            activation=self.activation,
            name="decoder_last",
        )(x)

        x = nn.Conv(self.output_dim, kernel_size=[1, 1], strides=1, padding="SAME")(x)
        x = nn.tanh(x)
        return x


class ResidualLayer(nn.Module):
    """A single residual network layer unit."""

    output_dim: int
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x) -> Any:
        identity = x

        # todo: always normalize
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = self.activation(x)

        x = nn.Dense(features=self.output_dim)(x)

        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = self.activation(x)

        x = nn.Dense(features=self.output_dim)(x)

        return x + identity


class Embedder(nn.Module):
    """Encodes an input observation tensor to a fixed lengthh feature vector."""

    dense_layers: Sequence[int]
    conv_layers: Sequence[int]
    use_running_average: bool = True
    activation: nn.Module = nn.leaky_relu

    @nn.compact
    def __call__(self, x):
        batch_size, image_count, height, width, channel_count = x.shape
        log.info(msg="Input shape to early-fusion cnn: %s" % str(x.shape))
        if image_count == 1:
            im = jnp.reshape(x, [batch_size, height, width, channel_count])
        else:
            im = jnp.reshape(
                jnp.transpose(x, axes=[0, 2, 3, 1, 4]),
                [batch_size, height, width, image_count * channel_count],
            )

        x, spatial_shapes = Encoder(
            self.conv_layers,
            self.dense_layers,
            self.use_running_average,
            self.activation,
        )(im)

        return x, spatial_shapes


if __name__ == "__main__":
    conv_layers = [16, 32, 64, 128, 128]
    dense_layers = [10, 20, 30]

    key = jax.random.PRNGKey(0)

    inp = jnp.ones((32, 64, 64, 3))

    encoder = Encoder(conv_layers, dense_layers)
    encoder_vars = encoder.init(key, inp)

    encoding, spatial_shapes = encoder.apply(encoder_vars, inp)

    decoder = Decoder(3, conv_layers, dense_layers)
    decoder_vars = decoder.init(key, encoding, spatial_shapes)

    decoded = decoder.apply(decoder_vars, encoding, spatial_shapes)

    assert decoded.shape == inp.shape

    res = ResidualLayer(3)
    res_vars = res.init(key, inp)

    emb = Embedder(dense_layers, conv_layers)
    inp = jnp.ones((32, 10, 64, 64, 3))
    emb_vars = emb.init(key, inp)

    embedded, dims = emb.apply(emb_vars, inp)
