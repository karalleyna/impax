"""Utilities for building neural networks."""
import functools
import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, Callable, Sequence, Union


# local
from impax.utils.logging_util import log

ModuleDef = Any




class TwinInference(nn.Module):
    inference_fn: Callable
    element_count: int
    element_embedding_length: int
    flat_element_length: int

    @nn.compact
    def __call__(self, x):
        remaining_length = self.flat_element_length - self.element_embedding_length
        if remaining_length <= 0:
            log.warning("Using less-tested option: single-tower in twin-tower.")
            remaining_length = self.flat_element_length

        explicit_embedding_length = remaining_length

        assert (
            self.element_embedding_length > 0,
            "Unsafe code: May not be possible to determine. Presence/absence of implicit parameters.",
        )

        prediction, embedding = self.inference_fn(
            x, self.element_count, explicit_embedding_length
        )  # TODO, model_config)

        if remaining_length > 0:
            implicit_parameters, implicit_embedding = self.inference_fn(
                observation,
                element_count,
                element_embedding_length,  # TODO, model_config)
            )
            prediction = jnp.concat([prediction, implicit_parameters], axis=2)
            embedding = jnp.concat([embedding, implicit_embedding], axis=1)
        return prediction, embedding


class EncoderLayer(nn.Module):
    """A single encoder layer."""
    output_dim: int
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.Conv(self.output_dim, kernel_size=[3, 3], strides=2, padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = self.activation(x)
        return x

class DecoderLayer(nn.Module):
    output_dim: int
    spatial_dims: Union[int, Sequence[int]]
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu

    @nn.compact
    def __call__(self, x) -> Any:

        assert (
            len(x.shape) not in [4, 5],
            "Unexpected input dimensionality: %i" % len(x.shape),
        )
        x = nn.Conv(self.output_dim, kernel_size=[5, 5], padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = self.activation(x)
        x = jax.image.resize(x, size=self.spatial_dims, antialias=True)

        return x



class Encoder(nn.Module):
    output_dim: int
    spatial_dim: int
    convolution_layers: Sequence[int]
    dense_layers : Sequence[int]
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu
    

    @nn.compact
    def __call__(self, x) -> Any:
        batch_size = x.shape[0]

        for output_dim in self.convolution_layers:
            x = EncoderLayer(output_dim, use_running_average=self.use_running_average, activation=self.activation)(x)

        cur_dim = (
            functools.reduce(lambda x, y: x * y,  x.shape[-1])
            * self.convolution_layers[-1]
        )

        x = jnp.reshape(x, [batch_size, cur_dim])
        
        for features in self.dense_layers:
            x = nn.Dense(features)(x)
            x = nn.BatchNorm(self.use_running_average)(x)
            x = self.activation(x)
        
        return x



class Decoder(nn.Module):
    """Decode a latent vector into an image."""
    width: int
    height: int
    output_dim: int
    spatial_dims: Sequence[Sequence[int, int]]
    convolution_layers: Sequence[int]
    dense_layers : Sequence[int]
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu
    

    @nn.compact
    def __call__(self, x) -> Any:
        batch_size = x.shape[0]
        for features in self.dense_layers:
            x = nn.Dense(features)(x)
            x = nn.BatchNorm(self.use_running_average)(x)
            x = self.activation(x)
        

        # width = model_config.hparams.w
        # height = model_config.hparams.h
        fc_width = self.width// 2 ** (len(self.convolution_layers))
        fc_height = self.height // 2 ** (len(self.convolution_layers)) 
        x = jnp.reshape(x, [batch_size, fc_height, fc_width, self.convolution_layers[-1]])

        
        for output_dim in reversed(self.convolution_layers):
            x = EncoderLayer(output_dim, use_running_average=self.use_running_average)(x)


        for output_dim, spatial_dim in reversed(zip(self.convolution_layers), self.spatial_dims[:-1])):
            x = DecoderLayer(output_dim, spatial_dim, use_running_average=self.use_running_average, activation=self.activation)(x)

        x = nn.Conv(output_dim,
            kernel_size=[1, 1],
            strides=1,
            padding="SAME"
        )(x)
        x = nn.tanh(x)
        return x


class ResidualLayer(nn.Module):
    """Decode a latent vector into an image."""
    width: int
    height: int
    output_dim: int
    spatial_dims: Sequence[Sequence[int, int]]
    convolution_layers: Sequence[int]
    dense_layers : Sequence[int]
    use_running_average: bool = True
    activation: Callable = nn.leaky_relu
    

    @nn.compact
    def __call__(self, x) -> Any:


def residual_layer(inputs, num_outputs, model_config):
    """A single residual network layer unit."""
    normalizer, normalizer_params = get_normalizer_and_mode(model_config)
    if normalizer is not None:
        output = normalizer(
            inputs,
            is_training=normalizer_params["is_training"],
            trainable=normalizer_params["trainable"],
        )
    else:
        output = inputs
    output = tf.nn.leaky_relu(output)
    output = contrib_layers.fully_connected(
        inputs=output, num_outputs=num_outputs, activation_fn=None, normalizer_fn=None
    )
    if normalizer is not None:
        output = normalizer(
            output,
            is_training=normalizer_params["is_training"],
            trainable=normalizer_params["trainable"],
        )
    output = tf.nn.leaky_relu(output)
    output = contrib_layers.fully_connected(
        inputs=output, num_outputs=num_outputs, activation_fn=None, normalizer_fn=None
    )
    return output + inputs
