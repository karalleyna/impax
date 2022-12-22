"""
References:
https://github.com/google/ldif/blob/master/ldif/util/math_util.py
"""

import jax.numpy as jnp
from jax import vmap


def nonzero_mean(x, eps=1e-8):
    """
    The mean over nonzero values in a tensor.
    """
    total_sum = jnp.sum(x)
    num_nonzeros = jnp.count_nonzero(x).astype(x.dtype)
    num_nonzeros = jnp.where(num_nonzeros == 0.0, eps, num_nonzeros)
    return jnp.divide(total_sum, num_nonzeros)


def increase_frequency(x, out_dim, flatten=False, interleave=True):
    """
    Maps elements of a tensor to a higher frequency, higher dimensional space.
    As shown in NeRF (https://arxiv.org/pdf/2003.08934.pdf), this can help
    networks learn higher frequency functions more easily since they are typically
    biased to low frequency functions. By increasing the frequency of the input
    signal, such biases are mitigated.
    Args:
      t: Tensor with any shape. Type tf.float32. The normalization of the input
        dictates how many dimensions are needed to avoid periodicity. The NeRF
        paper normalizes all ijnputs to the range [0, 1], which is safe.
      out_dim: How many (sine, cosine) pairs to generate for each element of t.
        Referred to as 'L' in NeRF. Integer.
      flatten: Whether to flatten the output tensor to have the same rank as t.
        Boolean. See returns section for details.
      interleave: Whether to interleave the sin and cos results, as described in
        the paper. If true, then the vector will contain [sin(2^0*t_i*pi),
        cos(2^0*t_i*pi), sin(2^1*t_i*pi), ...]. If false, some operations will be
        avoided, but the order will be [sin(2^0*t_i*pi), sin(2^1*t_i*pi), ...,
        cos(2^0*t_i*pi), cos(2^1*t_i*pi), ...].
    Returns:
      Tensor of type tf.float32. Has shape [..., out_dim*2] if flatten is false.
      If flatten is true, then if t has shape [..., N] then the output will have
      shape [..., N*out_dim*2].
    """
    # TODO(kgenova) Without a custom kernel this is somewhat less efficient,
    # because the sin and cos results have to be next to one another in the output
    # but tensorflow only allows computing them with two different ops. Thus it is
    # necessary to do some expensive tf.concats. It probably won't be a bottleneck
    # in most pipelines.

    x = jnp.pi * x
    # 2^0 ... 2^(out_dim - 1)
    scales = jnp.power(2, jnp.arange(out_dim, dtype=jnp.int32)).astype(jnp.float32)
    # rank
    x_rank = len(x.shape)
    # scale_shape = jnp.vstack((jnp.ones((rank, )), out_dim))
    scale_shape = [1] * x_rank + [out_dim]

    scales = jnp.reshape(scales, scale_shape)

    scaled = x[..., None] * scales

    sin_scaled = jnp.sin(scaled)
    cos_scaled = jnp.cos(scaled)
    output = jnp.concatenate([sin_scaled, cos_scaled], axis=-1)

    if interleave:

        output = vmap(
            lambda s, c: jnp.stack([s, c], axis=-1),
            in_axes=(-1, -1),
            out_axes=-2
        )(sin_scaled, cos_scaled)
        output = jnp.reshape(output, output.shape[:-2] + (-1,))

    if flatten:
        x_shape = x.shape
        output = jnp.reshape(output, x_shape[:-1] + (x_shape[-1] * out_dim * 2,))

    return output
