"""Generic Tensorflow utility functions."""

import jax.numpy as jnp


def tile_new_axis(t, axis, length):
    """Creates a new tensor axis and tiles it to a specified length.
    Args:
      t: Tensor with any shape.
      axis: The index for the new axis.
      length: The length of the new axis.
    Returns:
      Tensor with one extra dimension of length 'length' added to 't' at index
      'axis'.
    """
    t = jnp.expand_dims(t, axis=axis)
    cur_shape = t.shape
    tile_shape = [1] * len(cur_shape)
    tile_shape[axis] = length
    return jnp.tile(t, tile_shape)


def zero_by_mask(mask, vals, replace_with=0.0):
    """ "Sets the invalid part of vals to the value of replace_with.
    Args:
      mask: Boolean tensor with shape [..., 1].
      vals: Tensor with shape [..., channel_count].
      replace_with: Value to put in invalid locations, if not 0.0. Dtype should be
        compatible with that of vals. Can be a scalar tensor.
    Returns:
      Tensor with shape [..., channel_count].
    """
    mask_shape = mask.shape
    vals_shape = vals.shape
    assert mask.shape == (vals_shape[:-1] + [1])
    assert vals.shape == (mask_shape[:-1] + [vals_shape[-1]])
    return jnp.where(mask, vals, replace_with)


remove_element = jnp.delete
