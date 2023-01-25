"""
Utilties for working with sdf and pseudo-sdf functions.
References:
https://github.com/google/ldif/blob/master/ldif/util/sdf_util.py
"""

import jax
import jax.numpy as jnp


def apply_class_transfer(
    sdf, soft_transfer, offset, sigmoid_normalization: True, hardness=100.0, dtype=None
):
    """Applies a class label transformation to an input sdf.
    Args:
      sdf: Tensor of any shape. The sdfs to transform elementwise.
      model_config: A ModelConfig object.
      soft_transfer: Boolean. Whether the input gt should have a smooth
        classification transfer applied, or a hard one. A hard transformation
        destroys gradients.
      offset: The isolevel at which the surface lives.
      dtype: tf.float32 or tf.bool if specified. The output type. A soft transfer
        is always a float32, so this parameter is ignored if soft_transfer is
        true. If soft_transfer is false, a cast from bool to float32 is made if
        necessary. Defaults to tf.float32.
    Returns:
      classes: Tensor of the same shape as sdf.
    """
    # If the prediction defines the surface boundary at a location other than
    # zero, we have to offset before we apply the classification transfer:
    if offset:
        sdf -= offset
    if soft_transfer:
        # todo: this should enable trainable hardness but not possible
        if sigmoid_normalization:
            hardness = hardness
        else:
            hardness = hardness
        return jax.nn.sigmoid(hardness * sdf)
    else:
        if dtype is None or dtype == float:
            return jax.lax.convert_element_type(
                sdf > 0.0, float
            )  # tf.cast(sdf > 0.0, dtype=tf.float32)
        else:
            return sdf > 0.0


def apply_density_transfer(sdf):
    """Applies a density transfer function to an input sdf.
    The input sdf could either be from a prediction or the ground truth.
    Args:
      sdf: Tensor of any shape. The sdfs to transform elementwise.
    Returns:
      densities: Tensor of the same shape as sdf. Contains values in the range
      0-1, where it is 1 near the surface. The density is not a pdf, as it does
      not sum to 1 over the tensor.
    """
    # TODO(kgenova) This is one of the simplest possible transfer functions,
    # but is it the right one? Falloff rate should be controlled, and a 'signed'
    # density might even be relevant.
    return jnp.exp(-jnp.abs(sdf))
