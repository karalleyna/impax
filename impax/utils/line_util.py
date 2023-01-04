"""
Functions for differentiable line drawing.

References:
https://github.com/google/ldif/blob/master/ldif/util/line_util.py

"""

import jax.numpy as jnp
from jax import vmap
from jax import nn


def line_to_image(line_parameters, height, width, falloff=2.0):
    """Renders a 'line' (rectangle) image from its parameterization.
    Args:
      line_parameters: Tensor with shape [5]. Contains the angle of rotation, x
        center, y center, x thickness, and y thickness in order. Coordinates are
        specified in radians and screen space, respectively, with a top left
        origin.
      height: Int containing height in pixels of the desired output render.
      width: Int containing width in pixels of the desired output render.
      falloff: Float containing the soft falloff parameter for the line. Bigger
        values indicate a longer fade into grey outside the line borders. If None
        is passed instead, the line will be drawn with sharp lines.
    Returns:
      Tensor with shape [height, width, 1] containing the line image. Colors are
      in the range [0, 1]- 0 is entirely inside the line, and 0 is entirely
      outside the line.
    """
    # Initialize constant coordinates to be hit-tested:
    x_coords = jnp.linspace(0.5, width - 0.5, width)
    y_coords = jnp.linspace(0.5, height - 0.5, height)
    grid_x, grid_y = jnp.meshgrid(x_coords, y_coords, sparse=False, indexing="xy")
    coords = jnp.stack([grid_x, grid_y], axis=2)
    coords = coords.astype(jnp.float32)

    # Construct rectangle from ijnput parameters:
    angle_of_rotation, px, py, lx, ly = line_parameters
    angle_of_rotation = -angle_of_rotation
    center = line_parameters[1:3]

    v0 = jnp.stack([px + lx, py + ly], axis=0)
    v0 = rotate_about_point(angle_of_rotation, center, v0)

    v1 = jnp.stack([px + lx, py - ly], axis=0)
    v1 = rotate_about_point(angle_of_rotation, center, v1)

    v2 = jnp.stack([px - lx, py - ly], axis=0)
    v2 = rotate_about_point(angle_of_rotation, center, v2)

    coords = jnp.reshape(coords, [height * width, 1, 2])

    first_direction_insidedness = fractional_vector_projection(
        v1, v0, coords, falloff=falloff
    )
    second_direction_insidedness = fractional_vector_projection(
        v1, v2, coords, falloff=falloff
    )

    insidedness = first_direction_insidedness * second_direction_insidedness
    color = 1.0 - insidedness

    return jnp.reshape(color, [height, width, 1])


def fractional_vector_projection(e0, e1, p, falloff=2.0):
    """Returns a fraction describing whether p projects inside the segment e0 e1.
    If p projects inside the segment, the result is 1. If it projects outside,
    the result is a fraction that is always greater than 0 but monotonically
    decreasing as the distance to the inside of the segment increase.
    Args:
      e0: Tensor with two elements containing the first endpoint XY locations.
      e1: Tensor with two elements containing the second endpoint XY locations.
      p: Tensor with shape [batch_size, 2] containing the query points.
      falloff: Float or Scalar Tensor specifying the softness of the falloff of
        the projection. Larger means a longer falloff.
    """

    batch_size = p.shape[0]
    # p = jnp.reshape(p, [batch_size, 2])
    e0 = jnp.reshape(e0, [1, 2])
    e1 = jnp.reshape(e1, [1, 2])
    e01 = e1 - e0
    # Normalize for vector projection:
    e01_norm = jnp.sqrt(e01[0, 0] * e01[0, 0] + e01[0, 1] * e01[0, 1])
    e01_normalized = e01 / jnp.reshape(e01_norm, [1, 1])
    e0p = p - e0
    e0p_dot_e01_normalized = jnp.matmul(
        jnp.reshape(e0p, [1, batch_size, 2]),
        jnp.transpose(jnp.reshape(e01_normalized, [1, 1, 2]), axes=(0, 2, 1)),
    )
    e0p_dot_e01_normalized = (
        jnp.reshape(e0p_dot_e01_normalized, [batch_size]) / e01_norm
    )
    if falloff is None:
        left_sided_inside = jnp.logical_and(
            e0p_dot_e01_normalized >= 0, e0p_dot_e01_normalized <= 1
        ).astype(jnp.float32)

        return left_sided_inside

    # Now that we have done the left side, do the right side:
    e10_normalized = -e01_normalized
    e1p = p - e1
    e1p_dot_e10_normalized = jnp.matmul(
        jnp.reshape(e1p, [1, batch_size, 2]),
        jnp.transpose(jnp.reshape(e10_normalized, [1, 1, 2]), axes=(0, 2, 1)),
    )
    e1p_dot_e10_normalized = (
        jnp.reshape(e1p_dot_e10_normalized, [batch_size]) / e01_norm
    )

    # Take the maximum of the two projections so we face it from the positive
    # direction:
    proj = jnp.maximum(e0p_dot_e01_normalized, e1p_dot_e10_normalized)
    proj = jnp.maximum(proj, 1.0)

    # A projection value of 1 means at the border exactly.
    # Take the max with 1, to throw out all cases besides 'left' overhang.

    fractional_falloff = 1.0 / (jnp.power(falloff * (proj - 1), 2.0) + 1.0)
    return fractional_falloff


def rotate_about_point(angle_of_rotation, point, to_rotate):
    """Rotates a single input 2d point by a specified angle around a point."""

    cos_angle = jnp.cos(angle_of_rotation)
    sin_angle = jnp.sin(angle_of_rotation)
    rotation_matrix = jnp.array(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]]
    ).reshape((1, 2, 2))
    to_rotate = jnp.reshape(to_rotate, (1, 1, 2))
    point = jnp.reshape(point, (1, 1, 2))
    to_rotate = to_rotate - point
    to_rotate = jnp.matmul(rotation_matrix, jnp.transpose(to_rotate, axes=(0, 2, 1)))
    to_rotate = jnp.reshape(to_rotate, (1, 1, 2)) + point
    return to_rotate


def union_of_line_drawings(lines):
    """Computes the union image of a sequence of line predictions."""
    lines = jnp.stack(lines, axis=-1)
    lines = jnp.min(lines, axis=-1)
    return lines


def network_line_parameters_to_line(line_parameters, height, width):
    """Interprets a network's output as line parameters and calls line_to_image.
    Rescales to assume the network output is not resolution dependent, and
    clips to valid parameters.
    Args:
      line_parameters: Tensor with shape [batch_size, 5]. Contains the network
        output, to be interpreted as line parameters.
      height: Int containing output height.
      width: Int containing output width.
    Returns:
      An image with shape [batch_size, height, width, 1]. Contains a drawing of
      the network's output.
    """
    rotation, px, py, lx, ly = line_parameters.T
    px = jnp.minimum(nn.relu(px * width + width / 2), width)  # was leaky relu!
    py = jnp.minimum(nn.relu(py * height + height / 2), height)  # was leaky relu!
    lx = jnp.clip(jnp.abs(lx) * width, a_min=4.0, a_max=width / 3.0)
    ly = jnp.clip(jnp.abs(ly) * height, a_min=4.0, a_max=height / 3.0)
    line_parameters = jnp.stack([rotation, px, py, lx, ly], axis=1)
    batch_out = vmap(line_to_image, in_axes=(0, None, None))(
        line_parameters, height, width
    )
    return batch_out
