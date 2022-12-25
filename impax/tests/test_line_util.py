"""
Checks the results that are returned by the original implementation and ours are the same. For original functions,
please see https://github.com/google/ldif/blob/master/ldif/util/line_util.py.
"""
from jax import random
import numpy as np
import tensorflow as tf
from impax.utils.line_util import line_to_image
import jax.numpy as jnp


def _line_to_image(line_parameters, height, width, falloff=2.0):
    with tf.name_scope("line-to-image"):
        x_coords = np.linspace(0.5, width - 0.5, width)
        y_coords = np.linspace(0.5, height - 0.5, height)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, sparse=False, indexing="xy")
        coords = np.stack([grid_x, grid_y], axis=2)
        coords = tf.constant(coords, dtype=tf.float32)

        angle_of_rotation, px, py, lx, ly = tf.unstack(line_parameters)
        angle_of_rotation = -angle_of_rotation
        center = line_parameters[1:3]

        v0 = tf.stack([px + lx, py + ly], axis=0)
        v0 = _rotate_about_point(angle_of_rotation, center, v0)

        v1 = tf.stack([px + lx, py - ly], axis=0)
        v1 = _rotate_about_point(angle_of_rotation, center, v1)

        v2 = tf.stack([px - lx, py - ly], axis=0)
        v2 = _rotate_about_point(angle_of_rotation, center, v2)

        coords = tf.reshape(coords, [height * width, 1, 2])

        first_direction_insidedness = _fractional_vector_projection(
            v1, v0, coords, falloff=falloff
        )
        second_direction_insidedness = _fractional_vector_projection(
            v1, v2, coords, falloff=falloff
        )
        crease_corners = True
        if crease_corners:
            insidedness = first_direction_insidedness * second_direction_insidedness
        else:
            insidedness = tf.maximum(
                1.0
                - tf.sqrt(
                    (1.0 - first_direction_insidedness)
                    * (1.0 - first_direction_insidedness)
                    + (1.0 - second_direction_insidedness)
                    * (1.0 - second_direction_insidedness)
                ),
                tf.zeros_like(first_direction_insidedness),
            )

        color = 1.0 - insidedness
        return tf.reshape(color, [height, width, 1])


def _fractional_vector_projection(e0, e1, p, falloff=2.0):
    with tf.name_scope("fractional-vector-projection"):
        batch_size = p.shape[0]
        p = tf.reshape(p, [batch_size, 2])
        e0 = tf.reshape(e0, [1, 2])
        e1 = tf.reshape(e1, [1, 2])
        e01 = e1 - e0
        e01_norm = tf.sqrt(e01[0, 0] * e01[0, 0] + e01[0, 1] * e01[0, 1])
        e01_normalized = e01 / tf.reshape(e01_norm, [1, 1])
        e0p = p - e0
        e0p_dot_e01_normalized = tf.matmul(
            tf.reshape(e0p, [1, batch_size, 2]),
            tf.reshape(e01_normalized, [1, 1, 2]),
            transpose_b=True,
        )
        e0p_dot_e01_normalized = (
            tf.reshape(e0p_dot_e01_normalized, [batch_size]) / e01_norm
        )
        if falloff is None:
            left_sided_inside = tf.cast(
                tf.logical_and(
                    e0p_dot_e01_normalized >= 0, e0p_dot_e01_normalized <= 1
                ),
                dtype=tf.float32,
            )
            return left_sided_inside

        e10_normalized = -e01_normalized
        e1p = p - e1
        e1p_dot_e10_normalized = tf.matmul(
            tf.reshape(e1p, [1, batch_size, 2]),
            tf.reshape(e10_normalized, [1, 1, 2]),
            transpose_b=True,
        )
        e1p_dot_e10_normalized = (
            tf.reshape(e1p_dot_e10_normalized, [batch_size]) / e01_norm
        )
        proj = tf.maximum(e0p_dot_e01_normalized, e1p_dot_e10_normalized)
        proj = tf.maximum(proj, 1.0)

        falloff_is_relative = True
        if falloff_is_relative:
            fractional_falloff = 1.0 / (tf.pow(falloff * (proj - 1), 2.0) + 1.0)
            return fractional_falloff
        else:
            line_length = tf.reshape(e01_norm, [1])
            pixel_dist = tf.reshape(proj - 1, [-1]) * line_length
            zero_thresh_in_pixels = tf.reshape(
                tf.constant([8.0], dtype=tf.float32), [1]
            )
            relative_dist = pixel_dist / zero_thresh_in_pixels
            return 1.0 / (tf.pow(relative_dist, 3.0) + 1.0)


def _rotate_about_point(angle_of_rotation, point, to_rotate):
    with tf.name_scope("rotate-2d"):
        cos_angle = tf.cos(angle_of_rotation)
        sin_angle = tf.sin(angle_of_rotation)
        top_row = tf.stack([cos_angle, -sin_angle], axis=0)
        bottom_row = tf.stack([sin_angle, cos_angle], axis=0)
        rotation_matrix = tf.reshape(tf.stack([top_row, bottom_row], axis=0), [1, 2, 2])
        to_rotate = tf.reshape(to_rotate, [1, 1, 2])
        point = tf.reshape(point, [1, 1, 2])
        to_rotate = to_rotate - point
        to_rotate = tf.matmul(rotation_matrix, to_rotate, transpose_b=True)
        to_rotate = tf.reshape(to_rotate, [1, 1, 2]) + point
        return to_rotate


def _union_of_line_drawings(lines):
    with tf.name_scope("Union-of-Line-Images"):
        lines = tf.stack(lines, axis=-1)
        lines = tf.reduce_min(lines, axis=-1)
        return lines


def _network_line_parameters_to_line(line_parameters, height, width):
    rotation, px, py, lx, ly = tf.unstack(line_parameters, axis=1)
    px = tf.minimum(tf.nn.relu(px * width + width / 2), width)
    py = tf.minimum(tf.nn.relu(py * height + height / 2), height)
    lx = tf.clip_by_value(tf.abs(lx) * width, 4.0, width / 3.0)
    ly = tf.clip_by_value(tf.abs(ly) * height, 4.0, height / 3.0)
    line_parameters = tf.stack([rotation, px, py, lx, ly], axis=1)
    batch_out = []
    for batch_item in tf.unstack(line_parameters, axis=0):
        batch_out.append(_line_to_image(batch_item, height, width))
    return tf.stack(batch_out)


def test_line_to_image(key=random.PRNGKey(0)):
    key0, key1 = random.split(key)
    line_parameters = random.uniform(
        key0,
        shape=(5,),
    )
    height, width = random.randint(key1, shape=(2,), minval=2, maxval=10)
    ret = line_to_image(line_parameters, height, width)
    tf_arr = tf.convert_to_tensor(np.array(line_parameters))
    ground_truth = _line_to_image(tf_arr, int(height), int(width))
    assert jnp.allclose(ret, ground_truth.numpy())
