"""Camera utilities, pulled from diffren."""

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
from jax import random

from impax.utils import camera_util


def look_at(eye, center, world_up):
    """Computes camera viewing matrices.

    Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

    Args:
      eye: 2-D float32 Tensor (or convertible value) with shape [batch_size, 3]
        containing the XYZ world space position of the camera.
      center: 2-D float32 Tensor (or convertible value) with shape [batch_size, 3]
        containing a position along the center of the camera's gaze.
      world_up: 2-D float32 Tensor (or convertible value) with shape [batch_size,
        3] specifying the world's up direction; the output camera will have no
        tilt with respect to this direction.

    Returns:
      A [batch_size, 4, 4] float tensor containing a right-handed camera
      extrinsics matrix that maps points from world space to points in eye space.
    """
    eye = tf.convert_to_tensor(eye)
    center = tf.convert_to_tensor(center)
    world_up = tf.convert_to_tensor(world_up)
    batch_size = center.shape[0]

    vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = tf.norm(forward, ord="euclidean", axis=1, keepdims=True)
    with tf.control_dependencies(
        [
            tf.assert_greater(
                forward_norm,
                vector_degeneracy_cutoff,
                message="Camera matrix is degenerate because eye and center are close.",
            )
        ]
    ):
        forward = tf.divide(forward, forward_norm)

    to_side = tf.linalg.cross(forward, world_up)
    to_side_norm = tf.norm(to_side, ord="euclidean", axis=1, keepdims=True)
    with tf.control_dependencies(
        [
            tf.assert_greater(
                to_side_norm,
                vector_degeneracy_cutoff,
                message="{0} {1}".format(
                    "Camera matrix is degenerate because up and gaze are close or", "because up is degenerate."
                ),
            )
        ]
    ):
        to_side = tf.divide(to_side, to_side_norm)
        cam_up = tf.linalg.cross(to_side, forward)

    w_column = tf.constant(batch_size * [[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)  # [batch_size, 4]
    w_column = tf.reshape(w_column, [batch_size, 4, 1])
    view_rotation = tf.stack(
        [to_side, cam_up, -forward, tf.zeros_like(to_side, dtype=tf.float32)], axis=1
    )  # [batch_size, 4, 3] matrix
    view_rotation = tf.concat([view_rotation, w_column], axis=2)  # [batch_size, 4, 4]

    identity_batch = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
    view_translation = tf.concat([identity_batch, tf.expand_dims(-eye, 2)], 2)
    view_translation = tf.concat([view_translation, tf.reshape(w_column, [batch_size, 1, 4])], 1)
    camera_matrices = tf.matmul(view_rotation, view_translation)
    return camera_matrices


def look_at_np(eye, center, world_up):
    """Computes camera viewing matrices.

    Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

    Args:
      eye: 2-D float32 numpy array (or convertible value) with shape
        [batch_size, 3] containing the XYZ world space position of the camera.
      center: 2-D float32 array (or convertible value) with shape [batch_size, 3]
        containing a position along the center of the camera's gaze.
      world_up: 2-D float32 array (or convertible value) with shape [batch_size,
        3] specifying the world's up direction; the output camera will have no
        tilt with respect to this direction.

    Returns:
      A [batch_size, 4, 4] numpy array containing a right-handed camera
      extrinsics matrix that maps points from world space to points in eye space.
    """
    batch_size = center.shape[0]

    vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = np.linalg.norm(forward, ord=2, axis=1, keepdims=True)
    # assert forward_norm >= vector_degeneracy_cutoff
    forward = np.divide(forward, forward_norm)

    to_side = np.cross(forward, world_up)
    to_side_norm = np.linalg.norm(to_side, ord=2, axis=1, keepdims=True)
    # assert to_side_norm >= vector_degeneracy_cutoff
    to_side = np.divide(to_side, to_side_norm)
    cam_up = np.cross(to_side, forward)

    w_column = np.array(batch_size * [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)  # [batch_size, 4]
    w_column = np.reshape(w_column, [batch_size, 4, 1])
    view_rotation = np.stack(
        [to_side, cam_up, -forward, np.zeros_like(to_side, dtype=np.float32)], axis=1
    )  # [batch_size, 4, 3] matrix
    view_rotation = np.concatenate([view_rotation, w_column], axis=2)  # [batch_size, 4, 4]
    identity_singleton = np.eye(3, dtype=np.float32)[np.newaxis, ...]
    identity_batch = np.tile(identity_singleton, [batch_size, 1, 1])
    view_translation = np.concatenate([identity_batch, np.expand_dims(-eye, 2)], 2)
    view_translation = np.concatenate([view_translation, np.reshape(w_column, [batch_size, 1, 4])], 1)
    camera_matrices = np.matmul(view_rotation, view_translation)
    return camera_matrices


def roll_pitch_yaw_to_rotation_matrices(roll_pitch_yaw):
    """Converts roll-pitch-yaw angles to rotation matrices.

    Args:
      roll_pitch_yaw: Tensor (or convertible value) with shape [..., 3]. The last
        dimension contains the roll, pitch, and yaw angles in radians.  The
        resulting matrix rotates points by first applying roll around the x-axis,
        then pitch around the y-axis, then yaw around the z-axis.

    Returns:
       Tensor with shape [..., 3, 3]. The 3x3 rotation matrices corresponding to
       the input roll-pitch-yaw angles.
    """
    roll_pitch_yaw = tf.convert_to_tensor(roll_pitch_yaw)

    cosines = tf.cos(roll_pitch_yaw)
    sines = tf.sin(roll_pitch_yaw)
    cx, cy, cz = tf.unstack(cosines, axis=-1)
    sx, sy, sz = tf.unstack(sines, axis=-1)
    # pyformat: disable
    rotation = tf.stack(
        [
            cz * cy,
            cz * sy * sx - sz * cx,
            cz * sy * cx + sz * sx,
            sz * cy,
            sz * sy * sx + cz * cx,
            sz * sy * cx - cz * sx,
            -sy,
            cy * sx,
            cy * cx,
        ],
        axis=-1,
    )
    # pyformat: enable
    shape = tf.concat([tf.shape(rotation)[:-1], [3, 3]], axis=0)
    rotation = tf.reshape(rotation, shape)
    return rotation


@pytest.mark.parametrize("batch_size", [8, 16])
def test_look_at(batch_size, key=random.PRNGKey(0)):
    key0, key1, key2 = random.split(key, 3)
    shape = (batch_size, 3)

    eye = random.normal(key0, shape)
    center = random.normal(key1, shape)
    world_up = random.normal(key2, shape)

    ret = camera_util.look_at(eye, center, world_up)
    ground_truth = look_at(eye, center, world_up)

    assert jnp.allclose(ret, ground_truth.numpy(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("batch_size", [8, 16])
def test_look_at_np(batch_size, key=random.PRNGKey(0)):
    key0, key1, key2 = random.split(key, 3)
    shape = (batch_size, 3)

    eye = random.normal(key0, shape)
    center = random.normal(key1, shape)
    world_up = random.normal(key2, shape)

    ret = camera_util.look_at_np(eye, center, world_up)
    ground_truth = look_at_np(eye, center, world_up)

    assert jnp.allclose(ret, ground_truth, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("roll_pitch_yaw_shape", [(16, 16, 16)])
def test_roll_pitch_yaw_to_rotation_matrices(roll_pitch_yaw_shape, key=random.PRNGKey(0)):
    roll_pitch_yaw = random.normal(key, (*roll_pitch_yaw_shape, 3))

    ret = camera_util.roll_pitch_yaw_to_rotation_matrices(roll_pitch_yaw)
    ground_truth = roll_pitch_yaw_to_rotation_matrices(np.array(roll_pitch_yaw))

    assert jnp.allclose(ret, ground_truth)
