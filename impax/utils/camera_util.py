"""
References:
https://github.com/google/ldif/blob/master/ldif/util/camera_util.py
"""

import jax.numpy as jnp


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
    batch_size = center.shape[0]

    forward = center - eye
    forward_norm = jnp.linalg.norm(forward, ord=2, axis=1, keepdims=True)
    forward = jnp.divide(forward, forward_norm)

    to_side = jnp.cross(forward, world_up)
    to_side_norm = jnp.linalg.norm(to_side, ord=2, axis=1, keepdims=True)

    to_side = jnp.divide(to_side, to_side_norm)
    cam_up = jnp.cross(to_side, forward)

    w_column = jnp.array(
        batch_size * [[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32
    )  # [batch_size, 4]
    w_column = jnp.reshape(w_column, [batch_size, 4, 1])
    view_rotation = jnp.stack(
        [to_side, cam_up, -forward, jnp.zeros_like(to_side, dtype=jnp.float32)], axis=1
    )  # [batch_size, 4, 3] matrix
    view_rotation = jnp.concatenate(
        [view_rotation, w_column], axis=2
    )  # [batch_size, 4, 4]

    identity_batch = jnp.tile(jnp.expand_dims(jnp.eye(3), 0), [batch_size, 1, 1])
    view_translation = jnp.concatenate([identity_batch, jnp.expand_dims(-eye, 2)], 2)
    view_translation = jnp.concatenate(
        [view_translation, jnp.reshape(w_column, [batch_size, 1, 4])], 1
    )
    camera_matrices = jnp.matmul(view_rotation, view_translation)
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

    forward = center - eye
    forward_norm = jnp.linalg.norm(forward, ord=2, axis=1, keepdims=True)
    # assert forward_norm >= vector_degeneracy_cutoff
    forward = jnp.divide(forward, forward_norm)

    to_side = jnp.cross(forward, world_up)
    to_side_norm = jnp.linalg.norm(to_side, ord=2, axis=1, keepdims=True)
    # assert to_side_norm >= vector_degeneracy_cutoff
    to_side = jnp.divide(to_side, to_side_norm)
    cam_up = jnp.cross(to_side, forward)

    w_column = jnp.array(
        batch_size * [[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32
    )  # [batch_size, 4]
    w_column = jnp.reshape(w_column, [batch_size, 4, 1])
    view_rotation = jnp.stack(
        [to_side, cam_up, -forward, jnp.zeros_like(to_side, dtype=jnp.float32)], axis=1
    )  # [batch_size, 4, 3] matrix
    view_rotation = jnp.concatenate(
        [view_rotation, w_column], axis=2
    )  # [batch_size, 4, 4]
    identity_singleton = jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, ...]
    identity_batch = jnp.tile(identity_singleton, [batch_size, 1, 1])
    view_translation = jnp.concatenate([identity_batch, jnp.expand_dims(-eye, 2)], 2)
    view_translation = jnp.concatenate(
        [view_translation, jnp.reshape(w_column, [batch_size, 1, 4])], 1
    )
    camera_matrices = jnp.matmul(view_rotation, view_translation)
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

    cosines = jnp.cos(roll_pitch_yaw)
    sines = jnp.sin(roll_pitch_yaw)
    cx, cy, cz = cosines.T
    sx, sy, sz = sines.T

    cx, cy, cz = cx.T, cy.T, cz.T
    sx, sy, sz = sx.T, sy.T, sz.T

    # pyformat: disable
    rotation = jnp.stack(
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
    shape = list(jnp.shape(rotation)[:-1]) + [3, 3]
    rotation = jnp.reshape(rotation, shape)
    return rotation
