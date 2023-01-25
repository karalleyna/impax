"""
Utilities for randomly sampling in DeviceArrayflow.
References:
https://github.com/google/ldif/blob/master/ldif/util/random_util.py
"""

import importlib

import jax.numpy as jnp
from jax import random, vmap

# local
from impax.utils import geom_util

importlib.reload(geom_util)


def random_shuffle_along_dim(x, dim, key=random.PRNGKey(0)):
    """Randomly shuffles the elements of 'DeviceArray' along axis with index 'dim'."""
    if dim == 0:
        return random.shuffle(key, x)
    DeviceArray_rank = len(x.shape)
    axes = list(range(DeviceArray_rank))
    axes[dim], axes[0] = axes[0], axes[dim]
    x = jnp.transpose(x, axes=axes)
    x = random.shuffle(key, x)
    x = jnp.transpose(x, axes=axes)
    return x


def random_pan_rotations(batch_size, key=random.PRNGKey(0)):
    """Generates random 4x4 panning rotation matrices."""
    theta = random.uniform(key, shape=[batch_size], minval=0, maxval=2.0 * jnp.pi)
    z = jnp.zeros_like(theta)
    o = jnp.ones_like(theta)
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    m = jnp.stack([ct, z, st, z, z, o, z, z, -st, z, ct, z, z, z, z, o], axis=-1)
    return jnp.reshape(m, [batch_size, 4, 4])


def random_pan_rotation_jnp(key=random.PRNGKey(0)):
    theta = random.uniform(key, maxval=2.0 * jnp.pi)
    m = jnp.array(
        [
            [jnp.cos(theta), 0, jnp.sin(theta), 0],
            [0, 1, 0, 0],
            [-jnp.sin(theta), 0, jnp.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    return m


def random_rotations(batch_size, key=random.PRNGKey(0)):
    """Generates uniformly random 3x3 rotation matrices."""
    key0, key1, key2 = random.split(key, 3)
    theta = random.uniform(key0, shape=[batch_size], minval=0, maxval=2.0 * jnp.pi)
    phi = random.uniform(key1, shape=[batch_size], minval=0, maxval=2.0 * jnp.pi)
    z = random.uniform(key2, shape=[batch_size], minval=0, maxval=2.0)

    r = jnp.sqrt(z + 1e-8)
    v = jnp.stack(
        [r * jnp.sin(phi), r * jnp.cos(phi), jnp.sqrt(2.0 - z + 1e-8)], axis=-1
    )
    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    zero = jnp.zeros_like(st)
    one = jnp.ones_like(st)
    base_rot = jnp.stack([ct, st, zero, -st, ct, zero, zero, zero, one], axis=-1)
    base_rot = jnp.reshape(base_rot, [batch_size, 3, 3])
    v_outer = jnp.matmul(v[:, :, None], v[:, None, :])

    rotation_3x3 = jnp.matmul(
        v_outer - jnp.repeat(jnp.eye(3)[None, ...], batch_size, axis=0), base_rot
    )
    return rotation_to_tx(rotation_3x3)


def random_rotation_jnp(key=random.PRNGKey(0)):
    """Returns a uniformly random SO(3) rotation as a [3,3] numpy array."""
    vals = random.uniform(key, shape=(3,))
    theta = vals[0] * 2.0 * jnp.pi
    phi = vals[1] * 2.0 * jnp.pi
    z = 2.0 * vals[2]
    r = jnp.sqrt(z)
    v = jnp.stack([r * jnp.sin(phi), r * jnp.cos(phi), jnp.sqrt(2.0 * (1 - vals[2]))])
    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    base_rot = jnp.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]], dtype=jnp.float32)
    return (jnp.outer(v, v) - jnp.eye(3)).dot(base_rot)


def random_scales(batch_size, minval, maxval, key=random.PRNGKey(0)):
    scales = random.uniform(key, shape=[batch_size, 3], minval=minval, maxval=maxval)
    hom_coord = jnp.ones([batch_size, 1], dtype=jnp.float32)
    scales = jnp.concatenate([scales, hom_coord], axis=1)
    return vmap(jnp.diag)(scales)


def random_transformation(origin):
    batch_size = origin.shape[0]
    assert len(origin.shape) == 2
    center = translation_to_tx(-origin)
    rotate = random_rotations(batch_size)
    scale = random_scales(batch_size, 1, 4)
    tx = jnp.matmul(scale, jnp.matmul(rotate, center))
    return tx


def random_zoom_transformation(origin):
    batch_size = origin.shape[0]
    assert len(origin.shape) == 2
    center = translation_to_tx(-origin)
    scale = random_scales(batch_size, 3, 3)
    tx = jnp.matmul(scale, center)
    return tx


def translation_to_tx(x):
    """Maps three translation elements to a 4x4 homogeneous matrix.
    Args:
     x: DeviceArray with shape [..., 3].
    Returns:
      DeviceArray with shape [..., 4, 4].
    """
    batch_dims = x.shape[:-1]

    identity_matrix = jnp.eye(3)
    for dim in reversed(batch_dims):
        identity_matrix = jnp.repeat(identity_matrix[None, ...], dim, axis=0)

    rot = jnp.concatenate([identity_matrix, x[..., None]], axis=-1)

    identity_matrix = jnp.eye(4)
    for dim in reversed(batch_dims):
        identity_matrix = jnp.repeat(identity_matrix[None, ...], dim, axis=0)

    hom_row = identity_matrix[..., 3:4, :]
    return jnp.concatenate([rot, hom_row], axis=-2)


def rotation_to_tx(rot):
    """Maps a 3x3 rotation matrix to a 4x4 homogeneous matrix.
    Args:
      rot: DeviceArray with shape [..., 3, 3].
    Returns:
      DeviceArray with shape [..., 4, 4].
    """
    batch_dims = rot.shape[:-2]
    empty_col = jnp.zeros(batch_dims + (3, 1), dtype=jnp.float32)

    rot = jnp.concatenate([rot, empty_col], axis=-1)
    identity_matrix = jnp.eye(4)
    for dim in reversed(batch_dims):
        identity_matrix = jnp.repeat(identity_matrix[None, ...], dim, axis=0)

    hom_row = identity_matrix[..., 3:4, :]

    return jnp.concatenate([rot, hom_row], axis=-2)
