# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Utilities for randomly sampling in tensorflow."""

import importlib
import math

import numpy as np
import tensorflow as tf

import jax
import jax.numpy as jnp
from jax import random

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from ldif.util import geom_util
from ldif.util.file_util import log
# pylint: enable=g-bad-import-order

importlib.reload(geom_util)


def random_shuffle_along_dim(tensor, dim):
  """Randomly shuffles the elements of 'tensor' along axis with index 'dim'."""
  if dim == 0:
    # some random key
    key = random.PRNGKey(8484848)
    return random.shuffle(key, tensor) #.random_shuffle(tensor)
  tensor_rank = len(jnp.shape(tensor).as_list())
  perm = list(range(tensor_rank))
  perm[dim], perm[0] = perm[0], perm[dim]
  tensor = jnp.transpose(tensor, perm=perm)
  key = random.PRNGKey(8484849)
  tensor = random.shuffle(key, tensor)
  tensor = jnp.transpose(tensor, perm=perm)
  return tensor


def random_pan_rotations(batch_size):
  """Generates random 4x4 panning rotation matrices."""
  key = random.PRNGKey(8484841)
  theta = random.uniform(key, shape=[batch_size], minval=0, maxval=2.0 * math.pi)
  z = jnp.zeros_like(theta)
  o = jnp.ones_like(theta)
  ct = jnp.cos(theta)
  st = jnp.sin(theta)
  m = jnp.stack([ct, z, st, z, z, o, z, z, -st, z, ct, z, z, z, z, o], axis=-1)
  return jnp.reshape(m, [batch_size, 4, 4])


def random_pan_rotation_np():
  key = random.PRNGKey(8484842)
  theta = random.uniform(key, minval=0, maxval=2.0 * np.pi)
  m = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta), 0], [0, 1, 0, 0],
                [-jnp.sin(theta), 0, jnp.cos(theta), 0], [0, 0, 0, 1]],
               dtype=jnp.float32)
  return m


def random_rotations(batch_size):
  """Generates uniformly random 3x3 rotation matrices."""
  key = random.PRNGKey(8484843)
  theta = random.uniform(key, shape=[batch_size], minval=0, maxval=2.0 * math.pi)
  key = random.PRNGKey(8484844)
  phi = tf.random.uniform(key, shape=[batch_size], minval=0, maxval=2.0 * math.pi)
  key = random.PRNGKey(8484845)
  z = tf.random.uniform(key, shape=[batch_size], minval=0, maxval=2.0)

  r = jnp.sqrt(z + 1e-8)
  v = jnp.stack([r * jnp.sin(phi), r * jnp.cos(phi),
                jnp.sqrt(2.0 - z + 1e-8)],
               axis=-1)
  st = jnp.sin(theta)
  ct = jnp.cos(theta)
  zero = jnp.zeros_like(st)
  one = jnp.ones_like(st)
  base_rot = jnp.stack([ct, st, zero, -st, ct, zero, zero, zero, one], axis=-1)
  base_rot = jnp.reshape(base_rot, [batch_size, 3, 3])
  v_outer = jnp.matmul(v[:, :, tf.newaxis], v[:, tf.newaxis, :])
  rotation_3x3 = jnp.matmul(v_outer - jnp.eye(3, batch_shape=[batch_size]),
                           base_rot)
  return rotation_to_tx(rotation_3x3)


def random_rotation_np():
  """Returns a uniformly random SO(3) rotation as a [3,3] numpy array."""
  key = random.PRNGKey(8484846)
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


def random_scales(batch_size, minval, maxval):
  key = random.PRNGKey(8484847)
  scales = random.uniform(key,
      shape=[batch_size, 3], minval=minval, maxval=maxval)
  hom_coord = jnp.ones([batch_size, 1], dtype=jnp.float32)
  scales = jnp.concatenate([scales, hom_coord], axis=1)
  s = jnp.diag(scales)
  log.info(jnp.shape(s).as_list())
  return jnp.diag(scales)


def random_transformation(origin):
  batch_size = jnp.shape(origin).as_list()[0]
  assert len(jnp.shape(origin).as_list()) == 2
  center = translation_to_tx(-origin)
  rotate = random_rotations(batch_size)
  scale = random_scales(batch_size, 1, 4)
  tx = tf.matmul(scale, tf.matmul(rotate, center))
  return tx


def random_zoom_transformation(origin):
  batch_size = origin.get_shape().as_list()[0]
  assert len(origin.get_shape().as_list()) == 2
  center = translation_to_tx(-origin)
  scale = random_scales(batch_size, 3, 3)
  tx = jnp.matmul(scale, center)
  return tx


def translation_to_tx(t):
  """Maps three translation elements to a 4x4 homogeneous matrix.

  Args:
   t: Tensor with shape [..., 3].

  Returns:
    Tensor with shape [..., 4, 4].
  """
  batch_dims = jnp.shape(t).as_list()[:-1]
  empty_rot = jnp.eye(3, batch_shape=batch_dims)
  rot = jnp.concatenate([empty_rot, jnp.expand_dims(t, axis=-1)], axis=-1)
  hom_row = jnp.eye(4, batch_shape=batch_dims)[..., 3:4, :]
  return jnp.concatenate([rot, hom_row], axis=-2)


def rotation_to_tx(rot):
  """Maps a 3x3 rotation matrix to a 4x4 homogeneous matrix.

  Args:
    rot: Tensor with shape [..., 3, 3].

  Returns:
    Tensor with shape [..., 4, 4].
  """
  batch_dims = jnp.shape(rot).as_list()[:-2]
  empty_col = jnp.zeros(batch_dims + [3, 1], dtype=jnp.float32)
  rot = jnp.concatenate([rot, empty_col], axis=-1)
  hom_row = jnp.eye(4, batch_shape=batch_dims)[..., 3:4, :]
  return jnp.concatenate([rot, hom_row], axis=-2)
