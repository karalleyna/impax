"""
Helpers for differentiable interpolation.
References:
https://github.com/google/ldif/blob/master/ldif/util/interpolate_util.py
"""

import jax.numpy as jnp

from impax.utils.file_util import log


def interpolate_np(grid, samples, world2grid):
    """Returns the trilinearly interpolated SDF values using the grid.
    Args:
      grid: numpy array with shape [depth, height, width].
      samples: numpy array with shape [sample_count, 3].
      world2grid: numpy array with shape [4,4]. Rigid body transform.
    Returns:
      sdf: Tensor with shape [batch_size, sample_count, 1]. The ground truth
        sdf at the sample locations. Differentiable w.r.t. samples.
    """
    xyzw_samples = jnp.pad(
        samples, [[0, 0], [0, 1]], mode="constant", constant_values=1
    )
    grid_frame_samples = jnp.matmul(xyzw_samples, world2grid.T)[..., :3]
    lower_coords = jnp.floor(grid_frame_samples).astype(jnp.int32)
    upper_coords = jnp.ceil(grid_frame_samples).astype(jnp.int32)
    alpha = grid_frame_samples - lower_coords.astype(jnp.float32)

    lca = jnp.split(lower_coords, 3, axis=-1)[::-1]
    uca = jnp.split(upper_coords, 3, axis=-1)[::-1]
    aca = jnp.split(alpha, 3, axis=-1)[::-1]  # ?

    c00 = (
        grid[lca[0], lca[1], lca[2]] * (1 - aca[0])
        + grid[uca[0], lca[1], lca[2]] * aca[0]
    )
    c01 = (
        grid[lca[0], lca[1], uca[2]] * (1 - aca[0])
        + grid[uca[0], lca[1], uca[2]] * aca[0]
    )
    c10 = (
        grid[lca[0], uca[1], lca[2]] * (1 - aca[0])
        + grid[uca[0], uca[1], lca[2]] * aca[0]
    )
    c11 = (
        grid[lca[0], uca[1], uca[2]] * (1 - aca[0])
        + grid[uca[0], uca[1], uca[2]] * aca[0]
    )

    c0 = c00 * (1 - aca[1]) + c10 * aca[1]
    c1 = c01 * (1 - aca[1]) + c11 * aca[1]

    interp = c0 * (1 - aca[2]) + c1 * aca[2]

    log.info("interpolated:")
    log.info(interp.shape)
    log.info(interp)
    log.info("lower coords:")
    log.info(jnp.min(lower_coords))
    log.info(jnp.max(lower_coords))
    log.info(jnp.mean(lower_coords))
    log.info("upper coords:")
    log.info(jnp.min(upper_coords))
    log.info(jnp.max(upper_coords))
    log.info(jnp.mean(upper_coords))
    log.info("Interpolated SDF")
    log.info(jnp.min(interp))
    log.info(jnp.max(interp))
    log.info(jnp.mean(interp))
    log.info("Original SDF")
    log.info(jnp.min(grid))
    log.info(jnp.max(grid))
    log.info(jnp.mean(grid))
    log.info(jnp.histogram(interp))
    return interp
