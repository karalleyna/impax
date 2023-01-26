"""Converts a structured implicit function into a mesh."""

import numpy as jnp
from skimage import measure
import trimesh

# local
from impax.utils.logging_util import log


def marching_cubes(volume, mcubes_extent):
    """Maps from a voxel grid of implicit surface samples to a Trimesh mesh."""
    volume = jnp.squeeze(volume)
    length, height, width = volume.shape
    resolution = length
    # This function doesn't support non-cube volumes:
    assert resolution == height and resolution == width
    thresh = -0.07
    try:
        vertices, faces, *_ = measure.marching_cubes(volume, thresh)
        x, y, z = [jnp.array(x) for x in zip(*vertices)]
        xyzw = jnp.stack([x, y, z, jnp.ones_like(x)], axis=1)
        # Center the volume around the origin:
        xyzw += jnp.array(
            [[-resolution / 2.0, -resolution / 2.0, -resolution / 2.0, 0.0]]
        )
        # This assumes the world is right handed with y up; matplotlib's renderer
        # has z up and is left handed:
        # Reflect across z, rotate about x, and rescale to [-0.5, 0.5].
        xyzw *= jnp.array(
            [
                [
                    (2.0 * mcubes_extent) / resolution,
                    (2.0 * mcubes_extent) / resolution,
                    -1.0 * (2.0 * mcubes_extent) / resolution,
                    1,
                ]
            ]
        )
        y_up_to_z_up = jnp.array(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        xyzw = jnp.matmul(y_up_to_z_up, xyzw.T).T
        faces = jnp.stack([faces[..., 0], faces[..., 2], faces[..., 1]], axis=-1)
        world_space_xyz = xyzw[:, :3]
        mesh = trimesh.Trimesh(vertices=world_space_xyz, faces=faces)
        log.info("Generated mesh successfully.")
        return True, mesh
    except (ValueError, RuntimeError) as e:
        log.warning(f"Failed to extract mesh with error {e}. Setting to unit sphere.")
        return False, trimesh.primitives.Sphere(radius=0.5)
