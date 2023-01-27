"""Library code to create an LDIF example directory from a file."""

import os
import subprocess as sp

# pylint: enable=g-bad-import-order
import jax.numpy as jnp
import numpy as np

from impax.inference import example

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from impax.utils import file_util, gaps_util, path_util
from impax.utils.file_util import log


def remove_png_dir(d):
    """Removes a directory after verifying it contains only pngs."""
    pngs = os.listdir(d)
    pngs = [os.path.join(d, f) for f in pngs]
    files = [f for f in pngs if os.path.isfile(f)]
    if len(files) != len(pngs):
        raise ValueError(f"Expected the directory {d} to contain only pngs files.")
    for f in files:
        os.remove(f)
    os.rmdir(d)


def write_depth_and_normals_npz(dirpath, path_out):
    depth_images = gaps_util.read_depth_directory(f"{dirpath}/depth_images", 20)
    normal_images = gaps_util.read_normals_dir(f"{dirpath}/normals", 20)
    depth_images = depth_images[..., jnp.newaxis]
    arr = jnp.concatenate([depth_images, normal_images], axis=-1)
    np.savez_compressed(path_out, np.array(arr))
    # Delete the images, they are no longer needed:
    remove_png_dir(f"{dirpath}/depth_images")
    remove_png_dir(f"{dirpath}/normals")


def mesh_to_example(codebase_root_dir, mesh_path, dirpath, skip_existing):
    ldif_path = path_util.get_path_to_impax_root()
    if not skip_existing or not os.path.isfile(f"{dirpath}/depth_and_normals.npz"):
        sp.check_output(
            [
                f"{codebase_root_dir}/scripts/process_mesh_local.sh",
                mesh_path,
                dirpath,
                ldif_path,
            ]
        )
        write_depth_and_normals_npz(dirpath, f"{dirpath}/depth_and_normals.npz")
    else:
        log.info(
            f"Skipping shell script processing for {dirpath},"
            " the output already exists."
        )
    # Precompute the dodeca samples for later:
    e = example.InferenceExample.from_directory(dirpath)
    sample_path = e.precomputed_surface_samples_from_dodeca_path
    if not skip_existing or not os.path.isfile(sample_path):
        e.surface_sample_count = 100000
        precomputed_samples = e.surface_samples_from_dodeca
        assert precomputed_samples.shape[0] == 100000
        assert precomputed_samples.shape[1] == 6
        file_util.write_points(sample_path, precomputed_samples)
    else:
        log.info(
            f"Skipping surface sample precompution for {dirpath}, it's already done."
        )
