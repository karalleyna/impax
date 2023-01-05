"""
Utilities for geometric operations in NumPy.
References:
https://github.com/google/ldif/blob/master/ldif/util/geom_util_jnp.py
"""

import numpy as jnp

from impax.utils import jnp_util
from impax.utils.logging_util import log


def apply_4x4(arr, m, are_points=True, feature_count=0):
    """Applies a 4x4 matrix to 3D points/vectors.
    Args:
      arr: Numpy array with shape [..., 3 + feature_count].
      m: Matrix with shape [4, 4].
      are_points: Boolean. Whether to treat arr as points or vectors.
      feature_count: Int. The number of extra features after the points.
    Returns:
      Numpy array with shape [..., 3].
    """
    shape_in = arr.shape
    if are_points:
        hom = jnp.ones_like(arr[..., 0:1], dtype=jnp.float32)
    else:
        hom = jnp.zeros_like(arr[..., 0:1], dtype=jnp.float32)
    assert arr.shape[-1] == 3 + feature_count
    to_tx = jnp.concatenate([arr[..., :3], hom], axis=-1)
    to_tx = jnp.reshape(to_tx, [-1, 4])
    transformed = jnp.matmul(to_tx, m.T)[:, :3]
    if feature_count:
        flat_samples = jnp.reshape(arr[..., 3:], [-1, feature_count])
        transformed = jnp.concatenate([transformed, flat_samples], axis=1)
    return jnp.reshape(transformed, shape_in).astype(jnp.float32)


def batch_apply_4x4(arrs, ms, are_points=True):
    """Applies a batch of 4x4 matrices to a batch of 3D points/vectors.
    Args:
      arrs: Numpy array with shape [bs, ..., 3].
      ms: Matrix with shape [bs, 4, 4].
      are_points: Boolean. Whether to treat arr as points or vectors.
    Returns:
      Numpy array with shape [bs, ..., 3].
    """
    log("Input shapes to batch_apply_4x4: %s and %s" % (repr(arrs.shape), repr(ms.shape)))
    bs = arrs.shape[0]
    assert ms.shape[0] == bs
    assert len(ms.shape) == 3
    out = []
    for i in range(bs):
        out.append(apply_4x4(arrs[i, ...], ms[i, ...], are_points))
    return jnp.stack(out)


def transform_normals(normals, tx):
    """Transforms normals to a new coordinate frame (applies inverse-transpose).
    Args:
      normals: Numpy array with shape [batch_size, ..., 3].
      tx: Numpy array with shape [batch_size, 4, 4] or [4, 4]. Somewhat
        inefficient for [4,4] ijnputs (tiles across the batch dimension).
    Returns:
      Numpy array with shape [batch_size, ..., 3]. The transformed normals.
    """
    batch_size = normals.shape[0]
    assert normals.shape[-1] == 3
    normal_shape = list(normals.shape[1:-1])
    flat_normal_len = int(jnp.prod(normal_shape))  # 1 if []
    normals = jnp.reshape(normals, [batch_size, flat_normal_len, 3])
    assert len(tx.shape) in [2, 3]
    assert tx.shape[-1] == 4
    assert tx.shape[-2] == 4
    if len(tx.shape) == 2:
        tx = jnp.tile(tx[jnp.newaxis, ...], [batch_size, 1, 1])
    assert tx.shape[0] == batch_size

    normals_invalid = jnp.all(jnp.equal(normals, 0.0), axis=-1)
    tx_invt = jnp.linalg.inv(jnp.transpose(tx, axes=[0, 2, 1]))
    transformed = batch_apply_4x4(normals, tx_invt)
    transformed[normals_invalid, :] = 0.0
    norm = jnp.linalg.norm(transformed, axis=-1, keepdims=True)
    log.info("Norm shape, transformed shape: %s %s" % (repr(norm.shape), repr(transformed.shape)))
    transformed /= norm + 1e-8
    return jnp.reshape(transformed, [batch_size] + normal_shape + [3])


def world_xyzn_im_to_pts(world_xyz, world_n):
    """Makes a 10K long XYZN pointcloud from an XYZ image and a normal image."""
    # world im  + world normals -> world points+normals
    is_valid = jnp.logical_not(jnp.all(world_xyz == 0.0, axis=-1))
    world_xyzn = jnp.concatenate([world_xyz, world_n], axis=-1)
    world_xyzn = world_xyzn[is_valid, :]
    world_xyzn = jnp.reshape(world_xyzn, [-1, 6])
    jnp.random.shuffle(world_xyzn)
    point_count = world_xyzn.shape[0]
    assert point_count > 0
    log.info("The number of valid samples is: %i" % point_count)
    while point_count < 10000:
        world_xyzn = jnp.tile(world_xyzn, [2, 1])
        point_count = world_xyzn.shape[0]
    return world_xyzn[:10000, :]


def transform_r2n2_normal_cam_image_to_world_frame(normal_im, idx, e):
    is_valid = jnp.all(normal_im == 0.0, axis=-1)
    log.info(is_valid.shape)
    is_valid = is_valid.reshape([224, 224])
    world_im = apply_4x4(normal_im, jnp.linalg.inv(e.r2n2_cam2world[idx, ...]).T, are_points=False)
    world_im /= jnp.linalg.norm(world_im, axis=-1, keepdims=True) + 1e-8
    # world_im = jnp_util.zero_by_mask(is_valid, world_im).astype(jnp.float32)
    return world_im


def compute_argmax_image(xyz_image, decoder, embedding, k=1):
    """Uses the world space XYZ image to compute the maxblob influence image."""
    mask = jnp_util.make_pixel_mask(xyz_image)  # TODO(kgenova) Figure this out...
    assert len(mask.shape) == 2
    flat_xyz = jnp.reshape(xyz_image, [-1, 3])
    influences = decoder.rbf_influence_at_samples(embedding, flat_xyz)
    assert len(influences.shape) == 2
    rbf_image = jnp.reshape(influences, list(mask.shape) + [-1])
    # argmax_image = jnp.expand_dims(jnp.argmax(rbf_image, axis=-1), axis=-1)
    argmax_image = jnp.flip(jnp.argsort(rbf_image, axis=-1), axis=-1)
    argmax_image = argmax_image[..., :k]
    # TODO(kgenova) Insert an equivalence class map here.
    log.info(mask.shape)
    log.info(argmax_image.shape)
    argmax_image = jnp_util.zero_by_mask(mask, argmax_image, replace_with=-1)
    log.info(argmax_image.shape)
    return argmax_image.astype(jnp.int32)


def tile_world2local_frames(world2local, lyr):
    """Lifts from element_count world2local to effective_element_count frames."""
    world2local = world2local.copy()
    first_k = world2local[:lyr, :, :]
    refl = jnp.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=jnp.float32,
    )
    refl = jnp.tile(jnp.reshape(refl, [1, 4, 4]), [lyr, 1, 1])
    # log.info(refl.shape)
    first_k = jnp.matmul(first_k, refl)
    all_bases = jnp.concatenate([world2local, first_k], axis=0)
    # log.info(all_bases[1, ...])
    # log.info(all_bases[31, ...])
    # cand = jnp.array([1, 1, 1, 1], dtype=jnp.float32)
    # log.info(jnp.matmul(all_bases[1, ...], cand))
    # log.info(jnp.matmul(all_bases[31, ...], cand))
    return all_bases


def extract_local_frame_images(world_xyz_im, world_normal_im, embedding, decoder):
    """Computes local frame XYZ images for each of the world2local frames."""
    world2local = jnp.squeeze(decoder.world2local(embedding))
    log.info(world2local.shape)
    world2local = tile_world2local_frames(world2local, 15)
    log.info(world2local.shape)
    xyz_ims = []
    nrm_ims = []
    is_invalid = jnp.all(world_xyz_im == 0.0, axis=-1)
    # is_valid = jnp.logical_not(is_invalid)
    # plot(is_invalid)
    for i in range(world2local.shape[0]):
        m = world2local[i, :, :]
        local_xyz_im = apply_4x4(world_xyz_im, m, are_points=True)
        local_xyz_im[is_invalid, :] = 0.0
        local_nrm_im = apply_4x4(world_normal_im, jnp.linalg.inv(m).T, are_points=False)
        local_nrm_im /= jnp.linalg.norm(local_nrm_im, axis=-1, keepdims=True) + 1e-8
        local_nrm_im[is_invalid, :] = 0.0
        xyz_ims.append(local_xyz_im)
        nrm_ims.append(local_nrm_im)
    # log.info(xyz_ims[1][is_valid, :])
    # log.info(xyz_ims[31][is_valid, :])
    return jnp.stack(xyz_ims), jnp.stack(nrm_ims)


def select_top_k(argmax_im, xyzn_images):
    """Computes the h,w,k,6 image with the XYZN values of the argmax blobs."""
    # TODO(kgenova) The argmax im has to be [h, w, k]... right now it's just k
    argmax_im = argmax_im.copy()
    xyzn_images = xyzn_images.copy()
    k = argmax_im.shape[-1]
    argmax_im_nonneg = argmax_im.copy()
    # argmax_im_nonneg[argmax_im == -1] = 0
    argmax_im_nonneg = jnp.reshape(argmax_im_nonneg, [224, 224, k])

    chosen = jnp.zeros((224, 224, k, 6), dtype=jnp.float32)
    # ind_im = jnp.reshape(argmax_im, [224, 224])
    # log.info(jnp.min(argmax_im_nonneg))
    # log.info(jnp.max(argmax_im_nonneg))
    for i in range(224):
        for j in range(224):
            for ki in range(k):
                idx = argmax_im_nonneg[i, j, ki]
                if idx != -1:
                    chosen[i, j, ki, :] = xyzn_images[argmax_im_nonneg[i, j, ki], i, j, :]
    # log.info(chosen.shape)
    # plot(chosen[..., 1, :3])
    return chosen


def make_r2n2_gt_top_k_image(e, idx, encoder, decoder, k=2):
    """Computes the top-k influence image for a 3D-R2N2 example."""
    world_xyz = e.r2n2_xyz_images.copy()[idx, ...]
    world_n = e.r2n2_normal_world_images.copy()[idx, ...]
    # world_xyzn = jnp.concatenate([world_xyz, world_n], axis=-1)
    embedding = encoder.run_example(e)
    xyz_ims, nrm_ims = extract_local_frame_images(world_xyz, world_n, embedding, decoder)
    argmax_im = compute_argmax_image(world_xyz, decoder, embedding, k=k)

    # log.info(xyz_ims.shape)
    # log.info(nrm_ims.shape)
    # log.info(argmax_im.shape)
    top_k_im = select_top_k(argmax_im, jnp.concatenate([xyz_ims, nrm_ims], axis=-1))
    return top_k_im, argmax_im


def depth_to_xyz_image(depth_image, intrinsics, extrinsics):
    """Convert depth images to xyz images."""
    fx, fy, cx, cy = intrinsics
    mask = depth_image != 0
    z = depth_image
    u, v = jnp.meshgrid(jnp.arange(z.shape[1]), jnp.arange(z.shape[0]))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    cam2world = extrinsics
    flip_yz = jnp.eye(4)
    flip_yz[1], flip_yz[2] = -flip_yz[1], -flip_yz[2]
    transform = jnp.matmul(cam2world, flip_yz)
    xyz_image = jnp.stack([x, y, z, jnp.ones_like(x)], axis=-1)
    xyz_image = jnp.matmul(transform, jnp.transpose(xyz_image, (0, 2, 1)))
    xyz_image = jnp.transpose(xyz_image, (0, 2, 1))
    xyz_image = xyz_image[..., :3] / xyz_image[..., 3:]
    return xyz_image, mask


def depth_to_point_cloud(depth_image, intrinsics, extrinsics):
    xyz_image, mask = depth_to_xyz_image(depth_image, intrinsics, extrinsics)
    log.info(xyz_image.shape, mask.shape)
    xyz = xyz_image.reshape([-1, 3])[mask.ravel()]
    return xyz


def depth_image_to_cam_image(depth_image, fx=585.0, fy=585.0, cx=255.5, cy=255.5):
    """Converts a GAPS depth image to a camera-space image.
    Args:
      depth_image: Numpy array with shape [height, width] or [height, width, 1].
        The depth in units (not in 1000ths of units, which is what GAPS writes).
      fx: The x focal-length. Float.
      fy: The y focal-length. Float.
      cx: The x center. Float.
      cy: The y center. Float.
    Returns:
      cam_image: array with shape [height, width, 3].
    """
    is_invalid = jnp.squeeze(depth_image == 0.0)
    height, width = depth_image.shape[0:2]
    depth_image = jnp.reshape(depth_image, [height, width])
    pixel_coords = jnp_util.make_coordinate_grid(height, width, is_screen_space=True, is_homogeneous=False)
    # log.info(jnp.max(pixel_coords))
    # log.info(jnp.min(pixel_coords))
    nic_x = pixel_coords[:, :, 0]
    nic_y = pixel_coords[:, :, 1]
    nic_d = -depth_image

    # cam_x = (nic_x - 255.5) * (-nic_d) / 585.0
    # cam_y = (nic_y - 255.5) * (nic_d) / 585.0
    cam_x = (nic_x - cx) * (-nic_d) / fx
    cam_y = (nic_y - cy) * nic_d / fy
    cam_z = nic_d
    cam = jnp.stack([cam_x, cam_y, cam_z], axis=-1)
    cam[is_invalid, :] = 0.0
    return cam


def depth_images_to_cam_images(depth_images, fx=585.0, fy=585.0, cx=255.5, cy=255.5):
    """Converts depth images to xyz camera space images."""
    cam_ims = []
    add_one = depth_images.shape[-1] == 1
    assert len(depth_images.shape) == 3 + add_one
    for i in range(depth_images.shape[0]):
        cam_ims.append(depth_image_to_cam_image(depth_images[i, ...], fx, fy, cx, cy))
    return jnp.stack(cam_ims)
