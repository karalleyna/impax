"""Utilities to evaluate quadric implicit surface functions."""
import jax.numpy as jnp
from jax import vmap
from impax.utils import camera_util
from impax.utils import jax_util


NORMALIZATION_EPS = 1e-8
SQRT_NORMALIZATION_EPS = 1e-4
DIV_EPSILON = 1e-8


def sample_quadric_surface(quadric, center, samples):
    """Samples the algebraic distance to the input quadric at sparse locations.
    Args:
      quadric: DeviceArray with shape [..., 4, 4]. Contains the matrix of the quadric
        surface.
      center: DeviceArray with shape [..., 3]. Contains the [x,y,z] coordinates of the
        center of the coordinate frame of the quadric surface in NIC space with a
        top-left origin.
      samples: DeviceArray with shape [..., N, 3], where N is the number of samples to
        evaluate. These are the sample locations in the same space in which the
        quadric surface center is defined. Supports broadcasting the batching
        dimensions.
    Returns:
      distances: DeviceArray with shape [..., N, 1]. Contains the algebraic distance
        to the surface at each sample.
    """
    batching_dimensions = quadric.shape[:-2]
    batching_rank = len(batching_dimensions)

    # We want to transform the coordinates so that they are in the coordinate
    # frame of the conic section matrix, so we subtract the center of the
    # conic.
    samples = samples - jnp.expand_dims(center, axis=batching_rank)
    sample_count = samples.shape[-2]

    homogeneous_sample_ones = jnp.ones(samples.shape[:-1] + (1,), dtype=jnp.float32)
    homogeneous_sample_coords = jnp.concatenate(
        [samples, homogeneous_sample_ones], axis=-1
    )

    # When we transform the coordinates per-image, we broadcast on both sides-
    # the batching dimensions broadcast up the coordinate grid, and the
    # coordinate center broadcasts up along the height and width.
    # Per-pixel, the algebraic distance is v^T * M * v, where M is the matrix
    # of the conic section, and v is the homogeneous column vector [x y z 1]^T.
    length = len(homogeneous_sample_coords.shape)
    half_distance = jnp.matmul(
        quadric, jnp.swapaxes(homogeneous_sample_coords, length - 2, length - 1)
    )

    rank = batching_rank + 2
    half_distance = jnp.transpose(
        half_distance, axes=list(range(rank - 2)) + [rank - 1, rank - 2]
    )
    algebraic_distance = jnp.sum(
        jnp.multiply(homogeneous_sample_coords, half_distance), axis=-1
    )
    return jnp.reshape(algebraic_distance, batching_dimensions + (sample_count, 1))


def decode_covariance_roll_pitch_yaw(radius, invert=False):
    """Converts 6-D radus vectors to the corresponding covariance matrices.
    Args:
      radius: DeviceArray with shape [..., 6]. First three numbers are covariances of
        the three Gaussian axes. Second three numbers are the roll-pitch-yaw
        rotation angles of the Gaussian frame.
      invert: Whether to return the inverse covariance.
    Returns:
       DeviceArray with shape [..., 3, 3]. The 3x3 (optionally inverted) covariance
       matrices corresponding to the input radius vectors.
    """
    d = 1.0 / (radius[..., 0:3] + DIV_EPSILON) if invert else radius[..., 0:3]

    diag = vmap(jnp.diag)(d.reshape((-1, d.shape[-1])))
    diag = diag.reshape(d.shape + (d.shape[-1],))
    rotation = camera_util.roll_pitch_yaw_to_rotation_matrices(radius[..., 3:6])
    length = len(rotation.shape)

    return jnp.matmul(
        jnp.matmul(rotation, diag), jnp.swapaxes(rotation, length - 2, length - 1)
    )


def sample_cov_bf(center, radius, samples):
    """Samples gaussian radial basis functions at specified coordinates.
    Args:
      center: DeviceArray with shape [..., 3]. Contains the [x,y,z] coordinates of the
        RBF center in NIC space with a top-left origin.
      radius: DeviceArray with shape [..., 6]. First three numbers are covariances of
        the three Gaussian axes. Second three numbers are the roll-pitch-yaw
        rotation angles of the Gaussian frame.
      samples: DeviceArray with shape [..., N, 3],  where N is the number of samples to
        evaluate. These are the sample locations in the same space in which the
        quadric surface center is defined. Supports broadcasting the batching
        dimensions.
    Returns:
       DeviceArray with shape [..., N, 1]. The basis function strength at each sample
       location.
    """
    # Compute the samples' offset from center, then extract the coordinates.
    diff = samples - jnp.expand_dims(center, axis=-2)
    x, y, z = diff[..., -3], diff[..., -2], diff[..., -1]
    # Decode 6D radius vectors into inverse covariance matrices, then extract
    # unique elements.
    inv_cov = decode_covariance_roll_pitch_yaw(radius, invert=True)
    shape = inv_cov.shape[:-2] + (1, 9)
    inv_cov = jnp.reshape(inv_cov, shape)
    c00 = inv_cov[..., 0, 0][..., None]
    c01 = inv_cov[..., 0, 1][..., None]
    c02 = inv_cov[..., 0, 2][..., None]

    c11 = inv_cov[..., 1, 1][..., None]
    c12 = inv_cov[..., 1, 2][..., None]

    c22 = inv_cov[..., 2, 2][..., None]

    # Compute function value.
    dist = (
        x * (c00 * x + c01 * y + c02 * z)
        + y * (c01 * x + c11 * y + c12 * z)
        + z * (c02 * x + c12 * y + c22 * z)
    )
    dist = jnp.exp(-0.5 * dist)
    return dist


def sample_axis_aligned_bf(center, radius, samples):
    """Samples gaussian radial basis functions at specified coordinates.
    Args:
      center: DeviceArray with shape [..., 3]. Contains the [x,y,z] coordinates of the
        RBF center in NIC space with a top-left origin.
      radius: DeviceArray with shape [..., 3]. The covariance of the RBF in NIC space
        along the x, y, and z axes.
      samples: DeviceArray with shape [..., N, 3],  where N is the number of samples to
        evaluate. These are the sample locations in the same space in which the
        quadric surface center is defined. Supports broadcasting the batching
        dimensions.
    Returns:
       DeviceArray with shape [..., N, 1]. The basis function strength at each sample
       location.
    """
    diff = samples - jnp.expand_dims(center, axis=-2)
    squared_diff = jnp.square(diff)
    scale = jnp.minimum((-2) * jnp.expand_dims(radius, axis=-2), -NORMALIZATION_EPS)
    return jnp.exp(jnp.sum(squared_diff / scale, axis=-1, keepdims=True))


def sample_isotropic_bf(center, radius, samples):
    """Samples gaussian radial basis functions at specified coordinates.
    Args:
      center: DeviceArray with shape [..., 3]. Contains the [x,y,z] coordinates of the
        RBF center in NIC space with a top-left origin.
      radius: DeviceArray with shape [..., 1]. Twice the variance of the RBF in NIC
        space.
      samples: DeviceArray with shape [..., N, 3],  where N is the number of samples to
        evaluate. These are the sample locations in the same space in which the
        quadric surface center is defined. Supports broadcasting the batching
        dimensions.
    Returns:
       DeviceArray with shape [..., N, 1]. The RBF strength at each sample location.
    """
    batching_dimensions = center.shape[:-1]
    batching_rank = len(batching_dimensions)

    # Reshape the center to allow broadcasting over the sample domain:
    center = jnp.expand_dims(center, axis=batching_rank)
    samples -= center
    l2_norm = (
        samples[..., 0] * samples[..., 0]
        + samples[..., 1] * samples[..., 1]
        + samples[..., 2] * samples[..., 2]
    )
    # Ensure the radius is large enough to avoid numerical issues:
    radius = jnp.maximum(SQRT_NORMALIZATION_EPS, radius)
    weights = jnp.exp(-0.5 * l2_norm / radius)
    return jnp.expand_dims(weights, axis=-1)


def compute_shape_element_influences(quadrics, centers, radii, samples):
    """Computes the per-shape-element values at given sample locations.
    Args:
      quadrics: quadric parameters with shape [batch_size, quadric_count, 4, 4].
      centers: rbf centers with shape [batch_size, quadric_count, 3].
      radii: rbf radii with shape [batch_size, quadric_count, radius_length].
        radius_length can be 1, 3, or 6 depending on whether it is isotropic,
        anisotropic, or a general symmetric covariance matrix, respectively.
      samples: a grid of samples with shape [batch_size, quadric_count,
        sample_count, 3] or shape [batch_size, sample_count, 3].
    Returns:
      Two DeviceArrays (the quadric values and the RBF values, respectively), each
      with shape [batch_size, quadric_count, sample_count, 1]
    """
    # Select the number of samples along the ray. The larger this is, the
    # more memory that will be consumed and the slower the algorithm. But it
    # reduces warping artifacts and the likelihood of missing a thin surface.
    batch_size, quadric_count = quadrics.shape[:2]

    # We separate the isometric, axis-aligned, and general RBF functions.
    # The primary reason for this is that the type of basis function
    # affects the shape of many DeviceArrays, and it is easier to make
    # everything correct when the shape is known. Once the shape function is
    # set we can clean it up and choose one basis function.
    radii_shape = radii.shape
    if len(radii_shape) != 3:
        raise ValueError(
            "radii must have shape [batch_size, quadric_count, radii_values]."
        )
    elif radii_shape[2] == 1:
        rbf_sampler = sample_isotropic_bf
    elif radii_shape[2] == 3:
        rbf_sampler = sample_axis_aligned_bf
    elif radii_shape[2] == 6:
        rbf_sampler = sample_cov_bf
    else:
        raise ValueError("radii must have either 1, 3, or 6 elements.")

    # Ensure the samples have the right shape and tile in an axis for the
    # quadric dimension if it wasn't provided.
    sample_shape = samples.shape
    sample_rank = len(sample_shape)
    if (
        sample_rank not in [3, 4]
        or sample_shape[-1] != 3
        or sample_shape[0] != batch_size
    ):
        raise ValueError(
            f"Input DeviceArray samples must have shape [batch_size, quadric_count, sample_count, 3]"
            + "or shape [batch_size, sample_count, 3]. The input shape was {sample_shape}"
        )

    missing_quadric_dim = len(sample_shape) == 3
    if missing_quadric_dim:
        samples = jax_util.tile_new_axis(samples, axis=1, length=quadric_count)
    sample_count = sample_shape[-2]

    # Sample the quadric surfaces and the RBFs in world space, and composite
    # them.
    sampled_quadrics = sample_quadric_surface(quadrics, centers, samples)

    sampled_rbfs = rbf_sampler(centers, radii, samples)
    sampled_rbfs = jnp.reshape(
        sampled_rbfs, [batch_size, quadric_count, sample_count, 1]
    )
    return sampled_quadrics, sampled_rbfs
