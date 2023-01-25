"""
Utilities for manipulating images and image DeviceArrays.
References:
https://github.com/google/ldif/blob/master/ldif/util/image_util.py
"""
from functools import reduce

import jax.numpy as jnp
from jax import lax


def rgba_to_rgb(rgba, white_background: bool = True, smooth: bool = True):
    """Converts rgba to rgb images.
    Args:
      rgba: A DeviceArray with shape [..., 4]. Should be in the [0, 1] range and of
        type jnp.float32.
    Returns:
      A DeviceArray with shape [..., 3].
    """
    channel_count = rgba.shape[-1]
    assert channel_count == 4
    assert rgba.dtype == jnp.float32

    a = rgba[..., 3:4]
    background_color_value = float(white_background)
    if smooth:
        rgb = rgba[..., :3] * a + background_color_value * (1 - a)
    elif not white_background:
        rgb = rgba[..., :3]
    else:  # White, not smooth:
        rgb = jnp.where(a, rgba[..., :3], 1.0)
    return rgb


def downsample(images, exp=1):
    """Downsamples an image by a power of 2, averaging per pixel.
    Assumes that the images are already sufficiently blurred so as not to incur
    box filter artifacts.
    Args:
      images: DeviceArray with shape [..., height, width, channel_count]. The input
        image to downsample.
      exp: Integer specifying the number of times to halve the input resolution.
    Returns:
      DeviceArray with shape [..., height / 2^exp, width / 2^exp, channel_count].
    """
    for _ in range(exp):
        images = jnp.mean(
            jnp.stack(
                [
                    images[..., 0::2, 0::2, :],
                    images[..., 0::2, 1::2, :],
                    images[..., 1::2, 0::2, :],
                    images[..., 1::2, 1::2, :],
                ],
                axis=0,
            ),
            axis=0,
        )

    return images


def get_border_pixels(gt, threshold=0.1):
    """Returns a mask indicating whether each pixel is on the shape's border."""
    outside_pixels = gt >= threshold
    inside_pixels = gt <= -threshold
    border_pixels = jnp.logical_not(jnp.logical_or(outside_pixels, inside_pixels))
    return border_pixels


def hessian(sdf_im):
    """Computes the hessian matrix of a 2D distance function image."""
    batch_size, height, width = [sdf_im.shape[i] for i in range(3)]
    sdf_im = jnp.reshape(sdf_im, [batch_size, height, width, 1])
    # pyformat: disable
    xx_fda_kernel = jnp.reshape(
        jnp.array([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=jnp.float32),
        [3, 3, 1, 1],
    )
    yy_fda_kernel = jnp.reshape(
        jnp.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32),
        [3, 3, 1, 1],
    )
    xy_fda_kernel = jnp.reshape(
        jnp.array([[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]], dtype=jnp.float32),
        [3, 3, 1, 1],
    )
    # pyformat: enable
    fda_kernel = jnp.concatenate([xx_fda_kernel, xy_fda_kernel, xy_fda_kernel, yy_fda_kernel], axis=3)
    # NCHWD…, OIHWD…, NCHWD
    sdf_im = jnp.transpose(sdf_im, axes=[0, 3, 1, 2])
    fda_kernel = jnp.transpose(fda_kernel, axes=[3, 2, 0, 1])

    fda = lax.conv(sdf_im, fda_kernel, [1, 1], padding="SAME")
    # Fda should have shape [batch_size, height, width, 4], because we duplicated
    # the xy partial channels.
    fda = jnp.transpose(fda, axes=[0, 2, 3, 1])
    hess = jnp.reshape(fda, [batch_size, height, width, 2, 2])
    # Because we used an fda method, we don't have to symmetrize, so just return:
    return hess


def get_pil_formatted_image(image):
    """Converts the output of a mesh_renderer call to a numpy array for PIL.
    Args:
      image: a 1D numpy array containing an image using the coordinate scheme
          of mesh_renderer and containing RGBA values in the [0,1] range.
    Returns:
      A 3D numpy array suitable for input to PilImage.fromarray().
    """
    height, width, channel_count = image.shape
    if channel_count != 1:
        raise ValueError(
            "Single-channel input image was expected (dim 2), but " "input has shape %s" % (str(image.shape))
        )
    image = jnp.tile(image, [1, 1, 3])
    alpha = jnp.ones([height, width, 1], dtype=jnp.float32)
    image = jnp.concatenate([image, alpha], axis=2)
    out = jnp.clip(255.0 * image, 0.0, 255.0).astype(jnp.uint8).copy(order="K")
    if out.shape[0] != height or out.shape[1] != width or out.shape[2] != 4:
        raise AssertionError(
            "Internal error: output shape should be (%i, %i, 4) but " "is %s" % (height, width, str(out.shape))
        )
    return out


def images_are_near(baseline_image, result_image, max_outlier_fraction=0.005, pixel_error_threshold=0.04):
    """Compares two image arrays.
    The comparison is soft: the images are considered identical if fewer than
    max_outlier_fraction of the pixels differ by more than pixel_error_threshold
    of the full color value.
    Differences in JPEG encoding can produce pixels with pretty large variation,
    so by default we use 0.04 (4%) for pixel_error_threshold and 0.005 (0.5%) for
    max_outlier_fraction.
    Args:
      baseline_image: a numpy array containing the baseline image.
      result_image: a numpy array containing the result image.
      max_outlier_fraction: fraction of pixels that may vary by more than the
        error threshold. 0.005 means 0.5% of pixels.
      pixel_error_threshold: pixel values are considered to differ if their
        difference exceeds this amount. Range is 0.0 - 1.0.
    Returns:
      A (boolean, string) tuple where the first value is whether the images
      matched, and the second is a pretty-printed summary of the differences.
    """
    if baseline_image.shape != result_image.shape:
        return False, "Image shapes do not match"

    float_base = baseline_image.astype(float) / 255.0
    float_result = result_image.astype(float) / 255.0

    outlier_channels = jnp.abs(float_base - float_result) > pixel_error_threshold
    if len(baseline_image.shape) > 2:
        outlier_pixels = jnp.any(outlier_channels, axis=2)
    else:
        outlier_pixels = outlier_channels
    outlier_fraction = jnp.count_nonzero(outlier_pixels) / reduce(lambda x, y: x * y, baseline_image.shape[:2])
    images_match = outlier_fraction <= max_outlier_fraction
    message = " (%f of pixels are outliers, maximum allowed is %f) " % (
        outlier_fraction,
        max_outlier_fraction,
    )
    return images_match, message
