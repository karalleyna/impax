"""
Utilities for manipulating images and image DeviceArrays.
References:
https://github.com/google/ldif/blob/master/ldif/util/image_util.py
"""
import jax.numpy as jnp
from jax import lax

import io
import os
import six
from PIL import Image as PilImage


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
    batch_size, height, width = [sdf_im.shape[i].value for i in range(3)]
    sdf_im = jnp.reshape(sdf_im, [batch_size, height, width, 1])
    # pyformat: disable
    xx_fda_kernel = jnp.reshape(
        jnp.array(
            [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=jnp.float32
        ),
        [3, 3, 1, 1],
    )
    yy_fda_kernel = jnp.reshape(
        jnp.array(
            [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32
        ),
        [3, 3, 1, 1],
    )
    xy_fda_kernel = jnp.reshape(
        jnp.array(
            [[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]], dtype=jnp.float32
        ),
        [3, 3, 1, 1],
    )
    # pyformat: enable
    fda_kernel = jnp.concat(
        [xx_fda_kernel, xy_fda_kernel, xy_fda_kernel, yy_fda_kernel], axis=3
    )
    fda = lax.conv_general_dilated(sdf_im, fda_kernel, [1, 1, 1, 1], padding="SAME")
    # Fda should have shape [batch_size, height, width, 4], because we duplicated
    # the xy partial channels.
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
            "Single-channel input image was expected (dim 2), but "
            "input has shape %s" % (str(image.shape))
        )
    image = jnp.tile(image, [1, 1, 3])
    alpha = jnp.ones([height, width, 1], dtype=jnp.float32)
    image = jnp.concatenate([image, alpha], axis=2)
    out = jnp.clip(255.0 * image, 0.0, 255.0).astype(jnp.uint8).copy(order="C")
    if out.shape[0] != height or out.shape[1] != width or out.shape[2] != 4:
        raise AssertionError(
            "Internal error: output shape should be (%i, %i, 4) but "
            "is %s" % (height, width, str(out.shape))
        )
    return out


def images_are_near(
    baseline_image, result_image, max_outlier_fraction=0.005, pixel_error_threshold=0.04
):
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
    outlier_fraction = jnp.count_nonzero(outlier_pixels) / jnp.prod(
        baseline_image.shape[:2]
    )
    images_match = outlier_fraction <= max_outlier_fraction
    message = " (%f of pixels are outliers, maximum allowed is %f) " % (
        outlier_fraction,
        max_outlier_fraction,
    )
    return images_match, message


def expect_images_are_near_and_save_comparison(
    test,
    baseline_image,
    result_image,
    comparison_name,
    images_differ_message,
    max_outlier_fraction=0.005,
    pixel_error_threshold=0.04,
    save_format=".png",
):
    """A convenience wrapper around ImagesAreNear that saves comparison images.
    If the images differ, this function writes the
    baseline and result images into the test's outputs directory.
    Args:
      test: a python unit test instance.
      baseline_image: baseline image as a numpy array.
      result_image: the result image as a numpy array.
      comparison_name: a string naming this comparison. Names outputs for viewing
        in sponge.
      images_differ_message: the test message to display if the images differ.
      max_outlier_fraction: fraction of pixels that may vary by more than the
        error threshold. 0.005 means 0.5% of pixels.
      pixel_error_threshold: pixel values are considered to differ if their
        difference exceeds this amount. Range is 0.0 - 1.0.
      save_format: a text string defining the image format to save.
    """
    images_match, comparison_message = images_are_near(
        baseline_image, result_image, max_outlier_fraction, pixel_error_threshold
    )

    if not images_match:
        outputs_dir = os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]
        test.assertNotEmpty(outputs_dir)

        image_mode = "RGB" if baseline_image.shape[2] == 3 else "RGBA"

        baseline_output_path = os.path.join(
            outputs_dir, six.ensure_str(comparison_name) + "_baseline" + save_format
        )
        PilImage.fromarray(baseline_image, mode=image_mode).save(baseline_output_path)

        result_output_path = (
            os.path.join(outputs_dir, six.ensure_str(comparison_name) + "_result")
            + save_format
        )
        PilImage.fromarray(result_image, mode=image_mode).save(result_output_path)

    test.assertEqual(baseline_image.shape, result_image.shape)
    test.assertTrue(
        images_match, msg=six.ensure_str(images_differ_message) + comparison_message
    )


def expect_image_file_and_image_are_near(
    test,
    baseline_path,
    result_image_bytes_or_numpy,
    comparison_name,
    images_differ_message,
    max_outlier_fraction=0.005,
    pixel_error_threshold=0.04,
    resize_baseline_image=None,
):
    """Compares the input image bytes with an image on disk.
    The comparison is soft: the images are considered identical if fewer than
    max_outlier_fraction of the pixels differ by more than pixel_error_threshold
    of the full color value. If the images differ, the function writes the
    baseline and result images into the test's outputs directory.
    Uses ImagesAreNear for the actual comparison.
    Args:
      test: a python unit test instance.
      baseline_path: path to the reference image on disk.
      result_image_bytes_or_numpy: the result image, as either a bytes object or a
        numpy array.
      comparison_name: a string naming this comparison. Names outputs for viewing
        in sponge.
      images_differ_message: the test message to display if the images differ.
      max_outlier_fraction: fraction of pixels that may vary by more than the
        error threshold. 0.005 means 0.5% of pixels.
      pixel_error_threshold: pixel values are considered to differ if their
        difference exceeds this amount. Range is 0.0 - 1.0.
      resize_baseline_image: a (width, height) tuple giving a new size to apply to
        the baseline image, or None.
    """
    try:
        result_image = jnp.array(PilImage.open(io.BytesIO(result_image_bytes_or_numpy)))
    except IOError:
        result_image = result_image_bytes_or_numpy
    baseline_pil_image = PilImage.open(baseline_path)
    baseline_format = ("." + six.ensure_str(baseline_pil_image.format)).lower()

    if resize_baseline_image:
        baseline_pil_image = baseline_pil_image.resize(
            resize_baseline_image, PilImage.ANTIALIAS
        )
    baseline_image = jnp.array(baseline_pil_image)

    expect_images_are_near_and_save_comparison(
        test,
        baseline_image,
        result_image,
        comparison_name,
        images_differ_message,
        max_outlier_fraction,
        pixel_error_threshold,
        baseline_format,
    )
