import jax
import jax.numpy as jnp
import pytest
import tensorflow as tf
from jax import random
from ml_collections import ConfigDict

from impax.utils import image_util
from ldif.ldif.util import image_util as orig


@pytest.mark.parametrize("seed", [2, 4, 6, 8])
@pytest.mark.parametrize("white_background", [True, False])
@pytest.mark.parametrize("smooth", [True, False])
def test_rgba_to_rgb(seed, white_background, smooth):
    key = random.PRNGKey(seed)
    rgba = jax.random.normal(key, (32, 32, 32, 4))

    cfg = {
        "hparams": {"bg": ("w" if white_background else "b") + ("s" if smooth else "")}
    }

    model_config = ConfigDict(cfg)

    gnd = orig.rgba_to_rgb(model_config, tf.convert_to_tensor(rgba))

    ret = image_util.rgba_to_rgb(rgba, white_background, smooth)

    assert jnp.allclose(gnd.numpy(), ret)


@pytest.mark.parametrize("seed", [2, 4, 6, 8])
def test_downsample(seed):
    key = random.PRNGKey(seed)

    key, key2 = random.split(key)
    images = jax.random.normal(key, (32, 32, 32, 3))
    exp = jax.random.randint(key2, (1,), 0, 5)[0]

    gnd = orig.downsample(tf.convert_to_tensor(images), exp)

    ret = image_util.downsample(images, exp)

    assert jnp.allclose(gnd.numpy(), ret)


@pytest.mark.parametrize("seed", [2, 4, 6, 8])
def test_get_border_pixels(seed):
    key = random.PRNGKey(seed)
    gt = jax.random.normal(key, (32, 32, 32, 3))

    gnd = orig.get_border_pixels(tf.convert_to_tensor(gt), 0.1)
    ret = image_util.get_border_pixels(gt, 0.1)

    assert jnp.allclose(gnd.numpy(), ret)


@pytest.mark.parametrize("seed", [2, 4, 6, 8])
def test_hessian(seed):
    key = random.PRNGKey(seed)

    sdf_im = jax.random.normal(key, (64, 24, 24))

    gnd = orig.hessian(tf.convert_to_tensor(sdf_im))
    ret = image_util.hessian(sdf_im)

    assert jnp.allclose(gnd.numpy(), ret)


@pytest.mark.parametrize("seed", [2, 4, 6, 8])
def test_get_pil_formatted_image(seed):
    key = random.PRNGKey(seed)

    img = jax.random.normal(key, (32, 32, 1))

    gnd = orig.get_pil_formatted_image(img)

    ret = image_util.get_pil_formatted_image(img)

    assert jnp.allclose(gnd, ret)


@pytest.mark.parametrize("seed", [2, 4, 6, 8])
def test_images_are_near(seed):
    key = random.PRNGKey(seed)
    baseline_image = jax.random.normal(key, (32, 32, 3))
    result_image = jax.random.normal(key, (32, 32, 3))

    gnd, m1 = orig.images_are_near(
        baseline_image,
        result_image,
        max_outlier_fraction=0.005,
        pixel_error_threshold=0.04,
    )
    ret, m2 = image_util.images_are_near(
        baseline_image,
        result_image,
        max_outlier_fraction=0.005,
        pixel_error_threshold=0.04,
    )
    assert jnp.allclose(gnd, ret)
