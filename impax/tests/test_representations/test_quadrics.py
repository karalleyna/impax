"""Tests for Quadrics"""

import tensorflow as tf
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

import ldif.ldif.representation.quadrics as original
from impax.representations.quadrics import (
    compute_shape_element_influences,
    decode_covariance_roll_pitch_yaw,
    sample_axis_aligned_bf,
    sample_cov_bf,
    sample_isotropic_bf,
    sample_quadric_surface,
)


class QuadricsTest(parameterized.TestCase):
    """Test cases for quadrics representations."""

    def test_sample_quadric_surface(self, key=random.PRNGKey(0)):
        # array of  [..., 4, 4]
        key0, key1 = random.split(key)
        n, shape = 2, (3, 5)
        quadrics = random.normal(key0, shape=(*shape, 4, 4))
        quadrics_tf = tf.convert_to_tensor(quadrics)

        #  shape [..., 3]
        center = jnp.zeros((*shape, 3))
        center_tf = tf.convert_to_tensor(center)

        # samples:  [..., N, 3], where N is the number of samples
        samples = random.normal(key1, shape=(*shape, n, 3))
        samples_tf = tf.convert_to_tensor(samples)

        ground_truth = original.sample_quadric_surface(
            quadrics_tf, center_tf, samples_tf
        )

        ret = sample_quadric_surface(quadrics, center, samples)
        self.assertEqual(ret.shape, ground_truth.shape)
        assert jnp.allclose(ret, ground_truth.numpy())

    @parameterized.product(invert=(True, False))
    def test_decode_covariance_roll_pitch_yaw(self, invert, key=random.PRNGKey(0)):
        # array of  [..., 4, 4]
        shape = (3, 2)
        radius = random.normal(key, shape=(*shape, 6))
        radius_tf = tf.convert_to_tensor(radius)

        ground_truth = original.decode_covariance_roll_pitch_yaw(
            radius_tf, invert=invert
        )

        ret = decode_covariance_roll_pitch_yaw(radius, invert=invert)
        self.assertEqual(ret.shape, ground_truth.shape)
        assert jnp.allclose(ret, ground_truth.numpy())

    def test_sample_cov_bf(self, key=random.PRNGKey(0)):
        key0, key1 = random.split(key)
        n, shape = 2, (3, 5)
        quadrics = random.normal(key0, shape=(*shape, 3))
        quadrics_tf = tf.convert_to_tensor(quadrics)

        #  shape [..., 3]
        center = jnp.zeros((*shape, 6))
        center_tf = tf.convert_to_tensor(center)

        # samples:  [..., N, 3], where N is the number of samples
        samples = random.normal(key1, shape=(*shape, n, 3))
        samples_tf = tf.convert_to_tensor(samples)

        ground_truth = original.sample_cov_bf(quadrics_tf, center_tf, samples_tf)

        ret = sample_cov_bf(quadrics, center, samples)
        self.assertEqual(ret.shape, ground_truth.shape)
        assert jnp.allclose(ret, ground_truth.numpy())

    def test_sample_axis_aligned_bf(self, key=random.PRNGKey(0)):
        key0, key1 = random.split(key)
        n, shape = 2, (3, 5)
        quadrics = random.normal(key0, shape=(*shape, 3))
        quadrics_tf = tf.convert_to_tensor(quadrics)

        #  shape [..., 3]
        center = jnp.zeros((*shape, 3))
        center_tf = tf.convert_to_tensor(center)

        # samples:  [..., N, 3], where N is the number of samples
        samples = random.normal(key1, shape=(*shape, n, 3))
        samples_tf = tf.convert_to_tensor(samples)

        ground_truth = original.sample_axis_aligned_bf(
            quadrics_tf, center_tf, samples_tf
        )

        ret = sample_axis_aligned_bf(quadrics, center, samples)
        self.assertEqual(ret.shape, ground_truth.shape)
        assert jnp.allclose(ret, ground_truth.numpy())

    def test_sample_isotropic_bf(self, key=random.PRNGKey(0)):
        key0, key1 = random.split(key)
        n, shape = 2, (3, 5)
        quadrics = random.normal(key0, shape=(*shape, 3))
        quadrics_tf = tf.convert_to_tensor(quadrics)

        #  shape [..., 3]
        center = jnp.zeros((*shape, 1))
        center_tf = tf.convert_to_tensor(center)

        # samples:  [..., N, 3], where N is the number of samples
        samples = random.normal(key1, shape=(*shape, n, 3))
        samples_tf = tf.convert_to_tensor(samples)

        ground_truth = original.sample_isotropic_bf(quadrics_tf, center_tf, samples_tf)

        ret = sample_isotropic_bf(quadrics, center, samples)
        self.assertEqual(ret.shape, ground_truth.shape)
        assert jnp.allclose(ret, ground_truth.numpy())

    @parameterized.product(radius_length=(1, 3, 6))
    def test_compute_shape_element_influences(
        self, radius_length, key=random.PRNGKey(0)
    ):
        key0, key1 = random.split(key)
        batch_size, quadric_count, sample_count = 2, 3, 5
        quadrics = random.normal(key0, shape=(batch_size, quadric_count, 4, 4))
        quadrics_tf = tf.convert_to_tensor(quadrics)

        #  shape [..., 3]
        center = jnp.zeros((batch_size, quadric_count, 3))
        center_tf = tf.convert_to_tensor(center)

        # samples:  [..., N, 3], where N is the number of samples
        radii = random.normal(key1, shape=(batch_size, quadric_count, radius_length))
        radii_tf = tf.convert_to_tensor(radii)

        samples = random.normal(
            key1, shape=(batch_size, quadric_count, sample_count, 3)
        )
        samples_tf = tf.convert_to_tensor(samples)

        ground_truth_outputs = original.compute_shape_element_influences(
            quadrics_tf, center_tf, radii_tf, samples_tf
        )

        outputs = compute_shape_element_influences(quadrics, center, radii, samples)

        for ret, ground_truth in zip(outputs, ground_truth_outputs):
            self.assertEqual(ret.shape, ground_truth.shape)
            assert jnp.allclose(ret, ground_truth.numpy())


if __name__ == "__main__":
    absltest.main()
