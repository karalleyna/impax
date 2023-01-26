"""Tests for model.py."""
from absl.testing import absltest

import os
import jax.numpy as jnp

from impax.utils import image_util
from impax.utils import line_util
from impax.utils import path_util


class StructuredImplicitFunctionTest(absltest.TestCase):
    def setUp(self):
        super(StructuredImplicitFunctionTest, self).setUp()
        self.test_data_directory = os.path.join(
            path_util.get_path_to_ldif_root(), "test_data"
        )

    def test_render_centered_square(self):
        line_parameters = jnp.array([0.0, 64.0, 64.0, 32.0, 32.0], dtype=jnp.float32)
        image = line_util.line_to_image(
            line_parameters, height=128, width=128, falloff=None
        )
        target_image_name = "Centered_Square_0.png"
        baseline_image_path = os.path.join(self.test_data_directory, target_image_name)
        image = image_util.get_pil_formatted_image(image)
        image_util.expect_image_file_and_image_are_near(
            self,
            baseline_image_path,
            image,
            target_image_name,
            "%s does not match." % target_image_name,
        )

    def test_render_vertical_rectangle(self):
        line_parameters = jnp.array([0.0, 64.0, 64.0, 16.0, 48.0], dtype=jnp.float32)
        image = line_util.line_to_image(
            line_parameters, height=128, width=128, falloff=None
        )
        target_image_name = "Centered_Vertical_Rectangle_0.png"
        baseline_image_path = os.path.join(self.test_data_directory, target_image_name)
        image = image_util.get_pil_formatted_image(image)
        image_util.expect_image_file_and_image_are_near(
            self,
            baseline_image_path,
            image,
            target_image_name,
            "%s does not match." % target_image_name,
        )

    def test_render_offset_vertical_rectangle(self):
        line_parameters = jnp.array([0.0, 80.0, 49.0, 16.0, 48.0], dtype=jnp.float32)
        image = line_util.line_to_image(
            line_parameters, height=128, width=128, falloff=None
        )
        target_image_name = "Offset_Vertical_Rectangle_0.png"
        baseline_image_path = os.path.join(self.test_data_directory, target_image_name)
        image = image_util.get_pil_formatted_image(image)
        image_util.expect_image_file_and_image_are_near(
            self,
            baseline_image_path,
            image,
            target_image_name,
            "%s does not match." % target_image_name,
        )

    def test_render_offset_vertical_rectangle_rectangular_image(self):
        line_parameters = jnp.array([0.0, 80.0, 49.0, 16.0, 48.0], dtype=jnp.float32)
        image = line_util.line_to_image(
            line_parameters, height=130, width=120, falloff=None
        )
        target_image_name = "Offset_Vertical_Rectangle_1.png"
        baseline_image_path = os.path.join(self.test_data_directory, target_image_name)
        image = image_util.get_pil_formatted_image(image)
        image_util.expect_image_file_and_image_are_near(
            self,
            baseline_image_path,
            image,
            target_image_name,
            "%s does not match." % target_image_name,
        )

    def test_render_rotated_rectangle(self):
        line_parameters = jnp.array(
            [3.14159 / 4.0, 64.0, 64.0, 16.0, 48.0], dtype=jnp.float32
        )
        image = line_util.line_to_image(
            line_parameters, height=128, width=128, falloff=None
        )
        target_image_name = "Rotated_Rectangle_0.png"
        baseline_image_path = os.path.join(self.test_data_directory, target_image_name)
        image = image_util.get_pil_formatted_image(image)
        image_util.expect_image_file_and_image_are_near(
            self,
            baseline_image_path,
            image,
            target_image_name,
            "%s does not match." % target_image_name,
        )

    def test_render_centered_square_with_blur(self):
        line_parameters = jnp.array([0.0, 64.0, 64.0, 16.0, 16.0], dtype=jnp.float32)
        image = line_util.line_to_image(
            line_parameters, height=128, width=128, falloff=10.0
        )
        target_image_name = "Centered_Square_Blur_0.png"
        baseline_image_path = os.path.join(self.test_data_directory, target_image_name)
        image = image_util.get_pil_formatted_image(image)
        image_util.expect_image_file_and_image_are_near(
            self,
            baseline_image_path,
            image,
            target_image_name,
            "%s does not match." % target_image_name,
        )

    def test_render_rotated_rectangle_with_blur(self):
        line_parameters = jnp.array(
            [3.14159 / 4.0, 64.0, 64.0, 16.0, 48.0], dtype=jnp.float32
        )
        image = line_util.line_to_image(
            line_parameters, height=128, width=128, falloff=10.0
        )
        target_image_name = "Rotated_Rectangle_Blur_0.png"
        baseline_image_path = os.path.join(self.test_data_directory, target_image_name)
        image = image_util.get_pil_formatted_image(image)
        image_util.expect_image_file_and_image_are_near(
            self,
            baseline_image_path,
            image,
            target_image_name,
            "%s does not match." % target_image_name,
        )


if __name__ == "__main__":
    absltest.main()
