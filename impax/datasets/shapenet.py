import collections

import jax
import jax.numpy as jnp
import tensorflow as tf

# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from impax.utils import geom_util, image_util

# pylint: enable=g-bad-import-order
"""data_set = Shapenet.load(
    split="train",
    download_and_prepare_kwargs={"download_config": tfds.download.DownloadConfig(manual_dir="~/shapenet_base")},
)"""


"""for example in data_set.take(1):
    trimesh, label, model_id = example["trimesh"], example["label"], example["model_id"]
    print(label)"""

"""Wrapper for interfacing with ShapeNet specifics.

If code depends on the internals of ShapeNet, as compared to some other dataset,
it should be here.
"""


def ensure_shape_and_resize_if_needed(
    orig_renders, batch_size, frame_count, channel_count, orig_height, orig_width, target_height, target_width
):
    """Checks that the input tensor has an expected size, resizing if needed."""

    if target_height != orig_height or target_width != orig_width:
        renders = jnp.reshape(orig_renders.numpy(), [batch_size * frame_count, orig_height, orig_width, channel_count])
        renders = jax.image.resize(
            renders,
            (batch_size * frame_count, target_height, target_width, channel_count),
            method="nearest",
            antialias=True,
        )
        renders = jnp.reshape(renders, [batch_size, frame_count, target_height, target_width, channel_count])
    else:
        renders = orig_renders
    renders = jnp.reshape(renders, [batch_size, frame_count, target_height, target_width, channel_count])
    return renders


def select_image(images, indices):
    """Gathers an image from each sequence in a batch of image sequences.

    Args:
      images: Tensor with shape [batch_size, sequence_length, height, width,
        channel_count].
      indices: Tensor with shape [batch_size, image_count].
    Returns:
      Tensor with shape [batch_size, image_count, height, width, channel_count]
    """
    batch_size, _, height, width, channel_count = images.shape
    image_count = indices.shape[1]
    selected = jax.vmap(lambda x, y: jnp.take(x, y, axis=0), in_axes=(0, 0))(images, indices)
    assert selected.shape == (batch_size, image_count, height, width, channel_count)

    return selected


def apply_noise_to_depth(depth, stddev, key=jax.random.PRNGKey(0)):
    """Applies independent random gaussian noise to each valid depth pixel."""
    if stddev == 0.0:
        return depth
    assert stddev > 0.0
    noise = jax.random.normal(key, shape=depth.shape) * stddev
    noise = jnp.where(noise >= 0.0, noise + 1.0, 1.0 / (1.0 + jnp.abs(noise)))
    noise = jnp.where(depth > 0.0, noise, 1.0)
    return depth * noise


def add_noise_to_xyz(xyz, stddev, key=jax.random.PRNGKey(0)):
    if stddev == 0.0:
        return xyz
    noise_shape = xyz.shape
    noise = jax.random.normal(key, shape=noise_shape) * stddev

    noisy_pts = jnp.where(jnp.sum(xyz, axis=-1) == 0, xyz, xyz + noise)
    return noisy_pts


def add_noise_in_normal_direction(xyz, normals, stddev, key=jax.random.PRNGKey(0)):
    """Adds noise along normal direction."""
    if stddev == 0.0:
        return xyz
    noise_shape = xyz.shape[:-1] + [1]
    noise = jax.random.normal(key, shape=noise_shape) * stddev
    # normal_dir = pts[..., 3:]
    # noisy_pts = tf.concat([xyz[..., :3] + noise * normals, normals], axis=-1)
    return xyz + noise * normals
    # return noisy_pts


ShapeNetSparseProvider = collections.namedtuple(
    "ShapeNetSparseProvider",
    [
        "mesh_renders",
        "surface_point_samples",
        "mesh_name",
        "depth_renders",
        "bounding_box_samples",
        "near_surface_samples",
        "lum_renders",
        "depth_render",
        "xyz_render",
        "world2grid",
        "grid",
    ],
)


def build_placeholder_interface(model_config, proto="ShapeNetNSSDodecaSparseLRGMediumSlimPC"):
    """Return a placeholder clone of the interface the tf.Dataset() interface."""
    # TODO(kgenova) Add hyperparameters for RGB resolution. The 3D-R2N2 renders
    # are currently used for eval, and their resolution is baked in.
    mesh_renders = tf.placeholder(tf.float32, [model_config.batch_size, 6, 137, 137, 4])
    depth_renders = tf.placeholder(
        tf.float32, [model_config.batch_size, 20, model_config.gap_height, model_config.gap_width, 1]
    )
    depth_render = tf.placeholder(
        tf.float32, [model_config.batch_size, model_config.gap_height, model_config.gap_width, 1]
    )
    xyz_render = tf.placeholder(
        tf.float32, [model_config.batch_size, model_config.gap_height, model_config.gap_width, 3]
    )
    lum_renders = tf.placeholder(
        tf.float32, [model_config.batch_size, 20, model_config.gap_height, model_config.gap_width, 1]
    )
    # TODO(kgenova) The surface points have variable shape...
    surface_point_samples = tf.placeholder(tf.float32)
    bounding_box_samples = tf.placeholder(tf.float32, [model_config.batch_size, 100000, 4])
    near_surface_samples = tf.placeholder(tf.float32, [model_config.batch_size, 100000, 4])
    mesh_name = tf.placeholder(tf.string)
    world2grid = tf.placeholder(tf.float32, [model_config.batch_size, 4, 4])
    grid = tf.placeholder(tf.float32)  # [model_config.batch_size, 32, 32, 32])
    # TODO(kgenova) Maybe this needs to be some sort of 'DefaultProto' that
    # assumes everything.
    return {
        "dataset": ShapeNetSparseProvider(
            mesh_renders=mesh_renders,
            surface_point_samples=surface_point_samples,
            mesh_name=mesh_name,
            depth_renders=depth_renders,
            lum_renders=lum_renders,
            depth_render=depth_render,
            xyz_render=xyz_render,
            bounding_box_samples=bounding_box_samples,
            near_surface_samples=near_surface_samples,
            grid=grid,
            world2grid=world2grid,
        ),
        "split": "none",
        "proto": proto,
    }


class BoundingBox(object):
    """An axis-aligned bounding box."""

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    @classmethod
    def from_samples(cls, samples):
        """Computes the axis aligned bounding box enclosing a set of points.

        Args:
          samples: Tensor with shape [batch_size, sample_count, 3].

        Returns:
          Two Tensors with shape [batch_size, 1, 3]. The first 3-vector contains the
          minimum XYZ, and the second 3-vector contains the maximum XYZ.
        """
        lower_bound = tf.reduce_min(samples, axis=1, keepdims=True)
        upper_bound = tf.reduce_max(samples, axis=1, keepdims=True)
        return cls(lower_bound, upper_bound)


class ShapeNetExample(object):
    """A single ShapeNet shape as a training example."""

    def __init__(self, model_config, dataset, split):
        """Preprocesses a batch of shapenet training examples."""
        self._depth_renders = None
        self._lum_renders = None
        self._chosen_renders = None
        self._renders = None
        self._bounding_box = None
        self._random_depth_indices = None
        self._random_depth_images = None
        self._depth_to_world = None
        self._random_depth_to_worlds = None
        self._random_lum_render = None
        self._all_surface_points_from_depth = None
        self._dataset = dataset
        # if hasattr(ds, 'surface_point_samples'):
        # if (model_config.inputs['proto'] in
        #     [
        #         'ShapeNetNSSDodecaSparseLRGMediumSlimPC',
        #      'ShapeNetNSSDodecaSparseLRGMediumSlimPCExtra']):
        points_length = 10000
        # else:
        #   points_length = 100000
        self._full_zero_set_points_and_normals = tf.reshape(
            dataset.surface_point_samples, [model_config.batch_size, points_length, 6]
        )
        # else:
        #   self._full_zero_set_points_and_normals = None
        self.mesh_name = dataset.mesh_name

        self.split = split
        self._model_config = model_config
        self._full_point_count = 100000
        self._world2grid = dataset.world2grid
        if hasattr(dataset, "grid"):
            self._grid = dataset.grid
        else:
            self._grid = None

    @property
    def full_near_surface_samples(self):
        if not hasattr(self, "_full_near_surface_samples"):
            self._full_near_surface_samples = tf.ensure_shape(
                self._dataset.near_surface_samples, [self._model_config.batch_size, 100000, 4]
            )
        return self._full_near_surface_samples

    @property
    def full_uniform_samples(self):
        if not hasattr(self, "_full_uniform_samples"):
            self._full_uniform_samples = tf.ensure_shape(
                self._dataset.bounding_box_samples, [self._model_config.batch_size, 100000, 4]
            )
        return self._full_uniform_samples

    def _subsample(self, samples, sample_count):
        """Returns a uniform random subsampling of an input sample set."""
        tf.logging.info("Sample shape: %s", str(samples.get_shape().as_list()))
        max_sample = samples.get_shape().as_list()[1]
        # assert max_sample == 100000
        sample_indices = tf.random.uniform(
            [self._model_config.batch_size, sample_count], minval=0, maxval=max_sample - 1, dtype=tf.int32
        )
        subsamples = tf.batch_gather(samples, sample_indices)
        return self._finite_wrapper(subsamples)

    @property
    def proto_name(self):
        return self._model_config.proto

    @property
    def depth_render(self):
        """An input depth map. Only available if single-view depth is supported."""
        if not hasattr(self, "_depth_render"):
            depth_render = ensure_shape_and_resize_if_needed(
                tf.expand_dims(self._dataset.depth_render, axis=1),
                self._model_config.batch_size,
                1,
                1,
                self._model_config.gap_height,
                self._model_config.gap_width,
                self._model_config.input_height,
                self._model_config.input_width,
            )
            depth_render = tf.cast(depth_render, dtype=tf.float32)
            depth_render = apply_noise_to_depth(depth_render, self._model_config.depth_map_noise)
            self._depth_render = depth_render / 1000.0
        return self._depth_render

    def apply_transformation(self, tx):
        """Applies a transformation to a shape."""
        shape = tx.get_shape().as_list()
        if len(shape) != 3 or shape[0] != self._model_config.batch_size or shape[1] != 4 or shape[2] != 4:
            raise ValueError(f"Unexpected shape for example transformation: {shape}")
        # TODO(kgenova) We assert no access has happened because it is safest.
        # This way it is guaranteed the untransformed points are never accessed.
        assert not hasattr(self, "_tx")

        self._tx = tx

        # There are currently 10K surface points. For now, let's just transform
        # the whole point cloud to the local frame.
        all_surface_points = self.all_surface_points
        all_surface_normals = self.all_normals
        all_surface_points, all_surface_normals = geom_util.transform_points_with_normals(
            all_surface_points, tx=tx, normals=all_surface_normals
        )
        self._all_surface_points = all_surface_points
        self._all_surface_normals = all_surface_normals

        # TODO(kgenova) If we need the SDF itself that could get problematic because
        # it transforms as well if there's a scale in tx. Shouldn't matter though.
        if self._model_config.train or self._model_config.eval:
            self._full_near_surface_samples = geom_util.transform_featured_points(self.full_near_surface_samples, tx)
            self._full_uniform_samples = geom_util.transform_featured_points(self.full_uniform_samples, tx)
            world2local = tx
            local2world = tf.linalg.inv(world2local)
            local2grid = tf.matmul(self.world2grid, local2world)
            self._world2grid = local2grid

    def crop_input(self, crop_count=1024):
        self._all_surface_points, self._all_surface_normals = geom_util.extract_points_near_origin(
            self._all_surface_points, crop_count, features=self._all_surface_normals
        )

    def crop_supervision(self, crop_count=1024):
        self._full_near_surface_samples = geom_util.extract_points_near_origin(
            self._full_near_surface_samples, 2 * crop_count
        )
        self._full_uniform_samples = geom_util.extract_points_near_origin(self._full_uniform_samples, crop_count)

    @property
    def xyz_render(self):
        """An xyz map. Only available if SVD and extrinsics are both present."""
        if not hasattr(self, "_xyz_render"):
            xyz_render = ensure_shape_and_resize_if_needed(
                tf.expand_dims(self._dataset.xyz_render, axis=1),
                self._model_config.batch_size,
                1,
                3,
                self._model_config.gap_height,
                self._model_config.gap_width,
                self._model_config.input_height,
                self._model_config.input_width,
            )
            xyz_render = tf.cast(xyz_render, dtype=tf.float32)
            self._xyz_render = add_noise_to_xyz(xyz_render, self._model_config.xyz_noise)
        return self._xyz_render

    @property
    def depth_renders(self):
        """A stack of depth renders for the multiview case."""
        if self._depth_renders is None:
            depth_renders = ensure_shape_and_resize_if_needed(
                self._dataset.depth_renders,
                self._model_config.batch_size,
                20,
                1,
                self._model_config.gap_height,
                self._model_config.gap_width,
                self._model_config.input_height,
                self._model_config.input_width,
            )
            depth_renders = tf.cast(depth_renders, dtype=tf.float32)
            depth_renders = apply_noise_to_depth(depth_renders, self._model_config.depth_map_noise)
            self._depth_renders = depth_renders / 1000.0  # Was in 1000-ths.
        return self._finite_wrapper(self._depth_renders)

    @property
    def random_depth_indices(self):
        if self._random_depth_indices is None:
            self._random_depth_indices = tf.random_uniform(
                shape=[self._model_config.batch_size, self._model_config.num_random_images],
                minval=0,
                maxval=19,
                dtype=tf.int32,
            )
        return self._finite_wrapper(self._random_depth_indices)
        # return tf.constant([[2, 5, 3, 4, 9, 14, 17, 20, 1, 11]], dtype=tf.int32)

    @property
    def random_depth_images(self):
        if self._random_depth_images is None:
            self._random_depth_images = select_image(self.depth_renders, self.random_depth_indices)
        return self._finite_wrapper(self._random_depth_images)

    @property
    def random_depth_to_worlds(self):
        """A shuffled subset of the depth images per batch element."""
        if self._random_depth_to_worlds is None:
            self._random_depth_to_worlds = tf.gather(
                self.depth_to_world,
                tf.reshape(
                    self.random_depth_indices, [self._model_config.batch_size * self._model_config.num_random_images]
                ),
            )
            self._random_depth_to_worlds = tf.reshape(
                self._random_depth_to_worlds,
                [self._model_config.batch_size, self._model_config.num_random_images, 4, 4],
            )
        return self._finite_wrapper(self._random_depth_to_worlds)

    @property
    def xyz_renders(self):
        if not hasattr(self, "_xyz_renders"):
            self._xyz_renders = add_noise_to_xyz(
                geom_util.transform_depth_dodeca_to_xyz_dodeca(self.depth_renders), self._model_config.xyz_noise
            )
        return self._xyz_renders

    @property
    def depth_to_world(self):
        if self._depth_to_world is None:
            self._depth_to_world = tf.constant(geom_util.get_dodeca_camera_to_worlds(), dtype=tf.float32)
            self._depth_to_world = tf.ensure_shape(self._depth_to_world, [20, 4, 4])
        return self._finite_wrapper(self._depth_to_world)

    @property
    def renders(self):
        """Rendered images of the mesh."""
        if not hasattr(self._dataset, "mesh_renders"):
            raise ValueError("Trying to access RGB images that aren't in the proto.")
        if self._renders is None:
            # TODO(kgenova) Add a hyperparameter for RGB resolution.
            self._renders = ensure_shape_and_resize_if_needed(
                self._dataset.mesh_renders,
                self._model_config.batch_size,
                24,
                4,
                137,
                137,
                self._model_config.input_height,
                self._model_config.input_width,
            )
        return self._finite_wrapper(self._renders)

    @property
    def lum_renders(self):
        """Single-channel renders of the mesh."""
        if not hasattr(self._dataset, "lum_renders"):
            raise ValueError("Trying to access lum images that aren't in the proto.")
        if self._lum_renders is None:
            self._lum_renders = tf.cast(
                ensure_shape_and_resize_if_needed(
                    self._dataset.lum_renders,
                    self._model_config.batch_size,
                    20,
                    1,
                    self._model_config.gap_height,
                    self._model_config.gap_width,
                    self._model_config.input_height,
                    self._model_config.input_width,
                ),
                dtype=tf.float32,
            )
        return self._finite_wrapper(self._lum_renders)

    @property
    def chosen_renders(self):
        """Subsampled renders seen this batch."""
        if self._chosen_renders is None:
            chosen = self._subsample(self.renders, sample_count=1)
            chosen_rgba = tf.reshape(
                chosen,
                [self._model_config.batch_size, 1, self._model_config.input_height, self._model_config.input_width, 4],
            )
            chosen_rgb = image_util.rgba_to_rgb(self._model_config, chosen_rgba)
            self._chosen_renders = chosen_rgb
        return self._finite_wrapper(self._chosen_renders)

    @property
    def random_lum_render(self):
        """A random luminance image."""
        if self._random_lum_render is None:
            self._random_lum_render = self._subsample(self.lum_renders, sample_count=1)
            self._random_lum_render = tf.reshape(
                self._random_lum_render,
                [self._model_config.batch_size, 1, self._model_config.input_height, self._model_config.input_width, 1],
            )
        return self._random_lum_render

    @property
    def random_xyz_render(self):
        """A random XYZ image."""
        if not hasattr(self, "_random_xyz_render"):
            self._random_xyz_render = self._subsample(self.xyz_renders, sample_count=1)
            self._random_xyz_render = tf.reshape(
                self._random_xyz_render,
                [self._model_config.batch_size, 1, self._model_config.input_height, self._model_config.input_width, 3],
            )
            return self._random_xyz_render

    @property
    def grid(self):
        return self._finite_wrapper(self._grid)

    @property
    def world2grid(self):
        return self._finite_wrapper(self._world2grid)

    def _finite_wrapper(self, t):
        if self._model_config.debug_mode == "t":
            t = tf.debugging.check_numerics(t, message="inputs.py")
        return t

    @property
    def sample_bounding_box(self):
        """The bounding box of the uniform samples."""
        if self._bounding_box is None:
            bbox_samples = self.full_uniform_samples[..., :3]
            # Use only the inside samples:
            is_inside = self.full_uniform_samples[..., 3:4] < 0.0
            bbox_samples = tf.where_v2(is_inside, bbox_samples, 0.0)
            self._bounding_box = BoundingBox.from_samples(bbox_samples)
            # self._full_zero_set_points_and_normals[..., :3])
        return self._finite_wrapper(self._bounding_box)

    def sample_sdf_near_surface(self, sample_count):
        subsamples = self._subsample(self.full_near_surface_samples, sample_count)
        return self._finite_wrapper(tf.split(subsamples, [3, 1], axis=2))

    def sample_sdf_uniform(self, sample_count):
        subsamples = self._subsample(self.full_uniform_samples, sample_count)
        return self._finite_wrapper(tf.split(subsamples, [3, 1], axis=2))

    def all_uniform_samples(self):
        return self._finite_wrapper(tf.split(self.full_uniform_samples, [3, 1], axis=2))

    def _set_points_and_normals(self):
        points, normals = tf.split(self._full_zero_set_points_and_normals, [3, 3], axis=2)
        points = add_noise_in_normal_direction(points, normals, self._model_config.point_cloud_noise)
        self._all_surface_points = points
        self._all_surface_normals = normals

    @property
    def all_surface_points(self):
        if not hasattr(self, "_all_surface_points"):
            self._set_points_and_normals()
        return self._all_surface_points

    @property
    def all_normals(self):
        if not hasattr(self, "_all_surface_normals"):
            self._set_points_and_normals()
        return self._all_surface_normals

    def sample_zero_set_points_and_normals(self, sample_count):
        if self._full_zero_set_points_and_normals is None:
            raise ValueError("Trying to sample from surface points but they are not in the proto.")
        subsamples = self._subsample(self._full_zero_set_points_and_normals, sample_count)
        points, normals = tf.split(subsamples, [3, 3], axis=2)
        return self._finite_wrapper(points), self._finite_wrapper(normals)
