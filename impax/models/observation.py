import jax.numpy as jnp


class Observation(object):
    """An observation that is seen by a network."""

    def __init__(self, model_config, training_example):
        # Auxiliaries:
        sampling_scheme = model_config.sampling_scheme
        if "p" in sampling_scheme:
            # Then we have access to a point cloud as well:
            self._surface_points = jnp.array(
                training_example.all_surface_points.numpy()
            )
            sampling_scheme = sampling_scheme.replace("p", "")
        if "q" in sampling_scheme:
            self._surface_points = jnp.array(
                training_example.all_surface_points_from_depth.numpy()
            )
            sampling_scheme = sampling_scheme.replace("q", "")
        if "n" in sampling_scheme:
            # Then we have access to normals:
            if "q" in sampling_scheme:
                raise ValueError("Can't combine normals with xyz from depth.")
            self._normals = jnp.array(training_example.all_normals.numpy())
            sampling_scheme = sampling_scheme.replace("n", "")
        else:
            self._normals = None
        # Main input:
        idx = model_config.index_of_dodecahedron
        if sampling_scheme == "imd":
            tensor = training_example.depth_renders
            num_images = 20
            channel_count = 1
        elif sampling_scheme == "imxyz":
            tensor = training_example.xyz_renders
            num_images = 20
            channel_count = 3
        elif sampling_scheme == "rgb":
            tensor = training_example.chosen_renders
            num_images = 1
            channel_count = 3
        elif sampling_scheme == "im1d":
            if "ShapeNetNSSDodecaSparseLRGMediumSlimPC" in training_example.proto_name:
                tensor = training_example.depth_renders[:, idx : idx + 1, ...]
            elif (
                training_example.proto_name == "ShapeNetOneImXyzPC"
                or "Waymo" in training_example.proto_name
            ):
                tensor = training_example.depth_render
            num_images = 1
            channel_count = 1
        elif sampling_scheme == "im1l":
            tensor = training_example.lum_renders[:, idx : idx + 1, ...]
            num_images = 1
            channel_count = 1
        elif sampling_scheme == "im1xyz":
            if "ShapeNetNSSDodecaSparseLRGMediumSlimPC" in training_example.proto_name:
                tensor = training_example.xyz_renders[:, idx : idx + 1, ...]
            elif (
                training_example.proto_name == "ShapeNetOneImXyzPC"
                or "Waymo" in training_example.proto_name
            ):
                tensor = training_example.xyz_render
            num_images = 1
            channel_count = 3
        elif sampling_scheme == "imrd":
            tensor = training_example.random_depth_images
            num_images = model_config.rc
            channel_count = 1
        elif sampling_scheme == "imrxyz":
            tensor = training_example.random_xyz_render
            num_images = 1
            channel_count = 3
        elif sampling_scheme == "imrlum":
            tensor = training_example.random_lum_render
            num_images = 1
            channel_count = 1
        elif sampling_scheme == "imlum":
            num_images = 20
            channel_count = 1
            tensor = training_example.lum_renders
        else:
            raise ValueError(
                "Unrecognized samp: %s -> %s" % (model_config.samp, sampling_scheme)
            )
        tensor = jnp.reshape(
            tensor.numpy(),
            [
                model_config.batch_size,
                num_images,
                model_config.input_height,
                model_config.input_width,
                channel_count,
            ],
        )
        self._tensor = tensor
        self._one_image_one_channel_tensor = None
        self._model_config = model_config
        self._samp = sampling_scheme

    @property
    def tensor(self):
        return self._tensor

    @property
    def surface_points(self):
        return self._surface_points

    @property
    def normals(self):
        return self._normals

    @property
    def one_image_one_channel_tensor(self):
        """A summary that describes the observation with a single-channel."""
        # For now just return a fixed image and the first channel.
        # TODO(kgenova) Check if RGB, convert to luminance, etc., maybe tile and
        # resize to fit a grid as a single image.
        if self._one_image_one_channel_tensor is None:
            num_images = self.tensor.shape[1]
            if num_images == 1:
                self._one_image_one_channel_tensor = self.tensor[:, 0, :, :, :1]
            else:
                self._one_image_one_channel_tensor = self.tensor[:, 1, :, :, :1]
        return self._one_image_one_channel_tensor
