"""A class for an instance of a structured implicit function."""

import time

import jax
import jax.numpy as jnp
from jax import nn

from impax.representations import quadrics
from impax.utils import camera_util, geom_util, jax_util, jnp_util, sdf_util
from impax.utils.file_util import log

RADII_EPS = 1e-10


def _unflatten(vector, model_config):
    """Given a flat shape vector, separates out the individual parameter sets."""
    radius_shape = element_radius_shape(model_config)
    radius_len = radius_shape
    # assert len(radius_shape) == 1  # If the radius is a >1 rank tensor, unpack it.
    explicit_param_count = 1  # A single constant.
    # First, determine if there are implicit parameters present.
    total_explicit_length = explicit_param_count + 3 + radius_len
    provided_length = vector.shape[-1]
    # log.info('Provided len, explicit len: %i, %i' %
    # (provided_length, total_explicit_length))
    if provided_length > total_explicit_length:
        expected_implicit_length = model_config.implicit_parameter_length
        leftover_length = provided_length - total_explicit_length
        if leftover_length != expected_implicit_length:
            raise ValueError(
                ("Unable to interpret ijnput vector as either explicit or" " implicit+explicit vector. Shape: %s")
                % repr(vector.shape)
            )
        constant, center, radius, iparams = jnp.split(
            vector,
            [
                explicit_param_count,
                explicit_param_count + 3,
                explicit_param_count + 3 + radius_len,
            ],
            axis=-1,
        )
    elif provided_length == total_explicit_length:
        constant, center, radius = jnp.split(vector, [explicit_param_count, explicit_param_count + 3], axis=-1)
        iparams = None
    else:
        raise ValueError("Too few ijnput parameters for even explicit vector: %s" % repr(vector.shape))

    return constant, center, radius, iparams


def element_explicit_dof(model_config):
    """Determines the number of (analytic) degrees of freedom in a single blob."""
    # Constant:
    constant_dof = 1

    # Center:
    center_dof = 3  # in R^3

    # Radius:
    if model_config.coordinates == "iso":
        radius_dof = 1
    elif model_config.coordinates == "aa":
        radius_dof = 3
    elif model_config.coordinates == "cov":
        radius_dof = 6
    else:
        raise ValueError("Unrecognized radius hyperparameter: %s" % model_config.coordinates)
    return constant_dof + center_dof + radius_dof


def element_implicit_dof(model_config):
    if model_config.enable_implicit_parameters:
        return model_config.implicit_parameter_length
    return 0.0


def element_dof(model_config):
    """Returns the DoF of a single shape element."""
    return element_explicit_dof(model_config) + element_implicit_dof(model_config)


def element_constant_shape(model_config):
    # We will need the model config in the future if we enable other options:
    # pylint: disable=unused-argument
    return [1]


def element_center_shape(model_config):
    # We will need the model config in the future if we enable other options:
    # pylint: disable=unused-argument
    return [3]


def element_radius_shape(model_config):
    if model_config.coordinates == "iso":
        return 1
    elif model_config.coordinates == "aa":
        return 3
    elif model_config.coordinates == "cov":
        return 6
    else:
        raise ValueError("Unrecognized radius hyperparameter: %s" % model_config.coordinates)


def element_iparam_shape(model_config):
    if model_config.implicit_parameters == "no":
        return []
    elif model_config.implicit_parameters == "bp":
        return [model_config.implicit_parameter_length]
    else:
        raise ValueError("Unrecognized implicit parameter hyperparameter: %s" % model_config.implicit_parameters)


def ensure_net_if_needed(net, model_config):
    if model_config.implicit_parameters != "no" and net is None:
        raise ValueError(
            "Must provide network for sample decoding when using "
            "iparams (hparam ip=%s)." % model_config.implicit_parameters
        )


def sigma(x, a):
    return x / jnp.sqrt(a * a + x * x)


def ensure_jnp_shape(x, expected_shape, name):
    """Verifies a numpy array matches the expected shape. Supports symbols."""
    shape_ok = len(x.shape) == len(expected_shape)
    if shape_ok:
        symbols = {}
        for i in range(len(x.shape)):
            if isinstance(expected_shape[i], int) and expected_shape[i] != -1:
                shape_ok = shape_ok and x.shape[i] == expected_shape[i]
            elif expected_shape[i] == -1:
                continue
            else:
                if expected_shape[i] in symbols:
                    shape_ok = shape_ok and x.shape[i] == symbols[expected_shape[i]]
                else:
                    symbols[expected_shape[i]] = x.shape[i]
    else:
        raise ValueError(
            "Expected numpy array %s to have shape %s but it has shape %s." % (name, str(expected_shape), str(x.shape))
        )


class StructuredImplicitjnp(object):
    """A (batch of) numpy structured implicit functions(s)."""

    def __init__(self, structured_implicit):
        """Builds out the tensorflow graph portions that are needed."""
        self._initialized = False
        self._structured_implicit_tf = structured_implicit
        self._structured_implicit_ph = structured_implicit.as_placeholder()
        self._build_sample_on_grid()

    def initialize(self, session, feed_dict):
        """Initializes the data by evaluating the tensors in the session."""
        jnp_list = session.run(self._structured_implicit_tf.tensor_list, feed_dict=feed_dict)
        (self.element_constants, self.element_centers, self.element_radii) = jnp_list[:3]
        # pylint:disable=protected-access
        if not self._structured_implicit_tf._model_config.enable_implicit_parameters:
            # pylint:enable=protected-access
            # if len(jnp_list) == 4:
            self.element_iparams = jnp_list[3]
        else:
            if len(jnp_list) == 4:
                log.info(
                    "Warning: implicit parameters present but not enabled."
                    " Eliding them to avoid using untrained values."
                )
            self.element_iparams = None
        self._session = session
        self._initialized = True

    def ensure_initialized(self):
        if not self._initialized:
            raise AssertionError("A StructuredImplicitjnp must be initialized before use.")

    def flat_vector(self):
        self.ensure_initialized()
        ls = [self.element_constants, self.element_centers, self.element_radii]
        # if self._structured_implicit_tf._model_config.enable_implicit_parameters == 't':
        if self.element_iparams is not None:
            ls.append(self.element_iparams)
        return jnp.concatenate(ls, axis=-1)

    def _feed_dict(self):
        base_feed_dict = {
            self._structured_implicit_ph.element_constants: self.element_constants,
            self._structured_implicit_ph.element_centers: self.element_centers,
            self._structured_implicit_ph.element_radii: self.element_radii,
        }
        if self.element_iparams is not None:
            base_feed_dict[self._structured_implicit_ph.element_iparams] = self.element_iparams
        return base_feed_dict

    def sample_on_grid(self, sample_center, sample_size, resolution):
        """Evaluates the function on a grid."""
        self.ensure_initialized()
        # ensure_jnp_shape(sample_grid, ['res', 'res', 'res'], 'sample_grid')
        # mcubes_res = sample_grid.shape[0]
        if resolution % self._block_res:
            raise ValueError(
                "Ijnput resolution is %i, but must be a multiple of block size, %i." % (resolution, self._block_res)
            )
        block_count = resolution // self._block_res
        block_size = sample_size / block_count

        base_grid = jnp_util.make_coordinate_grid_3d(
            length=self._block_res,
            height=self._block_res,
            width=self._block_res,
            is_screen_space=False,
            is_homogeneous=False,
        ).astype(jnp.float32)
        lower_corner = sample_center - sample_size / 2.0
        start_time = time.time()
        l_block = []
        i = 0
        for li in range(block_count):
            l_min = lower_corner[2] + li * block_size
            h_block = []
            for hi in range(block_count):
                h_min = lower_corner[1] + hi * block_size
                w_block = []
                for wi in range(block_count):
                    w_min = lower_corner[0] + wi * block_size
                    offset = jnp.reshape(
                        jnp.array([w_min, l_min, h_min], dtype=jnp.float32),
                        [1, 1, 1, 3],
                    )
                    sample_locations = block_size * base_grid + offset
                    feed_dict = self._feed_dict()
                    feed_dict[self._sample_locations_ph] = sample_locations
                    grid_out_jnp = self._session.run(self._predicted_class_grid, feed_dict=feed_dict)
                    i += 1
                    w_block.append(grid_out_jnp)
                h_block.append(jnp.concatenate(w_block, axis=2))
            l_block.append(jnp.concatenate(h_block, axis=0))
        grid_out = jnp.concatenate(l_block, axis=1)
        compute_time = time.time() - start_time
        log.info("Post Initialization Time: %s" % compute_time)
        return grid_out, compute_time

    def _build_sample_on_grid(self):
        """Builds the graph nodes associated with sample_on_grid()."""
        block_res = 32
        self._block_res = block_res

        self._sample_locations_ph = jnp.zeros(dtype=jnp.float32, shape=[block_res, block_res, block_res, 3])
        samples = jnp.reshape(self._sample_locations_ph, [1, block_res**3, 3])
        predicted_class, _ = self._structured_implicit_ph.class_at_samples(samples)
        self._predicted_class_grid = jnp.reshape(predicted_class, [block_res, block_res, block_res])

    def extract_mesh(self):
        """Computes a mesh from the representation."""
        self.ensure_initialized()

    def render(self):
        """Computes an image of the representation."""
        self.ensure_initialized()

    def element_constants_as_quadrics(self):
        constants = self.element_constants
        quadric_shape = list(constants.shape[:-1]) + [4, 4]
        qs = jnp.zeros(quadric_shape, dtype=jnp.float32)
        qs[..., 3, 3] = jnp.squeeze(constants)
        return qs

    def write_to_disk(self, fnames):
        """Writes the representation to disk in the GAPS format."""
        self.ensure_initialized()
        eval_set_size = self.element_centers.shape[0]
        qs = self.element_constants_as_quadrics()
        for ei in range(eval_set_size):
            flat_quadrics = jnp.reshape(qs[ei, ...], [-1, 4 * 4])
            flat_centers = jnp.reshape(self.element_centers[ei, ...], [-1, 3])
            if self.element_radii.shape[-1] == 3:
                flat_radii = jnp.reshape(jnp.sqrt(self.element_radii[ei, ...]), [-1, 3])
            elif self.element_radii.shape[-1] == 6:
                flat_radii = jnp.reshape(self.element_radii[ei, ...], [-1, 6])
                flat_radii[..., :3] = jnp.sqrt(flat_radii[..., :3])
            if self.element_iparams is None:
                flat_params = jnp.concatenate([flat_quadrics, flat_centers, flat_radii], axis=1)
            else:
                flat_iparams = jnp.reshape(self.element_iparams[ei, ...], [-1, 32])
                flat_params = jnp.concatenate([flat_quadrics, flat_centers, flat_radii, flat_iparams], axis=1)
            jnp.savetxt(fnames[ei], flat_params)


def homogenize(m):
    """Adds homogeneous coordinates to a [..., N,N] matrix."""
    batch_rank = len(m.shape) - 2
    batch_dims = m.shape[:-2]
    n = m.shape[-1]
    assert m.shape[-2] == n
    right_col = jnp.zeros(batch_dims + (3, 1), dtype=jnp.float32)
    m = jnp.concatenate([m, right_col], axis=-1)
    lower_row = jnp.pad(
        jnp.zeros(batch_dims + (1, 3), dtype=jnp.float32),
        [(0, 0)] * batch_rank + [(0, 0), (0, 1)],
        mode="constant",
        constant_values=1.0,
    )
    return jnp.concatenate([m, lower_row], axis=-2)


def symgroup_equivalence_classes(model_config):
    """Generates the effective element indices for each symmetry group.
    Args:
      model_config: A ModelConfig object.
    Returns:
      A list of lists. Each sublist contains indices in the range [0,
        effective_element_count-1]. These indices map into tensors with dimension
        [..., effective_element_count, ...]. Each index appears in exactly one
        sublist, and the sublists are sorted in ascending order.
    """
    # Populate with the root elements.
    lists = [[i] for i in range(model_config.num_shape_elements)]

    left_right_sym_count = model_config.num_blobs
    if left_right_sym_count:
        start_idx = model_config.num_shape_elements
        for i in range(left_right_sym_count):
            idx = start_idx + i
            lists[i].extend(idx)
    return lists


def symgroup_class_ids(model_config):
    """Generates the equivalence class identifier for each effective element.
    Args:
      model_config: A ModelConfig object.
    Returns:
      A list of integers of length effective_element_count. Each element points to
      the 'true' (i.e. < shape count) identifier for each effective element
    """
    ls = list(range(model_config.num_shape_elements))

    left_right_sym_count = model_config.num_blobs
    if left_right_sym_count:
        ls.extend(list(range(left_right_sym_count)))
    return ls


def _tile_for_symgroups(elements, model_config):
    """Tiles an ijnput tensor along its element dimension based on symmetry.
    Args:
      model_config: A ModelConfig object.
      elements: Tensor with shape [batch_size, element_count, ...].
    Returns:
      Tensor with shape [batch_size, element_count + tile_count, ...]. The
      elements have been tiled according to the model configuration's symmetry
      group description.
    """
    left_right_sym_count = model_config.num_blobs
    assert len(elements.shape) >= 3
    # The first K elements get reflected with left-right symmetry (z-axis) as
    # needed.
    if left_right_sym_count:
        first_k = elements[:, :left_right_sym_count, ...]
        elements = jnp.concatenate([elements, first_k], axis=1)
    # TODO(kgenova) As additional symmetry groups are added, add their tiling.
    return elements


def get_effective_element_count(model_config):
    return model_config.num_shape_elements + model_config.num_blobs


def _generate_symgroup_samples(samples, model_config):
    """Duplicates and transforms samples as needed for symgroup queries.
    Args:
      model_config: A ModelConfig object.
      samples: Tensor with shape [batch_size, sample_count, 3].
    Returns:
      Tensor with shape [batch_size, effective_element_count, sample_count, 3].
    """
    assert len(samples.shape) == 3
    samples = jax_util.tile_new_axis(samples, axis=1, length=model_config.num_shape_elements)

    left_right_sym_count = model_config.num_blobs

    if left_right_sym_count:
        first_k = samples[:, :left_right_sym_count, :]
        first_k = geom_util.z_reflect(first_k)
        samples = jnp.concatenate([samples, first_k], axis=1)
    return samples


def constants_to_quadrics(constants):
    """Convert a set of constants to quadrics.
    Args:
      constants: Tensor with shape [..., 1].
    Returns:
      quadrics: Tensor with shape [..., 4,4]. All entries except the
      bottom-right corner are 0. That corner is the constant.
    """
    zero = jnp.zeros_like(constants)
    last_row = jnp.concatenate([zero, zero, zero, constants], axis=-1)
    zero_row = jnp.zeros_like(last_row)
    return jnp.stack([zero_row, zero_row, zero_row, last_row], axis=-2)


def _transform_samples(samples, tx):
    """Applies a 4x4 transformation to XYZ coordinates.
    Args:
      samples: Tensor with shape [..., sample_count, 3].
      tx: Tensor with shape [..., 4, 4].
    Returns:
      Tensor with shape [..., sample_count, 3]. The ijnput samples in a new
      coordinate frame.
    """
    # We assume the center is an XYZ position for this transformation:
    samples = geom_util.to_homogeneous(samples, is_point=True)
    n = len(tx.shape)
    samples = jnp.matmul(samples, jnp.swapaxes(tx, n - 1, n - 2))
    return samples[..., :3]


def get_sif_kwargs_from_activation(activation, net, model_config):
    """Parse a network activation into a structured implicit function."""
    ensure_net_if_needed(net, model_config)
    constant, center, radius, iparam = _unflatten(activation, model_config)

    if model_config.prediction_mode == "a":
        constant = -jnp.abs(constant)
    elif model_config.prediction_mode == "s":
        constant = nn.sigmoid(constant) - 1
    radius_var = nn.sigmoid(radius[..., 0:3])
    max_blob_radius = 0.15
    radius_var *= max_blob_radius
    radius_var = radius_var * radius_var
    if model_config.coordinates == "cov":
        # radius_rot = sigma(radius[..., 3:], 50.0)
        max_euler_angle = jnp.pi / 4.0
        radius_rot = jnp.clip(radius[..., 3:], -max_euler_angle, max_euler_angle)
        # radius_rot *= max_euler_angle
        radius = jnp.concatenate([radius_var, radius_rot], axis=-1)
    else:
        radius = radius_var
    center = center / 2.0
    return dict(constants=constant, centers=center, radius=radius, iparams=iparam)


def compute_world2local(_model_config, constants, centers, radius):
    """Computes a transformation to the local element frames for encoding."""
    # We assume the center is an XYZ position for this transformation:
    # TODO(kgenova) Update this transformation to account for rotation.
    ec, bs = constants.shape[1], constants.shape[0]

    if _model_config.transformations == "i":
        return jnp.repeat(
            jnp.repeat(jnp.eye(4)[None, None, ...], repeats=bs, axis=0),
            repeats=ec,
            axis=1,
        )

    if "c" in _model_config.transformations:
        tx = jnp.repeat(
            jnp.repeat(jnp.eye(3)[None, None, ...], repeats=bs, axis=0),
            repeats=ec,
            axis=1,
        )
        centers = jnp.reshape(centers, [bs, ec, 3, 1])
        tx = jnp.concatenate([tx, -centers], axis=-1)
        lower_row = jnp.tile(
            jnp.reshape(jnp.array([0.0, 0.0, 0.0, 1.0]), [1, 1, 1, 4]),
            [bs, ec, 1, 1],
        ).astype(jnp.float32)

        tx = jnp.concatenate([tx, lower_row], axis=-2)
    else:
        tx = jnp.repeat(
            jnp.repeat(jnp.eye(4)[None, None, ...], repeats=bs, axis=0),
            repeats=ec,
            axis=1,
        )

    # Compute a rotation transformation if necessary:
    if ("r" in _model_config.transformations) and (_model_config.coordinates == "cov"):
        # Apply the inverse rotation:
        rotation = jnp.linalg.inv(camera_util.roll_pitch_yaw_to_rotation_matrices(radius[..., 3:6]))
    else:
        rotation = jnp.repeat(
            jnp.repeat(jnp.eye(3)[None, None, ...], repeats=bs, axis=0),
            repeats=ec,
            axis=1,
        )

    # Compute a scale transformation if necessary:
    if ("s" in _model_config.transformations) and (_model_config.coordinates in ["aa", "cov"]):
        diag = radius[..., 0:3]
        diag = 1.0 / (jnp.sqrt(diag + 1e-8) + 1e-8)
        diag_shape = diag.shape
        diag = diag.reshape((-1, diag_shape[-1]))

        scale = jax.vmap(jnp.diag, in_axes=0)(diag).reshape(diag_shape + (diag_shape[-1],))
    else:
        scale = jnp.repeat(
            jnp.repeat(jnp.eye(3)[None, None, ...], repeats=bs, axis=0),
            repeats=ec,
            axis=1,
        )

    # Apply both transformations and return the transformed points.
    tx3x3 = jnp.matmul(scale, rotation)
    return jnp.matmul(homogenize(tx3x3), tx)


class StructuredImplicit(object):
    """A (batch of) structured implicit function(s)."""

    def __init__(self, constants, centers, radius, iparams, net, model_config):
        batching_dims = constants.shape[:-2]
        batching_rank = len(batching_dims)
        if batching_rank == 0:
            constants = jnp.expand_dims(constants, axis=0)
            radius = jnp.expand_dims(radius, axis=0)
            centers = jnp.expand_dims(centers, axis=0)
        self._constants = constants
        self._radii = radius
        self._centers = centers
        self._iparams = iparams
        self._model_config = model_config
        self._packed_vector = None
        self._flat_vector = None
        self._quadrics = None
        self._net = net
        self._world2local = None

    @classmethod
    def from_packed_vector(cls, packed_vector, net, model_config):
        """Parse an already packed vector (NOT a network activation)."""
        ensure_net_if_needed(net, model_config)
        constant, center, radius, iparam = _unflatten(packed_vector, model_config)
        return cls(constant, center, radius, iparam, net, model_config)

    @classmethod
    def from_activation(cls, activation, net, model_config):
        """Parse a network activation into a structured implicit function."""
        ensure_net_if_needed(net, model_config)
        constant, center, radius, iparam = _unflatten(activation, model_config)

        if model_config.prediction_mode == "a":
            constant = -jnp.abs(constant)
        elif model_config.prediction_mode == "s":
            constant = nn.sigmoid(constant) - 1
        radius_var = nn.sigmoid(radius[..., 0:3])
        max_blob_radius = 0.15
        radius_var *= max_blob_radius
        radius_var = radius_var * radius_var
        if model_config.coordinates == "cov":
            # radius_rot = sigma(radius[..., 3:], 50.0)
            max_euler_angle = jnp.pi / 4.0
            radius_rot = jnp.clip(radius[..., 3:], -max_euler_angle, max_euler_angle)
            # radius_rot *= max_euler_angle
            radius = jnp.concatenate([radius_var, radius_rot], axis=-1)
        else:
            radius = radius_var
        center = center / 2.0
        return cls(constant, center, radius, iparam, net, model_config)

    def force_valid_values(self):
        self._constants = jnp.minimum(self._constants, -1e-10)
        if self._model_config.coordinates == "cov":
            axisr, rotr = jnp.split(self._radii, [3], axis=-1)
            axisr = jnp.maximum(axisr, 1e-9)
            rotr = jnp.clip(rotr, -jnp.pi / 4.0, jnp.pi / 4.0)
            self._radii = jnp.concatenate([axisr, rotr], axis=-1)
        else:
            assert self._model_config.coordinates == "aa"
            self._radii = jnp.maximum(self._radii, 1e-9)

    @property
    def vector(self):
        """A vector with shape [batch_size, element_count, element_length]."""
        if self._packed_vector is None:
            to_pack = [self._constants, self._centers, self._radii]
            if self._iparams is not None:
                to_pack.append(self._iparams)
            self._packed_vector = jnp.concatenate(to_pack, axis=-1)
        return self._packed_vector

    @property
    def flat_vector(self):
        """A flattened vector with shape [batch_size, -1]."""
        if self._flat_vector is None:
            sc, sd = self.vector.shape[-2:]
            self._flat_vector = jnp.reshape(self.vector, [self._model_config.batch_size, sc, sd])
        return self._flat_vector

    @property
    def net(self):
        return self._net

    @property
    def element_constants(self):
        return self._constants

    @property
    def element_centers(self):
        return self._centers

    @property
    def element_radii(self):
        return self._radii

    @property
    def element_iparams(self):
        return self._iparams

    @property
    def constant_shape(self):
        return self._constants.shape[2:]

    @property
    def center_shape(self):
        return self._centers.shape[2:]

    @property
    def radius_shape(self):
        return self._radii.shape[2:]

    @property
    def iparam_shape(self):
        if self._iparams is None:
            return None
        else:
            return self._iparams.shape[2:]

    @property
    def tensor_list(self):
        return [x for x in [self._constants, self._centers, self._radii, self._iparams] if x is not None]

    @property
    def batch_size(self):
        return self._constants.shape[0]

    @property
    def element_count(self):
        return self._constants.shape[1]

    @property
    def constants_as_quadrics(self):
        if self._quadrics is None:
            self._quadrics = constants_to_quadrics(self._constants)
        return self._quadrics

    def zero_constants_by_threshold(self, threshold):
        """Zeros out constants 'under' (>=) the threshold in the representation.
        This is useful for removing tiny artifacts in the reconstruction.
        Args:
          threshold: A float, scalar numpy array, or scalar tensor.
        Returns:
          No return value.
        """
        self._constants = jnp.where(
            self._constants >= threshold,
            jnp.zeros_like(self._constants),
            self._constants,
        )

    def zero_constants_by_volume(self, volume_threshold):
        """Zeros out constants based on the volume of the associated ellipsoid.
        This is useful for removing tiny artifacts in the reconstruction.
        Args:
          volume_threshold: A threshold on the ellipsoid 'volume.' This is the
            volume of the ellipsoid at 1 (sqrt) radius length.
        Returns:
          No return.
        """
        sqrt_rads = jnp.sqrt(jnp.maximum(self._radii[..., 0:3], 0.0))
        volumes = (4.0 / 3.0 * jnp.pi) * jnp.prod(sqrt_rads, axis=-1, keepdims=True)
        should_zero = volumes < volume_threshold
        self._constants = jnp.where(should_zero, jnp.zeros_like(self._constants), self._constants)

    def as_placeholder(self):
        """Creates a doppleganger StructuredImplicit with zeros."""
        batch_size = self.batch_size
        element_count = self.element_count
        constants_ph = jnp.zeros(dtype=jnp.float32, shape=[batch_size, element_count] + self.constant_shape)
        centers_ph = jnp.zeros(dtype=jnp.float32, shape=[batch_size, element_count] + self.center_shape)
        radii_ph = jnp.zeros(dtype=jnp.float32, shape=[batch_size, element_count] + self.radius_shape)
        if self._iparams is None:
            iparams_ph = None
        else:
            iparams_ph = jnp.zeros(dtype=jnp.float32, shape=[batch_size, element_count] + self.iparam_shape)

        return StructuredImplicit(
            constants_ph,
            centers_ph,
            radii_ph,
            iparams_ph,
            self._net,
            self._model_config,
        )

    def set_iparams(self, iparams):
        """Adds LDIF embeddings to the SIF object."""
        ijnput_shape = iparams.shape
        expected_batch_dims = self.element_radii.shape[:-1]
        expected_shape = expected_batch_dims + (self._model_config.implicit_parameter_length,)
        if len(ijnput_shape) != len(expected_shape):
            raise ValueError(
                "Trying to set iparams with incorrect rank: %s in but %s expected."
                % (repr(ijnput_shape), repr(expected_shape))
            )
        for di, de in zip(ijnput_shape, expected_shape):
            if di != de:
                raise ValueError(
                    "Trying to set iparams with incorrect shape: %s in but %s expected."
                    % (repr(ijnput_shape), repr(expected_shape))
                )
        self._iparams = iparams

    @property
    def world2local(self):
        """The world2local transformations for each element. Shape [B, EC, 4, 4]."""
        if self._world2local is None:
            self._world2local = self._compute_world2local()
        return self._world2local

    def _compute_world2local(self):
        """Computes a transformation to the local element frames for encoding."""
        # We assume the center is an XYZ position for this transformation:
        # TODO(kgenova) Update this transformation to account for rotation.

        if self._model_config.transformations == "i":
            return jnp.repeat(
                jnp.repeat(jnp.eye(4)[None, None, ...], repeats=self.batch_size, axis=0),
                repeats=self.element_count,
                axis=1,
            )

        if "c" in self._model_config.transformations:
            tx = jnp.repeat(
                jnp.repeat(jnp.eye(3)[None, None, ...], repeats=self.batch_size, axis=0),
                repeats=self.element_count,
                axis=1,
            )
            centers = jnp.reshape(self._centers, [self.batch_size, self.element_count, 3, 1])
            tx = jnp.concatenate([tx, -centers], axis=-1)
            lower_row = jnp.tile(
                jnp.reshape(jnp.array([0.0, 0.0, 0.0, 1.0]), [1, 1, 1, 4]),
                [self.batch_size, self.element_count, 1, 1],
            ).astype(jnp.float32)

            tx = jnp.concatenate([tx, lower_row], axis=-2)
        else:
            tx = jnp.repeat(
                jnp.repeat(jnp.eye(4)[None, None, ...], repeats=self.batch_size, axis=0),
                repeats=self.element_count,
                axis=1,
            )

        # Compute a rotation transformation if necessary:
        if ("r" in self._model_config.transformations) and (self._model_config.coordinates == "cov"):
            # Apply the inverse rotation:
            rotation = jnp.linalg.inv(camera_util.roll_pitch_yaw_to_rotation_matrices(self._radii[..., 3:6]))
        else:
            rotation = jnp.repeat(
                jnp.repeat(jnp.eye(3)[None, None, ...], repeats=self.batch_size, axis=0),
                repeats=self.element_count,
                axis=1,
            )

        # Compute a scale transformation if necessary:
        if ("s" in self._model_config.transformations) and (self._model_config.coordinates in ["aa", "cov"]):
            diag = self._radii[..., 0:3]
            diag = 1.0 / (jnp.sqrt(diag + 1e-8) + 1e-8)
            diag_shape = diag.shape
            diag = diag.reshape((-1, diag_shape[-1]))

            scale = jax.vmap(jnp.diag, in_axes=0)(diag).reshape(diag_shape + (diag_shape[-1],))
        else:
            scale = jnp.repeat(
                jnp.repeat(jnp.eye(3)[None, None, ...], repeats=self.batch_size, axis=0),
                repeats=self.element_count,
                axis=1,
            )

        # Apply both transformations and return the transformed points.
        tx3x3 = jnp.matmul(scale, rotation)
        return jnp.matmul(homogenize(tx3x3), tx)

    def implicit_values(self, local_samples):
        """Computes the implicit values given local ijnput locations.
        Args:
          local_samples: Tensor with shape [..., effective_element_count,
            sample_count, 3]. The samples, which should already be in the coordinate
            frame of the local coordinates.
        Returns:
          values: Tensor with shape [..., effective_element_count, sample_count, 1]
            or something (like a scalar) that can broadcast to that type. The value
            decoded from the implicit parameters at each element.
        """
        if not self._model_config.enable_implicit_parameters:
            log.warning("Requesting implicit values when ipe='f'. iparams are None? %s" % repr(self._iparams is None))
            raise ValueError("Can't request implicit values when ipe='f'.")
        else:
            iparams = _tile_for_symgroups(self._iparams, self._model_config)
            eec = iparams.shape[-2]
            sample_eec = local_samples.shape[-3]
            if eec != sample_eec:
                raise ValueError(
                    "iparams have element count %i, local samples have element_count %i" % (eec, sample_eec)
                )
            values = self.net(iparams, local_samples)
            return values

    def rbf_influence_at_samples(self, samples):
        """Computes the per-effective-element RBF weights at the ijnput samples.
        Args:
          samples: Tensor with shape [..., sample_count, 3]. The ijnput samples.
        Returns:
          Tensor with shape [..., sample_count, effective_element_count]. The RBF
            weight of each *effective* element at each position. The effective
            elements are determined by the SIF's symmetry groups.
        """
        batching_dims = samples.shape[:-2]
        batching_rank = len(batching_dims)
        # For now:
        assert batching_rank == 1
        sample_count = samples.shape[-2]

        effective_constants = _tile_for_symgroups(self._constants, self._model_config)
        effective_centers = _tile_for_symgroups(self._centers, self._model_config)
        effective_radii = _tile_for_symgroups(self._radii, self._model_config)
        # Gives the samples shape [batch_size, effective_elt_count, sample_count, 3]
        effective_samples = _generate_symgroup_samples(samples, self._model_config)
        effective_element_count = get_effective_element_count(self._model_config)

        _, per_element_weights = quadrics.compute_shape_element_influences(
            constants_to_quadrics(effective_constants),
            effective_centers,
            effective_radii,
            effective_samples,
        )

        assert per_element_weights.shape == tuple(batching_dims + [effective_element_count, sample_count, 1])

        weights = per_element_weights
        weights = jnp.reshape(weights, batching_dims + [effective_element_count, sample_count])

        # To get to the desired output shape we need to swap the final dimensions:
        perm = list(range(len(weights.shape)))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        assert perm[-2] > perm[-1]
        weights = jnp.transpose(weights, perm=perm)
        return weights

    def class_at_samples(self, samples, apply_class_transfer=True):
        """Computes the function value of the implicit function at ijnput samples.
        Args:
          samples: Tensor with shape [..., sample_count, 3]. The ijnput samples.
          apply_class_transfer: Whether to apply a class transfer function to the
            predicted values. If false, will be the algebraic distance (depending on
            the selected reconstruction equations).
        Returns:
          A tuple: (global_decisions, local_information).
            global_decisions: Tensor with shape [..., sample_count, 1]. The
              classification value at each sample from the overall reconstruction.
            local_information: A tuple with two entries:
              local_decisions: A [..., element_count, sample_count, 1] Tensor. The
                output value at each sample from the individual shape elements. This
                value may not always be interpretable as a classification. If the
                global solution is an interpolation of local solutions, it will be.
                Otherwise, it may only be interpretable as a marginal contribution
                to the global classification decision.
              local_weights: A [..., element_count, sample_count, 1] Tensor. The
                influence weights of the local decisions.
        """
        batching_dims = samples.shape[:-2]
        batching_rank = len(batching_dims)
        # For now:
        assert batching_rank == 1
        # assert batching_rank in [0, 1]
        # if batching_rank == 0:
        #   batching_rank = 1
        #   batching_dims = [1]
        sample_count = samples.shape[-2]

        effective_constants = _tile_for_symgroups(self._constants, self._model_config)
        effective_centers = _tile_for_symgroups(self._centers, self._model_config)
        effective_radii = _tile_for_symgroups(self._radii, self._model_config)

        effective_samples = _generate_symgroup_samples(samples, self._model_config)
        # The samples have shape [batch_size, effective_elt_count, sample_count, 3]
        effective_element_count = get_effective_element_count(self._model_config)

        (per_element_constants, per_element_weights,) = quadrics.compute_shape_element_influences(
            constants_to_quadrics(effective_constants),
            effective_centers,
            effective_radii,
            effective_samples,
        )

        assert per_element_constants.shape == batching_dims + (
            effective_element_count,
            sample_count,
            1,
        )
        assert per_element_weights.shape == batching_dims + (
            effective_element_count,
            sample_count,
            1,
        )

        agg_fun_dict = {
            "s": jnp.sum,
            "m": jnp.max,
            "l": nn.logsumexp,
            "v": jnp.var,
        }
        agg_fun = agg_fun_dict[self._model_config.aggregation_method]

        if not self._model_config.enable_implicit_parameters:
            local_decisions = per_element_constants * per_element_weights
            local_weights = per_element_weights
            sdf = agg_fun(local_decisions, axis=batching_rank)

        else:
            effective_world2local = _tile_for_symgroups(self.world2local, self._model_config)
            local_samples = _transform_samples(effective_samples, effective_world2local)
            implicit_values = self.implicit_values(local_samples)

            multiplier = 1.0 if self._model_config.debug_implicit_parameters else 0.0
            residuals = 1 + multiplier * implicit_values
            # Each element is c * w * (1 + OccNet(x')):
            local_decisions = per_element_constants * per_element_weights * residuals
            local_weights = per_element_weights
            sdf = agg_fun(local_decisions, axis=batching_rank)

        # Need to map from the metaball influence to something that's < 0 inside.
        if apply_class_transfer:
            sdf = sdf_util.apply_class_transfer(
                sdf,
                self._model_config,
                soft_transfer=True,
                offset=self._model_config.isolevel,
            )

        assert sdf.shape == batching_dims + (sample_count, 1)
        assert local_decisions.shape == batching_dims + (effective_element_count, sample_count, 1)
        return sdf, (local_decisions, local_weights)
