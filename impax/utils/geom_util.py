"""Utilities for geometric operations in tensorflow."""

import jax.numpy as jnp
from jax import lax, random

# local
from impax.utils import jnp_util
from impax.utils import camera_util


def l2_normalize(arr, axis, epsilon=1e-12):
    """
    L2 normalize along a particular axis.
    Doc taken from tf.nn.l2_normalize:
    https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize
        output = x / (
            sqrt(
                max(
                    sum(x**2),
                    epsilon
                )
            )
        )
    """
    sq_arr = jnp.power(arr, 2)
    square_sum = jnp.sum(sq_arr, axis=axis, keepdims=True)
    max_weights = jnp.maximum(square_sum, epsilon)
    return jnp.divide(arr, jnp.sqrt(max_weights))


def ray_sphere_intersect(ray_start, ray_direction, sphere_center, sphere_radius, max_t):
    """Intersect rays with each of a set of spheres.
    Args:
      ray_start: Tensor with shape [batch_size, ray_count, 3]. The end point of
        the rays. In the same coordinate space as the spheres.
      ray_direction: Tensor with shape [batch_size, ray_count, 3]. The extant ray
        direction.
      sphere_center: Tensor with shape [batch_size, sphere_count, 3]. The center
        of the spheres.
      sphere_radius: Tensor with shape [batch_size, sphere_count, 1]. The radius
        of the spheres.
      max_t: The maximum intersection distance.
    Returns:
      intersections: Tensor with shape [batch_size, ray_count, sphere_count]. If
        no intersection is found between [0, max_t), then the value will be max_t.
    """
    # We apply the algebraic solution:
    batch_size, ray_count = ray_start.shape[:2]
    sphere_count = sphere_center.shape[1]
    ray_direction = jnp.reshape(ray_direction, [batch_size, ray_count, 1, 3])
    ray_start = jnp.reshape(ray_start, [batch_size, ray_count, 1, 3])
    sphere_center = jnp.reshape(sphere_center, [batch_size, 1, sphere_count, 3])
    sphere_radius = jnp.reshape(sphere_radius, [batch_size, 1, sphere_count, 1])
    a = 1.0
    b = 2.0 * ray_direction * (ray_start - sphere_center)
    ray_sphere_distance = jnp.sum(
        jnp.square(ray_start - sphere_center), axis=-1, keep_dims=True
    )
    c = ray_sphere_distance - jnp.square(sphere_radius)
    discriminant = jnp.square(b) - 4 * a * c
    # Assume it's positive, then zero out later:
    ta = jnp.divide((-b + jnp.sqrt(discriminant)), 2 * a)
    tb = jnp.divide((-b - jnp.sqrt(discriminant)), 2 * a)
    t0 = jnp.minimum(ta, tb)
    t1 = jnp.maximum(ta, tb)
    t = jnp.where(t0 > 0, t0, t1)
    intersection_invalid = jnp.logical_or(
        jnp.logical_or(discriminant < 0, t < 0), t > max_t
    )
    t = jnp.where(intersection_invalid, max_t * jnp.ones_like(t), t)
    return t


def to_homogeneous(t, is_point):
    """Makes a homogeneous space tensor given a tensor with ultimate coordinates.
    Args:
      t: Tensor with shape [..., K], where t is a tensor of points in
        K-dimensional space.
      is_point: Boolean. True for points, false for directions
    Returns:
      Tensor with shape [..., K+1]. t padded to be homogeneous.
    """
    padding = 1 if is_point else 0
    rank = len(t.shape)
    paddings = []
    for _ in range(rank):
        paddings.append([0, 0])
    paddings[-1][1] = 1
    return jnp.pad(t, jnp.array(paddings), mode="CONSTANT", constant_values=padding)


def transform_points_with_normals(points, tx, normals=None):
    """Transforms a pointcloud with normals to a new coordinate frame.
    Args:
      points: Tensor with shape [batch_size, point_count, 3 or 6].
      tx: Tensor with shape [batch_size, 4, 4]. Takes column-vectors from the
        current frame to the new frame as T*x.
      normals: Tensor with shape [batch_size, point_count, 3] if provided. None
        otherwise. If the points tensor contains normals, this should be None.
    Returns:
      If tensor 'points' has shape [..., 6], then a single tensor with shape
        [..., 6] in the new frame. If 'points' has shape [..., 3], then returns
        either one or two tensors of shape [..., 3] depending on whether 'normals'
        is None.
    """
    if len(points.shape) != 3:
        raise ValueError(f"Invalid points shape: {points.shape}")
    if len(tx.shape) != 3:
        raise ValueError(f"Invalid tx shape: {tx.shape}")
    are_concatenated = points.shape[-1] == 6
    if are_concatenated:
        points, normals = jnp.split(points, [3, 3], axis=-1)

    transformed_samples = apply_4x4(
        points, tx, are_points=True, batch_rank=1, sample_rank=1
    )
    if normals is not None:
        transformed_normals = apply_4x4(
            normals,
            jnp.linalg.inv(jnp.transpose(tx, perm=[0, 2, 1])),
            are_points=False,
            batch_rank=1,
            sample_rank=1,
        )
        transformed_normals = transformed_normals / (
            jnp.linalg.norm(transformed_normals, axis=-1, keepdims=True) + 1e-8
        )
    if are_concatenated:
        return jnp.concat([transformed_samples, transformed_normals], axis=-1)
    if normals is not None:
        return transformed_samples, transformed_normals
    return transformed_samples


def transform_featured_points(points, tx):
    """Transforms a pointcloud with features.
    Args:
      points: Tensor with shape [batch_size, point_count, 3+feature_count].
      tx: Tensor with shape [batch_size, 4, 4].
    Returns:
      Tensor with shape [batch_size, point_count, 3+feature_count].
    """
    feature_count = points.shape[-1] - 3
    if feature_count == 0:
        xyz = points
        features = None
    else:
        xyz, features = jnp.split(points, [3, feature_count], axis=2)

    xyz = apply_4x4(xyz, tx, are_points=True, batch_rank=1, sample_rank=1)
    if feature_count:
        return jnp.concat([xyz, features], axis=2)
    return xyz


def rotation_to_tx(rot):
    """Maps a 3x3 rotation matrix to a 4x4 homogeneous matrix.
    Args:
      rot: Tensor with shape [..., 3, 3].
    Returns:
      Tensor with shape [..., 4, 4].
    """
    batch_dims = rot.shape[:-2]
    empty_col = jnp.zeros(batch_dims + [3, 1], dtype=jnp.float32)
    rot = jnp.concat([rot, empty_col], axis=-1)
    hom_row = jnp.eye(4, batch_shape=batch_dims)[..., 3:4, :]
    return jnp.concat([rot, hom_row], axis=-2)


def extract_points_near_origin(points, count, features=None):
    """Returns the points nearest to the origin in a pointcloud.
    Args:
      points: Tensor with shape [batch_size, point_count, 3 or more].
      count: The number of points to extract.
      features: Tensor with shape [batch_size, point_count, feature_count] if
        present. None otherwise.
    Returns:
      Either one tensor of size [batch_size, count, 3 or 6] or two tensors of
      size [batch_size, count, 3], depending on whether normals was provided and
      the shape of the 'points' tensor.
    """
    are_concatenated = points.shape[-1] > 3
    if are_concatenated:
        feature_count = points.shape[-1] - 3
        original = points
        points, features = jnp.split(points, [3, feature_count], axis=-1)
    else:
        assert points.shape[-1] == 3

    candidate_dists = jnp.linalg.norm(points, axis=-1)
    _, selected_indices = lax.top_k(-candidate_dists, k=count, sorted=False)
    if are_concatenated:
        return lax.gather(original, selected_indices, dimension_numbers=1)
    else:
        selected_points = lax.gather(points, selected_indices, dimension_numbers=1)
        if features is not None:
            return selected_points, lax.gather(
                features, selected_indices, dimension_numbers=1
            )
        return selected_points


def local_views_of_shape(
    global_points,
    world2local,
    local_point_count,
    global_normals=None,
    global_features=None,
    zeros_invalid=False,
    zero_threshold=1e-6,
    expand_region=True,
    threshold=4.0,
):
    """Computes a set of local point cloud observations from a global observation.
    It is assumed for optimization purposes that
    global_point_count >> local_point_count.
    Args:
      global_points: Tensor with shape [batch_size, global_point_count, 3]. The
        ijnput observation point cloud in world space.
      world2local: Tensor with shape [batch_size, frame_count, 4, 4]. Each 4x4
        matrix maps from points in world space to points in a local frame.
      local_point_count: Integer. The number of points to output in each local
        frame. Whatever this value, the local_point_count closest points to each
        local frame origin will be returned.
      global_normals: Tensor with shape [batch_size, global_point_count, 3]. The
        ijnput observation point cloud's normals in world space. Optional.
      global_features: Tensor with shape [batch_size, global_point_count,
        feature_count]. The ijnput observation point cloud features, in any space.
        Optional.
      zeros_invalid: Whether to consider the vector [0, 0, 0] to be invalid.
      zero_threshold: Values less than this in magnitude are considered to be 0.
      expand_region: Whether to expand outward from the threshold region. If
        false, fill with zeros.
      threshold: The distance threshold.
    Returns:
      local_points: Tensor with shape [batch_size, frame_count,
        local_point_count, 3].
      local_normals: Tensor with shape [batch_size, frame_count,
        local_point_count, 3]. None if global_normals not provided.
      local_features: Tensor with shape [batch_size, frame_count,
        local_point_count, feature_count]. Unlike the local normals and points,
        these are not transformed because there may or may not be a good
        transformation to apply, depending on what the features are. But they will
        be the features associated with the local points that were chosen. None
        if global_features not provided.
    """
    # Example use case: batch_size = 64, global_point_count = 100000
    # local_point_count = 1000, frame_count = 25. Then:
    # global_points has size 64*100000*3*4 = 73mb
    # local_points has size 64*1000*25*3*4 = 18mb
    # If we made an intermediate tensor with shape [batch_size, frame_count,
    #   global_point_count, 3] -> 64 * 25 * 100000 * 3 * 4 = 1.8 Gb -> bad.

    batch_size, _, _ = global_points.shape
    if zeros_invalid:
        # If we just set the global points to be very far away, they won't be a
        # nearest neighbor
        abs_zero = False
        if abs_zero:
            is_zero = jnp.all(jnp.equal(global_points, 0.0), axis=-1, keepdims=True)
        else:
            is_zero = jnp.all(
                jnp.abs(global_points) < zero_threshold, axis=-1, keepdims=True
            )
        global_points = jnp.where(is_zero, 100.0, global_points)
    _, frame_count, _, _ = world2local.shape

    local2world = jnp.linalg.inv(world2local)

    # *sigh* oh well, guess we have to do the transform:
    tiled_global = jnp.tile(
        jnp.expand_dims(to_homogeneous(global_points, is_point=True), axis=1),
        [1, frame_count, 1, 1],
    )
    all_local_points = jnp.matmul(tiled_global, world2local, transpose_b=True)
    distances = jnp.norm(all_local_points, axis=-1)
    # thresh = 4.0
    # TODO(kgenova) This is potentially a problem because it could introduce
    # randomness into the pipeline at inference time.
    probabilities = random.uniform(key, distances.shape)
    is_valid = distances < threshold

    sample_order = jnp.where(is_valid, probabilities, -distances)
    _, top_indices = lax.top_k(sample_order, k=local_point_count, sorted=False)
    local_points = lax.gather(
        all_local_points, top_indices, dimension_numbers=2, axis=-2
    )

    is_valid = jnp.expand_dims(is_valid, axis=-1)
    points_valid = lax.gather(is_valid, top_indices, batch_dims=2, axis=-2)

    if not expand_region:
        local_points = jnp.where(points_valid, local_points, 0.0)

    if global_normals is not None:
        tiled_global_normals = jnp.tile(
            jnp.expand_dims(to_homogeneous(global_normals, is_point=False), axis=1),
            [1, frame_count, 1, 1],
        )
        # Normals get transformed by the inverse-transpose matrix:
        all_local_normals = jnp.matmul(
            tiled_global_normals, local2world, transpose_b=False
        )
        local_normals = lax.gather(
            all_local_normals, top_indices, batch_dims=2, axis=-2
        )
        # Remove the homogeneous coordinate now. It isn't a bug to normalize with
        # it since it's zero, but it's confusing.
        local_normals = l2_normalize(local_normals[..., :3], axis=-1)
    else:
        local_normals = None

    if global_features is not None:
        feature_count = global_features.shape[-1]
        local_features = lax.gather(global_features, top_indices, batch_dims=1, axis=-2)
    else:
        local_features = None
    return local_points, local_normals, local_features, points_valid


def chamfer_distance(pred, target):
    """Computes the chamfer distance between two point sets, in both directions.
    Args:
      pred: Tensor with shape [..., pred_point_count, n_dims].
      target: Tensor with shape [..., target_point_count, n_dims].
    Returns:
      pred_to_target, target_to_pred.
      pred_to_target: Tensor with shape [..., pred_point_count, 1]. The distance
        from each point in pred to the closest point in the target.
      target_to_pred: Tensor with shape [..., target_point_count, 1]. The distance
        from each point in target to the closet point in the prediction.
    """
    # batching_dimensions = pred.shape[:-2]
    # batching_rank = len(batching_dimensions)
    # pred_norm_squared = tf.matmul(pred, pred, transpose_b=True)
    # target_norm_squared = tf.matmul(target, target, transpose_b=True)
    # target_mul_pred_t = tf.matmul(pred, target, transpose_b=True)
    # pred_mul_target_t = tf.matmul(target, pred, transpose_b=True)

    differences = jnp.expand_dims(pred, axis=-2) - jnp.expand_dims(target, axis=-3)
    squared_distances = jnp.sum(differences * differences, axis=-1)
    # squared_distances = tf.matmul(differences, differences, transpose_b=True)
    # differences = pred - tf.transpose(target, perm=range(batching_rank) +
    #  [batching_rank+2, batching_rank+1])
    pred_to_target = jnp.min(squared_distances, axis=-1)
    target_to_pred = jnp.min(squared_distances, axis=-2)
    pred_to_target = jnp.expand_dims(pred_to_target, axis=-1)
    target_to_pred = jnp.expand_dims(target_to_pred, axis=-1)
    return jnp.sqrt(pred_to_target), jnp.sqrt(target_to_pred)


def dodeca_parameters(dodeca_idx):
    """Computes the viewpoint, centroid, and up vectors for the dodecahedron."""
    gr = (1.0 + jnp.sqrt(5.0)) / 2.0
    rgr = 1.0 / gr
    viewpoints = [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
        [0, gr, rgr],
        [0, gr, -rgr],
        [0, -gr, rgr],
        [0, -gr, -rgr],
        [rgr, 0, gr],
        [rgr, 0, -gr],
        [-rgr, 0, gr],
        [-rgr, 0, -gr],
        [gr, rgr, 0],
        [gr, -rgr, 0],
        [-gr, rgr, 0],
        [-gr, -rgr, 0],
    ]
    viewpoint = 0.6 * jnp.array(viewpoints[dodeca_idx], dtype=jnp.float32)
    centroid = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    world_up = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    return viewpoint, centroid, world_up


def get_camera_to_world(viewpoint, center, world_up):
    """Computes a 4x4 mapping from camera space to world space."""
    towards = center - viewpoint
    towards = towards / jnp.linalg.norm(towards)
    right = jnp.cross(towards, world_up)
    right = right / jnp.linalg.norm(right)
    cam_up = jnp.cross(right, towards)
    cam_up = cam_up / jnp.linalg.norm(cam_up)
    rotation = jnp.stack([right, cam_up, -towards], axis=1)
    rotation_4x4 = jnp.eye(4)
    rotation_4x4[:3, :3] = rotation
    camera_to_world = rotation_4x4.copy()
    camera_to_world[:3, 3] = viewpoint
    return camera_to_world


def get_dodeca_camera_to_worlds():
    camera_to_worlds = []
    for i in range(20):
        camera_to_worlds.append(get_camera_to_world(*dodeca_parameters(i)))
    camera_to_worlds = jnp.stack(camera_to_worlds, axis=0)
    return camera_to_worlds


def gaps_depth_render_to_xyz(model_config, depth_image, camera_parameters):
    """Transforms a depth image to camera space assuming its dodeca parameters."""
    # TODO(kgenova) Extract viewpoint, width, height from camera parameters.
    del camera_parameters
    depth_image_height, depth_image_width = depth_image.shape[1:3]
    if model_config.hparams.didx == 0:
        viewpoint = jnp.array([1.03276, 0.757946, -0.564739])
        towards = jnp.array([-0.737684, -0.54139, 0.403385])  #  = v/-1.4
        up = jnp.array([-0.47501, 0.840771, 0.259748])
    else:
        assert False
    towards = towards / jnp.linalg.norm(towards)
    right = jnp.cross(towards, up)
    right = right / jnp.linalg.norm(right)
    up = jnp.cross(right, towards)
    up = up / jnp.linalg.norm(up)
    rotation = jnp.stack([right, up, -towards], axis=1)
    rotation_4x4 = jnp.eye(4)
    rotation_4x4[:3, :3] = rotation
    camera_to_world = rotation_4x4.copy()
    camera_to_world[:3, 3] = viewpoint
    camera_to_world = jnp.array(camera_to_world.astype(jnp.float32))
    world_to_camera = jnp.reshape(jnp.linalg.inv(camera_to_world), [1, 4, 4])
    world_to_camera = jnp.tile(world_to_camera, [model_config.hparams.bs, 1, 1])
    xyz_image, _, _ = depth_image_to_xyz_image(depth_image, world_to_camera, xfov=0.5)
    xyz_image = jnp.reshape(
        xyz_image, [model_config.hparams.bs, depth_image_height, depth_image_width, 3]
    )
    return xyz_image


def angle_of_rotation_to_2d_rotation_matrix(angle_of_rotation):
    """Given a batch of rotations, create a batch of 2d rotation matrices.
    Args:
      angle_of_rotation: Tensor with shape [batch_size].
    Returns:
      Tensor with shape [batch_size, 2, 2]
    """
    c = jnp.cos(angle_of_rotation)
    s = jnp.sin(angle_of_rotation)
    top_row = jnp.stack([c, -s], axis=1)
    bottom_row = jnp.stack([s, c], axis=1)
    return jnp.stack([top_row, bottom_row], axis=1)


def fractional_vector_projection(e0, e1, p, falloff=2.0):
    """Returns a fraction describing whether p projects inside the segment e0 e1.
    If p projects inside the segment, the result is 1. If it projects outside,
    the result is a fraction that is always greater than 0 but monotonically
    decreasing as the distance to the inside of the segment increase.
    Args:
      e0: Tensor with two elements containing the first endpoint XY locations.
      e1: Tensor with two elements containing the second endpoint XY locations.
      p: Tensor with shape [batch_size, 2] containing the query points.
      falloff: Float or Scalar Tensor specifying the softness of the falloff of
        the projection. Larger means a longer falloff.
    """
    batch_size = p.shape[0].value
    p = jnp.reshape(p, [batch_size, 2])
    e0 = jnp.reshape(e0, [1, 2])
    e1 = jnp.reshape(e1, [1, 2])
    e01 = e1 - e0
    # Normalize for vector projection:
    e01_norm = jnp.sqrt(e01[0, 0] * e01[0, 0] + e01[0, 1] * e01[0, 1])
    e01_normalized = e01 / jnp.reshape(e01_norm, [1, 1])
    e0p = p - e0
    e0p_dot_e01_normalized = jnp.matmul(
        jnp.reshape(e0p, [1, batch_size, 2]),
        jnp.reshape(e01_normalized, [1, 1, 2]),
        transpose_b=True,
    )
    e0p_dot_e01_normalized = (
        jnp.reshape(e0p_dot_e01_normalized, [batch_size]) / e01_norm
    )
    if falloff is None:
        left_sided_inside = jnp.cast(
            jnp.logical_and(e0p_dot_e01_normalized >= 0, e0p_dot_e01_normalized <= 1),
            dtype=jnp.float32,
        )
        return left_sided_inside

    # Now that we have done the left side, do the right side:
    e10_normalized = -e01_normalized
    e1p = p - e1
    e1p_dot_e10_normalized = jnp.matmul(
        jnp.reshape(e1p, [1, batch_size, 2]),
        jnp.reshape(e10_normalized, [1, 1, 2]),
        transpose_b=True,
    )
    e1p_dot_e10_normalized = (
        jnp.reshape(e1p_dot_e10_normalized, [batch_size]) / e01_norm
    )

    # Take the maximum of the two projections so we face it from the positive
    # direction:
    proj = jnp.maximum(e0p_dot_e01_normalized, e1p_dot_e10_normalized)
    proj = jnp.maximum(proj, 1.0)

    # A projection value of 1 means at the border exactly.
    # Take the max with 1, to throw out all cases besides 'left' overhang.
    falloff_is_relative = True
    if falloff_is_relative:
        fractional_falloff = 1.0 / (jnp.pow(falloff * (proj - 1), 2.0) + 1.0)
        return fractional_falloff
    else:
        # Currently the proj value is given as a distance that is the fraction of
        # the length of the line. Instead, multiply by the length of the line
        # to get the distance in pixels. Then, set a target '0' distance, (i.e.
        # 10 pixels). Divide by that distance so we express distance in multiples
        # of the max distance that gets seen.
        # threshold at 1, and return 1 - that to get linear falloff from 0 to
        # the target distance.
        line_length = jnp.reshape(e01_norm, [1])
        pixel_dist = jnp.reshape(proj - 1, [-1]) * line_length
        zero_thresh_in_pixels = jnp.reshape(jnp.array([8.0], dtype=jnp.float32), [1])
        relative_dist = pixel_dist / zero_thresh_in_pixels
        return 1.0 / (jnp.pow(relative_dist, 3.0) + 1.0)


def rotate_about_point(angle_of_rotation, point, to_rotate):
    """Rotates a single ijnput 2d point by a specified angle around a point."""
    cos_angle = jnp.cos(angle_of_rotation)
    sin_angle = jnp.sin(angle_of_rotation)
    top_row = jnp.stack([cos_angle, -sin_angle], axis=0)
    bottom_row = jnp.stack([sin_angle, cos_angle], axis=0)
    rotation_matrix = jnp.reshape(jnp.stack([top_row, bottom_row], axis=0), [1, 2, 2])
    to_rotate = jnp.reshape(to_rotate, [1, 1, 2])
    point = jnp.reshape(point, [1, 1, 2])
    to_rotate = to_rotate - point
    to_rotate = jnp.matmul(rotation_matrix, to_rotate, transpose_b=True)
    to_rotate = jnp.reshape(to_rotate, [1, 1, 2]) + point
    return to_rotate


def interpolate_from_grid(samples, grid):
    grid_coordinates = (samples + 0.5) * 63.0
    return interpolate_from_grid_coordinates(grid_coordinates, grid)


def reflect(samples, reflect_x=False, reflect_y=False, reflect_z=False):
    """Reflects the sample locations across the planes specified in xyz.
    Args:
      samples: Tensor with shape [..., 3].
      reflect_x: Bool.
      reflect_y: Bool.
      reflect_z: Bool.
    Returns:
      Tensor with shape [..., 3]. The reflected samples.
    """
    assert isinstance(reflect_x, bool)
    assert isinstance(reflect_y, bool)
    assert isinstance(reflect_z, bool)
    floats = [-1.0 if ax else 1.0 for ax in [reflect_x, reflect_y, reflect_z]]
    mult = jnp.array(floats, dtype=jnp.float32)
    shape = samples.shape
    leading_dims = shape[:-1]
    assert shape[-1] == 3
    mult = mult.reshape([1] * len(leading_dims) + [3])
    mult = jnp.array(mult, dtype=jnp.float32)
    return mult * samples


def z_reflect(samples):
    """Reflects the sample locations across the XY plane.
    Args:
      samples: Tensor with shape [..., 3]
    Returns:
      reflected: Tensor with shape [..., 3]. The reflected samples.
    """
    return reflect(samples, reflect_z=True)


def get_world_to_camera(idx):
    assert idx == 1
    eye = jnp.array([[0.671273, 0.757946, -0.966907]], dtype=jnp.float32)
    look_at = jnp.zeros_like(eye)
    world_up = jnp.array([[0.0, 1.0, 0.0]], dtype=jnp.float32)
    world_to_camera = camera_util.look_at(eye, look_at, world_up)
    return world_to_camera


def transform_depth_dodeca_to_xyz_dodeca(depth_dodeca):
    """Lifts a dodecahedron of depth images to world space."""
    batch_size = depth_dodeca.shape[0]
    cam2world = get_dodeca_camera_to_worlds()
    cam2world = jnp.reshape(cam2world, [1, 20, 4, 4]).astype(jnp.float32)
    world2cams = jnp.linalg.inv(cam2world)
    world2cams = jnp.tile(world2cams, [batch_size, 1, 1, 1])
    world2cams = jnp.unstack(jnp.array(world2cams, dtype=jnp.float32), axis=1)
    depth_im_stack = jnp.unstack(depth_dodeca, axis=1)
    assert len(depth_im_stack) == 20
    assert len(world2cams) == 20
    xyz_images = []
    for i in range(20):
        world2cam = world2cams[i]
        depth_im = depth_im_stack[i]
        xyz_image = depth_image_to_xyz_image(depth_im, world2cam, xfov=0.5)[0]
        xyz_images.append(xyz_image)
    xyz_images = jnp.stack(xyz_images, axis=1)
    xyz_images = jnp.where(depth_dodeca > 0.0, xyz_images, 0.0)
    return xyz_images


def transform_depth_dodeca_to_xyz_dodeca_jnp(depth_dodeca):
    xyz_out = transform_depth_dodeca_to_xyz_dodeca(depth_dodeca)
    return xyz_out


def _unbatch(arr):
    if arr.shape[0] == 1:
        return arr.reshape(arr.shape[1:])
    return arr


def to_homogenous_jnp(arr, is_point=True):
    assert arr.shape[-1] in [2, 3]
    homogeneous_shape = list(arr.shape[:-1]) + [1]
    if is_point:
        coord = jnp.ones(homogeneous_shape, dtype=jnp.float32)
    else:
        coord = jnp.zeros(homogeneous_shape, dtype=jnp.float32)
    return jnp.concatenate([arr, coord], axis=-1)


def depth_to_cam_jnp(im, xfov=0.5):
    """Converts a gaps depth image to camera space."""
    im = _unbatch(im)
    height, width, _ = im.shape
    pixel_coords = jnp_util.make_coordinate_grid(
        height, width, is_screen_space=False, is_homogeneous=False
    )
    nic_x = jnp.reshape(pixel_coords[:, :, 0], [height, width])
    nic_y = jnp.reshape(pixel_coords[:, :, 1], [height, width])
    # GAPS nic coordinates have an origin at the center of the image, not
    # in the corner:
    nic_x = 2 * nic_x - 1.0
    nic_y = 2 * nic_y - 1.0
    nic_d = -jnp.reshape(im, [height, width])
    aspect = height / float(width)
    yfov = jnp.atan(aspect * jnp.tan(xfov))

    intrinsics_00 = 1.0 / jnp.tan(xfov)
    intrinsics_11 = 1.0 / jnp.tan(yfov)

    cam_x = nic_x * -nic_d / intrinsics_00
    cam_y = nic_y * nic_d / intrinsics_11
    cam_z = nic_d

    cam_xyz = jnp.stack([cam_x, cam_y, cam_z], axis=2)
    return cam_xyz


def apply_tx_jnp(samples, tx, is_point=True):
    shape_in = samples.shape
    flat_samples = jnp.reshape(samples, [-1, 3])
    flat_samples = to_homogenous_jnp(flat_samples, is_point=is_point)
    flat_samples = jnp.matmul(flat_samples, tx.T)
    flat_samples = flat_samples[:, :3]
    return jnp.reshape(flat_samples, shape_in)


def depth_image_to_sdf_constraints(im, cam2world, xfov=0.5):
    """Estimates inside/outside constraints from a gaps depth image."""
    im = _unbatch(im)
    cam2world = _unbatch(cam2world)
    height, width, _ = im.shape
    cam_xyz = depth_to_cam_jnp(im, xfov)
    world_xyz = apply_tx_jnp(cam_xyz, cam2world, is_point=True)
    ray_xyz = apply_tx_jnp(cam_xyz, cam2world, is_point=False)
    ray_xyz = ray_xyz / jnp.linalg.norm(ray_xyz, axis=-1, keepdims=True)
    delta = 0.005
    pos_constraint = world_xyz - delta * ray_xyz
    neg_constraint = world_xyz + delta * ray_xyz
    sample_shape = [height * width, 3]
    pos_constraint = jnp.reshape(pos_constraint, sample_shape)
    neg_constraint = jnp.reshape(neg_constraint, sample_shape)
    sdf_shape = [height * width, 1]
    zero = jnp.zeros(sdf_shape, dtype=jnp.float32)

    # Filter out the background
    is_valid = jnp.reshape(im, [-1]) != 0.0
    pos_constraint = pos_constraint[is_valid, :]
    neg_constraint = neg_constraint[is_valid, :]
    zero = zero[is_valid, :]

    samples = jnp.concatenate([pos_constraint, neg_constraint], axis=0)
    constraints = jnp.concatenate([zero + delta, zero - delta], axis=0)
    return samples, constraints


def depth_dodeca_to_sdf_constraints(depth_ims):
    """Estimates inside/outside constraints from a depth dodecahedron."""
    cam2world = jnp.split(get_dodeca_camera_to_worlds(), 20)
    depth_ims = jnp.split(_unbatch(depth_ims), 20)
    samps = []
    constraints = []
    for i in range(20):
        s, c = depth_image_to_sdf_constraints(depth_ims[i], cam2world[i])
        samps.append(s)
        constraints.append(c)
    samps = jnp.concatenate(samps)
    constraints = jnp.concatenate(constraints)
    return samps, constraints


def depth_dodeca_to_samples(dodeca):
    samples, sdf_constraints = depth_dodeca_to_sdf_constraints(dodeca)
    all_samples = jnp.concatenate([samples, sdf_constraints], axis=-1)
    return all_samples


def depth_image_to_class_constraints(im, cam2world, xfov=0.5):
    samples, sdf_constraints = depth_image_to_sdf_constraints(im, cam2world, xfov)
    class_constraints = sdf_constraints > 0
    return samples, class_constraints


def depth_image_to_samples(im, cam2world, xfov=0.5):  # pylint:disable=unused-argument
    """A wrapper for depth_image_to_sdf_constraints to return samples."""
    samples, sdf_constraints = depth_image_to_sdf_constraints(im, cam2world)
    all_samples = jnp.concatenate([samples, sdf_constraints], axis=-1)
    return all_samples


def apply_4x4(tensor, tx, are_points=True, batch_rank=None, sample_rank=None):
    """Applies a 4x4 matrix to 3D points/vectors.
    Args:
      tensor: Tensor with shape [batching_dims] + [sample_dims] + [3].
      tx: Tensor with shape [batching_dims] + [4, 4].
      are_points: Boolean. Whether to treat the samples as points or vectors.
      batch_rank: The number of leading batch dimensions. Optional, just used to
        enforce the shapes are as expected.
      sample_rank: The number of sample dimensions. Optional, just used to enforce
        the shapes are as expected.
    Returns:
      Tensor with shape [..., sample_count, 3].
    """
    expected_batch_rank = batch_rank
    expected_sample_rank = sample_rank
    batching_dims = tx.shape[:-2]
    batch_rank = len(batching_dims)
    if expected_batch_rank is not None:
        assert batch_rank == expected_batch_rank
    # flat_batch_count = int(jnp.prod(batching_dims))

    sample_dims = tensor.shape[batch_rank:-1]
    sample_rank = len(sample_dims)
    if expected_sample_rank is not None:
        assert sample_rank == expected_sample_rank
    flat_sample_count = int(jnp.prod(sample_dims))
    assert sample_rank >= 1
    assert batch_rank >= 0
    if sample_rank > 1:
        tensor = jnp.reshape(tensor, batching_dims + [flat_sample_count, 3])
    initializer = jnp.ones if are_points else jnp.zeros
    w = initializer(batching_dims + [flat_sample_count, 1], dtype=jnp.float32)
    tensor = jnp.concat([tensor, w], axis=-1)
    tensor = jnp.matmul(tensor, tx, transpose_b=True)
    tensor = tensor[..., :3]
    if sample_rank > 1:
        tensor = jnp.reshape(tensor, batching_dims + sample_dims + [3])
    return tensor


def depth_image_to_xyz_image(depth_images, world_to_camera, xfov=0.5):
    """Converts GAPS depth images to world space."""
    batch_size, height, width, channel_count = depth_images.shape
    assert channel_count == 1

    camera_to_world_mat = jnp.linalg.inv(world_to_camera)

    pixel_coords = jnp_util.make_coordinate_grid(
        height, width, is_screen_space=False, is_homogeneous=False
    )
    nic_x = jnp.tile(
        jnp.reshape(pixel_coords[:, :, 0], [1, height, width]), [batch_size, 1, 1]
    )
    nic_y = jnp.tile(
        jnp.reshape(pixel_coords[:, :, 1], [1, height, width]), [batch_size, 1, 1]
    )

    nic_x = 2 * nic_x - 1.0
    nic_y = 2 * nic_y - 1.0
    nic_d = -jnp.reshape(depth_images, [batch_size, height, width])

    aspect = height / float(width)
    yfov = jnp.atan(aspect * jnp.tan(xfov))

    intrinsics_00 = 1.0 / jnp.tan(xfov)
    intrinsics_11 = 1.0 / jnp.tan(yfov)

    nic_xyz = jnp.stack([nic_x, nic_y, nic_d], axis=3)
    flat_nic_xyz = jnp.reshape(nic_xyz, [batch_size, height * width, 3])

    camera_x = (nic_x) * -nic_d / intrinsics_00
    camera_y = (nic_y) * nic_d / intrinsics_11
    camera_z = nic_d
    homogeneous_coord = jnp.ones_like(camera_z)
    camera_xyz = jnp.stack([camera_x, camera_y, camera_z, homogeneous_coord], axis=3)
    flat_camera_xyzw = jnp.reshape(camera_xyz, [batch_size, height * width, 4])
    flat_world_xyz = jnp.matmul(flat_camera_xyzw, camera_to_world_mat, transpose_b=True)
    world_xyz = jnp.reshape(flat_world_xyz, [batch_size, height, width, 4])
    world_xyz = world_xyz[:, :, :, :3]
    return world_xyz, flat_camera_xyzw[:, :, :3], flat_nic_xyz


def interpolate_from_grid_coordinates(samples, grid):
    """Performs trilinear interpolation to estimate the value of a grid function.
    This function makes several assumptions to do the lookup:
    1) The grid is LHW and has evenly spaced samples in the range (0, 1), which
      is really the screen space range [0.5, {L, H, W}-0.5].
    Args:
      samples: Tensor with shape [batch_size, sample_count, 3].
      grid: Tensor with shape [batch_size, length, height, width, 1].
    Returns:
      sample: Tensor with shape [batch_size, sample_count, 1] and type float32.
      mask: Tensor with shape [batch_size, sample_count, 1] and type float32
    """
    batch_size, length, height, width = grid.shape[:4]
    # These asserts aren't required by the algorithm, but they are currently
    # true for the pipeline:
    assert length == height
    assert length == width
    sample_count = samples.shape[1]
    assert samples.shape == [
        batch_size,
        sample_count,
        3,
    ], "interpolate_from_grid:samples"

    assert grid.shape == [
        batch_size,
        length,
        height,
        width,
        1,
    ], "interpolate_from_grid:grid"

    offset_samples = samples  # Used to subtract 0.5
    lower_coords = jnp.floor(offset_samples).astype(jnp.int32)
    upper_coords = lower_coords + 1
    alphas = jnp.floormod(offset_samples, 1.0)

    maximum_value = grid.shape[1:4]
    size_per_channel = jnp.tile(
        jnp.reshape(jnp.array(maximum_value, dtype=jnp.int32), [1, 1, 3]),
        [batch_size, sample_count, 1],
    )
    # We only need to check that the floor is at least zero and the ceil is
    # no greater than the max index, because floor round negative numbers to
    # be more negative:
    is_valid = jnp.logical_and(lower_coords >= 0, upper_coords < size_per_channel)
    # Validity mask has shape [batch_size, sample_count] and is 1.0 where all of
    # x,y,z are within the [0,1] range of the grid.
    validity_mask = jnp.min(is_valid.astype(jnp.float32), axis=2, keep_dims=True)

    lookup_coords = [[[], []], [[], []]]
    corners = [[[], []], [[], []]]
    flattened_grid = jnp.reshape(grid, [batch_size, length * height * width])
    for xi, x_coord in enumerate([lower_coords[:, :, 0], upper_coords[:, :, 0]]):
        x_coord = jnp.clip_by_value(x_coord, 0, width - 1)
        for yi, y_coord in enumerate([lower_coords[:, :, 1], upper_coords[:, :, 1]]):
            y_coord = jnp.clip_by_value(y_coord, 0, height - 1)
            for zi, z_coord in enumerate(
                [lower_coords[:, :, 2], upper_coords[:, :, 2]]
            ):
                z_coord = jnp.clip_by_value(z_coord, 0, length - 1)
                flat_lookup = z_coord * height * width + y_coord * width + x_coord
                lookup_coords[xi][yi].append(flat_lookup)
                lookup_result = jnp.batch_gather(flattened_grid, flat_lookup)
                lookup_result = 1.0 * lookup_result
                corners[xi][yi].append(lookup_result)

    alpha_x = alphas[:, :, 0, ...]
    alpha_y = alphas[:, :, 1, ...]
    alpha_z = alphas[:, :, 2, ...]

    one_minus_alpha_x = 1.0 - alpha_x
    one_minus_alpha_y = 1.0 - alpha_y
    # First interpolate a face along x:
    f00 = corners[0][0][0] * one_minus_alpha_x + corners[1][0][0] * alpha_x
    f01 = corners[0][0][1] * one_minus_alpha_x + corners[1][0][1] * alpha_x
    f10 = corners[0][1][0] * one_minus_alpha_x + corners[1][1][0] * alpha_x
    f11 = corners[0][1][1] * one_minus_alpha_x + corners[1][1][1] * alpha_x
    # Next interpolate a long along y:
    l0 = f00 * one_minus_alpha_y + f10 * alpha_y
    l1 = f01 * one_minus_alpha_y + f11 * alpha_y

    # Finally interpolate a point along z:
    p = l0 * (1.0 - alpha_z) + l1 * alpha_z

    assert p.shape == [batch_size, sample_count], "interpolate_from_grid:p"

    p = jnp.reshape(p, [batch_size, sample_count, 1])
    validity_mask = jnp.reshape(validity_mask, [batch_size, sample_count, 1])
    return p, validity_mask
