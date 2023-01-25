import pytest
from jax import random
import jax.numpy as jnp
import tensorflow as tf

import ldif.ldif.util.geom_util as original
from impax.utils import geom_util


def test_ray_sphere_intersect(
    batch_size=4, ray_count=2, max_t=1.0, key=random.PRNGKey(0)
):
    key0, key1, key2, key3 = random.split(key, 4)

    ray_start = random.uniform(key0, shape=(batch_size, ray_count, 3))
    ray_direction = random.uniform(key1, shape=(batch_size, ray_count, 3))
    sphere_center = random.uniform(key2, shape=(batch_size, ray_count, 3))
    sphere_radius = random.uniform(key3, shape=(batch_size, ray_count, 1))

    ground_truth = original.ray_sphere_intersect(
        tf.convert_to_tensor(ray_start),
        tf.convert_to_tensor(ray_direction),
        tf.convert_to_tensor(sphere_center),
        tf.convert_to_tensor(sphere_radius),
        max_t,
    )

    ret = geom_util.ray_sphere_intersect(
        ray_start, ray_direction, sphere_center, sphere_radius, max_t
    )
    assert ret.shape == ground_truth.shape
    assert jnp.allclose(ret, ground_truth.numpy())


@pytest.mark.parametrize("is_point", [True, False])
def test_to_homogeneous(is_point, k=4, key=random.PRNGKey(0)):

    shape = (2, 3)
    x = random.uniform(key, shape=(*shape, k))

    ground_truth = original.to_homogeneous(tf.convert_to_tensor(x), is_point)

    ret = geom_util.to_homogeneous(x, is_point)
    assert ret.shape == ground_truth.shape
    assert jnp.allclose(ret, ground_truth.numpy())


@pytest.mark.parametrize("k", [3, 6])
@pytest.mark.parametrize("is_normals_none", [True, False])
def test_transform_points_with_normals(
    k, is_normals_none, batch_size=2, point_count=4, key=random.PRNGKey(0)
):
    key0, key1, key2 = random.split(key, 3)

    x = random.uniform(key0, shape=(batch_size, point_count, k))
    tx = random.normal(key1, shape=(batch_size, 4, 4))

    if is_normals_none:
        normals = None
    else:
        normals = random.uniform(key2, shape=(batch_size, point_count, 3))

    ground_truth = original.transform_points_with_normals(
        tf.convert_to_tensor(x),
        tf.convert_to_tensor(tx),
        None if is_normals_none else tf.convert_to_tensor(normals),
    )

    ret = geom_util.transform_points_with_normals(x, tx, normals)
    for x, y in zip(ret, ground_truth):
        assert jnp.allclose(x, y.numpy())


def test_transform_featured_points(
    batch_size=2, point_count=4, feature_count=3, key=random.PRNGKey(0)
):
    key0, key1 = random.split(key)

    x = random.uniform(key0, shape=(batch_size, point_count, 3 + feature_count))
    tx = random.normal(key1, shape=(batch_size, 4, 4))

    ground_truth = original.transform_featured_points(
        tf.convert_to_tensor(x),
        tf.convert_to_tensor(tx),
    )

    ret = geom_util.transform_featured_points(x, tx)
    assert ret.shape == ground_truth.shape
    assert jnp.allclose(ret, ground_truth.numpy())


def test_rotation_to_tx(key=random.PRNGKey(0)):
    x = random.uniform(key, shape=(3, 3))
    ground_truth = original.rotation_to_tx(
        tf.convert_to_tensor(x),
    )

    ret = geom_util.rotation_to_tx(x)
    assert ret.shape == ground_truth.shape
    assert jnp.allclose(ret, ground_truth.numpy())


@pytest.mark.parametrize("is_features_none", [True, False])
def test_extract_points_near_origin(
    is_features_none,
    batch_size=2,
    point_count=4,
    feature_count=4,
    key=random.PRNGKey(0),
):
    key0, key1 = random.split(key)

    x = random.uniform(key0, shape=(batch_size, point_count, 3))

    if is_features_none:
        features = None
    else:
        features = random.uniform(key1, shape=(batch_size, point_count, 3))

    ground_truth = original.extract_points_near_origin(
        tf.convert_to_tensor(x),
        feature_count,
        None if is_features_none else tf.convert_to_tensor(features),
    )

    ret = geom_util.extract_points_near_origin(x, feature_count, features)

    if isinstance(ground_truth, tuple):
        for r, g in zip(ret, ground_truth):
            assert r.shape == g.shape
            assert jnp.allclose(r, g.numpy())
    else:
        assert ret.shape == ground_truth.shape
        assert jnp.allclose(ret, ground_truth.numpy(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "dim",
    [
        3,
    ],
)
def test_chamfer_distance(
    dim, pred_point_count=2, target_point_count=4, key=random.PRNGKey(0)
):
    key0, key1 = random.split(key)
    x1 = random.uniform(key0, shape=(5, pred_point_count, dim))
    x2 = random.uniform(key1, shape=(5, target_point_count, dim))

    dist1_tf, dist2_tf = original.chamfer_distance(
        tf.convert_to_tensor(x1),
        tf.convert_to_tensor(x2),
    )

    dist1, dist2 = geom_util.chamfer_distance(x1, x2)
    assert dist1.shape == dist1_tf.shape
    assert jnp.allclose(dist1, dist1_tf.numpy())

    assert dist2.shape == dist2_tf.shape
    assert jnp.allclose(dist2, dist2_tf.numpy())


@pytest.mark.parametrize(
    "idx",
    [2, 4, 6],
)
def test_dodeca_index(idx):
    viewpoint_tf, centroid_tf, world_up_tf = original.dodeca_parameters(idx)

    viewpoint, centroid, world_up = geom_util.dodeca_parameters(idx)
    assert viewpoint.shape == viewpoint_tf.shape
    assert jnp.allclose(viewpoint, viewpoint_tf)

    assert centroid.shape == centroid_tf.shape
    assert jnp.allclose(centroid, centroid_tf)

    assert world_up.shape == world_up_tf.shape
    assert jnp.allclose(world_up, world_up_tf)


def test_get_dodeca_camera_to_worlds():
    ground_truth = original.get_dodeca_camera_to_worlds()
    ret = geom_util.get_dodeca_camera_to_worlds()

    assert ret.shape == ground_truth.shape
    assert jnp.allclose(ret, ground_truth)


def test_gaps_depth_render_to_xyz(
    batch_size=4,
    height=2,
    width=2,
    depth=1,
    key=random.PRNGKey(0),
):
    x = random.uniform(key, shape=(batch_size, height, width, depth))
    ground_truth = original.gaps_depth_render_to_xyz(
        tf.convert_to_tensor(x), batch_size, 0
    )

    ret = geom_util.gaps_depth_render_to_xyz(x, batch_size, 0)
    assert ret.shape == ground_truth.shape
    assert jnp.allclose(ret, ground_truth.numpy())


def test_angle_of_rotation_to_2d_rotation_matrix(rotation_angle=jnp.pi * 2 / 3):
    ground_truth = original.angle_of_rotation_to_2d_rotation_matrix(rotation_angle)
    ret = geom_util.angle_of_rotation_to_2d_rotation_matrix(rotation_angle)
    assert ret.shape == ground_truth.shape
    assert jnp.allclose(ret, ground_truth.numpy())


def test_depth_image_to_xyz_image(
    batch_size=4,
    height=3,
    width=3,
    xfov=0.5,
    key=random.PRNGKey(0),
):
    depth_images = random.uniform(key, shape=(batch_size, height, width, 1))
    world_to_camera = jnp.eye(4)
    ground_truth = original.depth_image_to_xyz_image(
        tf.convert_to_tensor(depth_images),
        tf.convert_to_tensor(world_to_camera),
        xfov=xfov,
    )

    ret = geom_util.depth_image_to_xyz_image(depth_images, world_to_camera, xfov=xfov)
    if isinstance(ground_truth, tuple):
        for r, g in zip(ret, ground_truth):
            assert r.shape == g.shape
            assert jnp.allclose(r, g.numpy())
    else:
        assert ret.shape == ground_truth.shape
        assert jnp.allclose(ret, ground_truth.numpy())


@pytest.mark.parametrize("is_normals_none", [True, False])
@pytest.mark.parametrize("is_features_none", [True, False])
@pytest.mark.parametrize("is_zeros_invalid", [True, False])
@pytest.mark.parametrize("expand_region", [True, False])
@pytest.mark.parametrize("threshold", [0, 1])
def test_local_views_of_shape(
    is_normals_none,
    is_features_none,
    is_zeros_invalid,
    expand_region,
    threshold,
    batch_size=2,
    num_global_points=4,
    num_frames=3,
    num_local_points=5,
    num_features=4,
    zero_threshold=1e-6,
    key=random.PRNGKey(0),
):
    key0, key1, key2, key3 = random.split(key, 4)
    # global_points: Array with shape [batch_size, global_point_count, 3]
    global_points = random.uniform(key0, shape=(batch_size, num_global_points, 3))
    # world2local: Array with shape [batch_size, frame_count, 4, 4]
    world2local = random.uniform(key1, shape=(batch_size, num_frames, 4, 4))
    if is_normals_none:
        # global_normals: Array with shape [batch_size, global_point_count, 3
        global_normals = random.uniform(key2, shape=(batch_size, num_global_points, 3))

    if is_features_none:
        # global_features: Array with shape [batch_size, global_point_count,
        # feature_count]
        global_features = random.uniform(
            key3, shape=(batch_size, num_global_points, num_features)
        )

    ground_truth = original.local_views_of_shape(
        tf.convert_to_tensor(global_points),
        tf.convert_to_tensor(world2local),
        num_local_points,
        global_normals=global_normals,
        global_features=global_features,
        zeros_invalid=is_zeros_invalid,
        zero_threshold=zero_threshold,
        expand_region=expand_region,
        threshold=threshold
    )

    ret = geom_util.local_views_of_shape(
        global_points,
        world2local,
        num_local_points,
        global_normals=global_normals,
        global_features=global_features,
        zeros_invalid=is_zeros_invalid,
        zero_threshold=zero_threshold,
        expand_region=expand_region,
        threshold=threshold
    )

    if isinstance(ground_truth, tuple):
        for r, g in zip(ret, ground_truth):
            assert r.shape == g.shape
            assert jnp.allclose(r, g.numpy())
    else:
        assert ret.shape == ground_truth.shape
        assert jnp.allclose(ret, ground_truth.numpy())


def test_interpolate_from_grid_coordinates(
    batch_size=4,
    height=3,
    width=3,
    sample_count=2,
    length=3,
    key=random.PRNGKey(0),
):
    key0, key1 = random.split(key)
    samples_shape = (batch_size, sample_count, 3)
    grid_shape = (
        batch_size,
        length,
        height,
        width,
        1,
    )
    samples = random.uniform(key0, shape=samples_shape)
    grid = random.uniform(key1, shape=grid_shape)
    ground_truth = original.interpolate_from_grid_coordinates(
        tf.convert_to_tensor(samples),
        tf.convert_to_tensor(grid),
    )

    ret = geom_util.interpolate_from_grid_coordinates(samples, grid)
    if isinstance(ground_truth, tuple):
        for r, g in zip(ret, ground_truth):
            assert r.shape == g.shape
            assert jnp.allclose(r, g.numpy())
    else:
        assert ret.shape == ground_truth.shape
        assert jnp.allclose(ret, ground_truth.numpy())
