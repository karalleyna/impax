"""Renders depth scans of an input mesh for the depth completion task."""

import imageio
import jax.numpy as jnp
import numpy as np
import pyrender
import trimesh
from absl import app, flags
from jax import random

from impax.utils import camera_util

FLAGS = flags.FLAGS

flags.DEFINE_string("input_mesh", None, "The path to the input mesh")
flags.DEFINE_string("output_npz", None, "The path to the output .npz")
flags.DEFINE_integer("height", 512, "The image height")
flags.DEFINE_integer("width", 512, "The image width")
flags.DEFINE_float("yfov", jnp.pi / 3.0, "The y field-of-view")


# The context is OpenGL- it's global and not threadsafe.
# Note that this context still requires a headed server even those it's
# offscreen. If you need a headless one, you can switch to egl/osmesa with
# pyrender via environment variables, but those can be trickier to setup.
# os.environ['PYOPENGL_PLATFORM'] = 'egl'


def render_depth_image(mesh, cam2world, context):
    scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_node = pyrender.Node(mesh=pyr_mesh, matrix=np.eye(4))
    scene.add_node(mesh_node)
    ar = FLAGS.width / float(FLAGS.height)
    cam = pyrender.PerspectiveCamera(yfov=FLAGS.yfov, aspectRatio=ar)
    scene.add_node(pyrender.Node(camera=cam, matrix=cam2world))
    color, depth = context.render(scene)
    return depth


def save_depth_image(depth, path):
    # depth = (depth * 1000).astype(np.uint16)
    imageio.imwrite(path, depth)


def clips_edge(depth):
    assert depth.shape[0] == FLAGS.height
    assert depth.shape[1] == FLAGS.width
    top_row = depth[0, :]
    bottom_row = depth[FLAGS.height - 1, :]
    left_col = depth[:, 0]
    right_col = depth[:, FLAGS.width - 1]
    edges = jnp.concatenate([top_row, bottom_row, left_col, right_col])
    return jnp.any(edges)


def get_cam2world(center, eye, world_up):
    eye = jnp.reshape(eye, [1, 3])
    world_up = jnp.reshape(world_up, [1, 3])
    center = jnp.reshape(center, [1, 3])
    world2cam = camera_util.look_at_np(eye=eye, center=center, world_up=world_up)
    return jnp.linalg.inv(world2cam[0, ...])


def find_critical_radius(
    mesh, dir_to_eye, center, world_up, context, min_radius=0.3, max_radius=3.0, iterations=10, fallback_radius=1.5
):
    def radius_clips(radius):
        cam2world = get_cam2world(eye=dir_to_eye * radius, center=center, world_up=world_up)
        depth = render_depth_image(mesh, cam2world, context)
        return clips_edge(depth)

    if radius_clips(max_radius) or not radius_clips(min_radius):
        return fallback_radius  # Critical radius is outside our bounded search.
    for i in range(iterations):
        midpoint = (min_radius + max_radius) / 2.0
        if radius_clips(midpoint):
            min_radius = midpoint
        else:
            max_radius = midpoint
    return (min_radius + max_radius) / 2.0


def sample_depth_image(mesh, context, key=random.PRNGKey(0)):
    key, key2, key3 = random.split(key, 3)

    center = random.normal(key, shape=(3,)).astype(jnp.float32) * 0.05
    world_up = jnp.array([0, 1, 0], dtype=jnp.float32)

    dir_to_eye = random.normal(key2, shape=(3,)).astype(jnp.float32) - 0.5
    dir_to_eye /= jnp.linalg.norm(dir_to_eye)

    critical_radius = find_critical_radius(mesh, dir_to_eye, center, world_up, context)
    radius_offset = random.normal(key3, shape=(1,)).astype(jnp.float32) * 0.2
    radius = critical_radius + radius_offset
    eye = dir_to_eye * radius
    cam2world = get_cam2world(eye=eye, center=center, world_up=world_up)
    return render_depth_image(mesh, cam2world, context), cam2world


def sample_depth_images(mesh, context, n_images=16):
    images = []
    cam2worlds = []
    for i in range(n_images):
        depth, cam2world = sample_depth_image(mesh, context)
        images.append(depth)
        cam2worlds.append(cam2world)
    return jnp.stack(images), jnp.stack(cam2worlds)


def get_projection_matrix():
    ar = FLAGS.width / float(FLAGS.height)
    cam = pyrender.PerspectiveCamera(yfov=FLAGS.yfov, aspectRatio=ar)
    return cam.get_projection_matrix()


def main(_):
    context = pyrender.OffscreenRenderer(viewport_width=FLAGS.width, viewport_height=FLAGS.height, point_size=1.0)
    mesh = trimesh.load(FLAGS.input_mesh)
    n_images_per_mesh = 16
    images, cam2world = sample_depth_images(mesh, context, n_images_per_mesh)
    projection_matrix = get_projection_matrix()
    # If space is a concern (currently ~5mb/shape), you could quantize the
    # depth images (e.g., to 16-bit pngs storing depth in millimeters) and
    # save them that way.
    np.savez_compressed(
        FLAGS.output_npz,
        {"depth": np.array(images), "cam2world": np.array(cam2world), "projection": np.array(projection_matrix)},
    )
    # If you want to visualize:
    # for i in range(n_images_per_mesh):
    #   save_depth_image(images[i, ...], f'./test-{i}.png')


if __name__ == "__main__":
    app.run(main)
