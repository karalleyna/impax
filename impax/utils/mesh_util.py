"""
Utilities for working with meshes.
References:
https://github.com/google/ldif/blob/master/ldif/util/mesh_util.py
"""

import io

import trimesh


def serialize(mesh):
    mesh_str = trimesh.exchange.ply.export_ply(
        mesh, encoding="binary", vertex_normal=False
    )
    return mesh_str


def deserialize(mesh_str):
    mesh_ply_file_obj = io.BytesIO(mesh_str)
    mesh = trimesh.Trimesh(**trimesh.exchange.ply.load_ply(mesh_ply_file_obj))
    return mesh


def remove_small_components(mesh, min_volume=5e-05):
    """Removes all components with volume below the specified threshold."""
    if mesh.is_empty:
        return mesh
    out = [m for m in mesh.split(only_watertight=False) if m.volume > min_volume]
    if not out:
        return mesh
    return trimesh.util.concatenate(out)
