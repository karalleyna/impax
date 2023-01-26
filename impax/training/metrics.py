"""Metrics for evaluating structured implicit functions."""

import jax.numpy as jnp

# local
from impax.utils import sdf_util


def point_iou(structured_implicit, sample_locations, sample_gt, model_config):
    """Estimates the mesh iou by taking the iou from uniform point samples."""
    assert model_config.hparams.bs == 1  # Otherwise the result would be wrong

    pred_class, _ = structured_implicit.class_at_samples(sample_locations)

    gt_is_inside = jnp.logical_not(
        sdf_util.apply_class_transfer(
            sample_gt, model_config, soft_transfer=False, offset=0.0, dtype=bool
        )
    )
    pred_is_inside = pred_class < 0.5
    intersection = jnp.logical_and(gt_is_inside, pred_is_inside).astype(jnp.float32)
    union = jnp.logical_or(gt_is_inside, pred_is_inside).astype(jnp.float32)

    iou = jnp.divide(jnp.sum(intersection), jnp.sum(union) + 1e-05)
    return iou
