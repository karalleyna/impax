"""Loss functions for training Structured Implicit Functions."""

import jax.numpy as jnp

from impax.datasets import shapenet
# LDIF is an internal package, should be imported last.
# pylint: disable=g-bad-import-order
from impax.utils import geom_util, interpolate_util, logging_util, sdf_util

# pylint: enable=g-bad-import-order


def bounding_box_constraint_error(samples, box):
    if not isinstance(box.lower, float) and len(box.lower.shape) < 3:
        box.lower = jnp.reshape(box.lower, [1, 1, 3])
        box.upper = jnp.reshape(box.upper, [1, 1, 3])
    lower_error = jnp.maximum(box.lower - samples, 0.0)
    upper_error = jnp.maximum(samples - box.upper, 0.0)
    constraint_error = lower_error * lower_error + upper_error * upper_error
    return constraint_error


def shape_element_center_magnitude_loss(model_config, _, structured_implicit):
    element_centers = structured_implicit.element_centers
    mse = model_config.hparams.cm * jnp.mean(jnp.square(element_centers + 1e-04)) + 1e-5
    return mse


def element_center_lowres_grid_direct_loss(model_config, training_example, structured_implicit):
    element_centers = structured_implicit.element_centers
    gt_sdf_at_centers, _ = interpolate_util.interpolate(
        training_example.grid, element_centers, training_example.world2grid
    )
    mse = model_config.hparams.gd * jnp.mean(gt_sdf_at_centers) + 1e-5
    return mse


def element_center_lowres_grid_squared_loss(model_config, training_example, structured_implicit):
    element_centers = structured_implicit.element_centers
    gt_sdf_at_centers, _ = interpolate_util.interpolate(
        training_example.grid, element_centers, training_example.world2grid
    )
    mse = model_config.hparams.gs * jnp.mean(jnp.sign(gt_sdf_at_centers) * jnp.square(gt_sdf_at_centers + 1e-04)) + 1e-5
    return mse


def element_center_lowres_grid_inside_loss(model_config, training_example, structured_implicit):
    """Loss that element centers should lie within a voxel of the GT inside."""
    element_centers = structured_implicit.element_centers
    gt_sdf_at_centers, _ = interpolate_util.interpolate(
        training_example.grid, element_centers, training_example.world2grid
    )
    gt_sdf_at_centers = jnp.where(gt_sdf_at_centers > model_config.hparams.igt, gt_sdf_at_centers, 0.0)
    mse = model_config.hparams.ig * jnp.mean(jnp.square(gt_sdf_at_centers + 1e-04)) + 1e-05
    return mse


def smooth_element_center_lowres_grid_inside_loss(model_config, training_example, structured_implicit):
    """Offset version of element_center_lowres_grid_inside_loss by voxel width."""
    element_centers = structured_implicit.element_centers
    gt_sdf_at_centers, _ = interpolate_util.interpolate(
        training_example.grid, element_centers, training_example.world2grid
    )
    gt_sdf_at_centers = jnp.maximum(gt_sdf_at_centers - model_config.hparams.igt, 0.0)
    mse = model_config.hparams.ig * jnp.mean(jnp.square(gt_sdf_at_centers + 1e-04)) + 1e-05
    return mse


def center_variance_loss(model_config, training_example, structured_implicit):  # pylint:disable=unused-argument
    """A loss on the -variance of the center locations."""
    # Training example present for interface uniformity
    element_centers = structured_implicit.element_centers
    center_shape = element_centers.shape
    if len(center_shape) != 3:
        raise ValueError(
            f"Expected the element centers to have shape [b, #, 3],"
            f" but they have shape {center_shape}. center_variance."
        )
    variance = jnp.var(element_centers, axis=[1, 2])
    loss_max = model_config.hparams.vt
    loss = model_config.hparams.vw * jnp.mean(jnp.maximum(loss_max - variance, 0.0))
    return loss


def center_nn_loss(model_config, training_example, structured_implicit):  # pylint:disable=unused-argument
    """A loss that decreases with the nearest neighbor center->center distance."""
    # Training example present for interface uniformity
    element_centers = structured_implicit.element_centers

    center_shape = element_centers.shape
    if len(center_shape) != 3:
        raise ValueError(
            f"Expected the element centers to have shape [b, #, 3],"
            f" but they have shape {center_shape}. Loss=center_nn."
        )
    batch_size, center_count, _ = center_shape
    sq_distances = jnp.sum(
        jnp.square(
            jnp.reshape(element_centers, [batch_size, center_count, 1, 3])
            - jnp.reshape(element_centers, [batch_size, 1, center_count, 3])
        ),
        axis=-1,
    )
    distances = jnp.sqrt(sq_distances + 1e-8)
    loss_max = model_config.hparams.nnt
    # We have to give the diagonal self -> self distances a high weight so they
    # aren't valid choices:
    diag_distances = jnp.diag(jnp.ones([center_count]) * (loss_max + 1))
    diag_distances = jnp.reshape(diag_distances, [1, center_count, center_count])
    distances = distances + diag_distances
    min_dists = jnp.min(distances, axis=-1)  # Shape [BS, #].
    assert len(min_dists.shape) == 2

    loss = jnp.mean(jnp.maximum(loss_max - min_dists, 0.0)) * model_config.hparams.nw
    return loss


def inside_box_loss(model_config, _, structured_implicit):
    """Loss that centers should be inside a fixed size bounding box."""
    element_centers = structured_implicit.element_centers
    if model_config.hparams.wm == "f":
        bounding_box = shapenet.BoundingBox(lower=-0.7, upper=0.7)
    elif model_config.hparams.wm == "t":
        bounding_box = shapenet.BoundingBox(
            lower=jnp.array([-0.75, -0.075, -0.75], dtype=jnp.float32),
            upper=jnp.array([0.75, 0.075, 0.75], dtype=jnp.float32),
        )

    if model_config.hparams.rsl != 1.0:
        bounding_box.lower *= model_config.hparams.rsl
        bounding_box.upper *= model_config.hparams.rsl

    bounding_box_error = jnp.mean(bounding_box_constraint_error(element_centers, bounding_box))
    outside_bounding_box_loss = model_config.hparams.ibblw * bounding_box_error
    return outside_bounding_box_loss


def shape_element_center_loss(model_config, training_example, structured_implicit):
    """Loss that centers should be inside the predicted surface."""
    element_centers = structured_implicit.element_centers
    logging_util.log.info("BID0: Shape Element Center Loss.")
    logging_util.log.info("Element Center Shape: %s", str(element_centers.shape))

    class_at_centers, _ = structured_implicit.class_at_samples(element_centers)

    bounding_box = training_example.sample_bounding_box
    bounding_box_error = jnp.mean(bounding_box_constraint_error(element_centers, bounding_box), axis=-1, keep_dims=True)
    center_is_inside_gt_box = bounding_box_error <= 0.0
    inside_prediction_weights = model_config.hparams.cc * jnp.array(center_is_inside_gt_box, dtype=jnp.float32)
    # bounding_box_error has shape [batch_size, center_count, 1]
    # inside_prediction_weights has shape [batch_size, center_count, 1]
    # class_at_centers has shape [batch_size, center_count, 1]. (Double check).

    # The class loss is 0 where the prediction is outside the bounding box,
    # because the bounding box loss is applied to those centers instead.
    class_loss = weighted_l2_loss(0.0, class_at_centers, inside_prediction_weights)

    outside_bounding_box_loss = model_config.hparams.ibblw * bounding_box_error

    final_loss = jnp.mean(class_loss + outside_bounding_box_loss)
    return final_loss


def old_shape_element_center_loss(model_config, training_example, structured_implicit):
    """Deprecated version of shape_element_center_loss()."""
    element_centers = structured_implicit.element_centers
    logging_util.log.info("Element Center Shape: %s", str(element_centers.shape))

    bounding_box = training_example.sample_bounding_box
    bounding_box_error = jnp.mean(bounding_box_constraint_error(element_centers, bounding_box))
    constraint_loss = model_config.hparams.ibblw * bounding_box_error
    class_at_centers, _ = structured_implicit.class_at_samples(element_centers)
    center_loss = jnp.mean((class_at_centers - 0) * (class_at_centers - 0))
    center_loss *= model_config.hparams.cclw
    return constraint_loss + center_loss


def weighted_l2_loss(gt_value, pred_value, weights):
    """Computers an l2 loss given broadcastable weights and inputs."""
    diff = pred_value - gt_value
    squared_diff = diff * diff
    if isinstance(gt_value, float):
        gt_shape = [1]
    else:
        gt_shape = gt_value.shape
    if isinstance(weights, float):
        weight_shape = [1]
    else:
        weight_shape = weights.shape
    logging_util.log.info(
        "gt vs pred vs weights shape: %s vs %s vs %s",
        str(gt_shape),
        str(pred_value.shape),
        str(weight_shape),
    )
    # TODO(kgenova) Consider using tf.losses.mean_squared_error. But need to
    # be careful about reduction method. Theirs is probably better since the
    # magnitude of the loss isn't affected by the weights. But it would need
    # hparam tuning, so it's left out in the first pass.
    return weights * squared_diff


def sample_loss(model_config, gt_sdf, structured_implicit, global_samples, name, apply_ucf):
    """Computes an l2 loss for predicted-vs-gt insidedness at samples."""
    gt_class = sdf_util.apply_class_transfer(gt_sdf, model_config, soft_transfer=False, offset=0.0)

    if model_config.hparams.lrf == "l":
        global_decisions, local_outputs = structured_implicit.class_at_samples(global_samples)
        local_decisions, local_weights = local_outputs
        predicted_class = local_decisions
        gt_class = jnp.tile(jnp.expand_dims(gt_class, axis=1), [1, model_config.hparams.sc, 1, 1])
        weights = local_weights
    elif model_config.hparams.lrf == "g":
        global_decisions, local_outputs = structured_implicit.class_at_samples(global_samples)
        predicted_class = global_decisions
        weights = 1.0
    elif model_config.hparams.lrf == "x":
        # TODO(kgenova) Don't forget we need more samples if lrf='x' than otherwise.
        local_samples, _, local_gt = geom_util.local_views_of_shape(
            global_samples,
            structured_implicit.world2local,
            local_point_count=model_config.hparams.spc,
            global_features=gt_class,
        )
        # This is an important distinction: With lrf='x', the implicit values are
        # required to be a classification decision *on their own*.
        predicted_class = structured_implicit.implicit_values(local_samples)
        gt_class = local_gt
        weights = 1.0
    if apply_ucf:
        is_outside = gt_class > 0.5
        weights *= jnp.where(is_outside, 1.0, model_config.hparams.ucf)
    loss = weighted_l2_loss(gt_class, predicted_class, weights)
    return jnp.mean(loss)


def uniform_sample_loss(model_config, training_example, structured_implicit):
    """Loss that uniformly sampled points should have the right insidedness."""
    sample_count = model_config.hparams.xsc if model_config.hparams.lrf == "x" else model_config.hparams.spc
    samples, gt_sdf = training_example.sample_sdf_uniform(sample_count=sample_count)
    logging_util.log.info("Building Uniform Sample Loss.")
    logging_util.log.info("Uni. Samples shape: %s", str(samples.shape))
    loss = model_config.hparams.l2w * sample_loss(
        model_config, gt_sdf, structured_implicit, samples, "uniform_sample", apply_ucf=True
    )
    return loss


def overlap_loss(model_config, training_example, structured_implicit):
    """A loss on the overlap between RBF weights."""
    sample_count = model_config.hparams.xsc if model_config.hparams.lrf == "x" else model_config.hparams.spc
    samples, _ = training_example.sample_sdf_near_surface(sample_count=sample_count)
    rbf_influences = structured_implicit.rbf_influence_at_samples(samples)
    assert len(rbf_influences.shape) == 3  # [b, sample_count, eec]
    loss = jnp.mean(jnp.linalg.norm(rbf_influences, ord=1, axis=2)) * model_config.hparams.ow
    return loss


def near_surface_sample_loss(model_config, training_example, structured_implicit):
    """An inside/outside loss that samples based on distance to the surface."""
    sample_count = model_config.hparams.xsc if model_config.hparams.lrf == "x" else model_config.hparams.spc
    samples, gt_sdf = training_example.sample_sdf_near_surface(sample_count=sample_count)
    logging_util.log.info("Building Near Surface Sample Loss.")
    logging_util.log.info("NS Samples shape: %s", str(samples.shape))
    # TODO(kgenova) Currently we set ucf=True here because that's how it was...
    # but go back and fix that because it seems bad.
    loss = model_config.hparams.a2w * sample_loss(
        model_config, gt_sdf, structured_implicit, samples, "ns_sample", apply_ucf=True
    )  # False)
    return loss


def compute_loss(model_config, training_example, structured_implicit):
    """Computes the overall loss based on the model configuration."""
    # The keys are kept so short because they are used to autogenerate a
    # tensorboard entry.
    loss_fun_dict = {
        "u": uniform_sample_loss,
        "ns": near_surface_sample_loss,
        "ec": shape_element_center_loss,
        "oc": old_shape_element_center_loss,
        "m": shape_element_center_magnitude_loss,
        "gd": element_center_lowres_grid_direct_loss,
        "gs": element_center_lowres_grid_squared_loss,
        "gi": element_center_lowres_grid_inside_loss,
        "gf": smooth_element_center_lowres_grid_inside_loss,
        "bb": inside_box_loss,
        "xv": center_variance_loss,
        "xp": center_nn_loss,
        "xw": overlap_loss,
    }

    losses = []
    for key, loss_fun in loss_fun_dict.items():
        if key in model_config.hparams.loss:
            loss = loss_fun(model_config, training_example, structured_implicit)
            losses.append(loss)
    # There must be at least one loss:
    assert losses
    return jnp.add(losses)


def set_loss(model_config, training_example, structured_implicit):
    # TODO(kgenova) Consider returning the add_n result as a tensor, setting
    # the loss in the launcher, and having a separate scalar summarizer in
    # summarize.py
    model_config.loss = compute_loss(model_config, training_example, structured_implicit)
