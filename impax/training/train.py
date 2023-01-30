"""
This script trains a SIF model.
The data is loaded using tensorflow_datasets.
"""

import sys
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
from flax.training import checkpoints, train_state
from jax import random
from tqdm import tqdm

from impax.configs.sif import get_config
from impax.datasets.local_inputs import _make_optimized_dataset
from impax.datasets.preprocess import preprocess
from impax.inference import example
from impax.models.model import StructuredImplicitModel
from impax.models.observation import Observation
from impax.training.loss import compute_loss
from impax.utils import gaps_util
from impax.utils.logging_util import log


def visualize_data(dataset, split):
    """Visualizes the dataset with two interactive visualizer windows."""
    (
        bounding_box_samples,
        depth_renders,
        mesh_name,
        near_surface_samples,
        grid,
        world2grid,
        surface_point_samples,
    ) = [
            jnp.array(dataset.bounding_box_samples[0]),
            jnp.array(dataset.depth_renders[0]),
            dataset.mesh_name.numpy()[0],
            jnp.array(dataset.near_surface_samples[0]),
            jnp.array(dataset.grid[0]),
            jnp.array(dataset.world2grid[0]),
            jnp.array(dataset.surface_point_samples[0]),
        ]
    
    gaps_util.ptsview([bounding_box_samples, near_surface_samples, surface_point_samples])
    mesh_name = mesh_name.decode(sys.getdefaultencoding())
    log.info(f"depth max: {jnp.max(depth_renders)}")
    log.info(f"Mesh name: {mesh_name}")
    assert "|" in mesh_name
    mesh_hash = mesh_name[mesh_name.find("|") + 1:]
    log.info(f"Mesh hash: {mesh_hash}")
    dyn_obj = example.InferenceExample(split, "airplane", mesh_hash, dynamic=True)

    gaps_util.gapsview(
        msh=dyn_obj.normalized_gt_mesh,
        pts=near_surface_samples[:, :3],
        grd=grid,
        world2grid=world2grid,
        grid_threshold=-0.07,
    )


def create_model(model_config, key):
    sif = StructuredImplicitModel(model_config, key)
    return sif


def initialized(key, model, model_config):
    dataset = _make_optimized_dataset(
        "/Users/burak/Desktop/repos/impax/impax/data2", model_config.batch_size, "train", "train", key=key
    )
    data = preprocess(model_config, next(dataset)[0], "train", key)
    observation = Observation(model_config, data)

    variables = model.init(key, observation)

    return variables["params"], variables["batch_stats"]


def get_optimizer(model_config):
    """Sets the train op for a single weights update."""
    optimizer = model_config.optimizer(model_config.learning_rate)
    return optimizer


def train_step(state, batch, model, model_config):
    """Perform a single training step."""
    obs, training_ex = batch
    model_config.train = True

    def loss_fn(params):
        """loss function used for training."""
        prediction, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            obs,
            mutable=["batch_stats"],
        )
        loss = compute_loss(model_config, training_ex, prediction.structured_implicit)
        # weight_penalty_params = jax.tree_util.tree_leaves(params)
        # weight_decay = 0.0001
        # weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
        # weight_penalty = weight_decay * 0.5 * weight_l2
        # loss = loss + weight_penalty
        return loss, (new_model_state, prediction)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.

    new_model_state, predictions = aux[1]

    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
    return new_state, aux[0]


def eval_step(state, batch, model_config):
    model_config.train = False

    model_config.batch_size = 1
    obs, training_ex = batch
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    predictions = state.apply_fn(variables, obs, mutable=False)
    loss = compute_loss(model_config, training_ex, predictions.structured_implicit)
    return loss


class TrainState(train_state.TrainState):
    batch_stats: Any


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


def create_train_state(rng, model_config: ml_collections.ConfigDict, model):
    """Create initial training state."""

    params, batch_stats = initialized(rng, model, model_config)
    tx = model_config.optimizer(model_config.learning_rate)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
    return state


def train_and_evaluate(model_config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      Final TrainState.
    """
    
    vis = True

    key = random.PRNGKey(0)

    key, train_data_key, eval_data_key = random.split(key, 3)

    train_dataset = _make_optimized_dataset(
        "/Users/burak/Desktop/repos/impax/impax/data2", model_config.batch_size, "train", "train", train_data_key
    )
    train_iter = map(lambda x_key: preprocess(model_config, x_key[0], split="train", key=x_key[1]), train_dataset)
    train_iter = map(lambda x: (Observation(model_config, x), x), train_iter)

    if vis:
        visualize_data(next(train_dataset)[0], "train")
    
    eval_dataset = _make_optimized_dataset(
        "/Users/burak/Desktop/repos/impax/impax/data2", 1, "eval", "val", eval_data_key
    )
    eval_iter = map(lambda x_key: preprocess(model_config, x_key[0], split="eval", key=x_key[1]), eval_dataset)
    eval_iter = map(lambda x: (Observation(model_config, x), x), eval_iter)

    # todo: do
    steps_per_epoch = 467 // model_config.batch_size
    num_steps = int(steps_per_epoch * 1)
    steps_per_eval = 79 // 1

    steps_per_checkpoint = steps_per_epoch * 10

    key, create_key = random.split(key)
    model = create_model(model_config, create_key)

    key, init_key = random.split(key)
    state = create_train_state(init_key, model_config, model)
    state = restore_checkpoint(state, workdir)

    model_config.train = True
    log.warning("Initial compilation, this might take some minutes...")

    for epoch in range(model_config.n_epochs):
        pbar: tqdm = tqdm(zip(range(0, steps_per_epoch), train_iter), total=steps_per_epoch)
        train_loss = []

        for step, batch in pbar:
            state, loss = train_step(state, batch, model, model_config)

            train_loss.append(loss)
            if (step + 1) % model_config.log_every_steps == 0:
                pbar.set_description(f"Epoch: {epoch} ")
                pbar.set_postfix({"train loss": sum(train_loss) / len(train_loss)})

            if (step + 1) % steps_per_epoch == 0:
                epoch = step // steps_per_epoch
                eval_loss = []

                # sync batch statistics across replicas
                pbs = model_config.batch_size
                model_config.batch_size = 1
                for _ in range(steps_per_eval):
                    eval_batch = next(eval_iter)
                    loss = eval_step(state, eval_batch, model_config)
                    eval_loss.append(loss)
                model_config.batch_size = pbs

                pbar.set_postfix(
                    {
                        "train loss": sum(train_loss[-steps_per_epoch:]) / steps_per_epoch,
                        "eval loss": sum(eval_loss) / len(eval_loss),
                    }
                )

            if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
                save_checkpoint(state, workdir)

    return state


if __name__ == "__main__":
    cfg = get_config()
    state = train_and_evaluate(cfg, "./output/")
