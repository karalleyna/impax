"""
This script trains a SIF model.
The data is loaded using tensorflow_datasets.
"""

from typing import Any

import jax
import ml_collections
from flax.training import checkpoints, train_state
from jax import random

from impax.configs.ldif import get_config
from impax.datasets.local_inputs import _make_optimized_dataset
from impax.datasets.preprocess import preprocess
from impax.models.model import StructuredImplicitModel
from impax.models.observation import Observation
from impax.training.loss import compute_loss
from impax.utils.logging_util import log


def create_model(model_config, key):
    sif = StructuredImplicitModel(model_config, key)
    return sif


def initialized(key, model, model_config):
    dataset = _make_optimized_dataset(
        "/Users/burak/Desktop/repos/impax/impax/data2", model_config.batch_size, "train", "train"
    )
    data = preprocess(model_config, next(dataset), "train")
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

    key = random.PRNGKey(0)

    train_iter = _make_optimized_dataset(
        "/Users/burak/Desktop/repos/impax/impax/data2", model_config.batch_size, "train", "train"
    )
    train_iter = map(lambda x: preprocess(model_config, x, split="train"), train_iter)
    train_iter = map(lambda x: (Observation(model_config, x), x), train_iter)

    eval_iter = _make_optimized_dataset("/Users/burak/Desktop/repos/impax/impax/data2", 1, "eval", "val")
    eval_iter = map(lambda x: preprocess(model_config, x, split="eval"), eval_iter)
    eval_iter = map(lambda x: (Observation(model_config, x), x), eval_iter)

    # todo: do
    steps_per_epoch = 24 // model_config.batch_size
    num_steps = int(steps_per_epoch * 1)
    steps_per_eval = 2 // 1

    steps_per_checkpoint = steps_per_epoch * 10

    key, create_key = random.split(key)
    model = create_model(model_config, create_key)

    key, init_key = random.split(key)
    state = create_train_state(init_key, model_config, model)
    state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)

    train_loss = []
    model_config.train = True
    log.warning("Initial compilation, this might take some minutes...")
    for step, batch in zip(range(step_offset, num_steps), train_iter):
        state, loss = train_step(state, batch, model, model_config)

        if step == step_offset:
            log.info("Initial compilation completed.")

        train_loss.append(loss)
        if (step + 1) % model_config.log_every_steps == 0:
            log.warning(
                "eval epoch: %d, loss: %.4f",
                step,
                sum(train_loss) / len(train_loss),
            )

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_loss = []

            # sync batch statistics across replicas
            for _ in range(steps_per_eval):
                eval_batch = next(eval_iter)
                loss = eval_step(state, eval_batch, model_config)
                eval_loss.append(loss)

            log.warning(
                "eval epoch: %d, loss: %.4f",
                epoch,
                sum(eval_loss) / len(eval_loss),
            )

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            save_checkpoint(state, workdir)

    return state


if __name__ == "__main__":
    cfg = get_config()
    state = train_and_evaluate(cfg, "./output/")
