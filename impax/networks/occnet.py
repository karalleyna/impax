"""An implementation of the OccNet architecture."""
# TODO: Discarded the unbatched layers and architectures, discuss if this is enough
import flax.linen as nn
import jax
import jax.numpy as jnp

from impax.utils import math_util  # todo: , net_util

SQRT_EPS = 1e-5


class CBNLayer(nn.Module):
    """Applies conditional batch norm to a batch of sample embeddings.
    The batch norm values are conditioned on shape embedding.
    Args:
        shape_embedding: Tensor with shape [batch_size, shape_embedding_length].
        sample_embeddings: Tensor with shape [batch_size, sample_count,
        sample_embedding_length.
    Returns:
        Tensor with shape [shape_embedding_length].
    """

    sample_embedding_length: int
    is_training: bool = True
    name: str = ""

    @nn.compact
    def __call__(self, shape_embedding, sample_embeddings):
        batch_size = shape_embedding.shape[0]

        assert sample_embeddings.shape[2] == self.sample_embedding_length

        beta = nn.Dense(features=self.sample_embedding_length)(shape_embedding)
        gamma = nn.Dense(features=self.sample_embedding_length)(shape_embedding)

        batch_mean = jnp.mean(sample_embeddings, axis=(1, 2))
        batch_variance = jnp.var(sample_embeddings, axis=(1, 2))

        assert batch_mean.shape == (batch_size,) and batch_variance.shape == (
            batch_size,
        )
        reduced_batch_mean = jnp.mean(batch_mean)
        reduced_batch_variance = jnp.mean(batch_variance)

        running_mean = self.variable(
            "stats", "running_mean_" + self.name + "rmean", init_fn=lambda: 0.0
        )
        running_variance = self.variable(
            "stats", "running_mean_" + self.name + "rvar", init_fn=lambda: 0.0
        )

        if self.is_training:
            running_mean.value = 0.995 * running_mean.value + 0.005 * reduced_batch_mean
            running_variance.value = (
                0.995 * running_variance.value + 0.005 * reduced_batch_variance
            )

        denom = jnp.sqrt(running_variance.value + SQRT_EPS)
        out = (
            gamma[:, None, ...] * ((sample_embeddings - running_mean.value) / denom)
            + beta[:, None, ...]
        )

        return out


class ResnetLayer(nn.Module):
    """Applies a fully connected resnet layer to the input.

    Args:
    shape_embedding: Tensor with shape [batch_size, shape_embedding_length].
    sample_embeddings: Tensor with shape [batch_size, sample_count,
        sample_embedding_length].

    Returns:
    Tensor with shape [sample_count, sample_embedding_length].
    """

    sample_embedding_length: int
    fon: str = "t"
    is_training: bool = True
    name: str = ""
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(self, shape_embedding, sample_embeddings):
        assert self.sample_embedding_length == sample_embeddings.shape[2]
        init_sample_embeddings = sample_embeddings
        sample_embeddings = CBNLayer(
            self.sample_embedding_length, self.is_training, self.name + "_cbn"
        )(shape_embedding, sample_embeddings)

        if self.fon == "t":
            init_sample_embeddings = sample_embeddings

        sample_embeddings = self.activation(sample_embeddings)
        sample_embeddings = nn.Dense(features=self.sample_embedding_length)(
            sample_embeddings
        )
        sample_embeddings = CBNLayer(
            sample_embedding_length=self.sample_embedding_length,
            is_training=self.is_training,
            name=self.name + "_cbn2",
        )(shape_embedding, sample_embeddings)
        sample_embeddings = self.activation(sample_embeddings)
        sample_embeddings = CBNLayer(
            sample_embedding_length=self.sample_embedding_length,
            is_training=self.is_training,
            name=self.name + "_cbn3",
        )(shape_embedding, sample_embeddings)
        return init_sample_embeddings + sample_embeddings


class Decoder(nn.Module):
    """Computes the OccNet output for the input embedding and its sample batch.

    Args:
    embedding: Tensor with shape [shape_embedding_length].
    samples: Tensor with shape [sample_count, 3].
    apply_sigmoid: Boolean. Whether to apply a sigmoid layer to the final linear
        activations.
    model_config: A ModelConfig object.

    Returns:
    Tensor with shape [sample_count, 1].
    """

    sample_embedding_length: int  # model_config.hparams.ips
    resnet_layer_count: int = 1  # model_config.hparams.orc
    apply_sigmoid: bool = True
    fon: str = "t"  # model_config.hparams.orc
    is_training: bool = True
    name: str = ""
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(self, embedding, samples):
        assert len(embedding.shape) == 2 and len(samples.shape) == 3

        batch_size, sample_count, _ = samples.shape

        sample_embeddings = nn.Dense(self.sample_embedding_length)(samples)

        assert sample_embeddings.shape == (
            batch_size,
            sample_count,
            self.sample_embedding_length,
        )
        for i in range(self.resnet_layer_count):
            sample_embeddings = ResnetLayer(
                sample_embedding_length=self.sample_embedding_length,
                fon=self.fon,
                is_training=self.is_training,
                name=self.name + f"_resnet_{i}",
                activation=self.activation,
            )(embedding, sample_embeddings)
        sample_embeddings = CBNLayer(
            sample_embedding_length=self.sample_embedding_length,
            is_training=self.is_training,
            name=self.name + "_cb1",
        )(embedding, sample_embeddings)
        vals = nn.Dense(1)(sample_embeddings)
        if self.apply_sigmoid:
            vals = nn.sigmoid(vals)
        return vals


class OCCNet(nn.Module):
    sample_embedding_length: int  # model_config.hparams.ips
    resnet_layer_count: int = 1  # model_config.hparams.orc
    is_training: bool = True
    apply_sigmoid: bool = False
    activation: nn.Module = nn.relu

    fon: str = "t"  # model_config.hparams.orc
    hyo: str = ""  # model_config.hparams.hyo
    dd: str = ""  # model_config.hparams.dd

    @nn.compact
    def __call__(self, embedding, samples):
        """Computes the OccNet output for the input embedding and its sample batch.

        Args:
        embedding: Tensor with shape [batch_size, shape_embedding_length].
        samples: Tensor with shape [batch_size, sample_count, 3].
        apply_sigmoid: Boolean. Whether to apply a sigmoid layer to the final linear
            activations.

        Returns:
        Tensor with shape [batch_size, sample_count, 1].
        """
        if self.hyo == "t":
            samples = math_util.nerfify(samples, 10, flatten=True, interleave=True)
            sample_len = 60
        else:
            sample_len = 3
        if self.dd == "t":
            print(
                "BID0: Running SS Occnet Decoder with input shapes embedding=%s, samples=%s",
                repr(embedding.shape),
                repr(samples.shape),
            )
            assert (
                embedding.shape[0] == 1
                and samples.shape[0] == 1
                and samples.shape[2] == 3
            )

            vals = Decoder(
                self.sample_embedding_length,
                self.resnet_layer_count,
                self.apply_sigmoid,
                self.fon,
                self.is_training,
                name="occnet1",
                activation=self.activation,
            )(embedding, samples)
            return vals

        batch_size, _ = embedding.shape
        assert samples.shape[0] == batch_size and samples.shape[2] == sample_len
        vals = Decoder(
            self.sample_embedding_length,
            self.resnet_layer_count,
            self.apply_sigmoid,
            self.fon,
            self.is_training,
            name="occnet2",
            activation=self.activation,
        )(embedding, samples)
        return vals


class Encoder(nn.Module):
    # TODO: How to use this?
    feature_length: int = 256

    @nn.compact
    def __call__(self, inputs):
        assert len(inputs.shape) == 5
        return jnp.ones((inputs.shape[:4] + [self.feature_length]))

        # return net_util.inputs_to_feature_vector(inputs, self.feature_length)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    occnet = OCCNet(32, 3, True, True, dd="t")
    occvars = occnet.init(key, jnp.ones((1, 32)), jnp.ones((1, 48, 3)))
    print(occvars)
