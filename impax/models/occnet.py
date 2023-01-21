"""An implementation of the OccNet architecture."""

import flax.linen as nn
import jax.numpy as jnp

from impax.utils import math_util, net_util

SQRT_EPS = 1e-5


def ensure_rank(tensor, rank):
    real_shape = tensor.shape
    real_rank = len(real_shape)
    if real_rank != rank:
        raise ValueError(
            "Expected a tensor with rank %i, but given a tensor "
            "with shape %s" % (real_rank, str(real_shape))
        )


def ensure_dims_match(t1, t2, dims):
    if isinstance(dims, int):
        dims = [dims]
    t1_shape = t1.shape
    t2_shape = t2.shape
    for dim in dims:
        if t1_shape[dim] != t2_shape[dim]:
            raise ValueError(
                "Expected tensors %s and %s to match along dims %s"
                % (str(t1_shape), str(t2_shape), str(dims))
            )


def ensure_shape(tensor, shape):
    """Raises a ValueError if the input tensor doesn't have the expected shape."""
    real_shape = tensor.shape
    failing = False
    if len(real_shape) != len(shape):
        failing = True
    if not failing:
        for dim, si in enumerate(shape):
            if si != -1 and si != real_shape[dim]:
                failing = True
    if failing:
        raise ValueError(
            "Expected tensor with shape %s to have shape %s."
            % (str(real_shape), str(shape))
        )


def dim_needs_broadcasting(a, b):
    return a != b and (a == 1 or b == 1)


def subset(shape, dims):
    """Returns the dims-th elements of shape."""
    out = []
    for dim in dims:
        out.append(shape[dim])
    return out


def remove_batch_dim(tensor):
    shape = tensor.shape
    assert shape[0] == 1
    return tensor[0]


def add_batch_dim(tensor):
    return tensor[None, ...]


def shapes_equal(a, b, dims=None):
    if dims is None:
        dims = list(range(len(a)))
    a_shape = subset(a.get_shape().as_list(), dims)
    b_shape = subset(b.get_shape().as_list(), dims)
    for sa, sb in zip(a_shape, b_shape):
        if sa != sb:
            return False
    return True


def ensure_is_scalar(t):
    if not jnp.is_scalar(t):
        s = t.shape
        is_scalar = len(s) == 1 and s[0] == 1
        if not is_scalar:
            raise ValueError("Expected tensor with shape %s to be a scalar tensor." % s)


def broadcast_if_necessary(a, b, dims):
    """Tiles shapes as necessary to match along a list of dims."""
    needs_broadcasting = False
    a_shape = a.shape
    b_shape = b.shape
    a_final_shape = []
    b_final_shape = []
    assert len(a_shape) == len(b_shape)
    for dim in range(len(a_shape)):
        if dim in dims and dim_needs_broadcasting(a_shape[dim], b_shape[dim]):
            needs_broadcasting = True
            dim_len = max(a_shape[dim], b_shape[dim])
            a_final_shape.append(dim_len)
            b_final_shape.append(dim_len)
        else:
            a_final_shape.append(a_shape[dim])
            b_final_shape.append(b_shape[dim])
    for dim in dims:
        if dim_needs_broadcasting(a_shape[dim], b_shape[dim]):
            needs_broadcasting = True
    if not needs_broadcasting:
        return a, b
    a = jnp.broadcast_to(a, a_final_shape)
    b = jnp.broadcast_to(b, b_final_shape)
    return a, b


class OCCNet(nn.Module):
    model_config: dict
    sample_embedding_length: int = model_config.hparams.ips
    running_mean: float = 0.0
    running_variance: float = 0.0
    resnet_layer_count: int = 1
    apply_sigmoid: bool = False

    def setup(self):
        self.beta_fc = nn.Dense(self.sample_embedding_length)
        self.gamma_fc = nn.Dense(self.sample_embedding_length)
        self.resnets = {}
        for i in range(self.resnet_layer_count):
            fc1 = nn.Dense(self.sample_embedding_length)
            fc2 = nn.Dense(self.sample_embedding_length)
            self.resnets[i] = (fc1, fc2)
        self.sample_resize_fc = nn.Dense(self.sample_embedding_length)
        self.final_act = nn.Dense(1)

    def __call__(self, inputs, samples):
        embedding = self.occnet_encoder(inputs)

        return self.occnet_decoder(embedding, samples, self.apply_sigmoid)

    def batched_cbn_layer(self, shape_embedding, sample_embeddings):
        """Applies conditional batch norm to a batch of sample embeddings.

        The batch norm values are conditioned on shape embedding.

        Args:
        shape_embedding: Tensor with shape [batch_size, shape_embedding_length].
        sample_embeddings: Tensor with shape [batch_size, sample_count,
            sample_embedding_length].

        Returns:
        Tensor with shape [shape_embedding_length].
        """
        batch_size = shape_embedding.shape[0]
        # sample_embedding_length = sample_embeddings.shape[2]

        beta = self.beta_fc(shape_embedding)
        gamma = self.gamma_fc(shape_embedding)

        batch_mean, batch_variance = jnp.mean(sample_embeddings, axis=(1, 2)), jnp.var(
            sample_embeddings, axis=(1, 2)
        )
        ensure_shape(batch_mean, [batch_size])
        reduced_batch_mean = jnp.mean(batch_mean)
        ensure_shape(batch_variance, [batch_size])
        reduced_batch_variance = jnp.mean(batch_variance)
        is_training = self.model_config.train
        if is_training:
            self.running_mean = 0.995 * self.running_mean + 0.005 * reduced_batch_mean
            self.running_variance = (
                0.995 * self.running_variance + 0.005 * reduced_batch_variance
            )

        denom = jnp.sqrt(self.running_variance + SQRT_EPS)
        out = jnp.expand_dims(gamma, axis=1) * jnp.divide(
            (sample_embeddings - self.running_mean), denom
        ) + jnp.expand_dims(beta, axis=1)
        return out

    def cbn_layer(self, shape_embedding, sample_embeddings, name, model_config):
        """Applies conditional batch norm to a batch of sample embeddings.

        The batch norm values are conditioned on shape embedding.

        Args:
        shape_embedding: Tensor with shape [shape_embedding_length].
        sample_embeddings: Tensor with shape [sample_count,
            sample_embedding_length].
        name: String naming the layer.
        model_config: A ModelConfig object.

        Returns:
        Tensor with shape [shape_embedding_length].
        """
        # sample_embedding_length = sample_embeddings.shape[1]
        beta = self.beta_fc(shape_embedding[None, ...]).flatten()
        gamma = self.gamma_fc(shape_embedding[None, ...]).flatten()
        batch_mean, batch_variance = jnp.mean(sample_embeddings, axis=(0, 1)), jnp.var(
            sample_embeddings, axis=(0, 1)
        )
        ensure_is_scalar(batch_mean)
        ensure_is_scalar(batch_variance)
        is_training = self.model_config.train
        if is_training:
            self.running_mean = 0.995 * self.running_mean + 0.005 * batch_mean
            self.running_variance = (
                0.995 * self.running_variance + 0.005 * batch_variance
            )

        denom = jnp.sqrt(self.running_variance + SQRT_EPS)
        out = gamma * jnp.divide((sample_embeddings - self.running_mean), denom) + beta
        return out

    def batched_occnet_resnet_layer(self, shape_embedding, sample_embeddings, fc):
        """Applies a fully connected resnet layer to the input.

        Args:
        shape_embedding: Tensor with shape [batch_size, shape_embedding_length].
        sample_embeddings: Tensor with shape [batch_size, sample_count,
            sample_embedding_length].

        Returns:
        Tensor with shape [sample_count, sample_embedding_length].
        """
        fc1, fc2 = fc
        # sample_embedding_length = sample_embeddings.shape[2]
        init_sample_embeddings = sample_embeddings
        sample_embeddings = self.batched_cbn_layer(shape_embedding, sample_embeddings)
        if self.model_config.hparams.fon == "t":
            init_sample_embeddings = sample_embeddings
        sample_embeddings = nn.relu(sample_embeddings)
        sample_embeddings = fc1(sample_embeddings)

        sample_embeddings = self.batched_cbn_layer(shape_embedding, sample_embeddings)
        sample_embeddings = nn.relu(sample_embeddings)
        sample_embeddings = fc2(sample_embeddings)
        return init_sample_embeddings + sample_embeddings

    def occnet_resnet_layer(self, shape_embedding, sample_embeddings, fc):
        """Applies a fully connected resnet layer to the input.

        Args:
        shape_embedding: Tensor with shape [shape_embedding_length].
        sample_embeddings: Tensor with shape [sample_count,
            sample_embedding_length].

        Returns:
        Tensor with shape [sample_count, sample_embedding_length].
        """
        (fc1, fc2) = fc
        # sample_embedding_length = sample_embeddings.shape[1]
        init_sample_embeddings = sample_embeddings
        sample_embeddings = self.cbn_layer(shape_embedding, sample_embeddings)
        if self.model_config.hparams.fon == "t":
            init_sample_embeddings = sample_embeddings
        sample_embeddings = nn.relu(sample_embeddings)
        sample_embeddings = fc1(sample_embeddings)
        sample_embeddings = self.cbn_layer(shape_embedding, sample_embeddings)

        sample_embeddings = nn.relu(sample_embeddings)
        sample_embeddings = fc2(sample_embeddings)
        return init_sample_embeddings + sample_embeddings

    def one_shape_occnet_decoder(self, embedding, samples, apply_sigmoid):
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
        ensure_rank(embedding, 1)
        ensure_rank(samples, 2)
        resnet_layer_count = self.model_config.hparams.orc
        sample_embeddings = self.sample_resize_fc(samples)
        for i in range(resnet_layer_count):
            sample_embeddings = self.occnet_resnet_layer(
                embedding, sample_embeddings, self.resnets[i]
            )
        sample_embeddings = self.cbn_layer(embedding, sample_embeddings)
        vals = self.final_act(sample_embeddings)
        if apply_sigmoid:
            vals = nn.sigmoid(vals)
        return vals

    def multishape_occnet_decoder(self, embedding, samples, apply_sigmoid):
        """Computes the OccNet output for the input embeddings and its sample batch.

        Args:
        embedding: Tensor with shape [batch_size, shape_embedding_length].
        samples: Tensor with shape [batch_size, sample_count, 3].
        apply_sigmoid: Boolean. Whether to apply a sigmoid layer to the final linear
            activations.
        model_config: A ModelConfig object.

        Returns:
        Tensor with shape [sample_count, 1].
        """
        ensure_rank(embedding, 2)
        ensure_rank(samples, 3)
        batch_size, sample_count = samples.shape[0:2]
        resnet_layer_count = self.model_config.hparams.orc
        sample_embeddings = self.sample_resize_fc(samples)
        ensure_shape(
            sample_embeddings, [batch_size, sample_count, self.sample_embedding_length]
        )
        for i in range(resnet_layer_count):
            sample_embeddings = self.batched_occnet_resnet_layer(
                embedding, sample_embeddings, self.resnets[i]
            )
        sample_embeddings = self.batched_cbn_layer(embedding, sample_embeddings)
        vals = self.final_act(sample_embeddings)
        if apply_sigmoid:
            vals = nn.sigmoid(vals)
        return vals

    def occnet_decoder(self, embedding, samples, apply_sigmoid):
        """Computes the OccNet output for the input embedding and its sample batch.

        Args:
        embedding: Tensor with shape [batch_size, shape_embedding_length].
        samples: Tensor with shape [batch_size, sample_count, 3].
        apply_sigmoid: Boolean. Whether to apply a sigmoid layer to the final linear
            activations.

        Returns:
        Tensor with shape [batch_size, sample_count, 1].
        """
        if self.model_config.hparams.hyo == "t":
            samples = math_util.nerfify(samples, 10, flatten=True, interleave=True)
            sample_len = 60
        else:
            sample_len = 3
        if self.model_config.hparams.dd == "t":
            print(
                "BID0: Running SS Occnet Decoder with input shapes embedding=%s, samples=%s",
                repr(embedding.shape),
                repr(samples.shape),
            )
            ensure_shape(embedding, [1, -1])
            ensure_shape(samples, [1, -1, 3])
            vals = self.one_shape_occnet_decoder(
                remove_batch_dim(embedding), remove_batch_dim(samples), apply_sigmoid
            )
            return add_batch_dim(vals)
        batch_size, embedding_length = embedding.shape
        ensure_shape(embedding, [batch_size, embedding_length])
        ensure_shape(samples, [batch_size, -1, sample_len])
        # Debugging:
        vals = self.multishape_occnet_decoder(embedding, samples, apply_sigmoid)
        return vals

    def occnet_encoder(self, inputs):
        ensure_rank(inputs, 5)
        # todo
        return net_util.inputs_to_feature_vector(inputs, 256, self.model_config)
